"""HippoRAG2 NarrativeQA inference-only against an existing index, using local
vLLM Qwen2.5-32B for both retrieval-time graph queries and rag_qa generation.

Assumes you already ran indexing and have a save_dir with
`openie_results_ner_qwen2_5_32b.json` + `qwen2_5_32b_nvidia_NV-Embed-v2/`.

Adds per-question token logging and writes a `<output>_summary.json` with
total / average prompt/completion tokens.
"""
import os
import sys
import argparse
import json
import asyncio
import threading
import time
import random
from datetime import datetime
from tqdm.asyncio import tqdm as async_tqdm
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

from hipporag import HippoRAG

from .main_for_nqa import load_nqa_questions, load_nqa_corpus
from .main_for_musique_split_llm import chunk_corpus


def process_question(hipporag, entry, args, idx, lock):
    question = entry['question']
    initial_k = args.k
    current_k = initial_k
    min_k = 1
    max_retries = 5

    for attempt in range(max_retries):
        try:
            with lock:
                retrieval_results = hipporag.retrieve(queries=[question], num_to_retrieve=current_k)
                if not retrieval_results:
                    raise Exception("No retrieval results")
                retrieved_docs = retrieval_results[0].docs
                qa_results = hipporag.rag_qa(retrieval_results)
                if not qa_results or len(qa_results) < 3:
                    raise Exception("No QA results")
                queries_solutions, _all_response, all_metadata = qa_results[0], qa_results[1], qa_results[2]
                if not queries_solutions:
                    raise Exception("Empty QA solutions")
                qa_result = queries_solutions[0]
                output = qa_result.answer
                meta = all_metadata[0] if all_metadata else {}

            retrieved_context = [
                {'docid': f'hippo_{idx}_{i}', 'text': t} for i, t in enumerate(retrieved_docs)
            ]

            if "Answer:" in output:
                justification, answer = output.split("Answer:", 1)
                justification, answer = justification.strip(), answer.strip()
            else:
                justification, answer = "N/A", output.strip()

            prompt_toks = int(meta.get('prompt_tokens', 0) or 0)
            completion_toks = int(meta.get('completion_tokens', 0) or 0)

            entry["model_response"] = answer
            entry["model_justification"] = justification
            entry["retrieved_context"] = retrieved_context
            entry["k_used"] = current_k
            entry["k_initial"] = initial_k
            entry["k_attempts"] = attempt + 1
            entry["prompt_tokens"] = prompt_toks
            entry["completion_tokens"] = completion_toks
            entry["total_tokens"] = prompt_toks + completion_toks
            return True

        except Exception as e:
            error_msg = str(e)
            is_token_error = any(kw in error_msg.lower() for kw in
                                 ['token', 'length', 'context', 'maximum', 'too long'])
            is_borrow_error = 'already borrowed' in error_msg.lower()
            if is_token_error and current_k > min_k:
                current_k = max(min_k, current_k - 2) if current_k > 4 else max(min_k, current_k - 1)
                continue
            if is_borrow_error:
                time.sleep(random.uniform(0.1, 0.5))
                continue
            entry.update({
                "model_response": None, "model_justification": None,
                "retrieved_context": None, "k_used": 0,
                "k_initial": initial_k, "k_attempts": attempt + 1,
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "error": error_msg,
            })
            return False

    entry.update({
        "model_response": None, "model_justification": None,
        "retrieved_context": None, "k_used": 0,
        "k_initial": initial_k, "k_attempts": max_retries,
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "error": "Exhausted retries",
    })
    return False


async def main_async(args, hipporag, data, lock):
    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def runner(idx, entry):
        async with semaphore:
            return await asyncio.to_thread(process_question, hipporag, entry, args, idx, lock)

    tasks = [runner(i, e) for i, e in enumerate(data)]
    results = []
    for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="QA"):
        results.append(await coro)
    return sum(results), len(results) - sum(results)


def main():
    p = argparse.ArgumentParser(description="HippoRAG2 NQA inference-only with vLLM Qwen32 + token logging")
    p.add_argument("--corpus", type=str, required=True)
    p.add_argument("--questions", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True,
                   help="Existing HippoRAG index dir (must contain openie_results_ner_<model>.json)")
    p.add_argument("--max_concurrent", type=int, default=4)
    p.add_argument("--k", type=int, default=9)
    p.add_argument("--embedding_model_name", type=str, default="nvidia/NV-Embed-v2")
    p.add_argument("--api_base", type=str, default="http://localhost:4330/v1")
    p.add_argument("--model_id", type=str, default="qwen2_5_32b")
    p.add_argument("--api_key", type=str, default="EMPTY")
    args = p.parse_args()

    os.environ["OPENAI_API_KEY"] = args.api_key

    print(f"Loading questions from {args.questions}")
    data = load_nqa_questions(args.questions)
    print(f"Loaded {len(data)} questions")

    print(f"Loading corpus from {args.corpus}")
    corpus_texts, corpus_docids = load_nqa_corpus(args.corpus)
    corpus_texts, corpus_docids = chunk_corpus(corpus_texts, corpus_docids)

    print(f"HippoRAG save_dir: {args.save_dir}")
    hipporag = HippoRAG(
        save_dir=args.save_dir,
        llm_model_name=args.model_id,
        embedding_model_name=args.embedding_model_name,
        llm_base_url=args.api_base,
    )
    # Loads/registers the persisted graph for retrieval. Idempotent on existing index.
    hipporag.index(docs=corpus_texts)

    lock = threading.Lock()
    processed, skipped = asyncio.run(main_async(args, hipporag, data, lock))
    print(f"Processed: {processed}, Failed: {skipped}")

    successful, failed = [], []
    for entry in data:
        rec = {
            'question_no': entry.get('question_no'),
            'question': entry.get('question'),
            'groundtruth': entry.get('groundtruth'),
            'gold_docs': entry.get('gold_docs', []),
            'evidence_docs': entry.get('evidence_docs', []),
            'model_response': entry.get('model_response'),
            'model_justification': entry.get('model_justification'),
            'retrieved_context': entry.get('retrieved_context'),
            'k_used': entry.get('k_used'),
            'k_initial': entry.get('k_initial'),
            'k_attempts': entry.get('k_attempts'),
            'total_evidence_tokens': entry.get('total_evidence_tokens'),
            'evidence_doc_count': entry.get('evidence_doc_count'),
            'prompt_tokens': entry.get('prompt_tokens', 0),
            'completion_tokens': entry.get('completion_tokens', 0),
            'total_tokens': entry.get('total_tokens', 0),
        }
        if 'error' in entry:
            rec['error'] = entry['error']
        (successful if rec['model_response'] is not None else failed).append(rec)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(successful, f, indent=4, ensure_ascii=False)
    if failed:
        failed_path = args.output.replace('.json', '_failed.json')
        with open(failed_path, 'w', encoding='utf-8') as f:
            json.dump(failed, f, indent=4, ensure_ascii=False)
        print(f"Failed: {failed_path}")

    n_success = len(successful)
    sum_prompt = sum(r['prompt_tokens'] for r in successful)
    sum_completion = sum(r['completion_tokens'] for r in successful)
    sum_total = sum_prompt + sum_completion
    summary = {
        'model_id': args.model_id,
        'api_base': args.api_base,
        'save_dir': args.save_dir,
        'num_questions': len(data),
        'num_successful': n_success,
        'num_failed': len(failed),
        'total_prompt_tokens': sum_prompt,
        'total_completion_tokens': sum_completion,
        'total_tokens': sum_total,
        'avg_prompt_tokens_per_question': (sum_prompt / n_success) if n_success else 0,
        'avg_completion_tokens_per_question': (sum_completion / n_success) if n_success else 0,
        'avg_total_tokens_per_question': (sum_total / n_success) if n_success else 0,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }
    summary_path = args.output.replace('.json', '_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    print(f"\n=== Token Usage Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"Saved {n_success} successful, {len(failed)} failed to {args.output}")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
