"""HippoRAG2 for MuSiQue with split LLM:

  - INDEX phase   : local vLLM (e.g. Qwen2.5-32B) — fast and free.
  - INFERENCE     : OpenRouter (e.g. Gemini 3 Flash) — used for rag_qa only.

The graph index is persisted in --save_dir; the inference HippoRAG instance is
re-constructed against the same save_dir and reuses the saved index.
"""
import os
import sys
import argparse
import json
import asyncio
import threading
from datetime import datetime
from transformers import AutoTokenizer
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

from hipporag import HippoRAG

from .main_for_musique import main_async

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../data_synthesis_pipeline'))
from musique_data_utils import (
    load_only_query_related_docs_musique,
    load_questions_with_evidence_docs_musique,
)


def patch_hipporag_retries(min_wait=4, max_wait=60, max_attempts=10, label=""):
    """Replace HippoRAG's hard-coded retry policy with longer exp-backoff.

    Used both for:
      - Index phase: tolerate vLLM restarts (set max_wait/max_attempts large enough
        to survive several minutes of downtime).
      - Inference phase: tolerate OpenRouter 429 rate limits.
    """
    import hipporag.llm.openai_gpt as _hp_llm
    from tenacity import wait_exponential as _wait_exp
    from tenacity import stop_after_attempt as _stop_after

    def _patched_wait_fixed(_seconds):
        return _wait_exp(multiplier=2, min=min_wait, max=max_wait)

    def _patched_stop_after_attempt(n):
        return _stop_after(max(n, max_attempts))

    _hp_llm.wait_fixed = _patched_wait_fixed
    _hp_llm.stop_after_attempt = _patched_stop_after_attempt
    print(f"[{label or 'patch'}] Patched retry: exp backoff {min_wait}-{max_wait}s, up to {max_attempts} attempts")


def chunk_corpus(corpus_texts, corpus_docids, max_tokens=131072 - 2048, overlap=256):
    print(f"\nChunking documents that exceed {max_tokens} tokens...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
    chunked_texts, chunked_docids, num_chunked = [], [], 0
    for text, docid in zip(corpus_texts, corpus_docids):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            chunked_texts.append(text)
            chunked_docids.append(docid)
            continue
        num_chunked += 1
        start, chunk_idx = 0, 0
        while start < len(token_ids):
            end = min(start + max_tokens, len(token_ids))
            chunked_texts.append(tokenizer.decode(token_ids[start:end], skip_special_tokens=True))
            chunked_docids.append(f"{docid}_chunk{chunk_idx}")
            chunk_idx += 1
            start = end - overlap if end < len(token_ids) else end
    print(f"Chunking: {len(corpus_texts)} -> {len(chunked_texts)} ({num_chunked} docs split)")
    return chunked_texts, chunked_docids


def main():
    p = argparse.ArgumentParser(description="HippoRAG2 MuSiQue with split index/inference LLMs")
    p.add_argument("--corpus", type=str, required=True)
    p.add_argument("--questions", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--save_dir", type=str, default=None,
                   help="HippoRAG index dir. Default: ./baselines/hipporag2/output_musique/<corpus_tag>_split_<ts>")
    p.add_argument("--max_concurrent", type=int, default=4)
    p.add_argument("--k", type=int, default=9)
    p.add_argument("--max_questions", type=int, default=None,
                   help="If set, only process the first N questions and only "
                        "load corpus docs referenced by them.")
    p.add_argument("--embedding_model_name", type=str, default="nvidia/NV-Embed-v2")

    # Index-phase LLM (local vLLM by default).
    p.add_argument("--index_api_base", type=str, default="http://localhost:4330/v1")
    p.add_argument("--index_model_id", type=str, default="qwen2_5_32b")
    p.add_argument("--index_api_key", type=str, default="EMPTY")

    # Inference-phase LLM (OpenRouter Gemini 3 Flash by default).
    p.add_argument("--inference_api_base", type=str, default="https://openrouter.ai/api/v1")
    p.add_argument("--inference_model_id", type=str, default="google/gemini-3-flash-preview")
    p.add_argument("--inference_api_key", type=str, default=None,
                   help="API key for inference LLM. Falls back to OPENROUTER_API_KEY_MIT or OPENROUTER_API_KEY env vars.")

    # Skip indexing if save_dir already has it (still re-instantiates for inference).
    p.add_argument("--skip_index", action="store_true",
                   help="Skip the indexing phase (reuse existing save_dir).")

    args = p.parse_args()

    inference_key = (args.inference_api_key
                     or os.environ.get("OPENROUTER_API_KEY_MIT")
                     or os.environ.get("OPENROUTER_API_KEY"))
    if not inference_key:
        raise SystemExit("Inference API key required: --inference_api_key or OPENROUTER_API_KEY_MIT env var")

    # Load data.
    print(f"\nLoading questions from {args.questions}...")
    data = load_questions_with_evidence_docs_musique(
        args.questions, max_num_questions=args.max_questions
    )
    print(f"Loaded {len(data)} questions.")

    print(f"\nLoading corpus from {args.corpus}...")
    corpus_texts, corpus_docids = load_only_query_related_docs_musique(
        args.corpus, args.questions, max_num_questions=args.max_questions
    )
    corpus_texts, corpus_docids = chunk_corpus(corpus_texts, corpus_docids)

    # Save dir.
    if args.save_dir is None:
        corpus_tag = os.path.splitext(os.path.basename(args.corpus))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"./baselines/hipporag2/output_musique/{corpus_tag}_split_{ts}"
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nHippoRAG save_dir: {save_dir}")

    # ---- Phase 1: INDEXING with vLLM ----
    if not args.skip_index:
        print(f"\n=== Phase 1: indexing via {args.index_api_base} (model={args.index_model_id}) ===")
        os.environ["OPENAI_API_KEY"] = args.index_api_key
        # Tolerate vLLM crashes/restarts during indexing: keep retrying for several
        # minutes per request before giving up and recording an empty triple.
        patch_hipporag_retries(min_wait=10, max_wait=120, max_attempts=30, label="index/vLLM")
        index_hipporag = HippoRAG(
            save_dir=save_dir,
            llm_model_name=args.index_model_id,
            embedding_model_name=args.embedding_model_name,
            llm_base_url=args.index_api_base,
        )
        print(f"Indexing {len(corpus_texts)} documents...")
        index_hipporag.index(docs=corpus_texts)
        print("Indexing complete.")
        # Free the index instance before instantiating the inference one.
        del index_hipporag
    else:
        print("\n=== Phase 1 skipped (--skip_index). Reusing existing save_dir. ===")

    # ---- Phase 2: INFERENCE via OpenRouter ----
    print(f"\n=== Phase 2: inference via {args.inference_api_base} (model={args.inference_model_id}) ===")
    os.environ["OPENAI_API_KEY"] = inference_key
    patch_hipporag_retries(min_wait=4, max_wait=60, max_attempts=10, label="inference/OpenRouter")

    hipporag = HippoRAG(
        save_dir=save_dir,
        llm_model_name=args.inference_model_id,
        embedding_model_name=args.embedding_model_name,
        llm_base_url=args.inference_api_base,
    )
    # Re-call index() so HippoRAG loads/registers the persisted graph for retrieval.
    # With the same save_dir + same docs, this is idempotent.
    hipporag.index(docs=corpus_texts)

    hipporag_lock = threading.Lock()

    processed, skipped = asyncio.run(main_async(args, hipporag, data, hipporag_lock))
    print(f"\nProcessed: {processed}, Failed: {skipped}")

    # Save outputs (mirrors main_for_musique.py).
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
        print(f"Failed results: {failed_path}")
    print(f"Saved {len(successful)} successful, {len(failed)} failed to {args.output}")


if __name__ == "__main__":
    main()
