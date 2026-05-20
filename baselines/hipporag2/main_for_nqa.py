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
from transformers import AutoTokenizer
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

from openai import OpenAI
from hipporag import HippoRAG
from baselines.utils.generate import generate_answer_vllm_sync


def load_nqa_questions(path):
    """Load pre-processed NQA questions (JSON list already in the expected schema)."""
    with open(path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    print(f"\n=== Loading Summary ===")
    print(f"Total questions loaded: {len(questions)}")
    return questions


def load_nqa_corpus(path):
    """Load an NQA corpus JSON list. Supports both schemas:
       - {idx, title, text} (chunked corpus)
       - {docid, text}      (unchunked corpus)
    Returns (corpus_texts, corpus_docids).
    """
    with open(path, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    corpus_texts = []
    corpus_docids = []
    for entry in entries:
        docid = entry.get('idx') or entry.get('docid')
        text = entry.get('text', '')
        corpus_texts.append(text)
        corpus_docids.append(docid)

    print(f"\n=== Corpus Loading Summary ===")
    print(f"Corpus entries loaded: {len(corpus_texts)}")
    if corpus_texts:
        print(f"Sample DocID: {corpus_docids[0]}")
        print(f"Sample text (first 200 chars): {corpus_texts[0][:200]}...")

    return corpus_texts, corpus_docids


def process_question(hipporag, entry, args, idx, lock, llm_client, llm_model_id):
    """Process a single question: HippoRAG retrieve + canonical QA via shared sync helper."""
    question = entry['question']

    initial_k = args.k

    print(f"\nidx:{idx} - initial k to be used: {initial_k}")

    current_k = initial_k
    min_k = 1
    max_retries = 5

    for attempt in range(max_retries):
        try:
            print(f"idx:{idx} - Attempt {attempt + 1}: trying with k={current_k}")

            _t_q_start = time.perf_counter()
            with lock:
                _t_ret_start = time.perf_counter()
                retrieval_results = hipporag.retrieve(queries=[question], num_to_retrieve=current_k)
                _t_ret = time.perf_counter() - _t_ret_start

                if not retrieval_results or len(retrieval_results) == 0:
                    print(f"idx:{idx} - No retrieval results returned")
                    raise Exception("No retrieval results")

                retrieved_docs = retrieval_results[0].docs

                _t_qa_internal = 0.0
                if getattr(args, "use_internal_rag_qa", False):
                    _t_qa_start = time.perf_counter()
                    qa_results = hipporag.rag_qa(retrieval_results)
                    _t_qa_internal = time.perf_counter() - _t_qa_start
                    if not qa_results or len(qa_results[0]) == 0:
                        raise Exception("No QA results from rag_qa()")
                    raw_output = qa_results[0][0].answer or ""
                    answer = (raw_output.split("Answer:", 1)[1].strip()
                              if "Answer:" in raw_output else raw_output.strip())

            _t_qa_canonical = 0.0
            if not getattr(args, "use_internal_rag_qa", False):
                _t_qa_start = time.perf_counter()
                answer = generate_answer_vllm_sync(
                    llm_client, llm_model_id, question, retrieved_docs, seed=args.seed
                )
                _t_qa_canonical = time.perf_counter() - _t_qa_start
            _t_qa = _t_qa_internal + _t_qa_canonical
            _t_total = time.perf_counter() - _t_q_start
            print(f"[TIMING] q_idx={idx} retrieve_s={_t_ret:.3f} qa_s={_t_qa:.3f} total_s={_t_total:.3f} k={current_k}")

            if not answer or not answer.strip():
                raise Exception("Empty model response")

            retrieved_context = [
                {'docid': f'hippo_{idx}_{doc_idx}', 'text': doc_text}
                for doc_idx, doc_text in enumerate(retrieved_docs)
            ]

            entry["model_response"] = answer
            entry["retrieved_context"] = retrieved_context
            entry["k_used"] = current_k
            entry["k_initial"] = initial_k
            entry["k_attempts"] = attempt + 1

            print(f"idx:{idx} - Success with k={current_k} (initial k={initial_k})")
            return True

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}".strip()
            try:
                _t_total_err = time.perf_counter() - _t_q_start
                print(f"[TIMING] q_idx={idx} retrieve_s={locals().get('_t_ret', 0.0):.3f} qa_s={(locals().get('_t_qa_internal', 0.0) + locals().get('_t_qa_canonical', 0.0)):.3f} total_s={_t_total_err:.3f} k={current_k} status=error")
            except Exception:
                pass
            print(f"idx:{idx} - Error with k={current_k}: {error_msg}")

            cls = type(e).__name__.lower()
            is_token_error = (
                any(keyword in error_msg.lower() for keyword in
                    ['token', 'length', 'context', 'maximum', 'too long'])
                or 'badrequest' in cls
            )
            is_borrow_error = 'already borrowed' in error_msg.lower()

            if is_token_error and current_k > min_k:
                current_k = max(min_k, current_k - 2) if current_k > 4 else max(min_k, current_k - 1)
                print(f"idx:{idx} - Reducing k to {current_k} and retrying...")
                continue
            elif is_borrow_error:
                wait_time = random.uniform(0.1, 0.5)
                print(f"idx:{idx} - Thread safety error, waiting {wait_time:.2f}s and retrying...")
                time.sleep(wait_time)
                continue
            else:
                print(f"idx:{idx} - Failed after {attempt + 1} attempts. Error: {error_msg}")
                entry["model_response"] = None
                entry["retrieved_context"] = None
                entry["k_used"] = 0
                entry["k_initial"] = initial_k
                entry["k_attempts"] = attempt + 1
                entry["error"] = error_msg
                return False

    print(f"idx:{idx} - Exhausted all {max_retries} retry attempts")
    entry["model_response"] = None
    entry["retrieved_context"] = None
    entry["k_used"] = 0
    entry["k_initial"] = initial_k
    entry["k_attempts"] = max_retries
    entry["error"] = "Exhausted retries"
    return False


async def main_async(args, hipporag, data, lock, llm_client, llm_model_id):
    print("Starting HippoRAG inference...")

    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def process_with_semaphore(idx, entry):
        async with semaphore:
            return await asyncio.to_thread(
                process_question, hipporag, entry, args, idx, lock,
                llm_client, llm_model_id,
            )

    tasks = [process_with_semaphore(idx, entry) for idx, entry in enumerate(data)]

    results = []
    for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing questions"):
        result = await coro
        results.append(result)

    processed_count = sum(results)
    skipped_count = len(results) - processed_count

    return processed_count, skipped_count


def main():
    parser = argparse.ArgumentParser(description="HippoRAG for NarrativeQA with NV-Embed-v2 and vLLM (Async)")
    parser.add_argument("--corpus", type=str, required=True, help="Path to NQA corpus JSON file")
    parser.add_argument("--questions", type=str, required=True, help="Path to NQA questions JSON file")
    parser.add_argument("--output", type=str, default="rag_results_hippo_nqa.json", help="Path to save output")
    parser.add_argument("--max_concurrent", type=int, default=32, help="Maximum concurrent processing")
    parser.add_argument("--k", type=int, default=9, help="Top k documents to retrieve")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory for HippoRAG index/graph output. If omitted, a timestamped "
                             "dir under ./baselines/hipporag2/output_nqa/<corpus_tag>_<ts> is used.")

    parser.add_argument("--api_base", type=str, default="http://localhost:4324/v1",
                        help="vLLM server URL")
    parser.add_argument("--model_id", type=str, default="qwen2_5_32b",
                        help="vLLM model name")
    parser.add_argument("--embedding_model_name", type=str, default="nvidia/NV-Embed-v2",
                        help="Embedding model name")
    # --- OpenRouter support (additive; default behavior unchanged) ---
    parser.add_argument("--provider", type=str, default="vllm", choices=["vllm", "openrouter"],
                        help="LLM provider. 'vllm' (default) uses local vLLM via --api_base/--model_id. "
                             "'openrouter' rewrites api_base/model_id to OpenRouter.")
    parser.add_argument("--openrouter_api_key", type=str, default=None,
                        help="OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.")
    parser.add_argument("--openrouter_base_url", type=str, default="https://openrouter.ai/api/v1",
                        help="OpenRouter OpenAI-compatible base URL.")
    parser.add_argument("--openrouter_model_id", type=str, default="google/gemini-3-flash",
                        help="OpenRouter model ID (e.g. google/gemini-3-flash).")
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed forwarded to the LLM API for stochastic generation reproducibility.")
    parser.add_argument("--api_key", type=str, default="EMPTY",
                        help="API key for the OpenAI-compatible endpoint. EMPTY for vLLM.")
    parser.add_argument("--use_internal_rag_qa", action="store_true",
                        help="Use HippoRAG's built-in rag_qa() with the library's original prompt.")

    args = parser.parse_args()

    # --- OpenRouter override block (no-op when provider == 'vllm') ---
    if args.provider == "openrouter":
        or_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        if not or_key:
            raise ValueError(
                "--provider=openrouter requires --openrouter_api_key or the OPENROUTER_API_KEY env var"
            )
        # HippoRAG's underlying OpenAI client reads OPENAI_API_KEY.
        os.environ["OPENAI_API_KEY"] = or_key
        args.api_base = args.openrouter_base_url
        args.model_id = args.openrouter_model_id
        args.api_key = or_key
        print(f"\n[OpenRouter] Using base_url={args.api_base}, model={args.model_id}")

        # Monkey-patch HippoRAG's retry policy for OpenRouter rate limits.
        # Upstream hard-codes wait_fixed(1) and ~5 attempts in
        # hipporag.llm.openai_gpt.dynamic_retry_decorator, which is far too
        # short for OpenRouter's rate-limit windows. The wrapper looks up
        # `wait_fixed` / `stop_after_attempt` via module globals at call
        # time, so patching those names takes effect without re-decorating
        # the already-wrapped CacheOpenAI.infer.
        import hipporag.llm.openai_gpt as _hp_llm
        from tenacity import wait_exponential as _tenacity_wait_exponential
        from tenacity import stop_after_attempt as _tenacity_stop_after_attempt

        _OPENROUTER_RETRY_MAX_ATTEMPTS = 10
        _OPENROUTER_RETRY_MIN_WAIT = 4
        _OPENROUTER_RETRY_MAX_WAIT = 60

        def _patched_wait_fixed(_seconds):
            return _tenacity_wait_exponential(
                multiplier=2,
                min=_OPENROUTER_RETRY_MIN_WAIT,
                max=_OPENROUTER_RETRY_MAX_WAIT,
            )

        def _patched_stop_after_attempt(n):
            return _tenacity_stop_after_attempt(max(n, _OPENROUTER_RETRY_MAX_ATTEMPTS))

        _hp_llm.wait_fixed = _patched_wait_fixed
        _hp_llm.stop_after_attempt = _patched_stop_after_attempt
        print(f"[OpenRouter] Patched HippoRAG retry: exponential backoff "
              f"{_OPENROUTER_RETRY_MIN_WAIT}s-{_OPENROUTER_RETRY_MAX_WAIT}s, "
              f"up to {_OPENROUTER_RETRY_MAX_ATTEMPTS} attempts.")

    # 1a. Load Questions (all of them — no subsetting for NQA)
    print(f"\nLoading questions from {args.questions}...")
    data = load_nqa_questions(args.questions)
    print(f"Loaded {len(data)} questions ready for processing.")

    # 1b. Load corpus
    print(f"\nLoading corpus from {args.corpus}...")
    corpus_texts, corpus_docids = load_nqa_corpus(args.corpus)
    print(f"Corpus loaded: {len(corpus_texts)} documents")

    print(f"Using k={args.k} for HippoRAG retrieval")

    # 1c. Chunk large documents to avoid exceeding vLLM max context length
    PROMPT_OVERHEAD_TOKENS = 2048
    MAX_CHUNK_TOKENS = 131072 - PROMPT_OVERHEAD_TOKENS

    print(f"\nChunking documents that exceed {MAX_CHUNK_TOKENS} tokens...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")

    chunked_texts = []
    chunked_docids = []
    num_chunked = 0
    for text, docid in zip(corpus_texts, corpus_docids):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= MAX_CHUNK_TOKENS:
            chunked_texts.append(text)
            chunked_docids.append(docid)
        else:
            num_chunked += 1
            overlap = 256
            start = 0
            chunk_idx = 0
            while start < len(token_ids):
                end = min(start + MAX_CHUNK_TOKENS, len(token_ids))
                chunk_text = tokenizer.decode(token_ids[start:end], skip_special_tokens=True)
                chunked_texts.append(chunk_text)
                chunked_docids.append(f"{docid}_chunk{chunk_idx}")
                chunk_idx += 1
                start = end - overlap if end < len(token_ids) else end

    print(f"Chunking complete: {len(corpus_texts)} docs -> {len(chunked_texts)} chunks ({num_chunked} docs were split)")
    corpus_texts = chunked_texts
    corpus_docids = chunked_docids

    # 2. Setup HippoRAG save_dir (timestamped + corpus-tagged for easy maintenance)
    if args.save_dir is None:
        corpus_tag = os.path.splitext(os.path.basename(args.corpus))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"./baselines/hipporag2/output_nqa/{corpus_tag}_{timestamp}"
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nHippoRAG save_dir: {save_dir}")

    print(f"Initializing HippoRAG with model: {args.model_id} and Embedding: {args.embedding_model_name}")

    hipporag = HippoRAG(
        save_dir=save_dir,
        llm_model_name=args.model_id,
        embedding_model_name=args.embedding_model_name,
        llm_base_url=args.api_base
    )

    if args.use_internal_rag_qa and hasattr(hipporag, "llm_model"):
        try:
            hipporag.llm_model.llm_config.generate_params["seed"] = args.seed
            hipporag.llm_model.llm_config.generate_params["temperature"] = 0.7
            print(f"[internal rag_qa] generate_params override: seed={args.seed}, temperature=0.7")
        except Exception as e:
            print(f"[warn] could not override library generate_params: {e}")

    # 3. Indexing
    print("\nIndexing corpus (Graph Construction)...")
    print(f"Indexing {len(corpus_texts)} documents...")
    _t_index_start = time.perf_counter()
    hipporag.index(docs=corpus_texts)
    _t_index = time.perf_counter() - _t_index_start
    print(f"[TIMING] indexing_seconds={_t_index:.3f} n_docs={len(corpus_texts)}")
    print("Indexing complete!")

    # 4. Thread lock
    hipporag_lock = threading.Lock()

    # 5. Run async processing
    llm_client = OpenAI(
        base_url=args.api_base,
        api_key=(args.api_key if args.api_key else "dummy"),
        timeout=120.0,
    )
    llm_model_id = args.model_id
    print(f"\nLLM client target: {args.api_base}, model={llm_model_id}, seed={args.seed}")

    _t_inf_start = time.perf_counter()
    processed_count, skipped_count = asyncio.run(
        main_async(args, hipporag, data, hipporag_lock, llm_client, llm_model_id)
    )
    _t_inf = time.perf_counter() - _t_inf_start
    print(f"[TIMING] inference_wall_seconds={_t_inf:.3f} n_questions={len(data)} max_concurrent={args.max_concurrent}")

    print(f"\nProcessed count: {processed_count}")
    print(f"Failed count: {skipped_count}")

    # 6. Filter and save
    successful_data = []
    failed_data = []

    for entry in data:
        entry_copy = {
            'question_no': entry.get('question_no'),
            'question': entry.get('question'),
            'groundtruth': entry.get('groundtruth'),
            'gold_docs': entry.get('gold_docs', []),
            'evidence_docs': entry.get('evidence_docs', []),
            'model_response': entry.get('model_response'),
            'retrieved_context': entry.get('retrieved_context'),
            'k_used': entry.get('k_used'),
            'k_initial': entry.get('k_initial'),
            'k_attempts': entry.get('k_attempts'),
            'total_evidence_tokens': entry.get('total_evidence_tokens'),
            'evidence_doc_count': entry.get('evidence_doc_count')
        }

        if 'error' in entry:
            entry_copy['error'] = entry['error']

        if entry_copy.get('model_response') is not None:
            successful_data.append(entry_copy)
        else:
            failed_data.append(entry_copy)

    # 7. Save outputs
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(successful_data, f, indent=4, ensure_ascii=False)

    failed_output = args.output.replace('.json', '_failed.json')
    if failed_data:
        with open(failed_output, 'w', encoding='utf-8') as f:
            json.dump(failed_data, f, indent=4, ensure_ascii=False)

    print(f"\n=== Final Summary ===")
    print(f"Results saved to {args.output}")
    print(f"Successful: {len(successful_data)} questions")
    print(f"Failed: {len(failed_data)} questions")
    if failed_data:
        print(f"Failed results saved to {failed_output}")


if __name__ == "__main__":
    main()
