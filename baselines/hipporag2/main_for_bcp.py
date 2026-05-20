import os
import sys
# Disable Flash Attention 2 to avoid import errors
# os.environ['VLLM_ATTENTION_BACKEND'] = 'TORCH_SDPA'
import argparse
import json
import os
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../data_synthesis_pipeline'))
from bcp_data_utils import (
    load_questions_with_evidence_docs,
    load_corpus_from_jsonl,
    load_only_query_related_docs,
    load_only_query_related_docs_with_negatives,
)

def process_question(hipporag, entry, args, idx, lock, llm_client, llm_model_id):
    """Process a single question synchronously: HippoRAG retrieve + canonical QA via shared async-style sync helper."""
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
                if args.use_internal_rag_qa:
                    # Library's built-in QA: original prompt template + "Answer:" parser.
                    _t_qa_start = time.perf_counter()
                    qa_results = hipporag.rag_qa(retrieval_results)
                    _t_qa_internal = time.perf_counter() - _t_qa_start
                    if not qa_results or len(qa_results[0]) == 0:
                        raise Exception("No QA results from rag_qa()")
                    raw_output = qa_results[0][0].answer or ""
                    answer = (raw_output.split("Answer:", 1)[1].strip()
                              if "Answer:" in raw_output else raw_output.strip())

            _t_qa_canonical = 0.0
            if not args.use_internal_rag_qa:
                # QA via canonical prompt — runs OUTSIDE the HippoRAG lock since it's an HTTP call.
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
            return True  # Indicates processed
            
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {e}".strip()
            try:
                _t_total_err = time.perf_counter() - _t_q_start
                print(f"[TIMING] q_idx={idx} retrieve_s={locals().get('_t_ret', 0.0):.3f} qa_s={(locals().get('_t_qa_internal', 0.0) + locals().get('_t_qa_canonical', 0.0)):.3f} total_s={_t_total_err:.3f} k={current_k} status=error")
            except Exception:
                pass
            print(f"idx:{idx} - Error with k={current_k}: {error_msg}")
            print(f"idx:{idx} - Traceback:\n{traceback.format_exc()}")

            cls = type(e).__name__.lower()
            # Check if it's a token limit error. BadRequestError from vLLM/OpenAI servers
            # is virtually always a context-overflow in this pipeline; treat its empty-str
            # variant as a token error so we still reduce k.
            is_token_error = (
                any(keyword in error_msg.lower() for keyword in
                    ['token', 'length', 'context', 'maximum', 'too long'])
                or 'badrequest' in cls
            )

            # Check if it's a thread-safety error (already borrowed)
            is_borrow_error = 'already borrowed' in error_msg.lower()
            
            if is_token_error and current_k > min_k:
                # Reduce k and retry
                current_k = max(min_k, current_k - 2) if current_k > 4 else max(min_k, current_k - 1)
                print(f"idx:{idx} - Reducing k to {current_k} and retrying...")
                continue
            elif is_borrow_error:
                # Retry without changing k (thread safety issue, not k issue)
                wait_time = random.uniform(0.1, 0.5)  # Random backoff
                print(f"idx:{idx} - Thread safety error, waiting {wait_time:.2f}s and retrying...")
                time.sleep(wait_time)
                continue
            else:
                # Non-token error or k is already at minimum - fail
                print(f"idx:{idx} - Failed after {attempt + 1} attempts. Error: {error_msg}")
                entry["model_response"] = None
                entry["retrieved_context"] = None
                entry["k_used"] = 0
                entry["k_initial"] = initial_k
                entry["k_attempts"] = attempt + 1
                entry["error"] = error_msg
                return False

    # Exhausted all retries
    print(f"idx:{idx} - Exhausted all {max_retries} retry attempts")
    entry["model_response"] = None
    entry["retrieved_context"] = None
    entry["k_used"] = 0
    entry["k_initial"] = initial_k
    entry["k_attempts"] = max_retries
    entry["error"] = "Exhausted retries"
    return False

def decompose_question(llm_client, model_id, question, tokenizer):
    """Use the LLM to decompose a question into sub-questions. Returns (sub_questions, token_counts)."""
    prompt = f"""Break down the following question into simpler, self-contained sub-questions that would help gather the information needed to answer the original question.

Requirements:
- Each sub-question must be fully self-contained (no pronouns or references to other questions)
- Focus on the most critical information needed
- Generate 3-5 sub-questions

Question: {question}

Output your response as JSON with the following format:
{{
    "sub_questions": [
        "fully self-contained sub-question 1",
        "fully self-contained sub-question 2"
    ]
}}

Do not wrap the JSON in markdown code blocks or backticks."""

    messages = [{"role": "user", "content": prompt}]
    input_tokens = len(tokenizer.encode(prompt))

    try:
        response = llm_client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )
        content = response.choices[0].message.content
        output_tokens = len(tokenizer.encode(content))

        # Parse JSON
        stripped = content.strip()
        if stripped.startswith("```"):
            import re
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        parsed = json.loads(stripped.strip())
        sub_questions = parsed.get('sub_questions', [])

        token_counts = {"decompose_input": input_tokens, "decompose_output": output_tokens}
        return sub_questions, token_counts
    except Exception as e:
        print(f"[Error] Question decomposition failed: {e}")
        return [], {"decompose_input": input_tokens, "decompose_output": 0}


def synthesize_final_answer(llm_client, model_id, original_question, qa_pairs, tokenizer):
    """Use the LLM to synthesize a final answer from all sub-question answers. Returns (answer, justification, token_counts)."""
    qa_block = "\n\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in qa_pairs])

    prompt = f"""You are given an original question and a set of related questions with their answers.
Use all the gathered information to provide a comprehensive answer to the original question.

## Original Question
{original_question}

## Gathered Information
{qa_block}

## Instructions
Synthesize the information above to answer the original question.
If the gathered information is insufficient, provide the best answer possible based on what is available.

Output your response as JSON:
{{
    "answer": "<your synthesized answer>",
    "justification": "<brief reasoning for your answer>"
}}

Do not wrap the JSON in markdown code blocks or backticks."""

    messages = [{"role": "user", "content": prompt}]
    input_tokens = len(tokenizer.encode(prompt))

    try:
        response = llm_client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=1024,
            temperature=0.3,
        )
        content = response.choices[0].message.content
        output_tokens = len(tokenizer.encode(content))

        stripped = content.strip()
        if stripped.startswith("```"):
            import re
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        parsed = json.loads(stripped.strip())

        answer = str(parsed.get('answer', 'N/A'))
        justification = str(parsed.get('justification', 'N/A'))
        token_counts = {"synthesis_input": input_tokens, "synthesis_output": output_tokens}
        return answer, justification, token_counts
    except Exception as e:
        print(f"[Error] Final synthesis failed: {e}")
        return "N/A", "N/A", {"synthesis_input": input_tokens, "synthesis_output": 0}


def process_question_multi_turn(hipporag, entry, args, idx, lock, llm_client, llm_model_id, tokenizer):
    """Multi-turn: decompose -> retrieve+QA per sub-question -> synthesize."""
    question = entry['question']
    initial_k = args.k
    token_log = {"sub_questions": [], "total_input_tokens": 0, "total_output_tokens": 0}

    # Step 1: Decompose original question into sub-questions
    print(f"\nidx:{idx} --- Multi-Turn Step 1: Decomposing question ---")
    sub_questions, decompose_tokens = decompose_question(llm_client, llm_model_id, question, tokenizer)
    token_log["total_input_tokens"] += decompose_tokens["decompose_input"]
    token_log["total_output_tokens"] += decompose_tokens["decompose_output"]
    token_log["decompose"] = decompose_tokens

    # Build full question list: [original_question] + sub_questions
    all_questions = [question] + sub_questions
    print(f"idx:{idx} - Will process {len(all_questions)} questions (1 original + {len(sub_questions)} sub-questions)")

    # Step 2: Retrieve + QA for each question
    print(f"idx:{idx} --- Multi-Turn Step 2: Retrieve + QA per question ---")
    qa_pairs = []
    all_retrieved_context = []

    for q_idx, q in enumerate(all_questions):
        label = "ORIGINAL" if q_idx == 0 else f"SUB-Q {q_idx}"
        print(f"idx:{idx} - [{label}]: {q}")
        sub_q_tokens = {"question": q, "label": label}

        # Count input tokens for this question
        q_input_tokens = len(tokenizer.encode(q))
        sub_q_tokens["qa_input_tokens"] = q_input_tokens

        try:
            with lock:
                retrieval_results = hipporag.retrieve(queries=[q], num_to_retrieve=initial_k)
                if not retrieval_results or len(retrieval_results) == 0:
                    raise Exception("No retrieval results")

                retrieved_docs = retrieval_results[0].docs
                qa_results = hipporag.rag_qa(retrieval_results)

                if not qa_results or len(qa_results[0]) == 0:
                    raise Exception("No QA results")

                qa_result = qa_results[0][0]
                output = qa_result.answer

            # Count output tokens
            q_output_tokens = len(tokenizer.encode(output))
            sub_q_tokens["qa_output_tokens"] = q_output_tokens

            # Parse answer
            if "Answer:" in output:
                answer = output.split("Answer:")[1].strip()
            else:
                answer = output.strip()

            qa_pairs.append({"question": q, "answer": answer})

            # Collect retrieved context
            for doc_idx, doc_text in enumerate(retrieved_docs):
                all_retrieved_context.append({
                    'docid': f'hippo_{idx}_{q_idx}_{doc_idx}',
                    'text': doc_text,
                    'source_question': q,
                })

            print(f"idx:{idx} - [{label}] Answer: {answer[:150]}...")

        except Exception as e:
            print(f"idx:{idx} - [{label}] Error: {e}")
            q_output_tokens = 0
            sub_q_tokens["qa_output_tokens"] = 0
            qa_pairs.append({"question": q, "answer": "N/A"})

        token_log["total_input_tokens"] += sub_q_tokens.get("qa_input_tokens", 0)
        token_log["total_output_tokens"] += sub_q_tokens.get("qa_output_tokens", 0)
        token_log["sub_questions"].append(sub_q_tokens)

    # Step 3: Synthesize final answer
    print(f"idx:{idx} --- Multi-Turn Step 3: Synthesizing final answer ---")
    final_answer, justification, synthesis_tokens = synthesize_final_answer(
        llm_client, llm_model_id, question, qa_pairs, tokenizer
    )
    token_log["total_input_tokens"] += synthesis_tokens["synthesis_input"]
    token_log["total_output_tokens"] += synthesis_tokens["synthesis_output"]
    token_log["synthesis"] = synthesis_tokens

    print(f"idx:{idx} - Final answer: {final_answer[:150]}...")
    print(f"idx:{idx} - Token totals: input={token_log['total_input_tokens']}, output={token_log['total_output_tokens']}")

    # Store results
    entry["model_response"] = final_answer
    entry["model_justification"] = justification
    entry["retrieved_context"] = all_retrieved_context
    entry["k_used"] = initial_k
    entry["k_initial"] = initial_k
    entry["k_attempts"] = 1
    entry["multi_turn_qa_pairs"] = qa_pairs
    entry["multi_turn_token_log"] = token_log

    return final_answer != "N/A"


async def main_async(args, hipporag, data, lock, llm_client, llm_model_id):
    """Async main function for HippoRAG loop."""
    print("Starting HippoRAG inference...")

    # Process questions with limited concurrency
    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def process_with_semaphore(idx, entry):
        async with semaphore:
            # HippoRAG is synchronous, so we use to_thread to run it concurrently
            return await asyncio.to_thread(
                process_question, hipporag, entry, args, idx, lock,
                llm_client, llm_model_id,
            )
    
    # Create tasks for all questions
    tasks = [process_with_semaphore(idx, entry) for idx, entry in enumerate(data)]
    
    # Run tasks with progress bar
    results = []
    for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing questions"):
        result = await coro
        results.append(result)
    
    processed_count = sum(results)
    skipped_count = len(results) - processed_count
    
    return processed_count, skipped_count

async def main_async_multi_turn(args, hipporag, data, lock, llm_client, llm_model_id, tokenizer):
    """Async main function for multi-turn HippoRAG loop."""
    print("Starting HippoRAG multi-turn inference...")

    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def process_with_semaphore(idx, entry):
        async with semaphore:
            return await asyncio.to_thread(
                process_question_multi_turn, hipporag, entry, args, idx, lock,
                llm_client, llm_model_id, tokenizer
            )

    tasks = [process_with_semaphore(idx, entry) for idx, entry in enumerate(data)]

    results = []
    for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing questions (multi-turn)"):
        result = await coro
        results.append(result)

    processed_count = sum(results)
    skipped_count = len(results) - processed_count

    return processed_count, skipped_count


def main():
    parser = argparse.ArgumentParser(description="HippoRAG with NV-Embed-v2 and vLLM (Async)")
    parser.add_argument("--corpus", type=str, required=True, help="Path to corpus JSONL file")
    parser.add_argument("--questions", type=str, required=True, help="Path to questions JSONL file")
    parser.add_argument("--refl_trace", type=str, default=None, help="Path to reflection traces JSON file")
    parser.add_argument("--output", type=str, default="rag_results_hippo.json", help="Path to save output")
    parser.add_argument("--max_questions", type=int, default=None, help="Number of questions to process")
    parser.add_argument("--max_concurrent", type=int, default=32, help="Maximum concurrent processing")
    parser.add_argument("--k", type=int, default=9, help="Top k documents to retrieve")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory for HippoRAG index/graph output. If omitted, a timestamped "
                             "dir under ./baselines/hipporag2/output_bcp/<corpus_tag>_<ts> is used.")

    # vLLM arguments
    parser.add_argument("--api_base", type=str, default="http://localhost:4324/v1", 
                        help="vLLM server URL")
    parser.add_argument("--model_id", type=str, default="qwen2_5_32b", 
                        help="vLLM model name")
    parser.add_argument("--embedding_model_name", type=str, default="nvidia/NV-Embed-v2",
                        help="Embedding model name")
    parser.add_argument("--mode", type=str, default="docs_only",
                        choices=["docs_only", "docs_with_refl", "refl_only"],
                        help="Corpus mode: docs_only (default), docs_with_refl (append QA answers to docs), refl_only (answers-only as docs)")
    parser.add_argument("--multi_turn", action='store_true',
                        help="Enable multi-turn inference: decompose question into sub-questions, retrieve+QA each, then synthesize")

    # --- OpenRouter support (additive; default behavior unchanged) ---
    parser.add_argument("--provider", type=str, default="vllm", choices=["vllm", "openrouter"],
                        help="LLM provider. 'vllm' (default) uses local vLLM via --api_base/--model_id. "
                             "'openrouter' rewrites api_base/model_id to OpenRouter.")
    parser.add_argument("--openrouter_api_key", type=str, default=None,
                        help="OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.")
    parser.add_argument("--openrouter_base_url", type=str, default="https://openrouter.ai/api/v1",
                        help="OpenRouter OpenAI-compatible base URL.")
    parser.add_argument("--openrouter_model_id", type=str, default="google/gemini-3-flash-preview",
                        help="OpenRouter model ID (e.g. google/gemini-3-flash-preview).")
    parser.add_argument("--include_negatives", action="store_true",
                        help="Include pre-selected negative docs in the corpus (uses "
                             "load_only_query_related_docs_with_negatives).")
    parser.add_argument("--neg_n", type=int, default=1, choices=[1, 2],
                        help="Negative-doc multiplier (1=N, 2=2N). Only used with --include_negatives.")
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed forwarded to the LLM API for stochastic generation reproducibility.")
    parser.add_argument("--api_key", type=str, default="EMPTY",
                        help="API key for the OpenAI-compatible endpoint. EMPTY for vLLM.")
    parser.add_argument("--use_internal_rag_qa", action="store_true",
                        help="Use HippoRAG's built-in rag_qa() with the library's original "
                             "prompt template (rag_qa_<dataset>, falling back to rag_qa_musique) "
                             "instead of our canonical JSON-output prompt via generate_answer_vllm_sync.")

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

    if args.mode in ("docs_with_refl", "refl_only") and not args.refl_trace:
        parser.error(f"--refl_trace is required when --mode={args.mode}")

    # ========== Load corpus and questions ==========

    # 1a. Load Questions
    print(f"\nLoading questions from {args.questions} (target: {args.max_questions} questions)...")
    data = load_questions_with_evidence_docs(
        args.questions,
        max_valid_questions=args.max_questions,
    )
    print(f"Loaded {len(data)} questions ready for processing.")

    # 1b. Load reflection traces if needed
    doc_qa_map = {}
    if args.refl_trace:
        print(f"\nLoading reflection traces from {args.refl_trace}...")
        with open(args.refl_trace, 'r', encoding='utf-8') as f:
            refl_data = json.load(f)

        for entry in refl_data['qa_pairs_cache']:
            if entry.get('type') == 'crossdoc':
                continue
            doc_id = entry['doc_id']
            for qa in entry.get('qa_pairs', []):
                doc_qa_map.setdefault(doc_id, []).append(qa['answer'])

        print(f"Loaded answers for {len(doc_qa_map)} unique docs (crossdoc entries skipped)")

    # 1c. Build corpus based on mode
    if args.mode == "refl_only":
        # Each doc is the concatenated answers for a given doc_id
        print("\nMode: refl_only — building corpus from reflection trace answers only")
        corpus_texts = []
        corpus_docids = []
        for doc_id, answers in doc_qa_map.items():
            corpus_texts.append("\n\n".join(answers))
            corpus_docids.append(doc_id)
        print(f"Corpus built: {len(corpus_texts)} docs from reflection traces")
    else:
        # Load original corpus docs
        print(f"\nLoading corpus from {args.corpus}...")
        corpus_loader = (
            load_only_query_related_docs_with_negatives if args.include_negatives
            else load_only_query_related_docs
        )
        print(f"Corpus loader: {corpus_loader.__name__}")
        loader_kwargs = {"max_valid_questions": args.max_questions}
        if args.include_negatives:
            loader_kwargs["neg_n"] = args.neg_n
            print(f"  neg_n={args.neg_n}")
        corpus_texts, corpus_docids = corpus_loader(args.corpus, args.questions, **loader_kwargs)
        print(f"Corpus loaded: {len(corpus_texts)} unique documents")

        if args.mode == "docs_with_refl":
            # Append QA answers to the end of each matching doc
            print("\nMode: docs_with_refl — appending reflection trace answers to docs")
            appended_count = 0
            for i, docid in enumerate(corpus_docids):
                if docid in doc_qa_map:
                    answers_block = "\n\n".join(doc_qa_map[docid])
                    corpus_texts[i] = corpus_texts[i] + "\n\n--- Related QA Pairs ---\n" + answers_block
                    appended_count += 1
            print(f"Appended answers to {appended_count}/{len(corpus_docids)} corpus docs")

    print(f"Using k={args.k} for HippoRAG retrieval")

    # ========== End of corpus loading ==========

    # 1c. Chunk large documents to avoid exceeding vLLM max context length
    # Reserve tokens for prompt template (system prompt, NER instructions, few-shot examples)
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
            # Split into overlapping chunks
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
        save_dir = f"./baselines/hipporag2/output_bcp/{corpus_tag}_{timestamp}"
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nHippoRAG save_dir: {save_dir}")

    print(f"\nInitializing HippoRAG with model: {args.model_id} and Embedding: {args.embedding_model_name}")
    
    # Initialize HippoRAG with the vLLM endpoint
    hipporag = HippoRAG(
        save_dir=save_dir,
        llm_model_name=args.model_id,
        embedding_model_name=args.embedding_model_name,
        llm_base_url=args.api_base
    )

    # When using the library's rag_qa(), force seed/temperature into its LLM config so
    # different --seed values actually produce different outputs (lib default is temp=0).
    if args.use_internal_rag_qa and hasattr(hipporag, "llm_model"):
        try:
            hipporag.llm_model.llm_config.generate_params["seed"] = args.seed
            hipporag.llm_model.llm_config.generate_params["temperature"] = 0.7
            print(f"[internal rag_qa] generate_params override: seed={args.seed}, temperature=0.7")
        except Exception as e:
            print(f"[warn] could not override library generate_params: {e}")

    # 3. Indexing - Use the FULL corpus (not just evidence docs)
    print("\nIndexing full corpus (Graph Construction)...")
    print(f"Indexing {len(corpus_texts)} documents...")
    _t_index_start = time.perf_counter()
    hipporag.index(docs=corpus_texts)
    _t_index = time.perf_counter() - _t_index_start
    print(f"[TIMING] indexing_seconds={_t_index:.3f} n_docs={len(corpus_texts)}")
    print("Indexing complete!")

    # 4. Create thread lock for HippoRAG
    hipporag_lock = threading.Lock()

    # 5. Run async processing loop
    # Single LLM client for QA across all paths (single- and multi-turn).
    llm_client = OpenAI(
        base_url=args.api_base,
        api_key=(args.api_key if hasattr(args, "api_key") and args.api_key else "dummy"),
        timeout=120.0,
    )
    llm_model_id = args.model_id
    print(f"\nLLM client target: {args.api_base}, model={llm_model_id}, seed={args.seed}")

    _t_inf_start = time.perf_counter()
    if args.multi_turn:
        processed_count, skipped_count = asyncio.run(
            main_async_multi_turn(args, hipporag, data, hipporag_lock, llm_client, llm_model_id, tokenizer)
        )
    else:
        processed_count, skipped_count = asyncio.run(
            main_async(args, hipporag, data, hipporag_lock, llm_client, llm_model_id)
        )
    _t_inf = time.perf_counter() - _t_inf_start
    print(f"[TIMING] inference_wall_seconds={_t_inf:.3f} n_questions={len(data)} max_concurrent={args.max_concurrent}")

    print(f"\nProcessed count: {processed_count}")
    print(f"Failed count: {skipped_count}")

    # 6. Filter and save output
    successful_data = []
    failed_data = []
    
    for entry in data:
        # Prepare entry for saving (create a clean copy)
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
        
        # Add multi-turn fields if present
        if 'multi_turn_qa_pairs' in entry:
            entry_copy['multi_turn_qa_pairs'] = entry['multi_turn_qa_pairs']
        if 'multi_turn_token_log' in entry:
            entry_copy['multi_turn_token_log'] = entry['multi_turn_token_log']

        # Add error field if present
        if 'error' in entry:
            entry_copy['error'] = entry['error']

        # Note: negative_docs is intentionally excluded
        
        # Separate successful vs failed
        if entry_copy.get('model_response') is not None:
            successful_data.append(entry_copy)
        else:
            failed_data.append(entry_copy)
    
    # 7. Save Outputs
    # Save successful results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(successful_data, f, indent=4, ensure_ascii=False)
    
    # Save failed results to separate file
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

def test_qa_indexing():
    """Quick test: load 100 QA pairs and run HippoRAG indexing only to inspect entity extraction."""
    parser = argparse.ArgumentParser(description="Test HippoRAG indexing on QA-formatted data")
    parser.add_argument("--qa_data", type=str,
                        default="baselines/data/bcp_subset300_numsamplingepochs1_crossdoc_entity_combination_v3.json",
                        help="Path to QA pairs JSON file")
    parser.add_argument("--num_docs", type=int, default=100, help="Number of doc_ids worth of QA pairs to load")
    parser.add_argument("--api_base", type=str, default="http://localhost:4327/v1", help="vLLM server URL")
    parser.add_argument("--model_id", type=str, default="qwen2_5_32b", help="vLLM model name")
    parser.add_argument("--embedding_model_name", type=str, default="nvidia/NV-Embed-v2", help="Embedding model name")
    args = parser.parse_args()

    # 1. Load QA pairs
    print(f"Loading QA pairs from {args.qa_data}...")
    with open(args.qa_data, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    entries = raw['qa_pairs_cache']
    print(f"Total entries in file: {len(entries)}")

    # Group answers only by doc_id, one document per unique doc_id
    from collections import OrderedDict
    doc_answers_map = OrderedDict()
    for entry in entries:
        doc_id = entry['doc_id']
        if doc_id not in doc_answers_map:
            doc_answers_map[doc_id] = []
        for qa in entry.get('qa_pairs', []):
            doc_answers_map[doc_id].append(qa['answer'])

    # Take first N doc_ids
    selected_doc_ids = list(doc_answers_map.keys())[:args.num_docs]
    qa_docs = []
    for doc_id in selected_doc_ids:
        qa_docs.append("\n\n".join(doc_answers_map[doc_id]))

    total_answers = sum(len(doc_answers_map[d]) for d in selected_doc_ids)
    print(f"Loaded {len(qa_docs)} docs ({total_answers} answers total) for indexing")
    print(f"\n--- Sample doc (doc_id={selected_doc_ids[0]}, truncated) ---\n{qa_docs[0][:500]}\n---")

    # 2. Initialize HippoRAG
    save_dir = "./baselines/hipporag2/output_qa_indexing_test"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nInitializing HippoRAG (model={args.model_id}, embedding={args.embedding_model_name})...")
    hipporag = HippoRAG(
        save_dir=save_dir,
        llm_model_name=args.model_id,
        embedding_model_name=args.embedding_model_name,
        llm_base_url=args.api_base,
    )

    # 3. Index only
    print(f"\nIndexing {len(qa_docs)} QA docs (entity extraction)...")
    hipporag.index(docs=qa_docs)
    print("Indexing complete!")
    print(f"Output saved to {save_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test_qa_indexing":
        sys.argv.pop(1)  # remove subcommand so argparse doesn't choke
        test_qa_indexing()
    else:
        main()