"""NV-Embed-v2 retrieval + LLM generation for BrowseComp+.

Combines:
  - BCP-specific data loaders (bcp_data_utils.load_questions_with_evidence_docs,
    load_only_query_related_docs) — same as bm25/main_for_bcp.py.
  - NV-Embed-v2 retrieval (encode_corpus + cosine top-k) — same as
    nv_embed/main_for_musique.py.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import re
import asyncio
import random
import gc
import sys
import torch
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from openai import AsyncOpenAI
from transformers import AutoModel

from ..utils.generate import generate_answer_vllm_async

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../data_synthesis_pipeline'))
from bcp_data_utils import (
    load_questions_with_evidence_docs,
    load_only_query_related_docs,
    load_only_query_related_docs_with_negatives,
)


def setup_retriever_model():
    print("Loading NV-Embed-v2...")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    try:
        model = AutoModel.from_pretrained(
            'nvidia/NV-Embed-v2',
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map={"": 0},
        )
    except Exception as e:
        print(f"CRITICAL: failed to load NV-Embed-v2: {e}")
        sys.exit(1)
    print("Model loaded.")
    return model


def encode_corpus_ultra_safe(model, corpus_texts):
    print(f"Encoding {len(corpus_texts)} corpus chunks (batch=1)...")
    all_embeddings = []
    for text in tqdm(corpus_texts):
        try:
            with torch.no_grad():
                emb = model.encode([text], batch_size=1, instruction="", max_length=4096)
                all_embeddings.append(emb.cpu())
            del emb
        except torch.OutOfMemoryError:
            print("OOM on chunk, clearing cache and skipping...")
            torch.cuda.empty_cache()
            gc.collect()
            continue
    torch.cuda.empty_cache()
    gc.collect()
    if not all_embeddings:
        raise ValueError("No embeddings generated!")
    return torch.cat(all_embeddings, dim=0)


def retrieve_safe(query_embedding, corpus_embeddings_cpu, k, device):
    scores = []
    chunk_size = 50000
    num_docs = corpus_embeddings_cpu.shape[0]
    with torch.no_grad():
        for i in range(0, num_docs, chunk_size):
            chunk = corpus_embeddings_cpu[i:i + chunk_size].to(device)
            scores.append(torch.matmul(query_embedding, chunk.transpose(0, 1)).cpu())
            del chunk
            torch.cuda.empty_cache()
    all_scores = torch.cat(scores, dim=1)
    k = min(k, all_scores.shape[1])
    _scores, top_k_indices = torch.topk(all_scores, k=k)
    return top_k_indices[0].tolist()


async def process_question(client, model_id, entry, retriever_model, corpus_embeddings_cpu,
                           corpus_texts, corpus_docids, args, idx, encode_lock, model_device):
    question = entry['question']
    initial_k = args.k
    k = initial_k
    min_k = 1
    max_retries = 8

    query_instruction = "Given a question, retrieve passages that answer the question"
    async with encode_lock:
        with torch.no_grad():
            query_embedding = retriever_model.encode(
                [question], instruction=query_instruction, max_length=4096
            )

    for attempt in range(max_retries):
        try:
            print(f"idx:{idx} - Attempt {attempt + 1}: k={k}")
            top_indices = retrieve_safe(query_embedding, corpus_embeddings_cpu, k, model_device)
            top_k_chunks = [corpus_texts[i] for i in top_indices]

            answer = await generate_answer_vllm_async(
                client, model_id, question, top_k_chunks
            )

            if not answer or not answer.strip():
                raise Exception("Empty model response")

            retrieved_with_docids = [
                {'docid': corpus_docids[i], 'text': corpus_texts[i]}
                for i in top_indices
            ]

            entry["model_response"] = answer
            entry["retrieved_context"] = retrieved_with_docids
            entry["k_used"] = k
            entry["k_initial"] = initial_k
            entry["k_attempts"] = attempt + 1
            return True

        except Exception as e:
            error_msg = str(e)
            print(f"idx:{idx} - Error with k={k}: {error_msg}")
            is_rate_limit = '429' in error_msg or 'rate limit' in error_msg.lower()
            if is_rate_limit and attempt < max_retries - 1:
                m = re.search(r"'X-RateLimit-Reset':\s*'(\d+)'", error_msg)
                wait_s = 5.0 * (2 ** attempt) + random.uniform(0, 2.0)
                if m:
                    import time as _t
                    reset_s = int(m.group(1)) / 1000.0 - _t.time()
                    wait_s = max(2.0, min(reset_s + random.uniform(0, 2.0), 90.0))
                wait_s = min(wait_s, 90.0)
                print(f"idx:{idx} - Rate limited, sleeping {wait_s:.1f}s before retry...")
                await asyncio.sleep(wait_s)
                continue
            is_token_error = any(s in error_msg.lower() for s in
                                 ['token', 'length', 'context', 'maximum', 'too long'])
            if is_token_error and k > min_k:
                k = max(min_k, k - 2) if k > 4 else max(min_k, k - 1)
                continue
            entry.update({
                "model_response": None,
                "retrieved_context": None, "k_used": 0,
                "k_initial": initial_k, "k_attempts": attempt + 1,
                "error": error_msg,
            })
            return False

    entry.update({
        "model_response": None,
        "retrieved_context": None, "k_used": 0,
        "k_initial": initial_k, "k_attempts": max_retries,
        "error": "Exhausted retries",
    })
    return False


async def main_async(args, retriever_model, corpus_embeddings_cpu, corpus_texts, corpus_docids, data):
    print(f"Connecting to LLM endpoint at {args.api_base}...")
    client = AsyncOpenAI(base_url=args.api_base, api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.max_concurrent)
    encode_lock = asyncio.Lock()
    model_device = retriever_model.device

    async def run(idx, entry):
        async with semaphore:
            return await process_question(client, args.model_id, entry, retriever_model,
                                          corpus_embeddings_cpu, corpus_texts, corpus_docids,
                                          args, idx, encode_lock, model_device)

    tasks = [run(i, e) for i, e in enumerate(data)]
    results = []
    for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
        results.append(await coro)
    return sum(results), len(results) - sum(results)


def main():
    parser = argparse.ArgumentParser(description="NV-Embed RAG for BrowseComp+ (Async)")
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--questions", type=str, required=True)
    parser.add_argument("--output", type=str, default="rag_results_nvembed_bcp.json")
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--max_concurrent", type=int, default=16)
    parser.add_argument("--k", type=int, default=9)
    parser.add_argument("--api_base", type=str, default="http://localhost:4327/v1")
    parser.add_argument("--model_id", type=str, default="qwen2_5_32b")
    parser.add_argument("--api_key", type=str, default="EMPTY",
                        help="API key for the OpenAI-compatible endpoint. Use EMPTY for local vLLM, "
                             "or pass an OpenRouter key when --api_base points at OpenRouter.")
    parser.add_argument("--include_negatives", action="store_true",
                        help="Include pre-selected negative docs in the corpus (uses "
                             "load_only_query_related_docs_with_negatives).")
    parser.add_argument("--neg_n", type=int, default=1, choices=[1, 2],
                        help="Negative-doc multiplier (1=N, 2=2N). Only used with --include_negatives.")
    parser.add_argument("--provider", type=str, default="vllm", choices=["vllm", "openrouter"],
                        help="LLM provider. 'vllm' (default) or 'openrouter'.")
    parser.add_argument("--openrouter_api_key", type=str, default=None,
                        help="OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.")
    parser.add_argument("--openrouter_base_url", type=str, default="https://openrouter.ai/api/v1")
    parser.add_argument("--openrouter_model_id", type=str, default="google/gemini-3-flash-preview")
    args = parser.parse_args()

    if args.provider == "openrouter":
        or_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        if not or_key:
            raise ValueError(
                "--provider=openrouter requires --openrouter_api_key or the OPENROUTER_API_KEY env var"
            )
        args.api_base = args.openrouter_base_url
        args.model_id = args.openrouter_model_id
        args.api_key = or_key
        print(f"\n[OpenRouter] Using base_url={args.api_base}, model={args.model_id}")

    print(f"\nLoading questions from {args.questions} (target: {args.max_questions} questions)...")
    data = load_questions_with_evidence_docs(args.questions, max_valid_questions=args.max_questions)
    print(f"Loaded {len(data)} questions.")

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
    corpus_texts, corpus_docids = corpus_loader(
        args.corpus, args.questions, **loader_kwargs
    )
    print(f"Corpus loaded: {len(corpus_texts)} unique documents")
    print(f"Using k={args.k} for NV-Embed retrieval")

    retriever_model = setup_retriever_model()
    corpus_embeddings_cpu = encode_corpus_ultra_safe(retriever_model, corpus_texts)
    torch.cuda.empty_cache()
    gc.collect()

    processed, skipped = asyncio.run(main_async(
        args, retriever_model, corpus_embeddings_cpu, corpus_texts, corpus_docids, data
    ))
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

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
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
