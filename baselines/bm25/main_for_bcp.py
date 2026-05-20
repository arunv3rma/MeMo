import argparse
import json
import csv
import bm25s
import re
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
from openai import AsyncOpenAI
from ..utils.generate import generate_answer_vllm_async
import os, sys
# UPDATED IMPORTS - use the new loader functions
from ..utils.read import load_corpus_from_jsonl, load_questions_with_evidence_docs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../data_synthesis_pipeline'))
from bcp_data_utils import load_only_query_related_docs_with_negatives

def setup_bm25(corpus_texts):
    """Initializes and indexes the BM25 retriever."""
    print("Tokenizing corpus and indexing BM25...")
    
    # Tokenize corpus
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")
    
    # Create and index retriever
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    
    return retriever

async def process_question(client, model_id, entry, retriever, corpus_texts, corpus_docids, args, idx):
    """Process a single question asynchronously with retry logic for token limits."""
    # Retrieve
    question = entry['question']
    query_tokens = bm25s.tokenize(question)
    
    # Use args.k as initial k (fixed value from command line)
    initial_k = args.k
    
    print(f"\nidx: {idx} - initial k to be used: {initial_k}")
    
    # Try with decreasing k values until success or k becomes too small
    k = initial_k
    min_k = 1  # Minimum k to try
    max_retries = 5  # Maximum number of retries
    
    for attempt in range(max_retries):
        try:
            print(f"idx:{idx} - Attempt {attempt + 1}: trying with k={k}")
            
            # Retrieve with current k
            retrieved_docs, scores = retriever.retrieve(query_tokens, corpus=corpus_texts, k=k)
            
            # retrieved_docs is shape (1, k)
            top_k_chunks = retrieved_docs[0].tolist()
            
            # Get the indices of retrieved documents to map back to docids
            # BM25 returns the actual text, we need to find their indices
            retrieved_indices = []
            for doc_text in top_k_chunks:
                try:
                    idx_in_corpus = corpus_texts.index(doc_text)
                    retrieved_indices.append(idx_in_corpus)
                except ValueError:
                    # Should not happen, but handle gracefully
                    retrieved_indices.append(-1)
            
            # Generate via vLLM asynchronously
            answer = await generate_answer_vllm_async(
                client, model_id, question, top_k_chunks
            )

            # Success! Store result with docids
            # Store as list of {docid, text} dicts
            retrieved_with_docids = []
            for corpus_idx, doc_text in zip(retrieved_indices, top_k_chunks):
                if corpus_idx >= 0:
                    retrieved_with_docids.append({
                        'docid': corpus_docids[corpus_idx],
                        'text': doc_text
                    })
                else:
                    # Fallback if index not found
                    retrieved_with_docids.append({
                        'docid': 'unknown',
                        'text': doc_text
                    })
            
            entry["model_response"] = answer
            entry["retrieved_context"] = retrieved_with_docids
            entry["k_used"] = k
            entry["k_initial"] = initial_k  # Track what we started with
            entry["k_attempts"] = attempt + 1  # Track how many attempts needed
            
            print(f"idx:{idx} - Success with k={k} (initial k={initial_k})")
            return True  # Indicates processed
            
        except Exception as e:
            error_msg = str(e)
            print(f"idx:{idx} - Error with k={k}: {error_msg}")
            
            # Check if it's a token limit error (adjust this based on your actual error messages)
            is_token_error = any(keyword in error_msg.lower() for keyword in 
                               ['token', 'length', 'context', 'maximum', 'too long'])
            
            if is_token_error and k > min_k:
                # Reduce k and retry
                # Strategy: reduce by half, or by 1 if k is small
                k = max(min_k, k - 2) if k > 4 else max(min_k, k - 1)
                print(f"idx:{idx} - Reducing k to {k} and retrying...")
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

async def main_async(args, corpus_texts, corpus_docids, data, retriever):
    """Async main function for RAG loop."""
    # Setup vLLM Async Client
    print(f"Connecting to vLLM at {args.api_base}...")
    client = AsyncOpenAI(
        base_url=args.api_base,
        api_key=args.api_key,
    )
    
    # RAG Loop with concurrency control
    print("Starting RAG inference...")
    
    # Process questions with limited concurrency
    semaphore = asyncio.Semaphore(args.max_concurrent)
    
    async def process_with_semaphore(idx, entry):
        async with semaphore:
            return await process_question(client, args.model_id, entry, retriever, corpus_texts, corpus_docids, args, idx)
    
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

def main():
    parser = argparse.ArgumentParser(description="RAG with BM25s and vLLM Server (Async)")
    parser.add_argument("--corpus", type=str, required=True, help="Path to corpus JSONL file")
    parser.add_argument("--questions", type=str, required=True, help="Path to questions JSONL file")
    parser.add_argument("--output", type=str, default="rag_results.json", help="Path to save output JSON")
    parser.add_argument("--max_questions", type=int, default=None, help="Number of valid questions to process")
    parser.add_argument("--max_concurrent", type=int, default=64, help="Maximum concurrent API requests")
    parser.add_argument("--k", type=int, default=9, help="Top k documents to search for")
    
    # vLLM specific arguments
    parser.add_argument("--api_base", type=str, default="http://localhost:4322/v1", 
                        help="vLLM server URL (e.g., http://localhost:4322/v1)")
    parser.add_argument("--model_id", type=str, default="qwen2_5_32b", 
                        help="The model name being served by vLLM")
    parser.add_argument("--api_key", type=str, default="EMPTY",
                        help="API key for the OpenAI-compatible endpoint. Use EMPTY for local vLLM, "
                             "or pass an OpenRouter key when --api_base points at OpenRouter.")
    parser.add_argument("--refl_trace", type=str, default=None,
                        help="Path to reflection traces JSON file (QA pairs)")
    parser.add_argument("--mode", type=str, default="docs_only",
                        choices=["docs_only", "docs_with_refl", "refl_only"],
                        help="Corpus mode: docs_only (default), docs_with_refl (append QA pairs to docs), refl_only (answers-only as docs)")
    parser.add_argument("--include_negatives", action="store_true",
                        help="Restrict corpus to query-related docs + pre-selected negatives "
                             "(uses load_only_query_related_docs_with_negatives).")
    parser.add_argument("--neg_n", type=int, default=1, choices=[1, 2],
                        help="Negative-doc multiplier (1=N, 2=2N). Only used with --include_negatives.")
    # --- OpenRouter support (additive; default behavior unchanged) ---
    parser.add_argument("--provider", type=str, default="vllm", choices=["vllm", "openrouter"],
                        help="LLM provider. 'vllm' (default) uses local vLLM via --api_base/--model_id. "
                             "'openrouter' rewrites api_base/api_key/model_id to OpenRouter.")
    parser.add_argument("--openrouter_api_key", type=str, default=None,
                        help="OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.")
    parser.add_argument("--openrouter_base_url", type=str, default="https://openrouter.ai/api/v1",
                        help="OpenRouter OpenAI-compatible base URL.")
    parser.add_argument("--openrouter_model_id", type=str, default="google/gemini-3-flash-preview",
                        help="OpenRouter model ID (e.g. google/gemini-3-flash-preview).")

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

    # ========== UPDATED: Load corpus and questions separately ==========

    # 1a. Load Corpus
    print(f"Loading corpus from {args.corpus}...")
    if args.include_negatives:
        print(f"Corpus loader: load_only_query_related_docs_with_negatives (neg_n={args.neg_n})")
        corpus_texts, corpus_docids = load_only_query_related_docs_with_negatives(
            args.corpus, args.questions, max_valid_questions=args.max_questions, neg_n=args.neg_n
        )
    else:
        corpus_texts, corpus_docids = load_corpus_from_jsonl(args.corpus)
    print(f"Corpus loaded: {len(corpus_texts)} unique documents")
    
    # 1b. Load Questions with evidence docs already attached
    print(f"\nLoading questions from {args.questions} (target: {args.max_questions} questions)...")
    data = load_questions_with_evidence_docs(
        args.questions,
        max_valid_questions=args.max_questions,
    )
    
    print(f"\nLoaded {len(data)} questions ready for processing.")
    print(f"Using k={args.k} for BM25 retrieval")
    
    # ========== End of updates ==========

    # 2. Setup Retriever
    retriever = setup_bm25(corpus_texts)

    # 3. Run async RAG loop
    processed_count, skipped_count = asyncio.run(main_async(args, corpus_texts, corpus_docids, data, retriever))

    print(f"Processed count: {processed_count}")
    print(f"Skipped count: {skipped_count}")
    
    # After main_async() completes, filter and transform the data
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
        
        # Add error field if present
        if 'error' in entry:
            entry_copy['error'] = entry['error']
        
        # Note: negative_docs is intentionally excluded
        # retrieved_context already has the {docid, text} format from process_question
        
        # Separate successful vs failed
        if entry_copy.get('model_response') is not None:
            successful_data.append(entry_copy)
        else:
            failed_data.append(entry_copy)
    
    # 4. Save Outputs
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

if __name__ == "__main__":
    main()