import argparse
import json
import csv
import bm25s
import re
from tqdm import tqdm
from openai import OpenAI
from ..utils.read import load_corpus, load_questions
from ..utils.generate import generate_answer_vllm

def setup_bm25(corpus_texts):
    """Initializes and indexes the BM25 retriever."""
    print("Tokenizing corpus and indexing BM25...")
    
    # Tokenize corpus
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")
    
    # Create and index retriever
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    
    return retriever

def main():
    parser = argparse.ArgumentParser(description="RAG with BM25s and vLLM Server")
    parser.add_argument("--corpus", type=str, required=True, help="Path to corpus JSON file")
    parser.add_argument("--questions", type=str, required=True, help="Path to questions JSON file")
    parser.add_argument("--output", type=str, default="rag_results.json", help="Path to save output JSON")
    parser.add_argument("--k", type=int, default=3, help="Number of passages to retrieve")
    
    # vLLM specific arguments
    parser.add_argument("--api_base", type=str, default="http://localhost:4322/v1", 
                        help="vLLM server URL (e.g., http://localhost:4322/v1)")
    parser.add_argument("--model_id", type=str, default="qwen2_5_32b", 
                        help="The model name being served by vLLM")
    
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading data...")
    corpus_texts = load_corpus(args.corpus, get_chunks=False, return_as_list=True)
    data = load_questions(args.questions)
    print(f"Loaded {len(corpus_texts)} corpus chunks and {len(data)} questions.")

    # 2. Setup Retriever
    retriever = setup_bm25(corpus_texts)

    # 3. Setup vLLM Client
    print(f"Connecting to vLLM at {args.api_base}...")
    # vLLM usually requires an API key argument, but "EMPTY" works for local instances
    client = OpenAI(
        base_url=args.api_base,
        api_key="EMPTY", 
    )

    # 4. RAG Loop
    print("Starting RAG inference...")
    for entry in tqdm(data):
        # Retrieve
        question = entry['question']
        query_tokens = bm25s.tokenize(question)
        
        # Helper: bm25s returns tuple (docs, scores). 
        # We pass corpus=corpus_texts to retrieve actual text instead of IDs.
        retrieved_docs, scores = retriever.retrieve(query_tokens, corpus=corpus_texts, k=args.k)
        
        # retrieved_docs is shape (1, k)
        top_k_chunks = retrieved_docs[0].tolist()
        
        # Generate via vLLM
        answer, justification = generate_answer_vllm(client, args.model_id, question, top_k_chunks, combine=True)
        
        # Store result
        entry["model_response"] = answer
        entry["model_justification"] = justification
        entry["retrieved_context"] = top_k_chunks

    # 5. Save Output
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Done! Results saved to {args.output}")

if __name__ == "__main__":
    main()