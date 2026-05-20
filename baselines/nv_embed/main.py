import os
# 1. Set memory configuration BEFORE importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import csv
import torch
import gc
import sys
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoModel
from ..utils.read import load_corpus, load_questions
from ..utils.generate import generate_answer_vllm

def setup_retriever_model():
    print("Loading NV-Embed-v2 model...")
    
    # Use bfloat16 for H100
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    try:
        model = AutoModel.from_pretrained(
            'nvidia/NV-Embed-v2', 
            trust_remote_code=True, 
            torch_dtype=dtype,
            device_map={"": 0}
        )
    except Exception as e:
        print("\nCRITICAL ERROR: Could not load model. You might not have enough free GPU memory.")
        print("If vLLM is running on this same GPU, you must stop it or run this script on a different GPU.")
        print(f"Error details: {e}")
        sys.exit(1)
        
    print("Model loaded.")
    return model

def encode_corpus_ultra_safe(model, corpus_texts):
    """
    Encodes with batch_size=1 to minimize memory usage.
    """
    print(f"Encoding {len(corpus_texts)} corpus chunks (Ultra-Safe Mode)...")
    
    all_embeddings = []
    
    # We process 1 by 1. 
    # It is slower, but it prevents the OOM caused by large activation spikes.
    
    for text in tqdm(corpus_texts):
        try:
            with torch.no_grad():
                # Encode single sample
                batch_emb = model.encode(
                    [text], 
                    batch_size=1, 
                    instruction="", 
                    max_length=4096
                )
                
                # Move to CPU immediately
                all_embeddings.append(batch_emb.cpu())
            
            # Clean up potential leftovers
            del batch_emb
            
        except torch.OutOfMemoryError:
            print("\nOOM encountered on a specific chunk! Clearing cache and retrying...")
            torch.cuda.empty_cache()
            gc.collect()
            # Skip this chunk or retry (here we skip to prevent crash loop)
            continue

    # Periodically clear cache during the loop if you want, 
    # but doing it per-item overhead is high.
    # We do a big clear at the end.
    torch.cuda.empty_cache()
    gc.collect()

    print("Concatenating embeddings on CPU...")
    if not all_embeddings:
        raise ValueError("No embeddings were generated!")
        
    corpus_embeddings = torch.cat(all_embeddings, dim=0)
    return corpus_embeddings

def retrieve_safe(query_embedding, corpus_embeddings_cpu, k, device):
    """Memory-safe retrieval."""
    scores = []
    chunk_size = 50000 
    num_docs = corpus_embeddings_cpu.shape[0]
    
    with torch.no_grad():
        for i in range(0, num_docs, chunk_size):
            corpus_chunk_cpu = corpus_embeddings_cpu[i : i + chunk_size]
            corpus_chunk_gpu = corpus_chunk_cpu.to(device)
            
            batch_scores = torch.matmul(query_embedding, corpus_chunk_gpu.transpose(0, 1))
            scores.append(batch_scores.cpu())
            
            del corpus_chunk_gpu
            # Clean up after every retrieval block
            torch.cuda.empty_cache()
    
    all_scores = torch.cat(scores, dim=1)
    top_k_scores, top_k_indices = torch.topk(all_scores, k=k)
    return top_k_indices[0].tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--questions", type=str, required=True)
    parser.add_argument("--output", type=str, default="rag_results_nvembed.json")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--api_base", type=str, default="http://localhost:4322/v1")
    parser.add_argument("--model_id", type=str, default="qwen2_5_32b")
    args = parser.parse_args()

    # 1. Load Data
    corpus_texts = load_corpus(args.corpus, get_chunks=False, return_as_list=True)
    data = load_questions(args.questions)
    
    # 2. Setup Model
    retriever_model = setup_retriever_model()
    
    # 3. Encode (Batch Size = 1)
    corpus_embeddings_cpu = encode_corpus_ultra_safe(retriever_model, corpus_texts)
    
    # Final cleanup before Inference loop
    torch.cuda.empty_cache()
    gc.collect()

    # 4. Connect to vLLM
    client = OpenAI(base_url=args.api_base, api_key="EMPTY")
    query_instruction = "Given a question, retrieve passages that answer the question"
    model_device = retriever_model.device

    print("Starting Inference...")
    for entry in tqdm(data):
        with torch.no_grad():
            question = entry['question']
            query_embedding = retriever_model.encode(
                [question], 
                instruction=query_instruction, 
                max_length=4096
            )
        
        top_indices = retrieve_safe(query_embedding, corpus_embeddings_cpu, args.k, model_device)
        top_k_docs = [corpus_texts[idx] for idx in top_indices]
        answer, justification = generate_answer_vllm(client, args.model_id, question, top_k_docs, combine=True)
        
        # Store result
        entry["model_response"] = answer
        entry["model_justification"] = justification
        entry["retrieved_context"] = top_k_docs

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    main()