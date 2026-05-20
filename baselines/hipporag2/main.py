import argparse
import json
import csv
import os
from hipporag import HippoRAG
from ..utils.read import load_corpus, load_questions

def main():
    parser = argparse.ArgumentParser(description="HippoRAG with NV-Embed-v2 and vLLM")
    parser.add_argument("--corpus", type=str, required=True, help="Path to corpus JSON file")
    parser.add_argument("--questions", type=str, required=True, help="Path to questions file")
    parser.add_argument("--output", type=str, default="rag_results_hippo.json", help="Path to save output")
    parser.add_argument("--k", type=int, default=3, help="Number of passages to retrieve")
    
    # vLLM arguments
    parser.add_argument("--api_base", type=str, default="http://localhost:4324/v1", 
                        help="vLLM server URL")
    parser.add_argument("--model_id", type=str, default="qwen2_5_32b", 
                        help="vLLM model name")
    
    args = parser.parse_args()

    # 1. Load Data
    docs = load_corpus(args.corpus, get_chunks=False, return_as_list=True)

    data = load_questions(args.questions)
    print(f"Loaded {len(docs)} documents and {len(data)} questions.")

    # 2. Setup HippoRAG
    # We create a specific directory for this run to avoid overwriting other indices
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Initializing HippoRAG with model: {args.model_id} and Embedding: nvidia/NV-Embed-v2")
    
    # Initialize HippoRAG with the vLLM endpoint
    hipporag = HippoRAG(
        save_dir=save_dir,
        llm_model_name=args.model_id,
        embedding_model_name="nvidia/NV-Embed-v2",
        llm_base_url=args.api_base
    )

    # 3. Indexing
    # HippoRAG builds a knowledge graph from the docs. This step can take time.
    print("Indexing corpus (Graph Construction)...")
    hipporag.index(docs=docs)

    # 4. Retrieval & QA
    print(f"Running Retrieval (k={args.k}) and QA...")
    
    # Step A: Retrieve
    # retrieval_results usually contains the retrieved nodes/docs and scores
    questions = [entry['question'] for entry in data]
    retrieval_results = hipporag.retrieve(queries=questions, num_to_retrieve=args.k)
    print("len(retrieval_results): ", len(retrieval_results))
    print("Sample retrieval result: ", retrieval_results[0] if len(retrieval_results) > 0 else "N/A")
    
    # Step B: QA
    # Pass the retrieval results to the QA module
    # HippoRAG handles the iteration internally for efficiency
    qa_results = hipporag.rag_qa(retrieval_results)
    print("len(qa_results[0]): ", len(qa_results[0]))
    print("qa_results: ", qa_results)

    # 5. Format and Save Output
    # We attempt to format the output to match previous scripts: {question, answer, retrieved_context}
    # Note: The exact structure of 'qa_results' depends on the HippoRAG version, 
    # but typically it aligns with the input queries.
    for i, entry in enumerate(data):
        qa_result = qa_results[0][i]
        output = qa_result.answer
        context = retrieval_results[i].docs
        
        # Check if justification is present in the output. 
        # It usually comes in the form of "Thought: [justification] Answer: [answer] or just "[justification] Answer: [answer]"
        if "Answer:" not in output or output.split("Answer:")[0].strip() == "":
            print(f"[ERROR OUTPUT] No justification in the model output:\n{output}")
        justification = output.split("Answer:")[0].strip() if "Answer:" in output else "N/A"
        answer = output.split("Answer:")[1].strip() if "Answer:" in output else output.strip()


        entry["model_response"] = answer
        entry["model_justification"] = justification
        entry["retrieved_context"] = context

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Done! Results saved to {args.output}")

if __name__ == "__main__":
    main()