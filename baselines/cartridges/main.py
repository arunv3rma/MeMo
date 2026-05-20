import argparse
import json
import sys
import os
import requests
from pathlib import Path
import yaml
import shutil
import torch
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../data_synthesis_pipeline'))
from bcp_data_utils import load_questions_with_evidence_docs
from musique_data_utils import load_questions_with_evidence_docs_musique

parser = argparse.ArgumentParser(description="Cartridges inference")
parser.add_argument("--questions", type=str, required=True, help="Path to questions JSONL file")
parser.add_argument("--max_questions", type=int, default=None, help="Number of valid questions to process (100, 200, or 300)")
parser.add_argument("--output", type=str, default="rag_results.json", help="Path to save output JSON")
parser.add_argument("--dataset", type=str, default="100papers_speculative_decoding", help="Dataset name")
parser.add_argument("--file_path", type=str, default="", help="File path identifier for cartridges")
parser.add_argument("--need_to_move_cartridges", action='store_true', help="Indicate this if the cartridges are newly trained and not moved to tokasaurus/cartridges yet")
parser.add_argument("--port", type=int, default=10223, help="Tokasaurus server port")
parser.add_argument("--seed", type=int, default=1, help="Seed forwarded to the cartridge server for reproducibility.")

args = parser.parse_args()


_CARTRIDGES_ROOT = Path(__file__).resolve().parent
CARTRIDGES_DIR = _CARTRIDGES_ROOT / "tokasaurus" / "cartridges"
DATASET_NAME = args.dataset
FILE_PATH = args.file_path
TRAIN_DATA_DIR = _CARTRIDGES_ROOT / "output" / FILE_PATH

def shard_cartridge(input_file, num_shards=2):
    print(f"Loading {input_file}...")

    data = torch.load(input_file, map_location="cpu", weights_only=False)

    tp0_data = {}
    tp1_data = {}

    # Inspect keys to ensure we are targeting the right tensors
    # In 'cartridges', weights are often inside a list or dict structure
    for key, val in data.items():
        tp0_list = []
        tp1_list = []
        for tensor in val:
            # Sharding logic: split heads (dim 1)
            num_heads = tensor.shape[1]
            mid = num_heads // 2
            tp0_list.append(tensor[:, :mid, ...].contiguous())
            tp1_list.append(tensor[:, mid:, ...].contiguous())
        tp0_data[key] = tp0_list
        tp1_data[key] = tp1_list

    # Save the shards
    base_path = input_file.replace(".pt", "")
    torch.save(tp0_data, f"{base_path}.tp0.pt")
    torch.save(tp1_data, f"{base_path}.tp1.pt")

    # Delete the original .pt file after sharding
    os.remove(input_file)
    print(f"Deleted original {input_file}")


def move_cartridges():
    # Ensure destination exists
    CARTRIDGES_DIR.mkdir(parents=True, exist_ok=True)

    # Search for all cache_last.pt files in the nested structure
    # Pattern: cat1_papers/{timestamp}/{uuid}/cache_last.pt
    for cache_path in TRAIN_DATA_DIR.glob("*/*/cache_last.pt"):
        config_path = cache_path.parent / "config.yaml"

        if not config_path.exists():
            print(f"Skipping: No config.yaml found for {cache_path}")
            continue

        try:
            # Parse the YAML to find the original paper name
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Extract the source path (e.g., /.../paper_9.json)
            text_source = config.get('kv_cache_initializer', {}).get('text_source', '')

            if not text_source:
                print(f"Skipping: Could not find text_source in {config_path}")
                continue

            # 1. Get the original paper name (e.g., paper_9)
            original_filename = Path(text_source).stem

            # 2. Define the new folder name (e.g., paper_9_cat1_papers.pt)
            folder_name = f"{original_filename}_{DATASET_NAME}".replace(".", "_")
            target_folder_path = CARTRIDGES_DIR / folder_name

            # 3. Create this specific folder
            target_folder_path.mkdir(parents=True, exist_ok=True)

            # 4. Define the final file path inside that folder
            destination_path = target_folder_path / "cartridge.pt"

            # Move and rename
            shutil.copy(str(cache_path), str(destination_path))
            print(f"Copied and Nested: {cache_path} -> {destination_path}")

            # Copy config.yaml file from the training run directory
            shutil.copy(str(config_path), str(target_folder_path))
            print(f"Copied config file to {target_folder_path}.")

            # Shard the original .pt file into 2 for Qwen2.5-32B-Instruct, served on 2 GPUs
            shard_cartridge(str(destination_path), num_shards=2)
            print(f"Successfully sharded {destination_path}.\n")

        except Exception as e:
            print(f"Error processing {cache_path}: {e}")

def generate_answer_cartridges(question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Output the answer in JSON format with strictly one string field: 'answer'. Do not wrap the JSON in markdown code blocks or backticks."},
        {"role": "user", "content": f"Question: {question}"}
    ]

    cartridges = []
    for f in os.listdir(CARTRIDGES_DIR):
        if DATASET_NAME in f:
            print(f"Loading cartridge folder: {f}")
            cartridges.append({
                "id": f,
                "source": "local",
            })

    try:
        completion = requests.post(f"http://localhost:{args.port}/custom/cartridge/chat/completions", json={
                'model': "Qwen/Qwen2.5-32B-Instruct",
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.7,
                "seed": args.seed,
                "cartridges": cartridges,
            })

        print("Response Status Code: ", completion.status_code)
        print("Response Content: ", completion.text)
        output = completion.json()["choices"][0]["message"]["content"]

        # Remove markdown code blocks if present
        if output.startswith("```"):
            output = output.split("```")[1]
            if output.startswith("json"):
                output = output[4:]
            output = output.strip()

        # Try direct JSON parsing
        try:
            parsed = json.loads(output)
            # Create a lowercase key mapping to handle case-insensitive lookups
            lowercase_keys = {k.lower(): v for k, v in parsed.items()}
            answer = str(lowercase_keys.get("answer", "N/A"))
            justification = str(lowercase_keys.get("justification", "N/A"))
            print("[Answer] ", answer)
            print("[Justification] ", justification)
            return answer, justification
        except json.JSONDecodeError:
            print("[Warning] Direct JSON parsing failed, attempting robust extraction...")

            # Fallback: Try to extract using simpler regex that doesn't get stuck
            answer = "N/A"
            justification = "N/A"

            # Extract answer - match content between quotes after "answer" key
            answer_match = re.search(r'"(?:answer|Answer)"\s*:\s*"([^"]*)"', output)
            if answer_match:
                answer = answer_match.group(1)

            # Extract justification - match content between quotes after "justification" key
            justification_match = re.search(r'"(?:justification|Justification)"\s*:\s*"([^"]*)"', output)
            if justification_match:
                justification = justification_match.group(1)

            # If extraction didn't work, try alternative patterns
            if answer == "N/A":
                # Try to find any content that looks like an answer
                answer_alt = re.search(r'answer["\']?\s*[:=]\s*["\']([^"\']*)["\']', output, re.IGNORECASE)
                if answer_alt:
                    answer = answer_alt.group(1)

            if justification == "N/A":
                # Try to find any content that looks like justification
                justif_alt = re.search(r'justification["\']?\s*[:=]\s*["\']([^"\']*)["\']', output, re.IGNORECASE)
                if justif_alt:
                    justification = justif_alt.group(1)

            print("[Answer] ", answer)
            print("[Justification] ", justification)
            return answer, justification

    except Exception as e:
        print("[Error] querying cartridges: ", str(e))
        return "N/A", "N/A"

def main():
    # 1. Move cartridges from output dir to cartridges/tokasaurus/cartridges dir
    if args.need_to_move_cartridges:
        move_cartridges()

    # 2. Load Questions using bcp_data_utils
    if "bcp" in args.dataset.lower():
        print(f"\nLoading questions from {args.questions} (max_questions={args.max_questions})...")
        data = load_questions_with_evidence_docs(
            args.questions,
            max_valid_questions=args.max_questions,
        )
        print(f"Loaded {len(data)} questions.")
    elif "musique" in args.dataset.lower():
        print(f"\nLoading musique questions from {args.questions} (max_questions={args.max_questions})...")
        data = load_questions_with_evidence_docs_musique(
            args.questions, max_num_questions=args.max_questions
        )
        print(f"Loaded {len(data)} questions.")
    else:
    # 2. Load Questions from NQA question file
        nqa_path = "baselines/data/nqa_question.json"
        print(f"\nLoading questions from {nqa_path}...")
        with open(nqa_path, "r") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} questions.")

    # 3. QA Loop
    print("Starting Cartridges inference...")
    for i, entry in enumerate(data):
        print(f"\n=== Question {i+1}/{len(data)} (id: {entry['question_no']}) ===")
        question = entry['question']
        answer, _ = generate_answer_cartridges(question)

        # Store result
        entry["model_response"] = answer
        entry["retrieved_context"] = "N/A"

    # 4. Save Output
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\nDone! Results saved to {args.output}")
    print(f"Total questions processed: {len(data)}")

if __name__ == "__main__":
    main()
