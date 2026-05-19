#!/usr/bin/env python3
"""
Download Tevatron/browsecomp-plus-corpus dataset from Hugging Face and save as JSONL
"""

from datasets import load_dataset
import json
import os

# Load the dataset
print("Loading dataset from Hugging Face...")
dataset = load_dataset("Tevatron/browsecomp-plus-corpus")

print(f"Dataset loaded successfully!")
print(f"Available splits: {list(dataset.keys())}")

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Process each split in the dataset
for split_name, split_data in dataset.items():
    output_file = f"output/full_corpus_{split_name}.jsonl"
    
    print(f"\nProcessing '{split_name}' split...")
    print(f"  - Number of records: {len(split_data)}")
    
    # Check the structure of the first record
    if len(split_data) > 0:
        print(f"  - Column names: {split_data.column_names}")
        print(f"  - First record sample: {split_data[0]}")
    
    # Write to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in split_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"  - Saved to: {output_file}")

print("\n✓ Download complete! All splits saved as JSONL files.")