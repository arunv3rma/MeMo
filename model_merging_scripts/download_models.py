#!/usr/bin/env python3
"""
Download fine-tuned MEMORY model checkpoints from HuggingFace.

Fill in `MODELS` with the (display_name, hf_repo_id) pairs you want to download,
and set `OUTPUT_DIR` to a writable local path.
"""

from huggingface_hub import snapshot_download

# === Configure before running =================================================
MODELS = {
    "memory_1": "<hf-user>/<finetuned-model-on-corpus-1>",
    "memory_2": "<hf-user>/<finetuned-model-on-corpus-2>",
    # "memory_3": "<hf-user>/<finetuned-model-on-corpus-3>",
}
OUTPUT_DIR = "<local-output-dir>"   # e.g. "./downloads"
# ==============================================================================

for name, repo_id in MODELS.items():
    print(f"Downloading {name} ({repo_id})...")
    path = snapshot_download(repo_id=repo_id, local_dir=f"{OUTPUT_DIR}/{name}")
    print(f"  Saved to: {path}\n")

print("Done.")
