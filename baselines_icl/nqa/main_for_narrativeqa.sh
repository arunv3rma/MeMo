#!/bin/bash

# --- OpenRouter models ---
MODELS=(
    "google/gemini-3-flash-preview"
    "google/gemini-2.5-flash-lite"
)
MAX_CONCURRENT=30
MAX_TOKENS=512
SPLIT=valid

for MODEL in "${MODELS[@]}"; do
    echo "=== NarrativeQA (full story) | model: $MODEL ==="
    python main_for_narrativeqa.py \
        --model "$MODEL" \
        --max_concurrent $MAX_CONCURRENT \
        --max_tokens $MAX_TOKENS \
        --split $SPLIT \
        --max_questions 500 \
        --max_docs 10
        # --max_questions 100  # uncomment to cap for testing
done

# --- vLLM (local server) ---
# VLLM_PORT=4325
# echo "=== NarrativeQA (full story, vLLM port $VLLM_PORT) ==="
# python main_for_narrativeqa.py \
#     --base_url "http://localhost:$VLLM_PORT/v1" \
#     --model auto \
#     --split $SPLIT \
#     --max_concurrent 10 \
#     --max_tokens $MAX_TOKENS \
#     --max_docs 10
#     # --model Qwen/Qwen2.5-14B-Instruct  # override if auto-detect fails
