#!/bin/bash

# black box models on openrouter
MODELS=(
    "google/gemini-3-flash-preview"
)
MAX_CONCURRENT=15
MAX_TOKENS=2048

for MODEL in "${MODELS[@]}"; do
    echo "=== MuSiQue (no context) | model: $MODEL ==="
    python main_for_musique_no_context.py \
        --model "$MODEL" \
        --max_concurrent $MAX_CONCURRENT \
        --max_tokens $MAX_TOKENS
done

# vLLM
VLLM_BASE_URL="http://localhost:4325/v1"
MAX_CONCURRENT=100

echo "=== MuSiQue (no context) | vLLM @ $VLLM_BASE_URL ==="
python main_for_musique_no_context.py \
    --model auto \
    --base_url "$VLLM_BASE_URL" \
    --max_concurrent $MAX_CONCURRENT \
    --max_tokens $MAX_TOKENS
