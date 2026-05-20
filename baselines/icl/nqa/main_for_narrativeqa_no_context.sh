#!/bin/bash

# --- OpenRouter models ---
MODELS=(
    "google/gemini-3-flash-preview"
)
MAX_CONCURRENT=30
MAX_TOKENS=512
SPLIT=valid

for MODEL in "${MODELS[@]}"; do
    echo "=== NarrativeQA (no context / closed-book) | model: $MODEL ==="
    python main_for_narrativeqa_no_context.py \
        --model "$MODEL" \
        --split $SPLIT \
        --max_concurrent $MAX_CONCURRENT \
        --max_tokens $MAX_TOKENS \
        --max_docs 10
done

# --- vLLM (local server) ---
VLLM_PORT=4323
echo "=== NarrativeQA (no context, vLLM port $VLLM_PORT) ==="
python main_for_narrativeqa_no_context.py \
    --base_url "http://localhost:$VLLM_PORT/v1" \
    --model auto \
    --split $SPLIT \
    --max_concurrent 100 \
    --max_tokens $MAX_TOKENS \
    --max_docs 10
