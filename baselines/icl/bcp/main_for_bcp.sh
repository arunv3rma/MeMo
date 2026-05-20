#!/bin/bash

QUESTIONS_FILE=baselines/data/browsecomp_plus_questions.jsonl

# --- OpenRouter models ---
MODELS=(
    "google/gemini-3-flash-preview"
)
MAX_CONCURRENT=8
MAX_TOKENS=2048

for MODEL in "${MODELS[@]}"; do
    echo "=== BrowseComp+ (evidence docs) | model: $MODEL ==="
    python main_for_bcp.py \
        --model "$MODEL" \
        --questions_file "$QUESTIONS_FILE" \
        --max_concurrent $MAX_CONCURRENT \
        --max_tokens $MAX_TOKENS
        # --max_questions 50  # uncomment to cap for testing
done

# --- vLLM (local server) ---
VLLM_PORT=4325
echo "=== BrowseComp+ (evidence docs, vLLM port $VLLM_PORT) ==="
python main_for_bcp.py \
    --base_url "http://localhost:$VLLM_PORT/v1" \
    --model auto \
    --questions_file "$QUESTIONS_FILE" \
    --max_concurrent 32 \
    --max_tokens $MAX_TOKENS
