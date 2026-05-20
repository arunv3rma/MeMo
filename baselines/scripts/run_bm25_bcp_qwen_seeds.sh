#!/bin/bash
set -euo pipefail


cd "$(cd "$(dirname "$0")/../.." && pwd)"

DATASET=bcp
CORPUS=baselines/data/browsecomp_plus/full_corpus_train.jsonl
QUESTIONS=baselines/data/decrypted.jsonl
CORPUS_TAG=$(basename "${CORPUS}" .jsonl)
CORPUS_TAG=$(basename "${CORPUS_TAG}" .json)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

API_BASE=http://localhost:4327/v1
MODEL_ID=qwen2_5_32b
API_KEY=EMPTY
PROVIDER_FLAGS="--provider vllm"
MAX_CONCURRENT=64

OUT_DIR=baselines/output_seeded/full_corpus_train
LOG_DIR=baselines/output_seeded/logs
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

SUMMARIES=()
for SEED in 1 2 3; do
    OUTPUT_FILE=${OUT_DIR}/bm25_qwen_seed${SEED}_${TIMESTAMP}.json
    LOG_FILE=${LOG_DIR}/bm25_${DATASET}_qwen_seed${SEED}_${TIMESTAMP}.log
    echo "=== Seed ${SEED}: output=${OUTPUT_FILE} ==="

    CUDA_VISIBLE_DEVICES=0 python -m baselines.bm25.main_for_bcp \
        --corpus "${CORPUS}" \
        --questions "${QUESTIONS}" \
    --max_questions 300 \
    --include_negatives \
    --neg_n 1 \
    --provider vllm \
    --api_base "${API_BASE}" \
    --model_id "${MODEL_ID}" \
    --api_key "${API_KEY}" \
        --max_concurrent "${MAX_CONCURRENT}" \
        --k 9 \
        --seed ${SEED} \
        --output "${OUTPUT_FILE}" \
        > "${LOG_FILE}" 2>&1

    EVAL=${OUTPUT_FILE%.json}_eval.json
    SUM=${OUTPUT_FILE%.json}_summary.json
    python baselines/utils/deepeval_via_algo_utils.py \
        --generated_file_path "${OUTPUT_FILE}" \
        --output_path "${EVAL}" \
        --summary_file_path "${SUM}"
    SUMMARIES+=("${SUM}")
done

COMBINED=${OUT_DIR}/combined_bm25_bcp_qwen_${TIMESTAMP}.json
python baselines/scripts/aggregate_seeds.py \
    --summary_files "${SUMMARIES[@]}" \
    --output "${COMBINED}"
echo "=== Combined: ${COMBINED} ==="
