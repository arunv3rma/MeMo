#!/bin/bash
set -euo pipefail

conda activate cartridges

cd "$(cd "$(dirname "$0")/../.." && pwd)"

DATASET=musique_1000_with_negatives_8192-samples
QUESTIONS=baselines/data/musique_questions_chunks_1000.jsonl
MAX_QUESTIONS=1000
OUTPUT_DIR=baselines/output_seeded/${DATASET}
LOG_DIR=baselines/output_seeded/logs
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARIES=()

for SEED in 1 2 3; do
    OUTPUT_FILE=${OUTPUT_DIR}/cartridges_qwen_seed${SEED}_${TIMESTAMP}.json
    LOG_FILE=${LOG_DIR}/cartridges_${DATASET}_qwen_seed${SEED}_${TIMESTAMP}.log
    echo "=== Seed ${SEED}: output=${OUTPUT_FILE} ==="
    CUDA_VISIBLE_DEVICES=0 python -m baselines.cartridges.main \
        --questions "${QUESTIONS}" \
        --max_questions ${MAX_QUESTIONS} \
        --dataset "${DATASET}" \
        --output "${OUTPUT_FILE}" \
        --port 10222 \
        --seed ${SEED} \
        > "${LOG_FILE}" 2>&1

    EVAL=${OUTPUT_FILE%.json}_eval.json
    SUM=${OUTPUT_FILE%.json}_summary.json
    python baselines/utils/deepeval_via_algo_utils.py \
        --generated_file_path "${OUTPUT_FILE}" \
        --output_path "${EVAL}" \
        --summary_file_path "${SUM}"
    SUMMARIES+=("${SUM}")
done

COMBINED=${OUTPUT_DIR}/combined_cartridges_qwen_${TIMESTAMP}.json
python baselines/scripts/aggregate_seeds.py \
    --summary_files "${SUMMARIES[@]}" \
    --output "${COMBINED}"
echo "=== Combined: ${COMBINED} ==="
