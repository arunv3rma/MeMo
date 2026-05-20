#!/bin/bash
set -euo pipefail

conda activate cartridges

cd "$(cd "$(dirname "$0")/../.." && pwd)"

DATASET=bcp_300_with_negatives_8192-samples
QUESTIONS=baselines/data/decrypted.jsonl
MAX_QUESTIONS=300
OUTPUT_DIR=baselines/output_runs/${DATASET}
LOG_DIR=baselines/output_runs/logs
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARIES=()

for RUN in 1 2 3; do
    OUTPUT_FILE=${OUTPUT_DIR}/cartridges_qwen_run${RUN}_${TIMESTAMP}.json
    LOG_FILE=${LOG_DIR}/cartridges_${DATASET}_qwen_run${RUN}_${TIMESTAMP}.log
    echo "=== Run ${RUN}: output=${OUTPUT_FILE} ==="
    CUDA_VISIBLE_DEVICES=0 python -m baselines.cartridges.main \
        --questions "${QUESTIONS}" \
        --max_questions ${MAX_QUESTIONS} \
        --dataset "${DATASET}" \
        --output "${OUTPUT_FILE}" \
        --port 10222 \
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
python baselines/scripts/aggregate_runs.py \
    --summary_files "${SUMMARIES[@]}" \
    --output "${COMBINED}"
echo "=== Combined: ${COMBINED} ==="
