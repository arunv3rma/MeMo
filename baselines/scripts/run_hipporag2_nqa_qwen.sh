#!/bin/bash
set -euo pipefail

conda activate hipporag

cd "$(cd "$(dirname "$0")/../.." && pwd)"

DATASET=nqa
CORPUS=baselines/data/narrativeqa_dev_10_doc_corpus.json
QUESTIONS=baselines/data/nqa_question.json
CORPUS_TAG=$(basename "${CORPUS}" .jsonl)
CORPUS_TAG=$(basename "${CORPUS_TAG}" .json)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

API_BASE=http://localhost:4327/v1
MODEL_ID=qwen2_5_32b
API_KEY=EMPTY
PROVIDER_FLAGS="--provider vllm"
MAX_CONCURRENT=16
export HIPPORAG_OPENIE_MAX_WORKERS=${HIPPORAG_OPENIE_MAX_WORKERS:-16}

OUT_DIR=baselines/output_runs/narrativeqa_dev_10_doc_corpus
LOG_DIR=baselines/output_runs/logs
mkdir -p "${OUT_DIR}" "${LOG_DIR}"
SAVE_DIR=baselines/hipporag2/output_nqa/${CORPUS_TAG}_runs_qwen_${TIMESTAMP}
mkdir -p "${SAVE_DIR}"

SUMMARIES=()
for RUN in 1 2 3; do
    OUTPUT_FILE=${OUT_DIR}/hipporag2_qwen_run${RUN}_${TIMESTAMP}.json
    LOG_FILE=${LOG_DIR}/hipporag2_${DATASET}_qwen_run${RUN}_${TIMESTAMP}.log
    echo "=== Run ${RUN}: output=${OUTPUT_FILE} ==="

    CUDA_VISIBLE_DEVICES=3 python -m baselines.hipporag2.main_for_nqa \
        --corpus "${CORPUS}" \
        --questions "${QUESTIONS}" \
    --provider vllm \
    --api_base "${API_BASE}" \
    --model_id "${MODEL_ID}" \
    --api_key "${API_KEY}" \
        --max_concurrent "${MAX_CONCURRENT}" \
        --k 9 \
    --save_dir "${SAVE_DIR}" \
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

COMBINED=${OUT_DIR}/combined_hipporag2_nqa_qwen_${TIMESTAMP}.json
python baselines/scripts/aggregate_runs.py \
    --summary_files "${SUMMARIES[@]}" \
    --output "${COMBINED}"
echo "=== Combined: ${COMBINED} ==="
