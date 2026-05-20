#!/bin/bash
set -euo pipefail

conda activate cartridges

# DATASET=bcp_300_single_doc_8192-samples
# QUESTIONS=baselines/data/decrypted.jsonl
# MAX_QUESTIONS=300

DATASET=bcp_300_with_negatives_8192-samples_8192-kvcachesize
QUESTIONS=baselines/data/decrypted.jsonl
MAX_QUESTIONS=300

# DATASET=musique_1000_single_doc_8192-samples
# QUESTIONS=baselines/data/musique_questions_chunks_1000.jsonl
# MAX_QUESTIONS=1000

# DATASET=musique_1000_with_negatives_8192-samples
# QUESTIONS=baselines/data/musique_questions_chunks_1000.jsonl
# MAX_QUESTIONS=1000

# DATASET=nqa_10_single_doc_8192samples
# QUESTIONS=baselines/data/nqa_question.json
# MAX_QUESTIONS=10
OUTPUT_DIR=baselines/output/${DATASET}
LOG_DIR=baselines/log

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

for run in 1 2 3; do
  if [[ ${run} -eq 1 ]]; then
    suffix=""
    move_flag="--need_to_move_cartridges"
  else
    suffix="_${run}"
    move_flag=""
  fi
  OUTPUT_FILE=${OUTPUT_DIR}/cartridges_output${suffix}.json
  LOG_FILE=${LOG_DIR}/cartridges_${DATASET}${suffix}.log

  echo "=== Run ${run}/3: output=${OUTPUT_FILE} log=${LOG_FILE} ==="
  CUDA_VISIBLE_DEVICES=0 python -m baselines.cartridges.main \
    --questions ${QUESTIONS} \
    --max_questions ${MAX_QUESTIONS} \
    --dataset ${DATASET} \
    --output ${OUTPUT_FILE} \
    --port 10222 \
    ${move_flag} \
    > "${LOG_FILE}" 2>&1
done

  # --multi_turn \
  # --need_to_move_cartridges \

# CUDA_VISIBLE_DEVICES=6 python -m baselines.cartridges.main \
#   --questions ${QUESTIONS} \
#   --output ${OUTPUT_FILE} \
#   --dataset ${DATASET} \
#   --need_to_move_cartridges \
#   > baselines/log/cartridges_${DATASET}.log 2>&1
  