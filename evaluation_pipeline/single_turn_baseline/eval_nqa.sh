MAX_NUM_DOCS=10
NUM_RUNS=3
EXPR_PREFIX_BASE=nqa_valid_split_${MAX_NUM_DOCS}docs_qwen32b_qwen14b-MEM_step1-5_batchsize256_trngepoch2
NQA_QNS_PATH=/path/to/narrativeqa_valid_questions_chunks.jsonl

EVAL_DIR=/path/to/eval_results_nqa/single_turn_eval
mkdir -p ${EVAL_DIR}

for RUN in $(seq 1 $NUM_RUNS); do
    EXPR_PREFIX=${EXPR_PREFIX_BASE}_run${RUN}
    EVAL_OUT_FILE=${EVAL_DIR}/${EXPR_PREFIX}.json
    echo "Starting run ${RUN}/${NUM_RUNS}: ${EXPR_PREFIX}"
    python3 eval_nqa_trng.py \
        --lm_port 4325 \
        --sm_port 4324 \
        --nqa_qns_path $NQA_QNS_PATH \
        --max_num_docs $MAX_NUM_DOCS \
        --output_path $EVAL_OUT_FILE
    echo "Completed run ${RUN}/${NUM_RUNS}"
done
