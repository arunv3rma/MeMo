MAX_NUM_QUESTIONS=300
NUM_RUNS=3
EXPR_PREFIX_BASE=bcp_${MAX_NUM_QUESTIONS}_qwen32b_qwen14b-MEM_step1-5_batchsize256_trngepoch2
BCP_QNS_PATH=/path/to/browsecomp_plus_questions.jsonl

EVAL_DIR=/path/to/eval_results_bcp/single_turn_eval
mkdir -p ${EVAL_DIR}

for RUN in $(seq 1 $NUM_RUNS); do
    EXPR_PREFIX=${EXPR_PREFIX_BASE}_run${RUN}
    EVAL_OUT_FILE=${EVAL_DIR}/${EXPR_PREFIX}.json
    echo "Starting run ${RUN}/${NUM_RUNS}: ${EXPR_PREFIX}"
    python3 eval_bcp_trng.py \
        --lm_port 4325 \
        --sm_port 4324 \
        --bcp_qns_path $BCP_QNS_PATH \
        --max_num_questions $MAX_NUM_QUESTIONS \
        --output_path $EVAL_OUT_FILE
    echo "Completed run ${RUN}/${NUM_RUNS}"
done
