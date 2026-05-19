MAX_NUM_QUESTIONS=1000
NUM_RUNS=3
EXPR_PREFIX_BASE=msq_${MAX_NUM_QUESTIONS}queries_qwen32b_qwen14b-MEM_step1-5_batchsize256_trngepoch2
MSQ_QNS_PATH=/path/to/musique_questions_1000.jsonl

EVAL_DIR=/path/to/eval_results_msq/single_turn_eval
mkdir -p ${EVAL_DIR}

for RUN in $(seq 1 $NUM_RUNS); do
    EXPR_PREFIX=${EXPR_PREFIX_BASE}_run${RUN}
    EVAL_OUT_FILE=${EVAL_DIR}/${EXPR_PREFIX}.json
    echo "Starting run ${RUN}/${NUM_RUNS}: ${EXPR_PREFIX}"
    python3 eval_msq_trng.py \
        --lm_port 4325 \
        --sm_port 4324 \
        --msq_qns_path $MSQ_QNS_PATH \
        --max_num_questions $MAX_NUM_QUESTIONS \
        --output_path $EVAL_OUT_FILE
    echo "Completed run ${RUN}/${NUM_RUNS}"
done
