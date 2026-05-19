NUM_RUNS=3
MAX_NUM_QUESTIONS=300
MAX_TURNS=15
lm_temperature=0.4
sm_temperature=0.1
loop_temperature=1.1
final_temperature=0.3
TEMP_CONFIG_NAME=lm${lm_temperature}-sm${sm_temperature}-loop${loop_temperature}-final${final_temperature}
EXPR_PREFIX_BASE=bcp_${MAX_NUM_QUESTIONS}queries_qwen32b_qwen14b-MEM-step1_5_batchsize256_trngepoch2
BCP_QNS_PATH=/path/to/browsecomp_plus_questions.jsonl

EVAL_DIR=/path/to/eval_results_bcp
mkdir -p ${EVAL_DIR}/multi_turn_eval/naive_multi_step_${MAX_TURNS}/${TEMP_CONFIG_NAME}

for RUN in $(seq 1 $NUM_RUNS); do
    EXPR_PREFIX=${EXPR_PREFIX_BASE}_run${RUN}
    EVAL_OUT_FILE=${EVAL_DIR}/multi_turn_eval/naive_multi_step_${MAX_TURNS}/${TEMP_CONFIG_NAME}/${EXPR_PREFIX}.json
    echo "Starting run ${RUN}/${NUM_RUNS}: ${EXPR_PREFIX}"
    python3 eval_bcp_trng_naive_multi_step.py \
        --lm_port 4325 \
        --sm_port 4324 \
        --bcp_qns_path $BCP_QNS_PATH \
        --max_num_questions $MAX_NUM_QUESTIONS \
        --output_path $EVAL_OUT_FILE \
        --max_turns $MAX_TURNS \
        --lm_temperature $lm_temperature \
        --sm_temperature $sm_temperature \
        --loop_temperature $loop_temperature \
        --final_temperature $final_temperature
    echo "Completed run ${RUN}/${NUM_RUNS}"
done
