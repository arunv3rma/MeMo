MSQ_QNS_PATH=/path/to/musique_questions_1000.jsonl
MAX_NUM_QUESTIONS=1000
MAX_ENTITY_TURNS=7
MAX_ANSWER_TURNS=8
DEAD_END_THRESHOLD=3
lm_grounding_temperature=0.4
sm_grounding_temperature=0.1
lm_entity_temperature=0.4
sm_entity_temperature=0.1
lm_answer_temperature=1.0
sm_answer_temperature=0.3
lm_final_temperature=0.3
TEMP_CONFIG_NAME=lmgrounding${lm_grounding_temperature}-smgrounding${sm_grounding_temperature}-lmentity${lm_entity_temperature}-smentity${sm_entity_temperature}-lmanswer${lm_answer_temperature}-smanswer${sm_answer_temperature}-lmfinal${lm_final_temperature}

EVAL_DIR=/path/to/eval_results_msq
mkdir -p $EVAL_DIR

## ---------------------------------------------------------------------------
## vLLM server config — update model paths / GPUs as needed
## ---------------------------------------------------------------------------
LM_PORT=4225
LM_GPU=6
LM_MODEL=Qwen/Qwen2.5-32B-Instruct
LM_MODEL_NAME=qwen2_5_32b
LFM_PORT=4330

SM_PORT=4224
SM_GPU=7
SM_MODEL_NAME=mem_model

## ---------------------------------------------------------------------------
## SM model pairs: parallel arrays of (model_path, expr_prefix_base)
## ---------------------------------------------------------------------------
SM_MODELS=(
    
)
SM_EXPR_PREFIXES=(
    # msq_${MAX_NUM_QUESTIONS}qns_qwen32b_qwen14b-MEM_step1-7_trngepoch1
)

export VLLM_DISABLE_COMPILE_CACHE=1
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

cleanup() {
    echo "Shutting down vLLM servers..."
    kill $LM_PID $SM_PID 2>/dev/null
    wait $LM_PID $SM_PID 2>/dev/null
    echo "Servers stopped."
}
trap cleanup EXIT

echo "Starting vLLM LM server (port ${LM_PORT}, GPU ${LM_GPU})..."
CUDA_VISIBLE_DEVICES=$LM_GPU python3 -m vllm.entrypoints.openai.api_server \
    --host localhost \
    --port $LM_PORT \
    --trust-remote-code \
    --model $LM_MODEL \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --served-model-name $LM_MODEL_NAME \
    --max-model-len 131072 \
    --rope-scaling '{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' &
LM_PID=$!

echo "Waiting for LM server to become healthy..."
for attempt in $(seq 1 60); do
    if curl -sf "http://localhost:${LM_PORT}/v1/models" > /dev/null 2>&1; then
        echo "✓ Port $LM_PORT ready"
        break
    fi
    if [ $attempt -eq 60 ]; then
        echo "✗ Port $LM_PORT failed to start. Exiting."
        exit 1
    fi
    sleep 10
done

mkdir -p ${EVAL_DIR}/structured_multi_step/entity${MAX_ENTITY_TURNS}_answer${MAX_ANSWER_TURNS}/${TEMP_CONFIG_NAME}

## ---------------------------------------------------------------------------
## Iterate over SM model pairs
## ---------------------------------------------------------------------------
for idx in "${!SM_MODELS[@]}"; do
    SM_MODEL=${SM_MODELS[$idx]}
    EXPR_PREFIX_BASE=${SM_EXPR_PREFIXES[$idx]}

    echo ""
    echo "=== SM model $((idx + 1))/${#SM_MODELS[@]}: ${SM_MODEL} ==="
    SM_MODEL_SHORT=$(echo "$SM_MODEL" | grep -oiP '(?<=[_-])\d+\.?\d*b(?=[_-])' | tail -1 | tr '[:upper:]' '[:lower:]')
    
    source /path/to/miniconda3/etc/profile.d/conda.sh
    if [[ "$EXPR_PREFIX_BASE" == *"lfm"* ]]; then
        echo "Found lfm in small model name! Activating env"
        conda activate lfm
        SM_MODEL_SHORT=lfm1.2b
    elif [[ "$EXPR_PREFIX_BASE" == *"gemma"* ]]; then
        echo "Found gemma in small model name"
        SM_MODEL_SHORT=gemma1b
    else
        echo "No special pattern found, defaulting to qwen"
        SM_MODEL_SHORT="qwen${SM_MODEL_SHORT}"
    fi
    EPOCH=$(echo "$EXPR_PREFIX_BASE" | grep -oP 'trngepoch\d+')

    echo "Starting vLLM SM server (port ${SM_PORT}, GPU ${SM_GPU})..."
    CUDA_VISIBLE_DEVICES=$SM_GPU python3 -m vllm.entrypoints.openai.api_server \
        --host localhost \
        --port $SM_PORT \
        --trust-remote-code \
        --model $SM_MODEL \
        --tensor-parallel-size 1 \
        --dtype bfloat16 \
        --served-model-name $SM_MODEL_NAME \
        --max_model_len 32678 &
    SM_PID=$!

    echo "Waiting for SM server to become healthy..."
    for attempt in $(seq 1 60); do
        if curl -sf "http://localhost:${SM_PORT}/v1/models" > /dev/null 2>&1; then
            echo "✓ Port $SM_PORT ready"
            break
        fi
        if [ $attempt -eq 60 ]; then
            echo "✗ Port $SM_PORT failed to start. Exiting."
            exit 1
        fi
        sleep 10
    done

    # qwen32b
    conda activate memo
    mkdir -p ${EVAL_DIR}/structured_multi_step/entity${MAX_ENTITY_TURNS}_answer${MAX_ANSWER_TURNS}/${TEMP_CONFIG_NAME}/qwen32b/${SM_MODEL_SHORT}/${EPOCH}
    NUM_RUNS=3
    for RUN in $(seq 1 $NUM_RUNS); do
        EXPR_PREFIX=${EXPR_PREFIX_BASE}_run${RUN}
        EVAL_OUT_FILE=${EVAL_DIR}/structured_multi_step/entity${MAX_ENTITY_TURNS}_answer${MAX_ANSWER_TURNS}/${TEMP_CONFIG_NAME}/qwen32b/${SM_MODEL_SHORT}/${EPOCH}/${EXPR_PREFIX}.json
        echo "Starting run ${RUN}/${NUM_RUNS}: ${EXPR_PREFIX}"
        python3 eval_msq_trng_structured_multi_step.py \
            --lm_port $LM_PORT \
            --sm_port $SM_PORT \
            --msq_qns_path $MSQ_QNS_PATH \
            --max_num_questions $MAX_NUM_QUESTIONS \
            --output_path $EVAL_OUT_FILE \
            --max_entity_turns $MAX_ENTITY_TURNS \
            --max_answer_turns $MAX_ANSWER_TURNS \
            --dead_end_threshold $DEAD_END_THRESHOLD \
            --lm_grounding_temperature $lm_grounding_temperature \
            --sm_grounding_temperature $sm_grounding_temperature \
            --lm_entity_temperature $lm_entity_temperature \
            --sm_entity_temperature $sm_entity_temperature \
            --lm_answer_temperature $lm_answer_temperature \
            --sm_answer_temperature $sm_answer_temperature \
            --lm_final_temperature $lm_final_temperature
        echo "Completed run ${RUN}/${NUM_RUNS}"
    done


    # OpenRouter variant (uncomment to use)
    NUM_RUNS=3
    MAIN_MODEL=gemini_3_flash_preview
    EXPR_PREFIX_BASE="${SM_EXPR_PREFIXES[$idx]/qwen32b/${MAIN_MODEL}}"

    mkdir -p ${EVAL_DIR}/structured_multi_step/entity${MAX_ENTITY_TURNS}_answer${MAX_ANSWER_TURNS}/${MAIN_MODEL}/${TEMP_CONFIG_NAME}/${SM_MODEL_SHORT}/${EPOCH}

    for RUN in $(seq 1 $NUM_RUNS); do
        EXPR_PREFIX=${EXPR_PREFIX_BASE}_run${RUN}
        EVAL_OUT_FILE=${EVAL_DIR}/structured_multi_step/entity${MAX_ENTITY_TURNS}_answer${MAX_ANSWER_TURNS}/${MAIN_MODEL}/${TEMP_CONFIG_NAME}/${SM_MODEL_SHORT}/${EPOCH}/${EXPR_PREFIX}.json
        LOG_FILE=${EVAL_DIR}/structured_multi_step/entity${MAX_ENTITY_TURNS}_answer${MAX_ANSWER_TURNS}/${MAIN_MODEL}/${TEMP_CONFIG_NAME}/${SM_MODEL_SHORT}/${EPOCH}/${EXPR_PREFIX}.log
        echo "Starting run ${RUN}/${NUM_RUNS}: ${EXPR_PREFIX}"
        python3 eval_msq_trng_structured_multi_step.py \
            --lm_model_name "google/gemini-3-flash-preview" \
            --sm_port $SM_PORT \
            --msq_qns_path $MSQ_QNS_PATH \
            --max_num_questions $MAX_NUM_QUESTIONS \
            --output_path $EVAL_OUT_FILE \
            --max_entity_turns $MAX_ENTITY_TURNS \
            --max_answer_turns $MAX_ANSWER_TURNS \
            --dead_end_threshold $DEAD_END_THRESHOLD \
            --lm_grounding_temperature $lm_grounding_temperature \
            --sm_grounding_temperature $sm_grounding_temperature \
            --lm_entity_temperature $lm_entity_temperature \
            --sm_entity_temperature $sm_entity_temperature \
            --lm_answer_temperature $lm_answer_temperature \
            --sm_answer_temperature $sm_answer_temperature \
            --lm_final_temperature $lm_final_temperature \
            --max_concurrent 15 \
            2>&1 | tee "$LOG_FILE"
        echo "Completed run ${RUN}/${NUM_RUNS}"
    done

    echo "Stopping SM server (PID ${SM_PID})..."
    kill $SM_PID 2>/dev/null
    wait $SM_PID 2>/dev/null
    echo "SM server stopped."
done
