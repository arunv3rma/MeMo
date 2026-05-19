#!/usr/bin/env bash
# narrativeqa_LOO_data_ablation.sh
#
# Full NarrativeQA data generation pipeline (steps 1a–5) with one step ablated.
# The ablated step is replaced by a --passthrough run: it reads its normal
# input, writes the expected output format with no LLM calls, and returns.
#
# Usage:
#   ./narrativeqa_LOO_data_ablation.sh <ABLATE_STEP>
#
# Valid ABLATE_STEP values: 1a  1b  2  3  4  5  (or "none" for full run)
# Note: the Helper (combine) step has no passthrough (pure local JSON combine, no LLM calls)
#
# Examples:
#   ./narrativeqa_LOO_data_ablation.sh 2      # skip consolidation
#   ./narrativeqa_LOO_data_ablation.sh 5      # skip cross-doc combination
#   ./narrativeqa_LOO_data_ablation.sh none   # run everything

cd "$(dirname "$0")/.."

ABLATE_STEP=${1:-"none"}

CORPUS_PATH=/path/to/narrativeqa_valid_corpus_chunks.jsonl
QNS_PATH=/path/to/narrativeqa_valid_questions_chunks.jsonl
MAX_NUM_DOCS=10  # NQA maps 1 doc → many questions, so we limit by doc count; passed as --max_num_questions, which for NQA resolves via SUBSET_MAP in nqa_data_utils.py (not a literal question count)
NUM_SAMPLING_EPOCHS=1
PORTS=(4325 4326)

OUTPUT_DIR=./ablation_NQA${MAX_NUM_DOCS}_skip_step${ABLATE_STEP}
LOG_DIR=./logs/ablation_nqa_skip_step${ABLATE_STEP}
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

echo "=========================================================="
echo " NarrativeQA Ablation Pipeline"
echo " Ablating step : ${ABLATE_STEP}"
echo " Output dir    : ${OUTPUT_DIR}"
echo "=========================================================="


## ---------------------------------------------------------------------------
## Step 1a — Direct fact extraction
## ---------------------------------------------------------------------------

DIRECT_FACT_EXPR_PREFIX="NQA_chunks_subset${MAX_NUM_DOCS}_numsamplingepochs${NUM_SAMPLING_EPOCHS}_directfact_extraction_cache"
DIRECT_FACT_OUTPUT_FILE_PATH=${OUTPUT_DIR}/${DIRECT_FACT_EXPR_PREFIX}.json
DIRECT_FACT_LOG_FILE="${LOG_DIR}/${DIRECT_FACT_EXPR_PREFIX}-log.txt"

STEP1A_PASSTHROUGH=""
[ "$ABLATE_STEP" = "1a" ] && STEP1A_PASSTHROUGH="--passthrough"

echo "--- $(date) --- Step 1a: direct fact extraction ${STEP1A_PASSTHROUGH}..." | tee -a ${DIRECT_FACT_LOG_FILE}
python3 generate_directfact_qa.py \
    --ports "${PORTS[@]}" \
    --output_file_path ${DIRECT_FACT_OUTPUT_FILE_PATH} \
    --corpus_path ${CORPUS_PATH} \
    --qns_path ${QNS_PATH} \
    --max_num_questions ${MAX_NUM_DOCS} \
    --num_epochs ${NUM_SAMPLING_EPOCHS} \
    --dataset nqa \
    --max_concurrent_generation 50 \
    --checkpoint_iter_freq 500 \
    ${STEP1A_PASSTHROUGH} > ${DIRECT_FACT_LOG_FILE}


## ---------------------------------------------------------------------------
## Step 1b — Indirect fact extraction
## ---------------------------------------------------------------------------

INDIRECT_FACT_EXPR_PREFIX="NQA_chunks_subset${MAX_NUM_DOCS}_numsamplingepochs${NUM_SAMPLING_EPOCHS}_indirectfact_extraction_cache"
INDIRECT_FACT_OUTPUT_FILE_PATH=${OUTPUT_DIR}/${INDIRECT_FACT_EXPR_PREFIX}.json
INDIRECT_FACT_LOG_FILE="${LOG_DIR}/${INDIRECT_FACT_EXPR_PREFIX}-log.txt"

STEP1B_PASSTHROUGH=""
[ "$ABLATE_STEP" = "1b" ] && STEP1B_PASSTHROUGH="--passthrough"

echo "--- $(date) --- Step 1b: indirect fact extraction ${STEP1B_PASSTHROUGH}..." | tee -a ${INDIRECT_FACT_LOG_FILE}
python3 generate_indirect_fact_qa.py \
    --ports "${PORTS[@]}" \
    --output_file_path ${INDIRECT_FACT_OUTPUT_FILE_PATH} \
    --corpus_path ${CORPUS_PATH} \
    --qns_path ${QNS_PATH} \
    --max_num_questions ${MAX_NUM_DOCS} \
    --num_epochs ${NUM_SAMPLING_EPOCHS} \
    --dataset nqa \
    --max_concurrent_generation 50 \
    --checkpoint_iter_freq 500 \
    ${STEP1B_PASSTHROUGH} > ${INDIRECT_FACT_LOG_FILE}


## ---------------------------------------------------------------------------
## (Helper) Combine direct + indirect
## ---------------------------------------------------------------------------

COMBINE_EXPR_PREFIX="NQA_chunks_subset${MAX_NUM_DOCS}_numsamplingepochs${NUM_SAMPLING_EPOCHS}_combine_direct_and_indirect_extraction_cache"
COMBINE_OUTPUT_FILE_PATH=${OUTPUT_DIR}/${COMBINE_EXPR_PREFIX}.json

echo "--- $(date) --- Helper: combine direct and indirect..."
python3 combine_direct_and_indirect.py \
    --direct_json_path ${DIRECT_FACT_OUTPUT_FILE_PATH} \
    --indirect_json_path ${INDIRECT_FACT_OUTPUT_FILE_PATH} \
    --output_json_path ${COMBINE_OUTPUT_FILE_PATH}


## ---------------------------------------------------------------------------
## Step 2 — Permutation consolidation
## ---------------------------------------------------------------------------

CONSOLIDATION_EXPR_PREFIX="NQA_chunks_subset${MAX_NUM_DOCS}_numsamplingepochs${NUM_SAMPLING_EPOCHS}_permutation_consolidation_extraction_cache"
CONSOLIDATION_OUTPUT_FILE_PATH=${OUTPUT_DIR}/${CONSOLIDATION_EXPR_PREFIX}.json
CONSOLIDATION_LOG_FILE="${LOG_DIR}/${CONSOLIDATION_EXPR_PREFIX}-log.txt"

STEP2_PASSTHROUGH=""
[ "$ABLATE_STEP" = "2" ] && STEP2_PASSTHROUGH="--passthrough"

echo "--- $(date) --- Step 2: consolidation ${STEP2_PASSTHROUGH}..." | tee -a ${CONSOLIDATION_LOG_FILE}
python3 generate_consolidation_cache.py \
    --ports "${PORTS[@]}" \
    --input_file_path ${COMBINE_OUTPUT_FILE_PATH} \
    --output_file_path ${CONSOLIDATION_OUTPUT_FILE_PATH} \
    --corpus_path ${CORPUS_PATH} \
    --dataset nqa \
    --min_qa_pairs 3 \
    --num_hedges 3 \
    --checkpoint_iter_freq 500 \
    ${STEP2_PASSTHROUGH} > ${CONSOLIDATION_LOG_FILE}


## ---------------------------------------------------------------------------
## Step 3 — Self-containment check + fix
## ---------------------------------------------------------------------------

SELF_CONTAINMENT_EXPR_PREFIX="NQA_chunks_subset${MAX_NUM_DOCS}_numsamplingepochs${NUM_SAMPLING_EPOCHS}_post_permutation_consolidation_verified_extraction_cache"
SELF_CONTAINMENT_OUTPUT_FILE_PATH=${OUTPUT_DIR}/${SELF_CONTAINMENT_EXPR_PREFIX}.json
SELF_CONTAINMENT_LOG_FILE="${LOG_DIR}/${SELF_CONTAINMENT_EXPR_PREFIX}-log.txt"

STEP3_PASSTHROUGH=""
[ "$ABLATE_STEP" = "3" ] && STEP3_PASSTHROUGH="--passthrough"

echo "--- $(date) --- Step 3: self-containment check ${STEP3_PASSTHROUGH}..." | tee -a ${SELF_CONTAINMENT_LOG_FILE}
python3 check_self_containment_post_combination.py \
    --ports "${PORTS[@]}" \
    --input_file_path ${CONSOLIDATION_OUTPUT_FILE_PATH} \
    --output_file_path ${SELF_CONTAINMENT_OUTPUT_FILE_PATH} \
    --corpus_path ${CORPUS_PATH} \
    --dataset nqa \
    --checkpoint_iter_freq 100 \
    ${STEP3_PASSTHROUGH} > ${SELF_CONTAINMENT_LOG_FILE}

echo "--- $(date) --- Retry attempt for checking self containment..." | tee -a ${SELF_CONTAINMENT_LOG_FILE}
SELF_CONTAINMENT_RETRY_LOG_FILE="${LOG_DIR}/${SELF_CONTAINMENT_EXPR_PREFIX}-retry_log.txt"
python3 check_self_containment_post_combination.py \
    --ports "${PORTS[@]}" \
    --input_file_path ${CONSOLIDATION_OUTPUT_FILE_PATH} \
    --resume_checkpoint ${SELF_CONTAINMENT_OUTPUT_FILE_PATH} \
    --output_file_path ${SELF_CONTAINMENT_OUTPUT_FILE_PATH} \
    --corpus_path ${CORPUS_PATH} \
    --dataset nqa \
    --retry_failed \
    --checkpoint_iter_freq 500 \
    ${STEP3_PASSTHROUGH} > ${SELF_CONTAINMENT_RETRY_LOG_FILE}


## ---------------------------------------------------------------------------
## Step 4 — Surface entity QA generation
## ---------------------------------------------------------------------------

SURFACE_ENTITY_EXPR_PREFIX="NQA_chunks_subset${MAX_NUM_DOCS}_numsamplingepochs${NUM_SAMPLING_EPOCHS}_surface_entity_cache"
SURFACE_ENTITY_OUTPUT_FILE_PATH=${OUTPUT_DIR}/${SURFACE_ENTITY_EXPR_PREFIX}.json
SURFACE_ENTITY_LOG_FILE="${LOG_DIR}/${SURFACE_ENTITY_EXPR_PREFIX}-log.txt"

STEP4_PASSTHROUGH=""
[ "$ABLATE_STEP" = "4" ] && STEP4_PASSTHROUGH="--passthrough"

echo "--- $(date) --- Step 4: surface entity generation ${STEP4_PASSTHROUGH}..." | tee -a ${SURFACE_ENTITY_LOG_FILE}
python3 generate_surface_entity_cache.py \
    --ports "${PORTS[@]}" \
    --input_file_path ${SELF_CONTAINMENT_OUTPUT_FILE_PATH} \
    --output_file_path ${SURFACE_ENTITY_OUTPUT_FILE_PATH} \
    --corpus_path ${CORPUS_PATH} \
    --dataset nqa \
    --include_source_qa_pairs \
    --checkpoint_iter_freq 500 \
    ${STEP4_PASSTHROUGH} > ${SURFACE_ENTITY_LOG_FILE}


## ---------------------------------------------------------------------------
## Step 5 — Cross-doc entity combination
## ---------------------------------------------------------------------------

CROSSDOC_EXPR_PREFIX="NQA_chunks_subset${MAX_NUM_DOCS}_numsamplingepochs${NUM_SAMPLING_EPOCHS}_crossdoc_entity_combination"
CROSSDOC_OUTPUT_FILE_PATH=${OUTPUT_DIR}/${CROSSDOC_EXPR_PREFIX}.json
CROSSDOC_LOG_FILE="${LOG_DIR}/${CROSSDOC_EXPR_PREFIX}-log.txt"

STEP5_PASSTHROUGH=""
[ "$ABLATE_STEP" = "5" ] && STEP5_PASSTHROUGH="--passthrough"

echo "--- $(date) --- Step 5: cross-doc entity combination ${STEP5_PASSTHROUGH}..." | tee -a ${CROSSDOC_LOG_FILE}
python3 generate_crossdoc_entity_combination_cache.py \
    --ports "${PORTS[@]}" \
    --input_file_path ${SURFACE_ENTITY_OUTPUT_FILE_PATH} \
    --output_file_path ${CROSSDOC_OUTPUT_FILE_PATH} \
    --qns_path ${QNS_PATH} \
    --max_num_questions ${MAX_NUM_DOCS} \
    --dataset nqa \
    --min_docs_with_qa 2 \
    --include_source_qa_pairs \
    --max_other_qa_per_batch 20 \
    --num_hedges 3 \
    --max_concurrent_generation 100 \
    --checkpoint_iter_freq 5000 \
    ${STEP5_PASSTHROUGH} > ${CROSSDOC_LOG_FILE}

echo "=========================================================="
echo " Done. Final output: ${CROSSDOC_OUTPUT_FILE_PATH}"
echo "=========================================================="
