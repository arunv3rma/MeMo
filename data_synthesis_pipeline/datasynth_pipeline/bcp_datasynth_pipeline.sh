#!/usr/bin/env bash
cd "$(dirname "$0")/.."

CORPUS_PATH=/path/to/full_corpus_train.jsonl
QNS_PATH=/path/to/browsecomp_plus_questions.jsonl
MAX_NUM_QUESTIONS=300
NUM_SAMPLING_EPOCHS=1
PORTS=(4325 4326)
LOG_DIR=./logs/step1_5_qwen32b_with_neg_bcp

OUTPUT_DIR=./with_N_neg_bcp${MAX_NUM_QUESTIONS}_datagen_step1_5
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}


## ---------------------------------------------------------------------------
## Step 1a — Direct fact extraction
## ---------------------------------------------------------------------------

DIRECT_FACT_EXPR_PREFIX="bcp_subset${MAX_NUM_QUESTIONS}_numsamplingepochs${NUM_SAMPLING_EPOCHS}_fact_extraction_cache"
DIRECT_FACT_OUTPUT_FILE_PATH=${OUTPUT_DIR}/${DIRECT_FACT_EXPR_PREFIX}.json
DIRECT_FACT_LOG_FILE="${LOG_DIR}/${DIRECT_FACT_EXPR_PREFIX}-log.txt"
echo "--- $(date) --- Initial direct fact extraction attempt with original parameters..." | tee -a $DIRECT_FACT_LOG_FILE
python3 with_neg_generate_directfact_qa.py \
    --ports "${PORTS[@]}" \
    --output_file_path ${DIRECT_FACT_OUTPUT_FILE_PATH} \
    --corpus_path $CORPUS_PATH \
    --qns_path $QNS_PATH \
    --max_num_questions $MAX_NUM_QUESTIONS \
    --num_epochs ${NUM_SAMPLING_EPOCHS} \
    --checkpoint_iter_freq 500 > ${DIRECT_FACT_LOG_FILE}


## ---------------------------------------------------------------------------
## Step 1b — Indirect fact extraction
## ---------------------------------------------------------------------------

INDIRECT_FACT_EXPR_PREFIX="bcp_subset${MAX_NUM_QUESTIONS}_numsamplingepochs${NUM_SAMPLING_EPOCHS}_indirectfact_extraction_cache"
INDIRECT_FACT_OUTPUT_FILE_PATH=${OUTPUT_DIR}/${INDIRECT_FACT_EXPR_PREFIX}.json
INDIRECT_FACT_LOG_FILE="${LOG_DIR}/${INDIRECT_FACT_EXPR_PREFIX}-log.txt"
echo "--- $(date) --- Initial indirect fact extraction attempt with original parameters..." | tee -a $INDIRECT_FACT_LOG_FILE
python3 with_neg_generate_indirect_fact_qa.py \
    --ports "${PORTS[@]}" \
    --output_file_path ${INDIRECT_FACT_OUTPUT_FILE_PATH} \
    --corpus_path $CORPUS_PATH \
    --qns_path $QNS_PATH \
    --max_num_questions $MAX_NUM_QUESTIONS \
    --num_epochs ${NUM_SAMPLING_EPOCHS} \
    --checkpoint_iter_freq 500 > ${INDIRECT_FACT_LOG_FILE}


## ---------------------------------------------------------------------------
## (Helper) Combine direct + indirect
## ---------------------------------------------------------------------------

COMBINE_DIRECT_AND_INDIRECT_EXPR_PREFIX="bcp_subset${MAX_NUM_QUESTIONS}_numsamplingepochs${NUM_SAMPLING_EPOCHS}_combine_direct_and_indirect_extraction_cache"
COMBINE_DIRECT_AND_INDIRECT_PATH=${OUTPUT_DIR}/${COMBINE_DIRECT_AND_INDIRECT_EXPR_PREFIX}.json
echo "--- $(date) --- Initial combination of direct and indirect fact extraction attempt with original parameters..."
python3 combine_direct_and_indirect.py \
    --direct_json_path $DIRECT_FACT_OUTPUT_FILE_PATH \
    --indirect_json_path $INDIRECT_FACT_OUTPUT_FILE_PATH \
    --output_json_path $COMBINE_DIRECT_AND_INDIRECT_PATH


## ---------------------------------------------------------------------------
## Step 2 — Permutation consolidation
## ---------------------------------------------------------------------------

PERMUTATION_CONSOLIDATION_EXPR_PREFIX="bcp_subset${MAX_NUM_QUESTIONS}_numsamplingepochs${NUM_SAMPLING_EPOCHS}_permutation_consolidation_extraction_cache"
PERMUTATION_CONSOLIDATION_OUTPUT_FILE_PATH=${OUTPUT_DIR}/${PERMUTATION_CONSOLIDATION_EXPR_PREFIX}.json
PERMUTATION_CONSOLIDATION_LOG_FILE="${LOG_DIR}/${PERMUTATION_CONSOLIDATION_EXPR_PREFIX}-log.txt"
echo "--- $(date) --- Initial attempt for permuation consolidation with original parameters..." | tee -a $PERMUTATION_CONSOLIDATION_LOG_FILE
python generate_consolidation_cache.py \
    --ports "${PORTS[@]}" \
    --input_file_path $COMBINE_DIRECT_AND_INDIRECT_PATH \
    --output_file_path $PERMUTATION_CONSOLIDATION_OUTPUT_FILE_PATH \
    --corpus_path $CORPUS_PATH \
    --min_qa_pairs 3 \
    --num_hedges 3 > ${PERMUTATION_CONSOLIDATION_LOG_FILE}


## ---------------------------------------------------------------------------
## Step 3 — Self-containment check + fix (first pass)
## ---------------------------------------------------------------------------

SELF_CONTAINMENT_VERIFICATION_EXPR_PREFIX="bcp_subset${MAX_NUM_QUESTIONS}_numsamplingepochs${NUM_SAMPLING_EPOCHS}_post_permutation_consolidation_verified_extraction_cache"
SELF_CONTAINMENT_VERIFICATION_OUTPUT_FILE_PATH=${OUTPUT_DIR}/${SELF_CONTAINMENT_VERIFICATION_EXPR_PREFIX}.json
SELF_CONTAINMENT_VERIFICATION_LOG_FILE="${LOG_DIR}/${SELF_CONTAINMENT_VERIFICATION_EXPR_PREFIX}-log.txt"
echo "--- $(date) --- Initial attempt for checking self containment with original parameters..." | tee -a $SELF_CONTAINMENT_VERIFICATION_LOG_FILE
python3 check_self_containment_post_combination.py \
    --ports "${PORTS[@]}" \
    --input_file_path ${PERMUTATION_CONSOLIDATION_OUTPUT_FILE_PATH} \
    --output_file_path ${SELF_CONTAINMENT_VERIFICATION_OUTPUT_FILE_PATH} \
    --corpus_path $CORPUS_PATH \
    --checkpoint_iter_freq 100 > ${SELF_CONTAINMENT_VERIFICATION_LOG_FILE}


## ---------------------------------------------------------------------------
## Step 3 — Self-containment retry (optional, re-queue failed pairs)
## ---------------------------------------------------------------------------

echo "--- $(date) --- Retry attempt for checking self containment with original parameters..." | tee -a $SELF_CONTAINMENT_VERIFICATION_LOG_FILE
SELF_CONTAINMENT_VERIFICATION_retry_LOG_FILE="${LOG_DIR}/${SELF_CONTAINMENT_VERIFICATION_EXPR_PREFIX}-retry_log.txt"
python3 check_self_containment_post_combination.py \
    --ports "${PORTS[@]}" \
    --input_file_path ${PERMUTATION_CONSOLIDATION_OUTPUT_FILE_PATH} \
    --resume_checkpoint $SELF_CONTAINMENT_VERIFICATION_OUTPUT_FILE_PATH \
    --output_file_path ${SELF_CONTAINMENT_VERIFICATION_OUTPUT_FILE_PATH} \
    --corpus_path $CORPUS_PATH \
    --retry_failed \
    --checkpoint_iter_freq 500 > ${SELF_CONTAINMENT_VERIFICATION_retry_LOG_FILE}


## ---------------------------------------------------------------------------
## Step 4 — Surface entity QA generation
## ---------------------------------------------------------------------------

SURFACE_ENTITY_EXPR_PREFIX="bcp_subset${MAX_NUM_QUESTIONS}_numsamplingepochs${NUM_SAMPLING_EPOCHS}_surface_entity_cache"
SURFACE_ENTITY_OUTPUT_FILE_PATH=${OUTPUT_DIR}/${SURFACE_ENTITY_EXPR_PREFIX}.json
SURFACE_ENTITY_LOG_FILE="${LOG_DIR}/${SURFACE_ENTITY_EXPR_PREFIX}-log.txt"
echo "--- $(date) --- Initial surface entity extraction attempt with original parameters..." | tee -a $SURFACE_ENTITY_LOG_FILE
python3 generate_surface_entity_cache.py \
    --ports "${PORTS[@]}" \
    --input_file_path ${SELF_CONTAINMENT_VERIFICATION_OUTPUT_FILE_PATH} \
    --output_file_path ${SURFACE_ENTITY_OUTPUT_FILE_PATH} \
    --corpus_path $CORPUS_PATH \
    --include_source_qa_pairs \
    --checkpoint_iter_freq 500 > ${SURFACE_ENTITY_LOG_FILE}


## ---------------------------------------------------------------------------
## Step 5 — Cross-doc entity combination (NEW)
##
## Groups docs by which query's evidence_docs they belong to.
## For each group, iterates anchor QA pairs from each doc against all other
## docs' QA pairs in batches and generates two types of cross-doc QA pairs:
##   TYPE A: converging_clues  — different facts across docs → same entity
##           Q: "Who [fact from doc X] and [fact from doc Y]?"
##           A: "[Entity Name]"
##   TYPE B: parallel_property — same fact/property → different entities
##           Q: "Which [entity type]s [shared property]?"
##           A: "[Entity A] and [Entity B]"
##
## With --include_source_qa_pairs, all step-6 entries are passed through
## unchanged followed by the new cross-doc entries.
## ---------------------------------------------------------------------------
CROSSDOC_COMBO_EXPR_PREFIX="bcp_subset${MAX_NUM_QUESTIONS}_numsamplingepochs${NUM_SAMPLING_EPOCHS}_crossdoc_entity_combination"
CROSSDOC_COMBO_OUTPUT_FILE_PATH=${OUTPUT_DIR}/${CROSSDOC_COMBO_EXPR_PREFIX}.json
CROSSDOC_COMBO_LOG_FILE="${LOG_DIR}/${CROSSDOC_COMBO_EXPR_PREFIX}-log.txt"
echo "--- $(date) --- Cross-doc entity combination step..." | tee -a $CROSSDOC_COMBO_LOG_FILE
python3 with_neg_generate_crossdoc_entity_combination_cache.py \
    --ports "${PORTS[@]}" \
    --input_file_path ${SURFACE_ENTITY_OUTPUT_FILE_PATH} \
    --output_file_path ${CROSSDOC_COMBO_OUTPUT_FILE_PATH} \
    --qns_path $QNS_PATH \
    --max_num_questions $MAX_NUM_QUESTIONS \
    --min_docs_with_qa 2 \
    --include_source_qa_pairs \
    --max_other_qa_per_batch 20 \
    --num_hedges 3 \
    --checkpoint_iter_freq 5000 > ${CROSSDOC_COMBO_LOG_FILE}
