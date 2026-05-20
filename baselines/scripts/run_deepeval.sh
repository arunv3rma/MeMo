#!/bin/bash
set -euo pipefail


export CUDA_VISIBLE_DEVICES=2

dataset="100papers_speculative_decoding_new_mcq"
# methods=("bm25" "hipporag2" "icl" "nv_embed")
# methods=("cartridges")
results_output_paths=(
    "baselines/output/bcp_300_with_negatives_8192-samples_8192-kvcachesize/cartridges_output.json"
    "baselines/output/bcp_300_with_negatives_8192-samples_8192-kvcachesize/cartridges_output_2.json"
    "baselines/output/bcp_300_with_negatives_8192-samples_8192-kvcachesize/cartridges_output_3.json"
)

for results_output_path in "${results_output_paths[@]}"; do
    eval_output_path=${results_output_path%.json}_eval_results_new.json
    summary_path=${results_output_path%.json}_summary_new.json

    echo "Starting deepeval for ${results_output_path}"
    python baselines/utils/deepeval_via_algo_utils.py \
        --generated_file_path ${results_output_path} \
        --output_path ${eval_output_path} \
        --summary_file_path ${summary_path}
    echo "Finished evaluating for $results_output_path"
done