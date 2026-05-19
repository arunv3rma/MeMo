#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=postrain_merge
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=14
#SBATCH --output=/home/%u/logs/%x.out
#SBATCH --account=<slurm-account>

# cd into the cloned copy of this repo
cd <path-to-this-repo>

methods=("linear" "task" "ties" "dare_ties" "dare_linear")

for method in "${methods[@]}"; do
    srun python3 main.py \
        --method $method \
        --models \
        <path-to-finetuned-model-1> \
        <path-to-finetuned-model-2> \
        <path-to-finetuned-model-3> \
        --weight 1 1 1 \
        --density 0.56 0.56 0.56 \
        --base <path-to-base-model>

    # Run evals based on the most recent match
    #base_pattern="<checkpoints-dir>/merge_"
    #latest_dir=$(ls -td ${base_pattern}*_HF 2>/dev/null | head -n1)
    #if [ -n "$latest_dir" ]; then
    #    autoeval eval "$latest_dir" tier1
    #    autoeval eval "$latest_dir" tier2
    #else
    #    echo "No matching directory found for pattern: ${base_pattern}*_HF"
    #fi
done
