export CUDA_VISIBLE_DEVICES=0,1

# if serving on H200, only one GPU is needed, and the tensor parallel size should be set to 1

conda activate memo
python3 -m vllm.entrypoints.openai.api_server \
    --host localhost \
    --port 4325 \
    --trust-remote-code \
    --model Qwen/Qwen2.5-32B-Instruct \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --served-model-name qwen2_5_32b \
    --max-model-len 131072 \
    --rope-scaling '{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'