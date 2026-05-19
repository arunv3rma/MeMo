# script for lora training
NUM_EPOCHS=3

# =============================================================================
# Qwen2.5-14B-Instruct  (r=16, alpha=32 → ~0.5% trainable params)
# =============================================================================
MODEL_PATH=Qwen/Qwen2.5-14B-Instruct

# BCP  (effective batch size: 64 * 4 * 2 = 512)
EXPR_SUFFIX=qwen25-14b-lora-sft-epoch${NUM_EPOCHS}-bcp-batchsize512
OUTPUT_DIR=/path/to/save/model
accelerate launch --config_file accelerate_config.yaml sft_lora_pipeline.py \
    --data_path /path/to/bcp_subset300_numsamplingepochs1_step1-5_with_N_neg_docs.json \
    --model_name $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --val_split 0.0 \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --max_seq_length 8096 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --logging_steps 100 \
    --save_only_model \
    --world_size 2 \
    --wandb_run_name $EXPR_SUFFIX

# NQA  (effective batch size: 64 * 4 * 2 = 512)
EXPR_SUFFIX=qwen25-14b-lora-sft-epoch${NUM_EPOCHS}-nqa-batchsize512
OUTPUT_DIR=/path/to/save/model
accelerate launch --config_file accelerate_config.yaml sft_lora_pipeline.py \
    --data_path /path/to/NQA_validsplit_subset10_numsamplingepochs1_step1-5.json \
    --model_name $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --val_split 0.0 \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --max_seq_length 8096 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --logging_steps 100 \
    --save_only_model \
    --world_size 2 \
    --wandb_run_name $EXPR_SUFFIX

# MSQ  (effective batch size: 64 * 2 * 2 = 256)
EXPR_SUFFIX=qwen25-14b-lora-sft-epoch${NUM_EPOCHS}-msq-batchsize256
OUTPUT_DIR=/path/to/save/model
accelerate launch --config_file accelerate_config.yaml sft_lora_pipeline.py \
    --data_path /path/to/msq_subset1000_numsamplingepochs1_step1-5_with_N_neg_docs.json \
    --model_name $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --val_split 0.0 \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --max_seq_length 8096 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --logging_steps 100 \
    --save_only_model \
    --world_size 2 \
    --wandb_run_name $EXPR_SUFFIX

# =============================================================================
# Qwen2.5-1.5B-Instruct  (r=8, alpha=16 → ~0.60% trainable params)
# =============================================================================
MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct

# BCP  (effective batch size: 64 * 4 * 2 = 512)
EXPR_SUFFIX=qwen25-1.5b-lora-sft-epoch${NUM_EPOCHS}-bcp-batchsize512
OUTPUT_DIR=/path/to/save/model
accelerate launch --config_file accelerate_config.yaml sft_lora_pipeline.py \
    --data_path /path/to/bcp_subset300_numsamplingepochs1_step1-5_with_N_neg_docs.json \
    --model_name $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --val_split 0.0 \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --max_seq_length 8096 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --logging_steps 100 \
    --save_only_model \
    --world_size 2 \
    --wandb_run_name $EXPR_SUFFIX

# NQA  (effective batch size: 64 * 4 * 2 = 512)
EXPR_SUFFIX=qwen25-1.5b-lora-sft-epoch${NUM_EPOCHS}-nqa-batchsize512
OUTPUT_DIR=/path/to/save/model
accelerate launch --config_file accelerate_config.yaml sft_lora_pipeline.py \
    --data_path /path/to/NQA_validsplit_subset10_numsamplingepochs1_step1-5.json \
    --model_name $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --val_split 0.0 \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --max_seq_length 8096 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --logging_steps 100 \
    --save_only_model \
    --world_size 2 \
    --wandb_run_name $EXPR_SUFFIX

# MSQ  (effective batch size: 64 * 2 * 2 = 256)
EXPR_SUFFIX=qwen25-1.5b-lora-sft-epoch${NUM_EPOCHS}-msq-batchsize256
OUTPUT_DIR=/path/to/save/model
accelerate launch --config_file accelerate_config.yaml sft_lora_pipeline.py \
    --data_path /path/to/msq_subset1000_numsamplingepochs1_step1-5_with_N_neg_docs.json \
    --model_name $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --val_split 0.0 \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --max_seq_length 8096 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --logging_steps 100 \
    --save_only_model \
    --world_size 2 \
    --wandb_run_name $EXPR_SUFFIX

# =============================================================================
# Gemma3-1B-IT  (r=8, alpha=16 → ~0.65% trainable params)
# NOTE: flash_attn is not used for Gemma3 — the pipeline auto-detects the model
# name and falls back to attn_implementation=eager.
# =============================================================================
MODEL_PATH=google/gemma-3-1b-it

# BCP  (effective batch size: 64 * 4 * 2 = 512)
EXPR_SUFFIX=gemma3-1b-lora-sft-epoch${NUM_EPOCHS}-bcp-batchsize512
OUTPUT_DIR=/path/to/save/model
accelerate launch --config_file accelerate_config.yaml sft_lora_pipeline.py \
    --data_path /path/to/bcp_subset300_numsamplingepochs1_step1-5_with_N_neg_docs.json \
    --model_name $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --val_split 0.0 \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --max_seq_length 8096 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --logging_steps 100 \
    --save_only_model \
    --world_size 2 \
    --wandb_run_name $EXPR_SUFFIX

# NQA  (effective batch size: 64 * 4 * 2 = 512)
EXPR_SUFFIX=gemma3-1b-lora-sft-epoch${NUM_EPOCHS}-nqa-batchsize512
OUTPUT_DIR=/path/to/save/model
accelerate launch --config_file accelerate_config.yaml sft_lora_pipeline.py \
    --data_path /path/to/NQA_validsplit_subset10_numsamplingepochs1_step1-5.json \
    --model_name $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --val_split 0.0 \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --max_seq_length 8096 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --logging_steps 100 \
    --save_only_model \
    --world_size 2 \
    --wandb_run_name $EXPR_SUFFIX

# MSQ  (effective batch size: 64 * 2 * 2 = 256)
EXPR_SUFFIX=gemma3-1b-lora-sft-epoch${NUM_EPOCHS}-msq-batchsize256
OUTPUT_DIR=/path/to/save/model
accelerate launch --config_file accelerate_config.yaml sft_lora_pipeline.py \
    --data_path /path/to/msq_subset1000_numsamplingepochs1_step1-5_with_N_neg_docs.json \
    --model_name $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --val_split 0.0 \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --max_seq_length 8096 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --logging_steps 100 \
    --save_only_model \
    --world_size 2 \
    --wandb_run_name $EXPR_SUFFIX

# =============================================================================
# LFM2.5-1.2B-Instruct
# =============================================================================
MODEL_PATH=LiquidAI/LFM2.5-1.2B-Instruct

# BCP  (effective batch size: 64 * 4 * 2 = 512)
EXPR_SUFFIX=lfm1.2b-lora-sft-epoch${NUM_EPOCHS}-bcp-batchsize512
OUTPUT_DIR=/path/to/save/model
accelerate launch --config_file accelerate_config.yaml sft_lora_pipeline.py \
    --data_path /path/to/bcp_subset300_numsamplingepochs1_step1-5_with_N_neg_docs.json \
    --model_name $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --val_split 0.0 \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --max_seq_length 8096 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --logging_steps 100 \
    --save_only_model \
    --world_size 2 \
    --wandb_run_name $EXPR_SUFFIX

# NQA  (effective batch size: 64 * 4 * 2 = 512)
EXPR_SUFFIX=lfm1.2b-lora-sft-epoch${NUM_EPOCHS}-nqa-batchsize512
OUTPUT_DIR=/path/to/save/model
accelerate launch --config_file accelerate_config.yaml sft_lora_pipeline.py \
    --data_path /path/to/NQA_validsplit_subset10_numsamplingepochs1_step1-5.json \
    --model_name $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --val_split 0.0 \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --max_seq_length 8096 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --logging_steps 100 \
    --save_only_model \
    --world_size 2 \
    --wandb_run_name $EXPR_SUFFIX

# MSQ  (effective batch size: 64 * 2 * 2 = 256)
EXPR_SUFFIX=lfm1.2b-lora-sft-epoch${NUM_EPOCHS}-msq-batchsize256
OUTPUT_DIR=/path/to/save/model
accelerate launch --config_file accelerate_config.yaml sft_lora_pipeline.py \
    --data_path /path/to/msq_subset1000_numsamplingepochs1_step1-5_with_N_neg_docs.json \
    --model_name $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --val_split 0.0 \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --max_seq_length 8096 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --logging_steps 100 \
    --save_only_model \
    --world_size 2 \
    --wandb_run_name $EXPR_SUFFIX
