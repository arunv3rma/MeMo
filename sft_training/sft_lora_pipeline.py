#!/usr/bin/env python3
"""
LoRA Supervised Fine-Tuning of a causal LM on QA pairs.

Identical data pipeline to mvp_sft_online_pipeline.py but trains only the
LoRA adapter weights.  After each checkpoint *and* at the end of training
the LoRA adapter is merged back into the base weights and the fully-merged
model is saved under <checkpoint>/merged/ (and <output_dir>/final/) so that
it can be loaded directly by vLLM without any PEFT runtime dependency.

Requirements:
    pip install torch transformers accelerate trl datasets peft deepspeed flash-attn
"""

import argparse
import json
import os
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA SFT on crossdoc QA pairs — checkpoints saved as merged weights"
    )

    # Data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--val_split", type=float, default=0.02)
    parser.add_argument("--crossdoc_only", action="store_true", default=False)
    parser.add_argument("--deduplicate_questions", action="store_true", default=False)

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./qwen25-14b-lora-sft")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha (scaling = alpha / r)")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="Comma-separated list of linear module names to apply LoRA to")
    parser.add_argument("--lora_bias", type=str, default="none",
                        choices=["none", "all", "lora_only"],
                        help="Which bias parameters to train alongside LoRA")

    # Training hyperparameters
    parser.add_argument("--max_seq_length", type=int, default=8096)
    parser.add_argument("--per_device_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to adapter checkpoint dir, or 'true' to auto-resume")
    parser.add_argument("--save_only_model", action="store_true", default=False)

    # Precision & efficiency
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--tf32", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--flash_attn", action="store_true", default=True)

    # Logging & saving
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=300)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--wandb_project", type=str, default="qwen25-lora-sft")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_api_key", type=str, default=None)
    parser.add_argument("--expr_suffix", type=str, default="crossdoc-lora")
    parser.add_argument("--world_size", type=int, default=1)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data helpers  (identical to mvp_sft_online_pipeline.py)
# ---------------------------------------------------------------------------

def extract_qa_pairs(
    data_path: str,
    crossdoc_only: bool = False,
    deduplicate: bool = False,
) -> list[dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "qa_pairs_cache" in data:
        records = data["qa_pairs_cache"]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(
            f"Unexpected data format in {data_path}: expected dict with "
            f"'qa_pairs_cache' or a list"
        )

    print(f"Loaded {len(records)} records from {data_path}")

    qa_rows: list[dict] = []
    seen_questions: set[tuple] = set()

    skipped_error = skipped_empty = skipped_type = skipped_dedup = 0

    for record in records:
        if "error" in record:
            skipped_error += 1
            continue

        entry_type = record.get("type")
        if crossdoc_only and entry_type != "crossdoc":
            skipped_type += 1
            continue

        doc_id = record.get("doc_id", "unknown")
        pairs = record.get("qa_pairs", [])
        if not isinstance(pairs, list) or not pairs:
            skipped_empty += 1
            continue

        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            question = (pair.get("question") or "").strip()
            answer = (pair.get("answer") or "").strip()
            if not question or not answer:
                continue

            if deduplicate:
                key = (doc_id, question)
                if key in seen_questions:
                    skipped_dedup += 1
                    continue
                seen_questions.add(key)

            qa_rows.append({
                "question": question,
                "answer": answer,
                "doc_id": doc_id,
                "type": entry_type or "surface",
            })

    print(
        f"  Skipped {skipped_error} error records, "
        f"{skipped_empty} empty-pair records, "
        f"{skipped_type} non-crossdoc records, "
        f"{skipped_dedup} duplicate questions"
    )
    print(f"  Total QA pairs extracted: {len(qa_rows)}")

    crossdoc_count = sum(1 for r in qa_rows if r["type"] == "crossdoc")
    print(
        f"  Breakdown: {crossdoc_count} crossdoc pairs, "
        f"{len(qa_rows) - crossdoc_count} surface-entity pairs"
    )
    return qa_rows


def format_as_chat(example: dict, tokenizer) -> dict:
    messages = [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ---------------------------------------------------------------------------
# Callback: merge adapter into base weights after every checkpoint save
# ---------------------------------------------------------------------------

class MergeAndSaveCallback(TrainerCallback):
    """
    After each checkpoint is written, load the adapter on top of the frozen
    base model (on CPU to avoid disturbing the training GPUs), merge the LoRA
    weights, and save a fully-merged HF model to <checkpoint_dir>/merged/.

    This lets vLLM (or any HF inference stack) load the checkpoint without
    knowing anything about PEFT.
    """

    def __init__(
        self,
        base_model_name: str,
        tokenizer,
        model_dtype: torch.dtype,
        attn_implementation: str,
    ):
        self.base_model_name = base_model_name
        self.tokenizer = tokenizer
        self.model_dtype = model_dtype
        self.attn_implementation = attn_implementation

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return

        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        merged_dir = os.path.join(checkpoint_dir, "merged")

        if not os.path.isdir(checkpoint_dir):
            print(
                f"[MergeAndSaveCallback] checkpoint dir not found: {checkpoint_dir} — skipping merge"
            )
            return

        print(f"\n[MergeAndSaveCallback] Merging LoRA into base weights → {merged_dir}")

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=self.model_dtype,
            trust_remote_code=True,
            device_map="cpu",
            attn_implementation=self.attn_implementation,
        )
        peft_model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        self.tokenizer.save_pretrained(merged_dir)

        del merged_model, peft_model, base_model
        torch.cuda.empty_cache()
        print(f"[MergeAndSaveCallback] Merged checkpoint saved to {merged_dir}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # -------------------------------------------------------------------------
    # 1. Initialise W&B
    # -------------------------------------------------------------------------
    import wandb
    from dotenv import load_dotenv
    load_dotenv()

    is_main_process = int(os.environ.get("RANK", "0")) == 0

    # Resolve attention implementation early — needed for both W&B config and
    # model loading. Gemma3 requires eager; all others default to flash_attention_2.
    is_gemma = "gemma" in args.model_name.lower()
    if is_gemma:
        attn_implementation = "eager"
    else:
        attn_implementation = "flash_attention_2" if args.flash_attn else "eager"

    if is_main_process:
        wandb.login(key=os.getenv("WANDB_API_KEY"))

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    import datetime
    import time

    timestamp = time.time()
    offset = datetime.timezone(datetime.timedelta(hours=8))
    dt = datetime.datetime.fromtimestamp(timestamp, tz=offset)
    formatted_date = dt.strftime("%d/%m/%Y %H:%M:%S")

    print(f"Original Timestamp: {timestamp}")
    print(f"Formatted Date (GMT+8): {formatted_date}")

    run_name = args.wandb_run_name or (
        f"{os.path.basename(args.model_name)}"
        f"_lora-r{args.lora_r}"
        f"_lr{args.learning_rate}"
        f"_bs{args.per_device_batch_size}x{args.gradient_accumulation_steps}"
        f"_ep{args.num_train_epochs}"
        f"_suffix{args.expr_suffix}"
        f"_start{formatted_date}"
    )

    if is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model_name": args.model_name,
                "data_path": args.data_path,
                "crossdoc_only": args.crossdoc_only,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lora_target_modules": args.lora_target_modules,
                "max_seq_length": args.max_seq_length,
                "per_device_batch_size": args.per_device_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "num_train_epochs": args.num_train_epochs,
                "warmup_ratio": args.warmup_ratio,
                "weight_decay": args.weight_decay,
                "lr_scheduler_type": args.lr_scheduler_type,
                "max_grad_norm": args.max_grad_norm,
                "bf16": args.bf16,
                "attn_implementation": attn_implementation,
                "gradient_checkpointing": args.gradient_checkpointing,
                "world_size": args.world_size,
            },
        )

    # -------------------------------------------------------------------------
    # 2. Load tokenizer
    # -------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"\nTokenizer: {args.model_name}")
    print(f"  eos={tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
    print(f"  pad={tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")

    # -------------------------------------------------------------------------
    # 3. Load & format dataset
    # -------------------------------------------------------------------------
    qa_rows = extract_qa_pairs(
        args.data_path,
        crossdoc_only=args.crossdoc_only,
        deduplicate=args.deduplicate_questions,
    )

    if not qa_rows:
        raise ValueError("No QA pairs found! Check your data file and --crossdoc_only flag.")

    hf_dataset = Dataset.from_list(qa_rows)
    hf_dataset = hf_dataset.map(
        lambda ex: format_as_chat(ex, tokenizer),
        desc="Applying chat template",
        remove_columns=["question", "answer", "doc_id", "type"],
    )

    def tokenize_length(example):
        ids = tokenizer(example["text"], truncation=False)["input_ids"]
        return {"length": len(ids)}

    hf_dataset = hf_dataset.map(tokenize_length, desc="Computing lengths")
    before = len(hf_dataset)
    hf_dataset = hf_dataset.filter(lambda ex: ex["length"] <= args.max_seq_length)
    after = len(hf_dataset)
    if before != after:
        print(f"Dropped {before - after} examples exceeding max_seq_length={args.max_seq_length}")
    hf_dataset = hf_dataset.remove_columns(["length"])

    print(f"\nFinal dataset size: {len(hf_dataset)} examples")

    if args.val_split > 0:
        split = hf_dataset.train_test_split(test_size=args.val_split, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"Split: {len(train_dataset)} train / {len(eval_dataset)} eval")
    else:
        train_dataset = hf_dataset
        eval_dataset = None

    # -------------------------------------------------------------------------
    # 4. Load base model
    # -------------------------------------------------------------------------
    # Gemma3 requires eager attention — flash_attention_2 produces incorrect
    # results or errors with Gemma3. All other models use flash_attention_2
    # when --flash_attn is set (the default).
    if is_gemma:
        print("\nNote: Gemma model detected — using attn_implementation='eager'")

    model_kwargs: dict = {
        "trust_remote_code": True,
        "torch_dtype": "auto",
        "attn_implementation": attn_implementation,
    }

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    # Resolve the actual dtype used (needed later for the merge callback)
    actual_dtype = next(model.parameters()).dtype

    # Disable caching — incompatible with gradient checkpointing
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # -------------------------------------------------------------------------
    # 5. Apply LoRA
    # -------------------------------------------------------------------------
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias=args.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )

    # -------------------------------------------------------------------------
    # 6. SFT Config
    # -------------------------------------------------------------------------
    training_args = SFTConfig(
        output_dir=args.output_dir,

        max_length=args.max_seq_length,
        packing=False,
        dataset_text_field="text",

        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        optim="adamw_torch_fused",

        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,

        bf16=args.bf16,
        tf32=args.tf32,

        logging_steps=args.logging_steps,
        logging_strategy="steps",
        logging_first_step=True,
        report_to=args.report_to,

        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="epoch",
        save_only_model=args.save_only_model,

        eval_strategy="epoch" if eval_dataset is not None else "no",
        eval_steps=args.save_steps if eval_dataset is not None else None,

        seed=42,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        ddp_find_unused_parameters=False,
    )

    # -------------------------------------------------------------------------
    # 7. Completion-only collator (same as full-SFT pipeline)
    # -------------------------------------------------------------------------
    # Gemma3 uses <start_of_turn>model\n; Qwen and most others use <|im_start|>assistant\n
    response_template = "<start_of_turn>model\n" if is_gemma else "<|im_start|>assistant\n"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)
    resp_len = len(response_template_ids)
    resp_tensor = torch.tensor(response_template_ids)

    class CompletionOnlyCollator:
        """Pads a batch and masks all label tokens before the assistant response."""

        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
            first = features[0]

            if "input_ids" not in first:
                texts = [f["text"] for f in features]
                encoded = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=training_args.max_length,
                    return_tensors="pt",
                )
            else:
                input_ids = [torch.tensor(f["input_ids"]) for f in features]
                if "attention_mask" in first:
                    attn = [torch.tensor(f["attention_mask"]) for f in features]
                else:
                    attn = [torch.ones(len(f["input_ids"]), dtype=torch.long) for f in features]
                encoded = {
                    "input_ids": torch.nn.utils.rnn.pad_sequence(
                        input_ids, batch_first=True,
                        padding_value=self.tokenizer.pad_token_id,
                    ),
                    "attention_mask": torch.nn.utils.rnn.pad_sequence(
                        attn, batch_first=True, padding_value=0,
                    ),
                }

            labels = encoded["input_ids"].clone()

            for i, seq in enumerate(labels):
                mask_until = len(seq)
                for j in range(len(seq) - resp_len, -1, -1):
                    if torch.equal(seq[j: j + resp_len], resp_tensor):
                        mask_until = j
                        break
                labels[i, :mask_until] = -100
                labels[i, encoded["attention_mask"][i] == 0] = -100

            encoded["labels"] = labels
            return encoded

    collator = CompletionOnlyCollator(tokenizer)

    # -------------------------------------------------------------------------
    # 8. Create trainer
    # -------------------------------------------------------------------------
    merge_callback = MergeAndSaveCallback(
        base_model_name=args.model_name,
        tokenizer=tokenizer,
        model_dtype=actual_dtype,
        attn_implementation=attn_implementation,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        peft_config=lora_config,
        callbacks=[merge_callback],
    )

    # -------------------------------------------------------------------------
    # 9. Train
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Starting LoRA SFT")
    print(f"  Model:               {args.model_name}")
    print(f"  Data:                {args.data_path}")
    print(f"  Train examples:      {len(train_dataset)}")
    print(f"  Eval examples:       {len(eval_dataset) if eval_dataset else 0}")
    print(f"  LoRA r:              {args.lora_r}")
    print(f"  LoRA alpha:          {args.lora_alpha}")
    print(f"  LoRA target modules: {args.lora_target_modules}")
    print(f"  Max seq length:      {args.max_seq_length}")
    print(f"  Per-device batch:    {args.per_device_batch_size}")
    print(f"  Grad accumulation:   {args.gradient_accumulation_steps}")
    print(f"  Learning rate:       {args.learning_rate}")
    print(f"  Training epochs:     {args.num_train_epochs}")
    print(f"  BF16:                {args.bf16}")
    print(f"  Attn implementation: {attn_implementation}")
    print(f"{'='*60}\n")

    if args.resume_from_checkpoint:
        ckpt = (
            True
            if args.resume_from_checkpoint.lower() == "true"
            else args.resume_from_checkpoint
        )
        trainer.train(resume_from_checkpoint=ckpt)
    else:
        trainer.train()

    # -------------------------------------------------------------------------
    # 10. Save — adapter only first, then merged for vLLM
    # -------------------------------------------------------------------------
    # (a) Save the raw adapter (lightweight, useful for further fine-tuning)
    adapter_dir = os.path.join(args.output_dir, "final_adapter")
    trainer.save_model(adapter_dir)

    # (b) Merge adapter into base weights and save a standalone HF model
    if is_main_process:
        final_merged_dir = os.path.join(args.output_dir, "final")
        print(f"\nMerging LoRA adapter into base weights → {final_merged_dir}")

        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=actual_dtype,
            trust_remote_code=True,
            device_map="cpu",
            attn_implementation=attn_implementation,
        )
        peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(final_merged_dir)

        # Save a clean tokenizer to avoid SPECIAL_TOKENS_ATTRIBUTES corruption
        clean_tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            use_fast=True,
        )
        if clean_tokenizer.pad_token is None:
            clean_tokenizer.pad_token = clean_tokenizer.eos_token
            clean_tokenizer.pad_token_id = clean_tokenizer.eos_token_id
        clean_tokenizer.save_pretrained(final_merged_dir)

        del merged_model, peft_model, base_model
        torch.cuda.empty_cache()

        print(f"Merged model saved to {final_merged_dir}")
        print(f"Raw adapter saved to  {adapter_dir}")

        wandb.finish()


if __name__ == "__main__":
    main()
