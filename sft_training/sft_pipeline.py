#!/usr/bin/env python3
"""
Supervised Fine-Tuning of Qwen2.5-14B-Instruct on QA pairs.

Reads the output from generate_crossdoc_entity_combination_cache.py and trains
on question/answer pairs using Qwen2.5's chat template.

Input format (from generate_crossdoc_entity_combination_cache.py):
  {
    "qa_pairs_cache": [
      {
        "doc_id": "...",
        "type": "crossdoc",          // present on cross-doc entries
        "query_id": "...",            // present on cross-doc entries
        "source_doc_ids": [...],      // present on cross-doc entries
        "qa_pairs": [
          {"question": "...", "answer": "..."},
          ...
        ]
      },
      ...
    ]
  }

  When --include_source_qa_pairs was passed to generate_crossdoc_entity_combination_cache.py,
  the file also contains original surface-entity entries (same structure, no "type" field).

Usage:
    # Multi-GPU (all GPUs on node):
    accelerate launch --config_file accelerate_config.yaml mvp_sft_online_pipeline.py \\
        --data_path /path/to/crossdoc_cache.json \\
        --output_dir /path/to/output

    # Single GPU / debug:
    python mvp_sft_online_pipeline.py --data_path /path/to/crossdoc_cache.json --output_dir ./sft-out

Requirements:
    pip install torch transformers accelerate trl datasets deepspeed flash-attn
"""

import argparse
import json
import os
from typing import Optional

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import Any

import torch
from transformers import DataCollatorWithPadding
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="SFT of Qwen2.5-14B-Instruct on crossdoc QA pairs")

    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to crossdoc entity combination cache JSON "
                             "(output of generate_crossdoc_entity_combination_cache.py)")
    parser.add_argument("--val_split", type=float, default=0.02,
                        help="Fraction of QA pairs to hold out for validation (0 = no eval)")
    parser.add_argument("--crossdoc_only", action="store_true", default=False,
                        help="If set, only use entries with type='crossdoc'; skip surface-entity entries")
    parser.add_argument("--deduplicate_questions", action="store_true", default=False,
                        help="If set, skip exact-duplicate questions across entries for the same doc_id")

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./qwen25-14b-sft")

    # Training hyperparameters
    parser.add_argument("--max_seq_length", type=int, default=8096,
                        help="Max tokens per example (question + answer). Longer examples are dropped.")
    parser.add_argument("--per_device_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=4e-6)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                    help="Path to a checkpoint dir to resume from, or 'true' to auto-resume latest")
    parser.add_argument("--save_only_model", action="store_true", default=False,
                    help="Only save model weights in checkpoints, skip optimizer/scheduler state")

    # Precision & efficiency
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--tf32", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--flash_attn", action="store_true", default=True,
                        help="Use Flash Attention 2 (requires flash-attn package)")

    # Logging & saving
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=300)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--wandb_project", type=str, default="qwen25-sft",
                        help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (auto-generated if not set)")
    parser.add_argument("--wandb_api_key", type=str, default=None,
                        help="W&B API key (or set WANDB_API_KEY env var)")
    parser.add_argument("--expr_suffix", type=str, default="crossdoc")
    parser.add_argument("--world_size", type=int, default=1,
                        help="Number of GPUs/processes used for training (logged to W&B)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def extract_qa_pairs(
    data_path: str,
    crossdoc_only: bool = False,
    deduplicate: bool = False,
) -> list[dict]:
    """
    Parse the crossdoc entity combination cache JSON and return a flat list of
    {"question": ..., "answer": ..., "doc_id": ..., "type": ...} dicts.

    Expected input format:
      {
        "qa_pairs_cache": [
          {
            "doc_id": "...",
            "type": "crossdoc",       // optional; absent on surface-entity entries
            "query_id": "...",         // optional
            "source_doc_ids": [...],   // optional
            "qa_pairs": [{"question": "...", "answer": "..."}, ...]
          },
          ...
        ]
      }

    Skips entries with errors or empty qa_pairs.
    If crossdoc_only=True, only entries with type="crossdoc" are used.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Unwrap top-level wrapper if present
    if isinstance(data, dict) and "qa_pairs_cache" in data:
        records = data["qa_pairs_cache"]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"Unexpected data format in {data_path}: expected dict with 'qa_pairs_cache' or a list")

    print(f"Loaded {len(records)} records from {data_path}")

    qa_rows = []
    seen_questions: set[tuple] = set()  # (doc_id, question) for dedup

    skipped_error = 0
    skipped_empty = 0
    skipped_type = 0
    skipped_dedup = 0

    for record in records:
        # Skip errored records
        if "error" in record:
            skipped_error += 1
            continue

        entry_type = record.get("type")  # "crossdoc" or absent for surface-entity

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

    print(f"  Skipped {skipped_error} error records, "
          f"{skipped_empty} empty-pair records, "
          f"{skipped_type} non-crossdoc records, "
          f"{skipped_dedup} duplicate questions")
    print(f"  Total QA pairs extracted: {len(qa_rows)}")

    crossdoc_count = sum(1 for r in qa_rows if r["type"] == "crossdoc")
    surface_count = len(qa_rows) - crossdoc_count
    print(f"  Breakdown: {crossdoc_count} crossdoc pairs, {surface_count} surface-entity pairs")

    return qa_rows


def format_as_chat(example: dict, tokenizer) -> dict:
    """
    Convert a QA pair into a chat-formatted string using the model's
    chat template. The assistant turn (answer) is what the model learns to predict.

    Returns {"text": "<formatted chat string>"}.
    """
    messages = [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    # apply_chat_template adds the special tokens and role markers
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,  # We're training, not inferring
    )
    return {"text": text}


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

    if is_main_process:
        wandb.login(key=os.getenv("WANDB_API_KEY"))

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    import time
    import datetime
    # The timestamp (e.g., current time)
    timestamp = time.time()

    # Define GMT+8 using timedelta
    offset = datetime.timezone(datetime.timedelta(hours=8))

    # Convert timestamp to a timezone-aware datetime object
    dt = datetime.datetime.fromtimestamp(timestamp, tz=offset)

    # Format to dd/mm/yyyy
    formatted_date = dt.strftime("%d/%m/%Y %H:%M:%S")

    print(f"Original Timestamp: {timestamp}")
    print(f"Formatted Date (GMT+8): {formatted_date}")
    run_name = args.wandb_run_name or (
        f"{os.path.basename(args.model_name)}"
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
                "flash_attn": args.flash_attn,
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

    # Convert to HuggingFace Dataset
    hf_dataset = Dataset.from_list(qa_rows)

    # Apply chat template formatting
    hf_dataset = hf_dataset.map(
        lambda ex: format_as_chat(ex, tokenizer),
        desc="Applying chat template",
        remove_columns=["question", "answer", "doc_id", "type"],
    )

    # Optional: filter out examples that are too long
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

    # Train/val split
    if args.val_split > 0:
        split = hf_dataset.train_test_split(test_size=args.val_split, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"Split: {len(train_dataset)} train / {len(eval_dataset)} eval")
    else:
        train_dataset = hf_dataset
        eval_dataset = None

    # -------------------------------------------------------------------------
    # 4. Load model
    # -------------------------------------------------------------------------
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": "auto",
    }
    if args.flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # -------------------------------------------------------------------------
    # 5. SFT Config
    # -------------------------------------------------------------------------
    # Key differences from CPT:
    #   - packing=False  — each example is a single QA pair, not packed documents
    #   - train_on_responses_only wraps the trainer so only answer tokens
    #     contribute to the loss (question tokens are masked to -100)
    training_args = SFTConfig(
        output_dir=args.output_dir,

        # SFT-specific: one example per sequence, no packing
        max_length=args.max_seq_length,
        packing=False,
        dataset_text_field="text",

        # Batch / accumulation
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # Optimizer
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        optim="adamw_torch_fused",

        # Schedule
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,

        # Precision
        bf16=args.bf16,
        tf32=args.tf32,

        # Logging
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        logging_first_step=True,
        report_to=args.report_to,

        # Saving
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="epoch",
        save_only_model=args.save_only_model,

        # Eval
        eval_strategy="epoch" if eval_dataset is not None else "no",
        eval_steps=args.save_steps if eval_dataset is not None else None,

        # Misc
        seed=42,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        ddp_find_unused_parameters=False,
    )

    # -------------------------------------------------------------------------
    # 6. Create trainer
    # -------------------------------------------------------------------------
    # Train ONLY on the assistant answer tokens — masks the user question turn
    # from the loss. We implement this manually since DataCollatorForCompletionOnlyLM
    # was removed in TRL 0.29. For each sequence we find the last occurrence of the
    # response_template token ids and mask everything before it to -100.
    response_template_ids = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    resp_len = len(response_template_ids)
    resp_tensor = torch.tensor(response_template_ids)

    class CompletionOnlyCollator:
        """Pads a batch and masks all label tokens before the assistant response."""

        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
            first = features[0]

            if "input_ids" not in first:
                # Raw text — tokenize now
                texts = [f["text"] for f in features]
                encoded = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=training_args.max_length,
                    return_tensors="pt",
                )
            else:
                # Already tokenized — pad manually
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
                # Find the last occurrence of the response template in this sequence
                mask_until = len(seq)  # default: mask everything (safety fallback)
                for j in range(len(seq) - resp_len, -1, -1):
                    if torch.equal(seq[j: j + resp_len], resp_tensor):
                        # Keep response_template itself; mask everything before it
                        mask_until = j
                        break
                labels[i, :mask_until] = -100
                # Also mask padding tokens
                labels[i, encoded["attention_mask"][i] == 0] = -100

            encoded["labels"] = labels
            return encoded

    collator = CompletionOnlyCollator(tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )

    # -------------------------------------------------------------------------
    # 7. Train
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Starting SFT")
    print(f"  Model:               {args.model_name}")
    print(f"  Data:                {args.data_path}")
    print(f"  Train examples:      {len(train_dataset)}")
    print(f"  Eval examples:       {len(eval_dataset) if eval_dataset else 0}")
    print(f"  Crossdoc only:       {args.crossdoc_only}")
    print(f"  Max seq length:      {args.max_seq_length}")
    print(f"  Per-device batch:    {args.per_device_batch_size}")
    print(f"  Grad accumulation:   {args.gradient_accumulation_steps}")
    print(f"  Learning rate:       {args.learning_rate}")
    print(f"  Training epochs:     {args.num_train_epochs}")
    print(f"  BF16:                {args.bf16}")
    print(f"  Flash Attention:     {args.flash_attn}")
    print(f"{'='*60}\n")

    if args.resume_from_checkpoint:
        ckpt = True if args.resume_from_checkpoint.lower() == "true" else args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=ckpt)
    else:
        trainer.train()

    # -------------------------------------------------------------------------
    # 8. Save
    # -------------------------------------------------------------------------
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    # Load a clean copy of the tokenizer to avoid SPECIAL_TOKENS_ATTRIBUTES corruption
    # caused by SFTTrainer mutating the tokenizer object during training setup
    clean_tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if clean_tokenizer.pad_token is None:
        clean_tokenizer.pad_token = clean_tokenizer.eos_token
        clean_tokenizer.pad_token_id = clean_tokenizer.eos_token_id

    clean_tokenizer.save_pretrained(final_dir)
    print(f"\nModel and tokenizer saved to {final_dir}")

    if is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
