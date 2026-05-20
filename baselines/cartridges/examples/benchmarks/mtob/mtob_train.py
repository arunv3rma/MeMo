import os
from pathlib import Path
import socket

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.data.mtob.evals import MTOBKalamangToEnglishGenerateDataset
from cartridges.initialization import KVFromText
from cartridges.train import GenerationEvalConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import DataSource, TrainDataset
from cartridges.utils.wandb import WandBConfig


NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "4096"))

MODEL = os.environ.get("MODEL", "llama")
if MODEL == "qwen":
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
    data_sources = [
        "hazyresearch/m07d28_mtob_synthesize_qwen3-4b_n65536-0",
        "hazyresearch/m07d28_mtob_synthesize_qwen3-4b_n65536-1"

    ]
    model=HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-4b",
        model_cls=FlexQwen3ForCausalLM,
    )
elif MODEL == "llama":
    from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
    # TODO: Note in the paper we also ran this on Llama-3.1-8B-Instruct. It is currently
    # on our backlog to run this experiment with Llama-3.1-8B-Instruct on the public, 
    # refactored repository. Leave an issue if you'd like to see the 8B datasets
    # and we will try to prioritize it.
    data_sources = [
        "hazyresearch/m07d28_mtob_synthesize_llama-3.2-3b_n65536-0",
        "hazyresearch/m07d28_mtob_synthesize_llama-3.2-3b_n65536-1"
    ]
    model = HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        model_cls=FlexLlamaForCausalLM,
    )
else:
    raise ValueError(f"Invalid model: {MODEL}")

configs = []
for lr in [2e-2]:
    config = TrainConfig(
        model=model,
        kv_cache_initializer=KVFromText.Config(
            max_tokens=NUM_TOKENS
        ),
        
        lr=lr,
        epochs=1,
        global_batch_size=32,

        dataset=TrainDataset.Config(
            data_sources=[DataSource(path=source, type="hf") for source in data_sources],
            top_k_logits=20,
            packed_seq_length=2048,
            packing_mode="truncate",
        ),

        save_every_n_steps=512,
        generate_eval_every_n_steps=128,
        generate_evals=[
            GenerationEvalConfig(
                name_for_wandb=f"mtob-ke-test",
                dataset=MTOBKalamangToEnglishGenerateDataset.Config(use_cot=False),
                batch_size=16,
                generate_max_new_tokens=128,
                num_samples=1,
                temperature=0,
            ),
        ],
        loss_eval_every_n_steps=512,
        loss_evals=[],
        distributed_backend="gloo",

        wandb=WandBConfig(tags=["train", "mtob"]),
        output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
        name=FormatStringVariable("mtob_train_lr{lr}_toks{kv_cache_initializer.max_tokens}"),
    )
    configs.append(config)

if __name__ == "__main__":
    pydrantic.main(configs)