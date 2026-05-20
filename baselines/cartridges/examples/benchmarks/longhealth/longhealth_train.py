import os
from pathlib import Path
import socket

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.initialization import KVFromText
from cartridges.train import GenerationEvalConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import TrainDataset, DataSource
from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset
from cartridges.utils.wandb import WandBConfig


NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]

NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "2048"))

MODEL = os.environ.get("MODEL", "llama")
if MODEL == "llama":
    from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
    data_sources = [
        "hazyresearch/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-0",
        "hazyresearch/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-1",
        "hazyresearch/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-2"
    ]
    model = HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        model_cls=FlexLlamaForCausalLM,
    )
elif MODEL == "qwen":
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
    data_sources = [
        "hazyresearch/m07d11_longhealth_synthesize_qwen3-4b_p10_n65536-0",
        "hazyresearch/m07d11_longhealth_synthesize_qwen3-4b_p10_n65536-1"
    ]
    model=HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-4b",
        model_cls=FlexQwen3ForCausalLM,
    )
else:
    raise ValueError(f"Invalid model: {MODEL}")


config = TrainConfig(
    model=model,
    kv_cache_initializer=KVFromText.Config(
        max_tokens=NUM_TOKENS
    ),
    
    lr=2e-2,
    epochs=2,
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
            dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                patient_ids=patient_ids,
            ),
            name_for_wandb=f"longhealth_{patients_str}",
            generate_max_new_tokens=512,
            batch_size=32,
            temperature=0.3,
        )
    ],
    distributed_backend="gloo",

    wandb=WandBConfig(tags=["train", "longhealth"]),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=FormatStringVariable("longhealth_train_lr{lr}_toks{kv_cache_initializer.max_tokens}"),
)


if __name__ == "__main__":
    pydrantic.main(config)