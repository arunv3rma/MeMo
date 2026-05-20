import os
from pathlib import Path
import socket

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.data.ruler.evals import NIAHGenerateDataset
from cartridges.initialization import KVFromText
from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
from cartridges.train import GenerationEvalConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import DataSource, TrainDataset
from cartridges.utils.wandb import WandBConfig

NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "1024"))

NUM_KEYS = (2, 2)

MODEL = os.environ.get("MODEL", "llama")
if MODEL == "llama":
    DATA_SOURCES = [
        "hazyresearch/m07d28_niah_synthesize_llama-3.2-3b_n65536_k1-0",
        "hazyresearch/m07d28_niah_synthesize_llama-3.2-3b_n65536_k1-1",
    ]
    model = HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        model_cls=FlexLlamaForCausalLM,
    )
else:
    raise ValueError(f"Invalid model: {MODEL}")


import os as _os
NIAH_PATH = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)),
    "..", "..", "..", "cartridges", "data", "ruler", "_data",
    "qwen3_4b-l100000-n1-k128-v1_1-essay-key_words-val_numbers-e83970e8.json",
)


config = TrainConfig(
    model=model,
    kv_cache_initializer=KVFromText.Config(max_tokens=NUM_TOKENS),
    
    lr=2e-2,
    epochs=2,
    global_batch_size=32,

    dataset=TrainDataset.Config(
        data_sources=[DataSource(path=source, type="hf") for source in DATA_SOURCES],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),

    save_every_n_steps=512,
    generate_eval_every_n_steps=128,
    generate_evals=[
        GenerationEvalConfig(
            dataset=NIAHGenerateDataset.Config(
                niah_path=NIAH_PATH,
                thinking=True,
            ),
            name_for_wandb=f"niah_mc",
            num_samples=8,
            override_max_tokens=256,
            temperature=0.3,
            batch_size=16,
        ),
        
    ],
    loss_eval_every_n_steps=512,
    loss_evals=[],
    distributed_backend="gloo",

    wandb=WandBConfig(tags=["train", "niah"]),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=FormatStringVariable("niah_train_lr{lr}_toks{kv_cache_initializer.max_tokens}"),
)


if __name__ == "__main__":
    pydrantic.main(config)