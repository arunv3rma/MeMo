import os
from pathlib import Path
import pydrantic

from cartridges.initialization import KVFromText
from cartridges.initialization.pretrained import KVFromPretrained
from cartridges.train import TrainConfig, LossEvalConfig, GenerationEvalConfig
from cartridges.evaluate import LossEvalRunConfig
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM, FlexLlamaForCausalLM
from cartridges.datasets import DataSource, GenerateEvalDataset, TrainDataset, LossEvalDataset




config = LossEvalRunConfig(
    # model=HFModelConfig(
    #     pretrained_model_name_or_path="Qwen/Qwen3-4b",
    #     model_cls=FlexQwen3ForCausalLM,
    # ),
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        model_cls=FlexLlamaForCausalLM,
    ),
    kv_cache_initializer=KVFromText.Config(
        text_source=os.path.join(os.environ["CARTRIDGES_DIR"], "examples/arxiv/cartridges.tex"),
        max_tokens=None
    ),


    batch_size=16,

    eval=LossEvalConfig(
        dataset=LossEvalDataset.Config(
            data_source=DataSource(
                path="hazyresearch/arxiv_synthesize_eval_gpt-5-mini-2025-08-07_n32-0",
                type="hf",
            ),

        ),
        name_for_wandb="arxiv_synthesize",
    ),

    name="cartridges-tutorial-baseline",
)


if __name__ == "__main__":
    pydrantic.main(config)