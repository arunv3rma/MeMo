import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

dotenv_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

import pydrantic

from cartridges.initialization import KVFromText
from cartridges.train import TrainConfig, LossEvalConfig, GenerationEvalConfig
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM, FlexQwen2ForCausalLM
from cartridges.datasets import DataSource, GenerateEvalDataset, TrainDataset, LossEvalDataset


def get_config(text_file, train_data_file):
    return TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen2.5-32B-Instruct",
        model_cls=FlexQwen2ForCausalLM,
    ),
    kv_cache_initializer=KVFromText.Config(
        text_source=os.path.join(os.environ["CARTRIDGES_DIR"], text_file),
        max_tokens=8192
    ),
    
    lr=2e-2,
    epochs=1,
    global_batch_size=32,

    dataset=TrainDataset.Config(
        data_sources=[
            # TODO: replace below with your own dataset you just synthesized and 
            # remove our huggingface dataset below
            DataSource(path=train_data_file, type="local"),
        ],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),

    # loss_eval_every_n_steps=16,
    # loss_evals=[
    #     LossEvalConfig(
    #         dataset=LossEvalDataset.Config(
    #             data_source=DataSource(
    #                 path="hazyresearch/arxiv_synthesize_eval_gpt-5-mini-2025-08-07_n32-0",
    #                 type="hf",
    #             ),
    #             packed_seq_length=2048,
    #         ),
    #         name_for_wandb="arxiv_synthesize",
    #     )
    # ],

    # generate_eval_every_n_steps=128,
    # generate_evals=[
    #     GenerationEvalConfig(
    #         dataset=GenerateEvalDataset.Config(
    #             data_source=DataSource(
    #                 path="hazyresearch/arxiv_synthesize_eval_gpt-5-mini-2025-08-07_n32-0",
    #                 type="hf",
    #             ),
    #         ),
    #         name_for_wandb="arxiv-train",
    #         batch_size=16
    #     )
    # ],
    distributed_backend="nccl",

    save_every_n_steps=512,
    name="cartridges-tutorial-train",
)


if __name__ == "__main__":
    # Configuration for training mode
    # training_mode = "grouped"  # "single" or "grouped"
    
    # if training_mode == "grouped":
    #     parent_dir = "output/browsecompplus"
    # else:
    num_of_samples=8192
    parent_dir = f"output/bcp_300_with_negatives_{num_of_samples}-samples_8192-kvcachesize"
    
    file_paths = []

    # Check if the parent directory exists to avoid errors
    if os.path.exists(parent_dir):
        # 1. Loop through child directories at the first level only
        for child in os.listdir(parent_dir):
            child_dir = os.path.join(parent_dir, child)
            if os.path.isdir(child_dir) and "synthesize" in child_dir:
                for grandchild in os.listdir(child_dir):
                    grandchild_dir = os.path.join(child_dir, grandchild)
                    config_file = os.path.join(grandchild_dir, f"Qwen2.5-32B-Instruct_n{num_of_samples}-0/config.yaml")
                    with open(config_file, 'r') as file:
                        config = yaml.safe_load(file)
                    text_file = config["synthesizer"]["resources"][0]["path"]
                    train_dataset_file = os.path.join(
                        grandchild_dir, 
                        f"Qwen2.5-32B-Instruct_n{num_of_samples}-0/artifact/dataset.parquet"
                    )
                    name = text_file.split("/")[-1] if "/" in text_file else text_file
                    file_paths.append([text_file, train_dataset_file])
    
    print("Check first entry of file_paths: ", file_paths[0] if file_paths else "No files found")
    print("Total number of synthesized paths: ", len(file_paths))
    
    # 3. Pass the list of configs to pydrantic
    for path in file_paths:
        print(f"Text file: {path[0]}\nTrain dataset file: {path[1]}\n")
        print(f"Running for: {path[0]}")
        config = get_config(text_file=path[0], train_data_file=path[1])
        pydrantic.main(config)