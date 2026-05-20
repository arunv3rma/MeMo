import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
dotenv_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.data.chunkers import TokenChunker
from cartridges.data.resources import TextFileResource
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.utils.wandb import WandBConfig
from cartridges.clients.tokasaurus import TokasaurusClient

# # 1. Setup the Client
client = TokasaurusClient.Config(
    url="http://localhost:10222",
    model_name="Qwen/Qwen2.5-32B-Instruct",
)

def get_config_for_file(file_path: Path):
    """
    Generates a SynthesizeConfig for a specific JSON file.
    Note: cartridges' TextFileResource usually expects raw text; 
    ensure your synthesizer can handle .json files via this resource.
    """
    return SynthesizeConfig(
        synthesizer=SelfStudySynthesizer.Config(
            client=client,
            max_rounds=1,
            prob_thinking=0.2,
            tools=[],
            resources=[
                TextFileResource.Config(
                    path=str(file_path),
                    seed_prompts=[
                        "question",
                        "summarization"
                    ],
                    chunker=TokenChunker.Config(
                        tokenizer=client.model_name,
                        min_tokens_per_chunk=512,
                        max_tokens_per_chunk=4096,
                    ),
                )
            ],
        ),
        num_samples=8192,  # Adjust as needed
        batch_size=1,  
        max_num_batches_in_parallel=128,
        # Dynamically name the run based on the JSON filename (stem)
        name=FormatStringVariable(f"{file_path.stem}_{{synthesizer.client.model_name}}_n{{num_samples}}"),
        run_id=FormatStringVariable("{name}"),
        output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
        upload_to_wandb=False,
        save_wandb_preview=False,
        upload_to_hf=False,
        hf_repo_id="hazyresearch/{wandb_run_id}",
    )

if __name__ == "__main__": 
    # Configuration - modify these variables as needed
    # mode = "grouped"  # "single" or "grouped"
    data_path = "data/bcp_300_with_negatives"
    # grouped_path = "data/browsecompplus_grouped"
    filter_prefix = None  # Set to "T" to filter files starting with T and beyond, or None for all

    # if mode == "grouped":
    #     # Process grouped files
    #     folder_path = Path(grouped_path)
    #     print(f"Processing grouped files from: {folder_path}")  
        
    #     # Check if grouped files exist, if not create them
    #     if not folder_path.exists() or not list(folder_path.glob("*.txt")):
    #         print(f"Grouped files not found. Creating groups of 10...")
    #         from cartridges.data.create_groups import create_groups_of_10
    #         source_path = Path(data_path)
    #         create_groups_of_10(str(source_path), str(folder_path), group_size=10)
    # else:

    # Process individual files
    folder_path = Path(data_path)
    print(f"Processing individual files from: {folder_path}")
    
    # 3. Filter for .txt files
    data_files = sorted(list(folder_path.glob("*.txt")))
    
    # Filter by prefix if specified
    if filter_prefix:
        data_files = [f for f in data_files if f.name[0].upper() >= filter_prefix.upper()]
        print(f"Filtered to {len(data_files)} files starting with '{filter_prefix}' and beyond")

    if not data_files:
        print(f"No files found in {folder_path}")
    else:
        print(f"Found {len(data_files)} files. Starting batch processing...")

        for file_path in data_files:
            print(f"\n>>> Processing: {file_path.name}")
            
            # Generate the specific config for this file
            current_config = get_config_for_file(file_path)
            
            # Execute via pydrantic
            try:
                pydrantic.main([current_config])
            except Exception as e:
                print(f"Failed to process {file_path.name}: {e}")
                raise