import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.data.chunkers import TokenChunker
from cartridges.data.resources import TextFileResource
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.utils.wandb import WandBConfig
from cartridges.data.resources import TextFileResource
from cartridges.clients.tokasaurus import TokasaurusClient

client = TokasaurusClient.Config(
    url=os.environ.get("CARTRIDGES_TOKASAURUS_QWEN3_4B_URL", "http://localhost:8000"),
    model_name="Qwen/Qwen3-4b",
)

config = SynthesizeConfig(

    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_thinking=0.2,
        tools=[],
        resources=[
            TextFileResource.Config(
                path=os.path.join(os.environ["CARTRIDGES_DIR"], "examples/arxiv/cartridges.tex"),
                seed_prompts=[
                    "structuring",
                    "summarization",
                    "question",
                    "use_case",
                    "creative",
                ],
                chunker=TokenChunker.Config(
                    tokenizer=client.model_name,
                    min_tokens_per_chunk=512,
                    max_tokens_per_chunk=1024,
                ),
            )
        ],
    ),

    num_samples=256, 
    batch_size=1,  
    max_num_batches_in_parallel=256,

    name=FormatStringVariable(f"{Path(__file__).stem}_{{synthesizer.client.model_name}}_n{{num_samples}}"),
    run_id=FormatStringVariable("{name}"),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    
    upload_to_wandb=False,
    save_wandb_preview=False,
    
    upload_to_hf=False,
    hf_repo_id="hazyresearch/{wandb_run_id}",
)


if __name__ == "__main__": 
    pydrantic.main([config])