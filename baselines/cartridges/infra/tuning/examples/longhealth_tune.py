import os
from pathlib import Path
import sys

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.data.longhealth.resources import LongHealthResource
from cartridges.utils.wandb import WandBConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tune_toka import EvaluateTokaConfig, TokaConfig


NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]

configs = []
# GLOBAL_BATCH_SIZE = 32
for batch_size in [32]:
    for num_workers in [4, 8, 16, 32]:
        for use_hydragen in [True]:
            port = 10100 + len(configs) * 10

            toka_config = TokaConfig(
                model="Qwen/Qwen3-4b",
                port=port,

                max_topk_logprobs=20,
                use_hydragen=use_hydragen,
                hydragen_min_group_size=32,
                cudagraph_max_size=16,
                
                wandb_enabled=True,
                wandb_project="tokasaurus",
                wandb_entity="hazy-research",
                
                wandb_run_name=FormatStringVariable(f"{Path(__file__).stem}_{patients_str}_bs{batch_size}_hydg{use_hydragen}_nw{num_workers}"),
            )


            synthesize_config = SynthesizeConfig(
                
                synthesizer=SelfStudySynthesizer.Config(
                    client=TokasaurusClient.Config(
                        url=f"http://0.0.0.0:{port}",
                        model_name="Qwen/Qwen3-4b",
                    ),
                    max_rounds=1,
                    prob_thinking=0.75,
                    use_tools_a=False, 
                    use_tools_b=False,
                    tools=[],
                    resources=[
                        LongHealthResource.Config(
                            seed_prompts=[
                                "structuring",
                                "summarization",
                                "question",
                                "use_case",
                                "creative",
                            ],
                            patient_ids=patient_ids,
                        )
                    ],
                ),
                output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
                num_samples=1024, 
                batch_size=batch_size,    # Smaller batches 
                
                max_num_batches_in_parallel=num_workers,

                name=FormatStringVariable(f"{Path(__file__).stem}_{patients_str}_bs{batch_size}_hydg{use_hydragen}_nw{num_workers}"),
                run_id=FormatStringVariable("{name}"),
                wandb=WandBConfig(
                    tags=[f"longhealth_synthesis"],
                ),
                upload_to_wandb=False,
                save_wandb_preview=False,
            )


        
            evaluate_config = EvaluateTokaConfig(
                synthesize=synthesize_config,
                tokasaurus=toka_config,
                conda_env="toka12",
            )

            configs.append(evaluate_config)





if __name__ == "__main__": 
    pydrantic.main(configs)