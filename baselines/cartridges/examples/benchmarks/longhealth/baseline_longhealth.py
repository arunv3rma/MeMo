import os
from pathlib import Path

import pydrantic

from cartridges.clients.openai import CartridgeConfig, OpenAIClient
from cartridges.data.longhealth.resources import LongHealthResource
from cartridges.evaluate import ICLBaseline, GenerationEvalRunConfig
from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset
from cartridges.evaluate import GenerationEvalConfig

from cartridges.utils.wandb import WandBConfig


base_url = os.environ.get("CARTRIDGES_VLLM_QWEN3_4B_URL", "http://localhost:8000")

client = OpenAIClient.Config(
    base_url=os.path.join(base_url, "v1"),
    model_name="Qwen/Qwen3-4b",
)


SYSTEM_PROMPT_TEMPLATE = f"""Please reference the patient medical records included below to answer the user's questions.

<patient-records>
{{content}}
</patient-records>

Do not think for too long (only a few sentences, you only have 512 tokens to work with).
"""


NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]

configs = [
    GenerationEvalRunConfig(
        name=f"longhealth_mc_{patients_str}",
        generator=ICLBaseline.Config(
            client=client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            temperature=0.3,
            max_completion_tokens=2048,
            context=LongHealthResource.Config(
                patient_ids=patient_ids,
            ),
        ),
        eval=GenerationEvalConfig(
            dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                patient_ids=patient_ids, 
                cot=True,
            ),
            name_for_wandb=f"longhealth_mc",
            num_samples=1,
            temperature=0.3,
        ),
        max_num_batches_in_parallel=32,
        batch_size=32,
        wandb=WandBConfig(
            tags=[f"longhealth", "genbaseline", f"patients_{patients_str}", "icl"],
        ),
    )
]

if __name__ == "__main__":
    pydrantic.main(configs)
