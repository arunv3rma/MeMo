This folder provides for running a modified version of the Ruler benchmark.

## Setup

Install dependencies and download the essays.
```bash
pip install -r requirements.txt
python download_paulgraham_essay.py
```

## Instructions by task

### Variable Tracking

1. First, run `python variable_tracking.py` to generate the variable tracking dataset.
This will output something like:
```bash
Saved 1 samples to <cartridges-dir>/cartridges/data/ruler/_data/llama_3.2_3b_instruct-l10000-n1-c16-h2-essay-words-1d31e1f5.json
```
Copy that path to your clipboard.

There's a bunch of configuration options for controlling how the task will be synthesized.
```python
class VariableTrackingConfig(BaseConfig):
    max_seq_length: int = 100_000
    num_samples: int = 1
    tokens_to_generate: int = 30
    tokenizer: str = "Qwen/Qwen3-4B"

    context_template: str = CONTEXT_TEMPLATE

    num_chains: int = 1
    num_hops: int = 4

    type_haystack: Literal['essay', 'noise'] = 'noise'
    type_value: Literal['numbers', 'words', 'uuids'] = 'numbers'
    type_vars: Literal['numbers', 'words', 'uuids'] = 'words'
    assignment_format: Literal['python', 'javascript', 'words', 'ruler'] = 'ruler'
    remove_newline_tab: bool = False
    
    model_template_token: int = 0
    seed: int = 42
```

Some are from the original RULER paper, some I've added (`type_value`, `type_vars`, `assignment_format`) in order to experiment with data formats that are more compatible with self-study.

2. Run the ICL baseline by running an `EvaluateConfig` like the one below:

```python 
from cartridges.evaluate import EvaluateConfig, ICLBaseline
from cartridges.train import GenerationEvalConfig

config = EvaluateConfig(
    generator=ICLBaseline.Config(
        system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
        context=VariableTrackingResource.Config(variable_tracking_path=VT_PATH),
        ...
    ),
    eval=GenerationEvalConfig(
        dataset=VariableTrackingGenerateDataset.Config(
            variable_tracking_path=VT_PATH,
            thinking=False,
        ),
        ...
    ),
    ...
)
```
Note, make sure that `variable_tracking_path` points to the path of the dataset you just generated above!
The above is not a complete config. See, here is an example config:
```bash 
python cartridges/configs/sabri/m07d29_vt_baseline.py
```

3. Run self-study synthesis by creating a `SynthesizeConfig` like the one below:
```python 
from cartridges.synthesize import SynthesizeConfig, SelfStudySynthesizer

VT_PATH = "path/to/your/dataset.json"
config = SynthesizeConfig(
    
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_thinking=0.5,
        use_tools_a=False, 
        use_tools_b=False,
        tools=[],
        resources=[
            VariableTrackingResource.Config(
                seed_prompts=[
                    "structuring",
                    "summarization",
                    "question",
                    "use_case",
                    "creative",
                ],
                variable_tracking_path=VT_PATH,
                sentences_per_chunk=(1, 1),
                chunks_per_prompt=(1, 8),
            )
        ],
    ),
    ...
)
```
Note, make sure that `variable_tracking_path` points to the path of the dataset you just generated above!

The above is not a complete config. Here is an example config:
```bash
python cartridges/configs/sabri/m07d29_vt_synthesize.py
```

It will output something like:
```bash
INFO:cartridges.synthesize [rank=0]:Final output saved to /data/sabri/cartridges/2025-08-04-14-36-31-m07d29_vt_synthesize/m07d29_vt_synthesize_llama-3.2-3b_n65536-0/artifact/dataset.pkl
```
Copy that path to your clipboard.

4. Run training by creating a `TrainConfig` like the one below:
```python
VT_PATH = "path/to/your/dataset.json"
DATA_SOURCE = "path/to/your/synthesized_dataset.pkl"
config = TrainConfig(
    model=model,
    kv_cache_initializer=KVFromRandomText.Config(
        max_tokens=NUM_TOKENS
    ),
    
    lr=2e-2,
    epochs=2,
    global_batch_size=32,

    dataset=CartridgeTrainDataset.Config(
        data_sources=[
            (source, None)
        ],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),

    save_every_n_steps=512,
    generate_every_n_steps=128,
    generate_evals=[
        GenerationEvalConfig(
            dataset=VariableTrackingGenerateDataset.Config(
                variable_tracking_path=vt_path,
                thinking=False,
            ),
            name_for_wandb=f"variable_tracking",
            ...
        ),
    ],
    ...
)

```



