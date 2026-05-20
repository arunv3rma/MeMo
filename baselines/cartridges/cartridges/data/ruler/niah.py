# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

"""
Create a dataset jsonl file for needle in a haystack.

python niah.py \
    --save_dir=./ \
    --save_name=niah_single \
    --tokenizer_path=tokenizer.model \
    --tokenizer_type=nemo \
    --max_seq_length=4096 \
    --tokens_to_generate=128 \
    --num_samples=10 \
    --template="Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text? The special magic {type_needle_v} for {query} mentioned in the provided text are"
"""
from copy import copy
from dataclasses import dataclass
import hashlib
import os
import re
import json
from typing import List, Literal, Tuple
import uuid
import numpy as np
from pathlib import Path
import pydrantic
from tqdm import tqdm
import random
import sys
from collections import defaultdict
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize
import logging
from pydrantic import BaseConfig, RunConfig

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


CONTEXT_TEMPLATE = """Some special magic keys are hidden within the following text. Make sure to memorize them. I will quiz you about the keys afterwards.
{context}"""


QUERY_TEMPLATE = """What are all the special magic {type_needle_v} for {query} mentioned in the provided text?"""

ANSWER_PROMPT_TEMPLATE = """Please answer with the following format.
"The special magic {type_needle_v} for {query} mentioned in the provided text are: {{ {placeholder} }} "
Do not include any other text in your answer."""



from cartridges.data.ruler.constants import TASKS

class NIAHConfig(BaseConfig):
    max_seq_length: int = 100_000
    num_samples: int = 1
    tokens_to_generate: int = 128
    tokenizer: str = "Qwen/Qwen3-4B"

    context_template: str = CONTEXT_TEMPLATE
    query_template: str = QUERY_TEMPLATE

    num_needle_k: int = 1
    num_needle_v: int | Tuple[int, int] = 1

    type_haystack: Literal['essay', 'noise', 'needle'] = 'essay'
    type_needle_k: Literal['numbers', 'words', 'uuids'] = 'words'
    type_needle_v: Literal['numbers', 'words', 'uuids'] = 'numbers'
    
    model_template_token: int = 0
    seed: int = 42
    

class GenerateNIAHConfig(RunConfig):
    niah: NIAHConfig
    save_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")

    def run(self):
        # Set random seeds
        random.seed(self.niah.seed)
        np.random.seed(self.niah.seed)
                
        # Load tokenizer
        tokenizer = HFTokenizer(self.niah.tokenizer)


        # get a hash of the config
        tokenizer_str = self.niah.tokenizer.split("/")[-1].replace("-", "_").lower()
        config_hash = hashlib.sha256(str(self.niah.model_dump()).encode()).hexdigest()[:8]
        num_needle_v_str = f"{self.niah.num_needle_v[0]}_{self.niah.num_needle_v[1]}" if isinstance(self.niah.num_needle_v, tuple) else self.niah.num_needle_v
        config_str = f"{tokenizer_str}-l{self.niah.max_seq_length}-n{self.niah.num_samples}-k{self.niah.num_needle_k}-v{num_needle_v_str}-{self.niah.type_haystack}-key_{self.niah.type_needle_k}-val_{self.niah.type_needle_v}-{config_hash}"
        save_file = Path(self.save_dir) / f'{config_str}.json'
        save_file.parent.mkdir(parents=True, exist_ok=True)
        samples = generate_samples(
            config=self.niah,
            tokenizer=tokenizer
        )
        # Helper function to convert dataclass instances to dictionaries recursively
        def dataclass_to_dict(obj):
            if isinstance(obj, list):
                return [dataclass_to_dict(item) for item in obj]
            elif hasattr(obj, "__dataclass_fields__"):
                return {field: dataclass_to_dict(getattr(obj, field)) for field in obj.__dataclass_fields__}
            else:
                return obj

        # Convert samples to JSON format
        samples_json = {
            "config": self.niah.model_dump(),
            "samples": [dataclass_to_dict(sample) for sample in samples]
        }
        
        # Write the JSON data to the specified file
        with open(save_file, 'w') as f:
            json.dump(samples_json, f, indent=4)
        
        # write_manifest(save_file, write_jsons)
        print(f"Saved {len(samples_json)} samples to {save_file}")



@dataclass
class NIAHQuery:
    query: str
    answers: List[str]
    answer_prompt: str

@dataclass
class NIAHSample:
    context: str
    queries: List[NIAHQuery]

class HFTokenizer:
    def __init__(self, model_path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def text_to_tokens(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def tokens_to_text(self, tokens: List[int]) -> str:
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return text



def get_haystack(type_haystack: str):
    """Get haystack content based on type."""
    needle = "One of the special magic {type_needle_v} for {key} is: {value}."
    if type_haystack == 'essay':
        essay = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data/PaulGrahamEssays.json")
        essay = json.load(open(essay))['text']
        haystack = re.sub(r'\s+', " ", essay).split(" ")
    elif type_haystack == 'noise':
        haystack = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    elif type_haystack == 'needle':
        haystack = needle
    else:
        raise NotImplementedError(f'{type_haystack} is not implemented.')
    return haystack


# Words
import wonderwords
nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
# verbs = wonderwords.random_word._get_words_from_text_file("verblist.txt")
words = [f"{adj}-{noun}" for adj in adjs for noun in nouns]
words = sorted(list(set(words)))


# Positions
DEPTHS = list(np.round(np.linspace(0, 100, num=512, endpoint=True)).astype(int))


def generate_random_number(num_digits=7):
    lower_bound = 10**(num_digits - 1)
    upper_bound = 10**num_digits - 1
    return str(random.randint(lower_bound, upper_bound))

def generate_random_word():
    word = random.choice(words)
    return word

def generate_random_uuid():
    return str(uuid.UUID(int=random.getrandbits(128), version=4))

def generate_random(type_needle: str):
    if type_needle == 'numbers':
        return generate_random_number()
    elif type_needle == 'words':
        return generate_random_word()
    elif type_needle == 'uuids':
        return generate_random_uuid()
    else:
        raise NotImplementedError(f'{type_needle} is not implemented.')


@dataclass
class Needle:
    key: str
    values: str
    sentence: str

def generate_input_output(num_haystack, config: NIAHConfig):
    needle = "One of the special magic {type_needle_v} for {key} is: {value}."
    haystack = get_haystack(config.type_haystack)
    
    needles: List[Needle] = []
    for _ in range(config.num_needle_k):
        key = generate_random(config.type_needle_k)

        if isinstance(config.num_needle_v, int):
            num_needle_v = config.num_needle_v
        else:
            num_needle_v = random.randint(*config.num_needle_v)

        for _ in range(num_needle_v):
            value = generate_random(config.type_needle_v)
            needles.append(Needle(key=key, values=value, sentence=needle.format(
                type_needle_v=config.type_needle_v,
                key=key,
                value=value,
            )))

    random.Random(config.seed).shuffle(needles)

    # Context
    if config.type_haystack == 'essay':
        text = " ".join(haystack[:num_haystack])
        if num_haystack <= len(haystack):
            text = " ".join(haystack[:num_haystack])
        else:
            # Repeat haystack as many times as needed and slice to num_haystack
            repeats = (num_haystack + len(haystack) - 1) // len(haystack)  # Ceiling division
            text = " ".join((haystack * repeats)[:num_haystack])
        document_sents = sent_tokenize(text.strip())
        insertion_positions = [0] + \
                              sorted([int(len(document_sents) * (depth / 100)) for depth in random.sample(DEPTHS, len(needles))]) + \
                              [len(document_sents)]
        document_sents_list = []
        for i in range(1,len(insertion_positions)):
            last_pos = insertion_positions[i-1]
            next_pos = insertion_positions[i]
            document_sents_list.append(" ".join(document_sents[last_pos:next_pos]))
            if i-1 < len(needles):
                document_sents_list.append(needles[i-1].sentence)
        context = " ".join(document_sents_list)

    else:
        if config.type_haystack == 'noise':
            sentences = [haystack] * num_haystack
        elif config.type_haystack == 'needle':
            sentences = [haystack.format(
                type_needle_v=config.type_needle_v,
                key=generate_random(config.type_needle_k),
                value=generate_random(config.type_needle_v),
            ) for _ in range(num_haystack)]


        indexes = sorted(random.sample(range(num_haystack), len(needles)), reverse=True)
        for index, element in zip(indexes, needles):
            sentences.insert(index, element)
        context = "\n".join(sentences)

    # group by key
    needles_by_key = defaultdict(list)
    for needle in needles:
        needles_by_key[needle.key].append(needle)

    ## Query and Answer
    queries = []
    for key, needles in needles_by_key.items():
        answers = [needle.values for needle in needles]

        if len(answers) == 1:
            query_template = QUERY_TEMPLATE.replace('are all', 'is')
            type_needle_v = config.type_needle_v[:-1] # remove "s"
        else:
            query_template = QUERY_TEMPLATE
            type_needle_v = config.type_needle_v
        query_str = query_template.format(type_needle_v=type_needle_v, query=key)

        answer_prompt = ANSWER_PROMPT_TEMPLATE.format(
            type_needle_v=type_needle_v, 
            query=key, 
            placeholder="your comma separated answers" if len(answers) > 1 else "your answer"
        )

        if len(answers) == 1:
            query_str = query_str.replace('are all', 'is')
        queries.append(
            NIAHQuery(
                query=query_str,
                answers=answers,
                answer_prompt=answer_prompt
            )
        )
        
    

    context_template = config.context_template

    context = context_template.format(context=context)
    
    sample = NIAHSample(
        context=context,
        queries=queries,
    )

    return sample


def generate_samples(config: NIAHConfig, tokenizer: HFTokenizer, incremental: int = 500):
    write_jsons = []
    max_seq_length = config.max_seq_length - config.model_template_token

    if config.type_haystack == 'essay':
        incremental = 4096
    elif config.type_haystack == 'noise':
        incremental = 25
    elif config.type_haystack == 'needle':
        incremental = 25

    if config.type_haystack != 'essay' and config.max_seq_length < 4096:
        incremental = 5

    # Estimate tokens per question to determine reasonable upper bound
    curr_config = copy(config)
    curr_config.num_needle_k = int(incremental / max_seq_length * curr_config.num_needle_k)
    sample: NIAHSample = generate_input_output(incremental, curr_config)
    sample_tokens = len(tokenizer.text_to_tokens(sample.context + sample.queries[0].query))
    tokens_per_haystack = sample_tokens / incremental

    # Let's do 3x to allow for some slack since we can get unlucky due to sampling.
    # NOTE: We should test this for really large sequence lengths to make sure it's reasonable.
    estimated_max_questions = int((max_seq_length / tokens_per_haystack) * 3)

    # Binary search for optimal haystack size
    lower_bound = incremental
    upper_bound = max(estimated_max_questions, incremental * 2)  # Ensure upper_bound is reasonable

    optimal_num_haystack = None

    logger.info(f"Estimated {tokens_per_haystack:.1f} tokens per haystack")
    logger.info(f"Starting binary search with bounds: {lower_bound} to {upper_bound}")

    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound) // 2
        curr_config = copy(config)
        curr_config.num_needle_k = int(mid / max_seq_length * curr_config.num_needle_k)
        sample: NIAHSample = generate_input_output(mid, curr_config)
        total_tokens = len(tokenizer.text_to_tokens(sample.context + sample.queries[0].query)) + config.tokens_to_generate

        logger.info(f"Testing haystack size: {mid}, resulting tokens: {total_tokens}/{max_seq_length}")

        if total_tokens <= max_seq_length:
            # This size works, can we go larger?
            optimal_num_haystack = mid
            lower_bound = mid + 1
        else:
            # Too large, need to go smaller
            upper_bound = mid - 1

    num_haystack = optimal_num_haystack if optimal_num_haystack is not None else incremental
    logger.info(f'Final optimal haystack size (number of haystack): {num_haystack}')


    # Generate samples
    samples = []
    for index in tqdm(range(config.num_samples)):
        used_haystack = num_haystack
        print("start generating samples")
        for _ in range (1000):
            try:
                sample: NIAHSample = generate_input_output(used_haystack, config)
                length = len(tokenizer.text_to_tokens(sample.context + sample.queries[0].query)) + config.tokens_to_generate
                print(f"length: {length}, max_seq_length: {max_seq_length}")
                assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                break
            except Exception as e:
                print(f"Error: {e}")
                if used_haystack > incremental:
                    used_haystack -= incremental
        else:
            raise ValueError("Failed to generate samples")
        print("end generating samples")
        samples.append(sample)
    return samples




if __name__ == "__main__":
    config = GenerateNIAHConfig(
        niah=NIAHConfig(
            seed=42,
            context_template=CONTEXT_TEMPLATE,
            query_template=QUERY_TEMPLATE,
            num_needle_k=128,
            num_needle_v=(2,2),
            type_haystack='essay',
            type_needle_k='words',
            tokens_to_generate=128,
            num_samples=1,
            tokenizer="Qwen/Qwen3-4B",
        ),
    )
    pydrantic.main([config])
