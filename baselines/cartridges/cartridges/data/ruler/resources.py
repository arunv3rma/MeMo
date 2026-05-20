import json
import os
from typing import Dict, List, Optional, Tuple, Any
import random

from cartridges.data.ruler.niah import NIAHConfig, NIAHQuery, NIAHSample
from cartridges.data.ruler.variable_tracking import VariableTrackingConfig, VariableTrackingQuery, VariableTrackingSample
from cartridges.utils import get_logger
from cartridges.data.resources import Resource, sample_seed_prompts, SEED_TYPES
from cartridges.data.longhealth.utils import load_longhealth_dataset
logger = get_logger(__name__)

SYSTEM_PROMPT_TEMPLATE = """\
Please read the following chunks of text from a broader context and be prepared to answer questions about it.

{context}"""

DEFAULT_NAIH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data", "qwen3_4b-l32768-n1-k64-v1_2-essay-key_words-val_numbers-8995738516693885058.json")
DEFAULT_VARIABLE_TRACKING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data", "qwen3_4b-l100000-n1-c128-h3-noise-74c2a599.json")


class NIAHResource(Resource):
    class Config(Resource.Config):
        niah_path: Optional[str] = DEFAULT_NAIH_PATH
        sample_idx: int = 0

        sentences_per_chunk: Tuple[int, int] = (1, 4)
        chunks_per_prompt: Tuple[int, int] = (1, 1)
        
        seed_prompts: List[SEED_TYPES] = ["generic"]
        
    
    def __init__(self, config: Config):
        self.config = config

        with open(self.config.niah_path, "r") as f:
            self.data = json.load(f)
        
        # self.niah_config = NIAHConfig(**self.data["config"])

        sample = self.data["samples"][self.config.sample_idx]
        self.sample = NIAHSample(
            context=sample["context"],
            queries=[NIAHQuery(**query) for query in sample["queries"]]
        )

        from nltk.tokenize import sent_tokenize

        self.sentences = sent_tokenize(self.sample.context)
        

    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        num_chunks = random.randint(self.config.chunks_per_prompt[0], self.config.chunks_per_prompt[1])

        str_ctx = ""
        for _ in range(num_chunks):
            num_sentences = random.randint(self.config.sentences_per_chunk[0], self.config.sentences_per_chunk[1])
            chunk_start = random.randint(0, len(self.sentences) - num_sentences)
            
            str_ctx += " ".join(self.sentences[chunk_start:chunk_start + num_sentences])
            str_ctx += "\n\n"

        ctx = SYSTEM_PROMPT_TEMPLATE.format(context=str_ctx)
        seed_prompts = sample_seed_prompts(self.config.seed_prompts, batch_size)
        return ctx, seed_prompts
    
    def to_string(self) -> str:
        return self.sample.context


class VariableTrackingResource(Resource):
    class Config(Resource.Config):
        variable_tracking_path: Optional[str] = DEFAULT_VARIABLE_TRACKING_PATH
        sample_idx: int = 0

        sentences_per_chunk: Tuple[int, int] = (1, 4)
        chunks_per_prompt: Tuple[int, int] = (1, 1)
        
        seed_prompts: List[SEED_TYPES] = ["generic"]
    
    def __init__(self, config: Config):
        self.config = config

        with open(self.config.variable_tracking_path, "r") as f:
            self.data = json.load(f)
        
        sample = self.data["samples"][self.config.sample_idx]
        self.sample = VariableTrackingSample(
            context=sample["context"],
            queries=[VariableTrackingQuery(**query) for query in sample["queries"]]
        )

        from nltk.tokenize import sent_tokenize

        self.sentences = sent_tokenize(self.sample.context)
        

    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        num_chunks = random.randint(self.config.chunks_per_prompt[0], self.config.chunks_per_prompt[1])

        str_ctx = ""
        for _ in range(num_chunks):
            num_sentences = random.randint(self.config.sentences_per_chunk[0], self.config.sentences_per_chunk[1])
            chunk_start = random.randint(0, len(self.sentences) - num_sentences)
            
            str_ctx += " ".join(self.sentences[chunk_start:chunk_start + num_sentences])
            str_ctx += "\n\n"

        ctx = SYSTEM_PROMPT_TEMPLATE.format(context=str_ctx)
        seed_prompts = sample_seed_prompts(self.config.seed_prompts, batch_size)
        return ctx, seed_prompts
    
    def to_string(self) -> str:
        return self.sample.context
