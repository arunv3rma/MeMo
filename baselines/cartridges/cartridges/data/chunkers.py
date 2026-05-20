from abc import ABC, abstractmethod

import random
from typing import List, Optional

from pydrantic import ObjectConfig
from transformers import AutoTokenizer


class Chunker(ABC):

    class Config(ObjectConfig):
        _pass_as_config: bool = True

    def __init__(self, text: str):
        self.text = text

    @abstractmethod
    def sample_chunk(self, text: str) -> List[str]:
        raise NotImplementedError


class TokenChunker(Chunker):

    class Config(Chunker.Config):
        tokenizer: str
        min_tokens_per_chunk: Optional[int] = 512
        max_tokens_per_chunk: int = 1024

    def __init__(self, config: Config, text: str):
        super().__init__(text)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)

        self.tokens = self.tokenizer.encode(text)

    def sample_chunk(self) -> List[str]:
        if self.config.min_tokens_per_chunk is None:
            tokens_in_chunk = self.config.max_tokens_per_chunk
        else:
            tokens_in_chunk = random.randint(
                self.config.min_tokens_per_chunk, 
                self.config.max_tokens_per_chunk
            )
        if tokens_in_chunk > len(self.tokens):
            return self.text
            
        start_idx = random.randint(0, len(self.tokens) - tokens_in_chunk)
        end_idx = start_idx + tokens_in_chunk
        print(f"tokens_in_chunk: {tokens_in_chunk}, start_idx: {start_idx}, end_idx: {end_idx}")
        return self.tokenizer.decode(self.tokens[start_idx:end_idx])


class CharacterChunker(Chunker):

    class Config(Chunker.Config):
        chunk_size: int = 1000
        overlap: int = 100

    def __init__(self, config: Config, text: str):
        super().__init__(text)
        self.config = config

    def sample_chunk(self) -> str:
        if len(self.text) <= self.config.chunk_size:
            return self.text
            
        max_start = len(self.text) - self.config.chunk_size
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + self.config.chunk_size
        return self.text[start_idx:end_idx]