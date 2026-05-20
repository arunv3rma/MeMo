from __future__ import annotations
import random
from typing import List, Literal

from transformers import AutoTokenizer

from cartridges.data.mtob.baseline import prompt_generic
from cartridges.data.resources import SEED_TYPES, Resource, sample_seed_prompts


class MTOBResource(Resource):
    class Config(Resource.Config):
        setup: Literal["latex_and_sentences", "medium_and_sentences"]
        seed_prompts: List[SEED_TYPES] = ["generic"]
        tokenizer: str = "Qwen/Qwen3-4b"
        min_chunk_size: int = 512
        max_chunk_size: int = 4096
        
    def __init__(self, config: Config):
        self.config = config
        self.content = prompt_generic(
            grammar_book=(
                "medium" if config.setup == "medium_and_sentences" else "latex"
            ),
            include_wordlist=False,
            include_sentences=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)

        self.tokens = self.tokenizer.encode(self.content)


    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        chunk = self._sample_chunk()
        seed_prompts = sample_seed_prompts(self.config.seed_prompts, batch_size)
        return chunk, seed_prompts

    def to_string(self) -> str:
        return self.content

    def _sample_chunk(self) -> str:
        chunk_size = random.randint(self.config.min_chunk_size, self.config.max_chunk_size)
        chunk_start = random.randint(0, len(self.tokens) - chunk_size)
        chunk_end = chunk_start + chunk_size
        chunk = self.tokenizer.decode(self.tokens[chunk_start:chunk_end])

        desc = "The following is an excerpt from a grammar book about the Kalamang language."
        chunk = f"{desc}\n\n{chunk}"
        
        return chunk



TEMPLATE = """Textbook
{textbook}

---
Wordlist
{word_list}

---
Sentences
{sentences}
"""


