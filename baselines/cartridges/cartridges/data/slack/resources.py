
import os
import random
from typing import List
from .base import Resource


class SlackResource(Resource):

    class Config(Resource.Config):
        path: str = "database/slack"
        seed: int = 42
        text: str = ""
        prompts: List[str] = []


    def __init__(self, config: Config):
        self.config = config

        random.seed(self.config.seed)
        files = os.listdir(self.config.path)
        file = random.choice(files)
        with open(f"{self.config.path}/{file}", "r") as f:
            self.text = f.read()

        self.prompts = SLACK_PROMPTS


    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        return self.text, random.choices(self.prompts, k=batch_size)



SLACK_PROMPTS = [
    (
        "You are analyzing the Slack discussion around implementing a GEMM kernel on AMD GPUs. Please identify a key insight that was shared in the discussion and generate a question that can be used to test the user's understanding of the insight."
        "Be sure to include details (ids, concepts, names, titles, dates, etc.) in the question to make it clear what you are asking about. "
        "Answer only with the question, do not include any other text."
    ),

    (
        "You are analyzing the Slack discussion around implementing a GEMM kernel on AMD GPUs. Please identify something the user learned from the AMD experts and generate a question that can be used to test the user's understanding of the lesson."
        "Be sure to include details (ids, concepts, names, titles, dates, etc.) in the question to make it clear what you are asking about. "
        "Answer only with the question, do not include any other text."
    ),

    (
        "You are analyzing the Slack discussion around implementing a GEMM kernel on AMD GPUs. Please identify a key difficulties that the user faced during their work and generate a question that can be used to test the user's understanding of the difficulty."
        "Be sure to include details (ids, concepts, names, titles, dates, etc.) in the question to make it clear what you are asking about. "
            "Answer only with the question, do not include any other text."
    ),

    (
        "You are analyzing the Slack discussion around implementing a GEMM kernel on AMD GPUs. Please identify a key insight that was shared in the discussion and generate a question that can be used to test the user's understanding of the insight."
        "Be sure to include details (ids, concepts, names, titles, dates, etc.) in the question to make it clear what you are asking about. "
        "Answer only with the question, do not include any other text."
    ),

    (
        "You are analyzing the Slack discussion around implementing a GEMM kernel on AMD GPUs. Please generate a question that can be used to test the user's understanding of the discussion."
        "Be sure to include details (ids, concepts, names, titles, dates, etc.) in the question to make it clear what you are asking about. "
        "Answer only with the question, do not include any other text."
    ),

    (
        "You are analyzing the Slack discussion around implementing a GEMM kernel on AMD GPUs."
        "Please generate a single chat message instructing an LLM to summarize one sub section of the discussion."
    )
]


