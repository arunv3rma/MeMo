from collections import defaultdict
import os
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from cartridges.clients.base import (
    Client,
    Sample,
    SelectedToken,
    ClientConfig,
    ClientResponse,
)
from cartridges.clients.usage import Usage
from cartridges.utils import get_logger

from together import Together
from pydrantic import BaseConfig


# class TogetherClient:
#     def __init__(self, config: TogetherClientConfig):
#         self.config = config
#         self.client = Together(api_key=os.environ["TOGETHER_API_KEY"])


class TogetherClient(Client):

    class Config(ClientConfig):
        """Configuration options for the TogetherClient."""

        model_name: str = (
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"  # Default model name
        )
        api_key: Optional[str] = None

    def __init__(self, config: Config):
        """
        Initialize the Together client with the provided config.
        If config.api_key is set, it will override TOGETHER_API_KEY in the environment.
        """
        self.config = config
        self.api_key = (
            config.api_key if config.api_key else os.getenv("TOGETHER_API_KEY")
        )
        self.logger = get_logger("TogetherClient")

        # Initialize the Together API client (replace with actual Together client initialization if available)
        self.client = self.initialize_client()

    def initialize_client(self):
        """
        Initialize and return the Together API client.
        Replace this with the actual Together client initialization code.
        """
        # Example placeholder for Together client initialization
        # Replace with the correct Together client setup
        return Together(api_key=self.api_key)  # Placeholder for Together API client

    def complete(
        self,
        prompts: List[Union[str, List[int]]],
        temperature: float = 0.6,
        stop: List[str] = [],
        max_completion_tokens: int = 1,
        **kwargs,
    ) -> Sample:
        """
        Together does not directly support a `complete` API. Raise a `NotImplementedError` as a placeholder.
        """
        raise NotImplementedError("TogetherClient does not support a completion API.")

    def chat(
        self,
        chats: List[Dict[str, Any]],
        temperature: float = 0.6,
        max_completion_tokens: Optional[int] = None,
        stop: List[str] = [],
        top_logprobs: Optional[int] = None,
        **kwargs,
    ) -> ClientResponse:
        """
        Handle chat completions using the Together API in parallel using a thread pool.
        """
        assert len(chats) > 0, "Messages cannot be empty."
        assert (
            top_logprobs is None or top_logprobs == 1
        ), "Together does not support top_logprobs."

        samples = []
        usage = Usage()

        # Define a worker function to process each chat
        def process_chat(chat):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=chat,
                    max_tokens=max_completion_tokens,
                    temperature=temperature,
                    api_key=self.api_key,
                    stop=stop if stop else None,
                    # SE (03/01/2025): Together does not support top_logprobs.
                    logprobs=1 if top_logprobs is None else top_logprobs,
                )

                choice = response.choices[0]
                # Create Token objects for each token in the response
                
                if choice.logprobs is None:
                    print("Unexpected, investigate!")
                    breakpoint()

                tokens = [
                    SelectedToken(
                        text=token_text, id=token_id, logprob=logprob, top_logprobs=None
                    )
                    for token_text, token_id, logprob in zip(
                        choice.logprobs.tokens,
                        choice.logprobs.token_ids,
                        choice.logprobs.token_logprobs,
                    )
                ]

                sample = Sample(
                    text=choice.message.content,
                    tokens=tokens,
                    stop_reason=choice.finish_reason,
                )

                return sample, Usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                )

            except Exception as e:
                self.logger.error(f"Error during Together API call: {e}")
                raise

        # Use ThreadPoolExecutor to process chats in parallel
        with ThreadPoolExecutor(
            # TODO: do not hardcode this
            max_workers=16,
        ) as executor:
            # Submit all tasks and get futures
            futures = [executor.submit(process_chat, chat) for chat in chats]

            if self.config.show_progress_bar:
                for future in tqdm(
                    as_completed(futures),
                    desc="Waiting for responses from Together API",
                    total=len(futures),
                ):
                    pass

            # Process results as they complete
            for future in futures:
                try:
                    sample, chat_usage = future.result()
                    samples.append(sample)
                    usage += chat_usage
                except Exception as e:
                    self.logger.error(f"Error processing chat: {e}")
                    raise

        return ClientResponse(samples=samples, usage=usage)
