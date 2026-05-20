import asyncio
import aiohttp
import json
import time
from typing import Any, Dict, List, Literal, Optional
import requests
import base64


import numpy as np
from openai.types.chat.chat_completion import ChatCompletion

from cartridges.clients.base import (
    Client,
    ClientSample,
    ClientConfig,
    ClientResponse,
    TopLogprobs,
    CartridgeConfig
)
from cartridges.clients.usage import Usage
from cartridges.utils import get_logger
from cartridges.utils.thinking import MODEL_TO_THINKING_OVERRIDES, add_thinking_prompt


logger = get_logger(__name__)


class TokasaurusClient(Client):
    """Client for Tokasaurus with async gather support for batch calls."""

    class Config(ClientConfig):
        """Configuration options for the TokasaurusClient."""

        model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
        url: str
        
        # we have pretty robust timeout and retry logic in the client because
        # we have found that sometimes requests to the server just hang. 
        # having a growing timeout is useful, because it allows us to have a relatively
        # short initial timeout, which will quickly catch these hangs, without 
        # breaking the longer-running requests.
        max_retries: int = 10
        base_timeout: int = 90
        timeout_multiplier: float = 1.5

        on_failure: Literal["raise", "continue"] = "raise"

        cartridges: Optional[List[CartridgeConfig]] = None

    def __init__(self, config: Config):
        """Initialize the Tokasaurus client with the provided config."""
        self.config = config
        self.logger = get_logger("TokasaurusClient")

        # Ensure that the Tokasaurus server is running the correct model
        if self.config.model_name != "default":
            try:
                r = requests.get(f"{self.config.url}/v1/models")
                r.raise_for_status()
                data = r.json()
                assert len(data["data"]) == 1, "Expected exactly one model"
                model_id = data["data"][0]["id"]
                if model_id.lower() != self.config.model_name.lower():
                    raise ValueError(f"Expected model {self.config.model_name}, got {model_id} from tokasaurus")
            except Exception as e:
                self.logger.error(f"Failed to get model id: {e}")
                raise e

        if self.config.cartridges is not None:
            self.cartridges = [c.model_dump() for c in self.config.cartridges]
        else:
            self.cartridges = None

    async def _send_requests(self, requests: list[dict], modal_upstream_id: Optional[str] = None, use_cartridge_endpoint: bool = False) -> dict:
        """Send a single request to the server with retries."""
        if modal_upstream_id is not None:
            headers = {
                # "X-Modal-Flash-Upstream": modal_upstream_id,
            }
        else:
            headers = {}
        response = None
        for retry_idx in range(self.config.max_retries):
            try:
                timeout = self.config.base_timeout * (self.config.timeout_multiplier ** retry_idx)
                
                import pickle
                t0 = time.time()
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                    endpoint = "/custom/synchronous-batch-completions" if not use_cartridge_endpoint else "/batch/cartridge/chat/completions"
                    async with session.post(
                        f"{self.config.url}{endpoint}",
                        json={"requests": requests},
                        headers=headers,
                    ) as resp:
                        if resp.status != 200:
                            # Get response text for better error info
                            error_text = await resp.text()
                            error_msg = f"HTTP {resp.status}: {error_text}"
                            raise aiohttp.ClientResponseError(
                                request_info=resp.request_info,
                                history=resp.history,
                                status=resp.status,
                                message=error_msg
                            )
                        response = await resp.content.read()
                
                # print(f"batch/chat/completions took {time.time() - t0} seconds")
                t0 = time.time()
                response = pickle.loads(response)
                # print(f"pickle.loads took {time.time() - t0} seconds")
                break
            except Exception as e:
                logger.warning(f"Error sending request (retry {retry_idx + 1}/{self.config.max_retries}): {type(e).__name__}: {e}")
                response = e
                # Also use exponential backoff for sleep: 1 * (2 ** retry_idx) seconds
                sleep_duration = 1 * (2 ** retry_idx)
                await asyncio.sleep(sleep_duration)
                continue
    
        if response is None or isinstance(response, Exception):
            logger.error(f"Failed to get response after {self.config.max_retries} retries")
            if self.config.on_failure == "raise":
                raise Exception("Failed to get response from server")
        
        return response

    def _extract_fingerprint_logprobs(self, fingerprint_data: dict) -> Optional[TopLogprobs]:
        """Extract logprobs data from the fingerprint if available."""
        try:
            def decode_array(encoded_str):
                """Decode base64 encoded numpy array."""
                return np.frombuffer(base64.b64decode(encoded_str), dtype=np.float32)
            
            # Check if logprobs data is available
            if not fingerprint_data.get("packed_chosen_logprobs") or not fingerprint_data.get("packed_topk_indices"):
                return None
            
            # For single sequence, take the first element
            packed_chosen_logprobs = fingerprint_data["packed_chosen_logprobs"][0]
            packed_topk_indices = fingerprint_data["packed_topk_indices"][0] 
            packed_topk_logprobs = fingerprint_data["packed_topk_logprobs"][0]
            
            # Decode the arrays
            chosen_logprobs = decode_array(packed_chosen_logprobs)
            topk_indices = np.frombuffer(base64.b64decode(packed_topk_indices), dtype=np.int32)
            topk_logprobs = np.frombuffer(base64.b64decode(packed_topk_logprobs), dtype=np.float32)
            
            # Reshape topk arrays - they should be (num_tokens, topk_size)
            num_tokens = len(chosen_logprobs)
            if num_tokens == 0:
                return None
                
            topk_size = len(topk_indices) // num_tokens
            topk_indices = topk_indices.reshape(num_tokens, topk_size)
            topk_logprobs = topk_logprobs.reshape(num_tokens, topk_size)
            
            return TopLogprobs(
                logprobs=topk_logprobs,
                token_ids=topk_indices,
            )
        except Exception as e:
            logger.warning(f"Failed to extract logprobs from fingerprint: {e}")
        return None


    async def chat(
        self,
        chats: List[List[Dict[str, Any]]],
        max_completion_tokens: int,
        temperature: float = 0.6,
        stop: Optional[List[str]] = None,
        top_logprobs: Optional[int] = None,
        modal_upstream_id: Optional[str] = None,
        enable_thinking: bool = False,
        cartridges: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ClientResponse:
        """
        Send batch chat requests using async gather.
        
        Args:
            chats: List of chat conversations, each containing message dicts
            max_completion_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences (not supported)
            top_logprobs: Number of top logprobs to return
            modal_upstream_id: Optional ID which will determine which modal replica to 
                use. If provided, two requests with the same modal_upstream_id will be 
                sent to the same modal replica. This is useful for utilizing the 
                KV cache. This will work with Tokasaurus servers launched with 
                `modal deploy infra/modal_deploy_tokasaurus.py`
            **kwargs: Additional arguments
            
        Returns:
            ClientResponse with samples and usage information
        """
        assert stop is None, "stop is not supported by Tokasaurus batch endpoint"
        
        t0 = time.time()
        logger.info(f"[batch={modal_upstream_id}] Sending batch chat request")

        if self.config.model_name.lower() in MODEL_TO_THINKING_OVERRIDES:
            # this provides the kwargs needed for the apply_chat_template to enable
            # thinking. 
            thinking_overrides = MODEL_TO_THINKING_OVERRIDES[self.config.model_name.lower()](enable_thinking)
        elif enable_thinking:
            # if the model is not in the MODEL_TO_THINKING_OVERRIDES, we add a
            # thinking prompt to the last message of the chat. 
            thinking_overrides = {}
            for chat in chats:
                chat[-1]["content"] = add_thinking_prompt(chat[-1]["content"])
        else:
            thinking_overrides = {}


        def _construct_request(chat: List[Dict[str, Any]]) -> dict:
            request = {
                "messages": chat,
                "model": self.config.model_name,
                "max_completion_tokens": max_completion_tokens,
                "temperature": temperature,
                "apply_chat_template_overrides": thinking_overrides,
                "logprobs_in_fingerprint": True,
            }
            all_cartridges = []
            if self.cartridges is not None:
                all_cartridges.extend(self.cartridges)
            if cartridges is not None:
                all_cartridges.extend(cartridges)
            if len(all_cartridges) > 0:
                request["cartridges"] = all_cartridges
            if top_logprobs is not None:
                request["logprobs"] = True
                request["top_logprobs"] = top_logprobs
            return request
        response = await self._send_requests(
            [_construct_request(chat) for chat in chats], modal_upstream_id, 
            use_cartridge_endpoint=cartridges is not None or self.cartridges is not None
        )
        # SE (07/07): Running validation with ChatCompletion Pydantic model is very slow.
        # So we use model_construct to create the objects.
        responses: List[ChatCompletion] = [ChatCompletion.model_construct(**r) if isinstance(r, dict) else r for r in response]
        logger.info(f"[batch={modal_upstream_id}] Responses received")
        
        samples = []
        usage = Usage(prompt_tokens=0, completion_tokens=0)
        for i, response in enumerate(responses):
            if isinstance(response, Exception) or response is None:
                logger.error(f"Request {i} failed: {response}")
                samples.append(
                    ClientSample(
                        text="", 
                        token_ids=None, 
                        top_logprobs=None
                    )
                )
                continue
            
            # Extract usage information
            if hasattr(response, 'usage') and response.usage:
                usage += Usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                )
            
            # Extract token IDs from fingerprint if available
            fingerprint_data = json.loads(response.system_fingerprint)
            samples.append(
                ClientSample(
                    text=response.choices[0].message.content,
                    token_ids=fingerprint_data["completion_ids"][0],
                    top_logprobs=(
                        None if top_logprobs is None else 
                        self._extract_fingerprint_logprobs(fingerprint_data)
                    )
                )
            )
        
        logger.info(f"[batch={modal_upstream_id}] Batch chat completed in {time.time() - t0:.2f} seconds")
        
        assert len(samples) == len(chats), f"Expected {len(chats)} samples, got {len(samples)}"
        return ClientResponse(samples=samples, usage=usage)

