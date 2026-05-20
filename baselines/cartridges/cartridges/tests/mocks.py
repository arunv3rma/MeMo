"""Mock implementations for testing synthesizers and tools."""
import asyncio
from typing import Any, Dict, List, Optional, Union
import numpy as np

from cartridges.clients.base import Client, ClientConfig, ClientResponse, ClientSample, TopLogprobs
from cartridges.clients.usage import Usage
from cartridges.data.tools import Tool, ToolInput, ToolOutput
from cartridges.data.resources import Resource


class MockClient(Client):
    """Mock client that returns pre-programmed responses for testing."""
    
    class Config(ClientConfig):
        """Configuration for MockClient."""
        model_name: str = "mock-model"
        responses: List[str] = []
        response_tokens: List[int] = []
        top_logprobs_data: Optional[Dict[str, Any]] = None
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.config: MockClient.Config = config
        self.call_count = 0
        self.call_history = []
    
    def complete(
        self, 
        prompts: List[Union[str, List[int]]], 
        **kwargs
    ) -> ClientResponse:
        """Mock completion method."""
        self.call_count += 1
        self.call_history.append({"method": "complete", "prompts": prompts, "kwargs": kwargs})
        
        samples = []
        for i, prompt in enumerate(prompts):
            response_idx = min(i, len(self.config.responses) - 1) if self.config.responses else 0
            response_text = self.config.responses[response_idx] if self.config.responses else f"Mock response {i}"
            num_tokens = self.config.response_tokens[response_idx] if self.config.response_tokens else 10
            
            top_logprobs_result = None
            if self.config.top_logprobs_data:
                top_logprobs_result = TopLogprobs(
                    num_input_tokens=self.config.top_logprobs_data.get("num_input_tokens", 10),
                    token_ids=np.array(self.config.top_logprobs_data.get("token_ids", [1, 2, 3, 4, 5])),
                    top_logprobs=np.array(self.config.top_logprobs_data.get("top_logprobs", [[-1.0, -2.0, -3.0]])),
                    top_ids=np.array(self.config.top_logprobs_data.get("top_ids", [[100, 200, 300]]))
                )
            
            samples.append(ClientSample(
                output_text=response_text,
                num_output_tokens=num_tokens,
                top_logprobs=top_logprobs_result
            ))
        
        usage = Usage(
            prompt_tokens=sum(len(str(p)) for p in prompts),
            completion_tokens=sum(s.num_output_tokens for s in samples)
        )
        
        return ClientResponse(samples=samples, usage=usage)
    
    def chat(
        self, 
        chats: List[List[Dict[str, Any]]], 
        temperature: float = 0.6, 
        stop: List[str] = [], 
        max_completion_tokens: Optional[int] = None,
        frequency_penalty: float = 0.0,
        top_logprobs: int = 1,
        logprobs_start_message: Optional[int] = None,
        **kwargs
    ) -> ClientResponse:
        """Mock chat method."""
        self.call_count += 1
        self.call_history.append({
            "method": "chat", 
            "chats": chats, 
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "top_logprobs": top_logprobs,
            "logprobs_start_message": logprobs_start_message,
            "kwargs": kwargs
        })
        
        samples = []
        for i, chat in enumerate(chats):
            response_idx = min(i, len(self.config.responses) - 1) if self.config.responses else 0
            response_text = self.config.responses[response_idx] if self.config.responses else f"Mock chat response {i}"
            num_tokens = self.config.response_tokens[response_idx] if self.config.response_tokens else 10
            
            top_logprobs_result = None
            if top_logprobs and top_logprobs > 0 and self.config.top_logprobs_data:
                top_logprobs_result = TopLogprobs(
                    num_input_tokens=self.config.top_logprobs_data.get("num_input_tokens", 10),
                    token_ids=np.array(self.config.top_logprobs_data.get("token_ids", [1, 2, 3, 4, 5])),
                    top_logprobs=np.array(self.config.top_logprobs_data.get("top_logprobs", [[-1.0, -2.0, -3.0]])),
                    top_ids=np.array(self.config.top_logprobs_data.get("top_ids", [[100, 200, 300]]))
                )
            
            samples.append(ClientSample(
                output_text=response_text,
                num_output_tokens=num_tokens,
                top_logprobs=top_logprobs_result
            ))
        
        usage = Usage(
            prompt_tokens=sum(len(str(chat)) for chat in chats),
            completion_tokens=sum(s.num_output_tokens for s in samples)
        )
        
        return ClientResponse(samples=samples, usage=usage)
    
    async def chat_async(
        self,
        chats: List[List[Dict[str, Any]]],
        temperature: float = 0.6,
        stop: Optional[List[str]] = None,
        max_completion_tokens: Optional[int] = None,
        top_logprobs: Optional[int] = None,
        logprobs_start_message: Optional[int] = None,
        **kwargs,
    ) -> ClientResponse:
        """Async version of chat method."""
        # Simulate async delay
        await asyncio.sleep(0.01)
        return self.chat(
            chats=chats,
            temperature=temperature,
            stop=stop or [],
            max_completion_tokens=max_completion_tokens,
            top_logprobs=top_logprobs or 1,
            logprobs_start_message=logprobs_start_message,
            **kwargs
        )


class MockToolInput(ToolInput):
    """Mock tool input for testing."""
    query: str


class MockTool(Tool):
    """Mock tool for testing tool integration."""
    
    ToolInput = MockToolInput
    
    class Config(Tool.Config):
        responses: List[str] = ["Mock tool response"]
        should_fail: bool = False
        failure_message: str = "Mock tool failure"
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.config: MockTool.Config = config
        self.call_count = 0
        self.call_history = []
    
    async def run_tool(self, input: MockToolInput) -> ToolOutput:
        """Mock tool execution."""
        self.call_count += 1
        self.call_history.append(input.query)
        
        if self.config.should_fail:
            return ToolOutput(
                input=input,
                success=False,
                error=self.config.failure_message,
                response=None
            )
        
        response_idx = min(self.call_count - 1, len(self.config.responses) - 1)
        response = self.config.responses[response_idx]
        
        return ToolOutput(
            input=input,
            success=True,
            error=None,
            response=response
        )
    
    @property
    def name(self) -> str:
        return "mock_tool"
    
    @property
    def description(self) -> str:
        return "A mock tool for testing purposes"


class MockResource(Resource):
    """Mock resource for testing."""
    
    class Config(Resource.Config):
        context: str = "Mock context"
        prompts: List[str] = ["Mock prompt"]
    
    def __init__(self, config: Config):
        self.config: MockResource.Config = config
        self.call_count = 0
    
    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        """Mock prompt sampling."""
        self.call_count += 1
        
        # Return the configured context and repeat prompts to match batch size
        prompts = []
        for i in range(batch_size):
            prompt_idx = i % len(self.config.prompts)
            prompts.append(self.config.prompts[prompt_idx])
        
        return self.config.context, prompts