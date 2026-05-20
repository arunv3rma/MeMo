from __future__ import annotations
from abc import ABC, abstractmethod
import abc
import asyncio
from typing import Any, List, Optional
from pydantic import BaseModel
from pydrantic import ObjectConfig

class ToolInput(BaseModel, ABC):
    pass

class ToolOutput(BaseModel):
    input: Optional[ToolInput]
    success: bool
    error: Optional[str]
    response: Optional[str]


class Tool(ABC):

    class Config(ObjectConfig):
        _pass_as_config = True

    
    def __init__(self, config: Config):
        self.config = config
    
    async def setup(self):
        pass

    async def cleanup(self):
        pass
    
    @abstractmethod
    async def run_tool(self, input: ToolInput) -> ToolOutput:
        raise NotImplementedError()
    
    async def batch_run_tool(self, inputs: List[ToolInput]) -> List[ToolOutput]:
        results = await asyncio.gather(*(self.run_tool(input) for input in inputs))
        return results
        
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def description(self) -> str:
        raise NotImplementedError()

    @property
    def definition(self) -> dict:
        schema = self.ToolInput.model_json_schema()
        del schema["title"]
        return {
            "name": self.name,
            "description": self.description,
            "parameters": schema,
        }



class ToolSet(ABC):

    class Config(ObjectConfig):
        _pass_as_config = True
    
    def __init__(self, config: Config):
        self.config = config
    
    async def setup(self):
        pass

    @abstractmethod
    def tools(self) -> List[Tool]:
        raise NotImplementedError()

    async def cleanup(self):
        pass



async def instantiate_tools(
    tools: List[Tool.Config | ToolSet.Config]
) -> List[Tool | ToolSet]:
    objs = [tool.instantiate() for tool in tools]

    # Setup all tools and toolsets (they all have setup methods now)
    setup_tasks = []
    for obj in objs:
        setup_tasks.append(obj.setup())
    
    if setup_tasks:
        await asyncio.gather(*setup_tasks)
    
    tools = []
    for obj in objs:
        if isinstance(obj, Tool):
            tools.append(obj)
        elif isinstance(obj, ToolSet):
            tools.extend(obj.tools)
    
    return tools, [obj.cleanup for obj in objs]