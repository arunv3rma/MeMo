import os
from typing import List
from cartridges.data.tools import ToolSet, Tool
from cartridges.data.mcp.tools import MCPToolSet
from pydantic import BaseModel

class Message(BaseModel):
    id: str
    subject: str
    from_address: str
    to_addresses: List[str]
    date: str
    snippet: str
    content: str
    raw: dict

class Thread(BaseModel):
    id: str
    messages: List[Message]

INCLUDED_TOOLS = [
    'list_labels',
    'fetch_threads'
]

class GmailToolSet(MCPToolSet):

    class Config(ToolSet.Config):
        email: str = ""  # Optional email identifier
        
    def __init__(self, config: Config):
        # Create MCP config for Gmail using custom server
        mcp_config = MCPToolSet.Config(
            command="python",
            args=[
                "-m",
                "cartridges.resources.gmail.server"
            ],
            env={
                "CARTRIDGES_DIR": os.environ.get("CARTRIDGES_DIR", os.getcwd()),
                "CARTRIDGES_OUTPUT_DIR": os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
            }
        )
        super().__init__(mcp_config)

    @property
    def tools(self) -> List[Tool]:
        # Return filtered tools from MCP connection
        tools = []
        for tool in super().tools:
            if tool.name in INCLUDED_TOOLS:
                tools.append(tool)
        return tools


