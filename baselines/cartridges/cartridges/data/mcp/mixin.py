from cartridges.data.tools import ToolSet, Tool, ToolInput, ToolOutput
from typing import List, Dict, Any
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

    

class MCPMixin:

    def __init__(self):
        self._tools = []
        self.exit_stack = None
        self.session = None
        self._is_connected = False
    

    async def setup(self):
        """Connect to an MCP server"""
        if self._is_connected:
            return
            
        try:
            self.exit_stack = AsyncExitStack()
            
            server_params = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=self.env
            )
            
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            
            await self.session.initialize()
            self._is_connected = True
            
        except Exception as e:
            await self.cleanup()
            raise e

    @property
    def tools(self) -> List[Tool]:
        return [
            MCPTool(
                name=tool.name,
                description=tool.description,
                parameters=tool.inputSchema if hasattr(tool, 'inputSchema') and tool.inputSchema else {},
                mcp_toolset=self  # Pass reference to toolset for execution
            )
            for tool in self._tools
        ]
    
    async def cleanup(self):
        """Clean up MCP server connection"""
        if self.exit_stack is not None:
            try:
                await self.exit_stack.aclose()
            except Exception as e:
                print(f"Warning: Error during MCP cleanup: {e}")
            finally:
                self.exit_stack = None
                self.session = None
                self._is_connected = False
    
    async def __aenter__(self):
        await self.setup()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
        return False  # Don't suppress exceptions
    