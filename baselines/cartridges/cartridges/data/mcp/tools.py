from cartridges.data.tools import ToolSet, Tool, ToolInput, ToolOutput
from typing import List, Dict, Any
import asyncio
from contextlib import AsyncExitStack
from cartridges.data.mcp.mixin import MCPMixin

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    # Handle missing MCP dependency gracefully
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None

class MCPTool(Tool):

    def __init__(self, name: str, description: str, parameters: dict, mcp_toolset=None):
        from pydantic import create_model

        self._name = name
        self._description = description
        self.parameters = parameters
        self.mcp_toolset = mcp_toolset  # Reference to parent toolset for execution

        # Dynamically create a Pydantic model for ToolInput based on parameters
        properties = self.parameters.get("properties", {})
        field_definitions = {}
        
        for param, param_spec in properties.items():
            param_type = param_spec.get("type", "string")
            required = param in self.parameters.get("required", [])
            
            # Map JSON Schema types to Python types
            if param_type == "string":
                python_type = str
            elif param_type == "integer":
                python_type = int
            elif param_type == "number":
                python_type = float
            elif param_type == "boolean":
                python_type = bool
            else:
                python_type = str  # Default to string
            
            # Set field with default if not required
            if required:
                field_definitions[param] = (python_type, ...)
            else:
                field_definitions[param] = (python_type, None)
        
        # If no parameters, create empty model
        if not field_definitions:
            field_definitions["__empty__"] = (bool, False)  # Dummy field to make model valid
        
        self.ToolInput = create_model(
            f"{self._name.replace('-', '_')}Input",
            __base__=ToolInput,  # Inherit from ToolInput
            **field_definitions
        )
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    async def run_tool(self, input: ToolInput) -> ToolOutput:
        """Execute the tool via MCP server"""
        if not self.mcp_toolset or not self.mcp_toolset.session:
            return ToolOutput(
                input=input,
                success=False,
                error="MCP session not available",
                response=None
            )
        
        try:
            # Convert ToolInput to arguments dict
            args = input.model_dump() if hasattr(input, 'model_dump') else {}
            
            # Call the MCP server tool
            result = await self.mcp_toolset.session.call_tool(
                name=self._name,
                arguments=args
            )
            
            # Extract content from result
            content = []
            if hasattr(result, 'content') and result.content:
                for item in result.content:
                    if hasattr(item, 'text'):
                        content.append(item.text)
                    elif hasattr(item, 'data'):
                        content.append(str(item.data))
            content = "\n".join(content)
            
            return ToolOutput(
                input=input,
                success=True,
                error=None,
                response=content
            )
            
        except Exception as e:
            return ToolOutput(
                input=input,
                success=False,
                error=f"Tool execution failed: {str(e)}",
                response=None
            )
    
# it is important to 
class MCPToolSet(MCPMixin, ToolSet):

    class Config(ToolSet.Config):
        command: str
        args: List[str]
        env: Dict[str, str]

    def __init__(self, config: Config):
        ToolSet.__init__(self, config)
        MCPMixin.__init__(self)
        self.command: str = config.command
        self.args: List[str] = config.args
        self.env: Dict[str, str] = config.env
    

    async def setup(self):
        """Connect to an MCP server"""
        if self._is_connected:
            return
        await MCPMixin.setup(self)
        response = await self.session.list_tools()
        self._tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in self._tools])
        self._is_connected = True
            
    @property
    def tools(self) -> List[Tool]:
        """Return list of tools available from the MCP server"""
        if not hasattr(self, '_tools') or not self._tools:
            return []
        
        return [
            MCPTool(
                name=tool.name,
                description=tool.description,
                parameters=tool.inputSchema if hasattr(tool, 'inputSchema') and tool.inputSchema else {},
                mcp_toolset=self  # Pass reference to toolset for execution
            )
            for tool in self._tools
        ]

