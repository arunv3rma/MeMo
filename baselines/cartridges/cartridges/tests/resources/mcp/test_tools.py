import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from cartridges.data.mcp.tools import MCPToolSet, MCPTool
from cartridges.data.tools import ToolInput, ToolOutput


class TestMCPToolSet:
    """Test MCPToolSet initialization"""

    def test_config_creation(self):
        """Test MCPToolSet configuration creation"""
        config = MCPToolSet.Config(
            command="echo",
            args=["test"],
            env={}
        )
        assert config.command == "echo"
        assert config.args == ["test"]
        assert config.env == {}

    def test_initialization(self):
        """Test MCPToolSet initialization"""
        config = MCPToolSet.Config(
            command="echo",
            args=["test"],
            env={}
        )
        mcp_tools = MCPToolSet(config)
        
        # Test that tools property is accessible
        assert hasattr(mcp_tools, 'tools')
        assert isinstance(mcp_tools.tools, list)

    @pytest.mark.asyncio
    async def test_mcp_tool_run(self):
        """Test MCPTool run_tool method"""
        # Create a mock MCPTool
        tool = MCPTool(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                }
            }
        )
        
        # Create a mock input
        class TestInput(ToolInput):
            input: str = "test"
        
        test_input = TestInput()
        
        # Test the run_tool method (should return not implemented error)
        result = await tool.run_tool(test_input)
        
        assert isinstance(result, ToolOutput)
        assert result.success is False
        assert "not implemented" in result.error.lower()
        assert result.input == test_input

    @pytest.mark.asyncio
    async def test_mcp_server_connection_mock(self):
        """Test MCPToolSet server connection with mocks"""
        config = MCPToolSet.Config(
            command="echo",
            args=["test"],
            env={}
        )
        mcp_tools = MCPToolSet(config)
        
        # Mock the MCP client components
        with patch('cartridges.resources.mcp.tools.stdio_client') as mock_stdio_client, \
             patch('cartridges.resources.mcp.tools.ClientSession') as mock_session_class:
            
            # Setup mocks
            mock_transport = (AsyncMock(), AsyncMock())
            mock_stdio_client.return_value.__aenter__ = AsyncMock(return_value=mock_transport)
            mock_stdio_client.return_value.__aexit__ = AsyncMock(return_value=None)
            
            mock_session = AsyncMock()
            mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Mock tool list response
            mock_tool = MagicMock()
            mock_tool.name = "test_tool"
            mock_tool.description = "Test tool"
            mock_tool.inputSchema = MagicMock()
            mock_tool.inputSchema.model_dump.return_value = {"type": "object"}
            
            mock_session.list_tools.return_value = MagicMock(tools=[mock_tool])
            
            # Test connection
            await mcp_tools.connect_to_server()
            
            # Verify connection was established
            assert len(mcp_tools._tools) == 1
            assert mcp_tools._tools[0].name == "test_tool"
            
            # Test tools property
            tools = mcp_tools.tools
            assert len(tools) == 1
            assert isinstance(tools[0], MCPTool)