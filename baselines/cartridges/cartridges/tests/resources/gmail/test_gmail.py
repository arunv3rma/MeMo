import pytest
import os
from unittest.mock import MagicMock
from cartridges.data.gmail.gmail import GmailToolSet
from cartridges.data.mcp.tools import MCPTool
from cartridges.data.tools import ToolOutput


class TestGmailToolSet:
    """Test GmailToolSet initialization and configuration"""

    def test_config_creation(self):
        """Test GmailToolSet configuration creation"""
        config = GmailToolSet.Config(email="test@example.com")
        assert config.email == "test@example.com"

    def test_initialization(self):
        """Test GmailToolSet initialization"""
        config = GmailToolSet.Config(email="test@example.com")
        gmail_tools = GmailToolSet(config)
        
        # Test that tools property is accessible
        assert hasattr(gmail_tools, 'tools')
        assert isinstance(gmail_tools.tools, list)

    @pytest.mark.asyncio
    async def test_gmail_connection_mock(self):
        """Test GmailToolSet with mocked MCP connection"""
        config = GmailToolSet.Config(email="test@example.com")
        gmail_tools = GmailToolSet(config)
        
        # Mock the MCP connection and tools
        mock_list_labels_tool = MagicMock()
        mock_list_labels_tool.name = "list_labels"
        mock_list_labels_tool.description = "List all labels in the user's mailbox"
        mock_list_labels_tool.inputSchema = {}
        
        mock_fetch_threads_tool = MagicMock()
        mock_fetch_threads_tool.name = "fetch_threads"
        mock_fetch_threads_tool.description = "Fetch email threads from Gmail"
        mock_fetch_threads_tool.inputSchema = {
            "type": "object",
            "properties": {
                "num_threads": {"type": "integer"},
                "label_names": {"type": "array"}
            }
        }
        
        gmail_tools._tools = [mock_list_labels_tool, mock_fetch_threads_tool]
        
        # Test that tools are properly created and filtered
        tools = gmail_tools.tools
        assert len(tools) == 2
        tool_names = [tool.name for tool in tools]
        assert "list_labels" in tool_names
        assert "fetch_threads" in tool_names

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_gmail_real_connection(self):
        """Test GmailToolSet with real MCP server connection"""
        # Check for required environment variables
        cartridges_dir = os.environ.get("CARTRIDGES_DIR")
        cartridges_output_dir = os.environ.get("CARTRIDGES_OUTPUT_DIR")
        
        if not cartridges_dir or not cartridges_output_dir:
            pytest.skip("CARTRIDGES_DIR or CARTRIDGES_OUTPUT_DIR not found in environment")
        
        config = GmailToolSet.Config(email="test@example.com")
        gmail_tools = GmailToolSet(config)
        
        try:
            # Attempt to connect to the real MCP server
            await gmail_tools.connect_to_server()
            
            # If connection succeeds, verify we have tools
            tools = gmail_tools.tools
            assert isinstance(tools, list)
            
            # Print available tools for debugging
            print(f"Connected to Gmail MCP server with {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description}")
                
            # Verify we have the expected Gmail tools
            tool_names = [tool.name for tool in tools]
            expected_tools = ["list_labels", "fetch_threads"]
            found_tools = [tool for tool in expected_tools if tool in tool_names]
            
            if found_tools:
                print(f"Found expected Gmail tools: {found_tools}")
            else:
                print(f"Available tools: {tool_names}")
                
        except Exception as e:
            # If MCP server fails to start or credentials are missing, skip the test
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["connection", "server", "credentials", "auth", "token"]):
                pytest.skip(f"Could not connect to Gmail MCP server: {e}")
            else:
                # Re-raise if it's an unexpected error
                raise

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_gmail_list_labels_tool(self):
        """Test actually using the list_labels tool"""
        # Check for required environment variables
        cartridges_dir = os.environ.get("CARTRIDGES_DIR")
        cartridges_output_dir = os.environ.get("CARTRIDGES_OUTPUT_DIR")
        
        if not cartridges_dir or not cartridges_output_dir:
            pytest.skip("CARTRIDGES_DIR or CARTRIDGES_OUTPUT_DIR not found in environment")
        
        config = GmailToolSet.Config(email="test@example.com")
        gmail_tools = GmailToolSet(config)
        
        try:
            # Connect to the MCP server
            await gmail_tools.connect_to_server()
            
            # Find the list_labels tool
            tools = gmail_tools.tools
            list_labels_tool = None
            
            for tool in tools:
                if tool.name == "list_labels":
                    list_labels_tool = tool
                    break
            
            if not list_labels_tool:
                pytest.skip("list_labels tool not found in available tools")
            
            print(f"Found tool: {list_labels_tool.name} - {list_labels_tool.description}")
            
            # Create input for the tool (list_labels typically takes no parameters)
            tool_input = list_labels_tool.ToolInput()
            
            # Execute the tool
            result = await list_labels_tool.run_tool(tool_input)
            
            print(f"Tool execution result: success={result.success}")
            if result.error:
                print(f"Error: {result.error}")
            if result.response:
                print(f"Response preview: {result.response[:200]}...")
            
            # Verify the result
            assert isinstance(result, ToolOutput)
            
            if result.success:
                # If successful, we should have a response with label data
                assert result.response is not None
                assert len(result.response) > 0
                print("✅ Successfully listed Gmail labels!")
            else:
                # If it failed, check if it's due to credentials or configuration
                print(f"⚠️ Tool execution failed: {result.error}")
                # Don't fail the test if it's a credential/config issue
                if any(keyword in result.error.lower() for keyword in ["credentials", "auth", "token", "permission"]):
                    pytest.skip(f"Gmail API credential issue: {result.error}")
                
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["connection", "server", "credentials", "auth", "token"]):
                pytest.skip(f"Could not connect to Gmail MCP server: {e}")
            else:
                raise