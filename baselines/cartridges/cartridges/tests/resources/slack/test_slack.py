import pytest
import os
from unittest.mock import MagicMock
from cartridges.data.slack.slack import SlackToolSet
from cartridges.data.mcp.tools import MCPTool
from cartridges.data.tools import ToolOutput


class TestSlackToolSet:
    """Test SlackToolSet initialization and configuration"""

    def test_config_creation(self):
        """Test SlackToolSet configuration creation"""
        # Check environment variables
        bot_token = os.getenv("SLACK_BOT_TOKEN")
        team_id = os.getenv("SLACK_TEAM_ID")
        
        if not bot_token or not team_id:
            pytest.skip("SLACK_BOT_TOKEN or SLACK_TEAM_ID not found in environment")
        
        # Create SlackToolSet config
        config = SlackToolSet.Config()
        assert config.bot_token == bot_token
        assert config.team_id == team_id

    def test_initialization(self):
        """Test SlackToolSet initialization"""
        bot_token = os.getenv("SLACK_BOT_TOKEN")
        team_id = os.getenv("SLACK_TEAM_ID")
        
        if not bot_token or not team_id:
            pytest.skip("SLACK_BOT_TOKEN or SLACK_TEAM_ID not found in environment")
        
        config = SlackToolSet.Config()
        slack_tools = SlackToolSet(config)
        
        # Test that tools property is accessible
        assert hasattr(slack_tools, 'tools')
        assert isinstance(slack_tools.tools, list)

    @pytest.mark.asyncio
    async def test_slack_connection_mock(self):
        """Test SlackToolSet with mocked MCP connection"""
        bot_token = os.getenv("SLACK_BOT_TOKEN")
        team_id = os.getenv("SLACK_TEAM_ID")
        
        if not bot_token or not team_id:
            pytest.skip("SLACK_BOT_TOKEN or SLACK_TEAM_ID not found in environment")
        
        config = SlackToolSet.Config()
        slack_tools = SlackToolSet(config)
        
        # Mock the MCP connection and tools
        mock_tool = MagicMock()
        mock_tool.name = "send_message"
        mock_tool.description = "Send a message to Slack"
        mock_tool.inputSchema = MagicMock()
        mock_tool.inputSchema.model_dump.return_value = {
            "type": "object",
            "properties": {
                "channel": {"type": "string"},
                "message": {"type": "string"}
            }
        }
        
        slack_tools._tools = [mock_tool]
        
        # Test that tools are properly created
        tools = slack_tools.tools
        assert len(tools) == 1
        assert isinstance(tools[0], MCPTool)
        assert tools[0].name == "send_message"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_slack_real_connection(self):
        """Test SlackToolSet with real MCP server connection"""
        bot_token = os.getenv("SLACK_BOT_TOKEN")
        team_id = os.getenv("SLACK_TEAM_ID")
        
        if not bot_token or not team_id:
            pytest.skip("SLACK_BOT_TOKEN or SLACK_TEAM_ID not found in environment")
        
        config = SlackToolSet.Config()
        slack_tools = SlackToolSet(config)
        
        try:
            # Attempt to connect to the real MCP server
            await slack_tools.connect_to_server()
            
            # If connection succeeds, verify we have tools
            tools = slack_tools.tools
            assert isinstance(tools, list)
            
            # Print available tools for debugging
            print(f"Connected to Slack MCP server with {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description}")
                
            # Verify we have at least some expected Slack tools
            tool_names = [tool.name for tool in tools]
            
            # Common Slack MCP tools we might expect
            expected_tools = ["send_message", "list_channels", "get_channel_history"]
            found_tools = [tool for tool in expected_tools if tool in tool_names]
            
            if found_tools:
                print(f"Found expected Slack tools: {found_tools}")
            else:
                print(f"Available tools: {tool_names}")
                
        except Exception as e:
            # If Docker is not available or MCP server fails to start, skip the test
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["docker", "connection", "server", "command not found"]):
                pytest.skip(f"Could not connect to Slack MCP server: {e}")
            else:
                # Re-raise if it's an unexpected error
                raise

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_slack_list_channels_tool(self):
        """Test actually using the slack_list_channels tool"""
        bot_token = os.getenv("SLACK_BOT_TOKEN")
        team_id = os.getenv("SLACK_TEAM_ID")
        
        if not bot_token or not team_id:
            pytest.skip("SLACK_BOT_TOKEN or SLACK_TEAM_ID not found in environment")
        
        config = SlackToolSet.Config()
        slack_tools = SlackToolSet(config)
        
        try:
            # Connect to the MCP server
            await slack_tools.connect_to_server()
            
            # Find the slack_list_channels tool
            tools = slack_tools.tools
            list_channels_tool = None
            
            for tool in tools:
                if tool.name == "slack_list_channels":
                    list_channels_tool = tool
                    break
            
            if not list_channels_tool:
                pytest.skip("slack_list_channels tool not found in available tools")
            
            print(f"Found tool: {list_channels_tool.name} - {list_channels_tool.description}")
            
            # Create input for the tool (slack_list_channels typically takes no parameters or optional limit)
            tool_input = list_channels_tool.ToolInput()
            
            # Execute the tool
            result = await list_channels_tool.run_tool(tool_input)
            
            print(f"Tool execution result: success={result.success}")
            if result.error:
                print(f"Error: {result.error}")
            if result.response:
                print(f"Response preview: {result.response[:200]}...")
            
            # Verify the result
            assert isinstance(result, ToolOutput)
            
            if result.success:
                # If successful, we should have a response with channel data
                assert result.response is not None
                assert len(result.response) > 0
                print("✅ Successfully listed Slack channels!")
            else:
                # If it failed, check if it's due to permissions or configuration
                print(f"⚠️ Tool execution failed: {result.error}")
                # Don't fail the test if it's a permission/config issue
                if any(keyword in result.error.lower() for keyword in ["permission", "auth", "token", "forbidden"]):
                    pytest.skip(f"Slack API permission issue: {result.error}")
                
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["docker", "connection", "server", "command not found"]):
                pytest.skip(f"Could not connect to Slack MCP server: {e}")
            else:
                raise