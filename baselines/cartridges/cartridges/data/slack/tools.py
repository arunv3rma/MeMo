from cartridges.data.tools import ToolSet, Tool
from cartridges.data.mcp.tools import MCPToolSet
import os
from typing import List

INCLUDED_TOOLS = [
    'slack_list_channels', 
    'slack_get_channel_history', 
    'slack_get_thread_replies', 
    'slack_get_users', 
    'slack_get_user_profile'
]
class SlackToolSet(MCPToolSet):

    class Config(ToolSet.Config):
        bot_token: str = os.getenv("SLACK_BOT_TOKEN")
        team_id: str = os.getenv("SLACK_TEAM_ID")
        
    def __init__(self, config: Config):
        # Create MCP config for Slack
        mcp_config = MCPToolSet.Config(
            command="docker",
            args=[
                "run",
                "-i",
                "--rm",
                "-e", "SLACK_BOT_TOKEN",
                "-e", "SLACK_TEAM_ID",
                "mcp/slack"
            ],
            env={
                "SLACK_BOT_TOKEN": config.bot_token,
                "SLACK_TEAM_ID": config.team_id
            }
        )
        super().__init__(mcp_config)

    @property
    def tools(self) -> List[Tool]:
        # Return tools from MCP connection
        tools = []
        for tool in super().tools:
            if tool.name in INCLUDED_TOOLS:
                tools.append(tool)
        return tools
