from collections import defaultdict
from typing import List, Literal, Optional, Tuple, Any
import json
import uuid
import re

from jinja2 import Template
from pydantic import BaseModel, Field

QWEN_TOOL_TEMPLATE = r"""
{%- if tools %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- endif %}
"""


MODEL_TO_TOOL_TEMPLATE = defaultdict(
    lambda: QWEN_TOOL_TEMPLATE,
    {
        "Qwen/Qwen2.5-Coder-32B-Instruct": QWEN_TOOL_TEMPLATE,
    }
)


def render_tool_template(tools: List[dict], template: str):
    template: Template = Template(template)
    def to_json(value):
        return json.dumps(value, ensure_ascii=False)
    tool_strs = [to_json(tool) for tool in tools]

    return template.render(tools=tool_strs)

# These interfaces are borrowed from openai
class FunctionCall(BaseModel):
    name: str
    arguments: dict[str, Any]

class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: Literal["function"] = "function"
    function: FunctionCall


def parse_tool_calls_hermes(response: str) -> List[ToolCall]:
    # Borrowed from vllm.entrypoints.openai.tool_parsers.hermes_tool_parser
    HERMES_TOOL_CALL_REGEX = r"<tool_call>(.*?)<\/tool_call>|<tool_call>(.*)"
    # Critical to include re.DOTALL since there are newlines in the tool calls
    tool_call_regex = re.compile(HERMES_TOOL_CALL_REGEX, re.DOTALL)

    function_call_tuples = tool_call_regex.findall(response)

    def parse(match: Tuple[str, str]) -> Optional[ToolCall]:
        try:
            function_call = json.loads(match[0] if match[0] else match[1])
            return ToolCall(
                type="function",
                function=FunctionCall(
                    name=function_call["name"],
                    # function call args are JSON but as a string
                    arguments=function_call["arguments"]
                )
            )
        except Exception as e:
            print(f"Error parsing function call: {e}")
            return None

    # load the JSON, and then use it to build the Function and
    # Tool Call
    tool_calls = [parse(match) for match in function_call_tuples]
    tool_calls = [call for call in tool_calls if call is not None]
    return tool_calls



MODEL_TO_TOOL_CALL_PARSER = defaultdict(
    lambda: parse_tool_calls_hermes,
    {
        "Qwen/Qwen2.5-Coder-32B-Instruct": parse_tool_calls_hermes,
    }
)
