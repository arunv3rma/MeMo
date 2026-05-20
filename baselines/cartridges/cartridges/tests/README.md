# Testing Framework for SelfStudySynthesizer Tool Integration

This directory contains comprehensive tests for the SelfStudySynthesizer's tool integration capabilities, along with mock implementations that allow testing without actual model API calls.

## Overview

The testing framework includes:

1. **Mock Client** (`mocks.py`): A programmable mock that simulates LLM API responses
2. **Mock Tool** (`mocks.py`): A configurable tool for testing tool integration
3. **Mock Resource** (`mocks.py`): A mock data source for testing prompt sampling
4. **Comprehensive Tests** (`synthesizers/test_self_study.py`): Full test suite covering tool integration scenarios

## Key Components

### MockClient

The `MockClient` allows you to program specific responses for testing:

```python
from cartridges.tests.mocks import MockClient

# Configure responses
client_config = MockClient.Config(
    model_name="test-model",
    responses=["First response", "Second response"],
    response_tokens=[10, 15],
    top_logprobs_data={
        "num_input_tokens": 5,
        "token_ids": [1, 2, 3, 4, 5],
        "top_logprobs": [[-1.0, -2.0], [-1.5, -2.5]],
        "top_ids": [[100, 200], [101, 201]]
    }
)

client = client_config.instantiate()
```

### MockTool

The `MockTool` allows you to simulate tool execution with configurable responses and failures:

```python
from cartridges.tests.mocks import MockTool

# Configure tool behavior
tool_config = MockTool.Config(
    responses=["Tool executed successfully"],
    should_fail=False,
    failure_message="Tool execution failed"
)

tool = tool_config.instantiate()
```

### MockResource

The `MockResource` provides controlled data sampling for testing:

```python
from cartridges.tests.mocks import MockResource

# Configure resource data
resource_config = MockResource.Config(
    context="Test context for conversations",
    prompts=["Prompt 1", "Prompt 2"]
)

resource = resource_config.instantiate()
```

## Test Coverage

The test suite covers the following scenarios:

### 1. **Basic Integration** (`test_synthesizer_initialization`)
- Verifies that the synthesizer properly initializes with mock components
- Tests that tools, clients, and resources are correctly instantiated

### 2. **Tool Usage** (`test_sample_convos_with_tools`)
- Tests conversation generation with tool usage enabled
- Verifies that tools are called and responses are integrated into conversations

### 3. **No Tool Usage** (`test_tool_integration_without_tools`)
- Tests synthesizer behavior when tools are disabled
- Ensures conversations are generated without tool calls

### 4. **Error Handling** (`test_tool_failure_handling`)
- Tests how the synthesizer handles tool execution failures
- Verifies graceful degradation when tools fail

### 5. **Parsing Errors** (`test_tool_call_parsing_error`)
- Tests handling of malformed tool call responses
- Ensures the synthesizer continues operating despite parsing failures

### 6. **Multiple Tool Calls** (`test_multiple_tool_calls_per_round`)
- Tests scenarios where multiple tools are called in a single round
- Verifies proper batching and result aggregation

### 7. **Response Formatting** (`test_tool_responses_to_str`)
- Tests conversion of tool outputs to string format
- Verifies proper XML formatting for tool results

### 8. **Client Interaction Tracking** (`test_client_interaction_tracking`)
- Tests that client calls are properly tracked
- Verifies call history and metrics

### 9. **Batch Processing** (`test_different_batch_sizes`)
- Tests the synthesizer with various batch sizes
- Ensures consistent behavior across different scales

## Running Tests

### Run all tests:
```bash
python -m pytest cartridges/tests/synthesizers/test_self_study.py -v
```

### Run specific test:
```bash
python -m pytest cartridges/tests/synthesizers/test_self_study.py::TestSelfStudySynthesizerToolIntegration::test_tool_integration_without_tools -v
```

### Run with coverage:
```bash
python -m pytest cartridges/tests/synthesizers/test_self_study.py --cov=cartridges.synthesizers.self_study
```

## Example Usage

See `example_usage.py` for a complete example of how to use the mock components:

```bash
python cartridges/tests/example_usage.py
```

## Key Testing Patterns

### 1. **Configurable Responses**
Mock clients can be programmed with specific responses to test different conversation flows:

```python
client_config = MockClient.Config(
    responses=["Bot A response", "Bot B response"],
    response_tokens=[25, 35]
)
```

### 2. **Tool Call Mocking**
Tests use `unittest.mock.patch` to mock the tool call parsing and template rendering:

```python
with patch('cartridges.synthesizers.self_study.MODEL_TO_TOOL_CALL_PARSER') as mock_parser:
    def mock_parser_func(text):
        return [ToolCall(function=FunctionCall(name="mock_tool", arguments={"query": "test"}))]
    mock_parser.__getitem__.return_value = mock_parser_func
```

### 3. **Failure Simulation**
Tools can be configured to simulate various failure conditions:

```python
tool.config.should_fail = True
tool.config.failure_message = "Network timeout"
```

### 4. **Assertion Patterns**
Tests verify both successful execution and proper metadata tracking:

```python
assert len(examples) == batch_size
assert example.metadata["tool_calls"] != []
assert all(call["success"] for call in tool_calls)
```

## Adding New Tests

When adding new tools to the system, follow these patterns:

1. **Create a mock version** of your tool in `mocks.py`
2. **Add test cases** covering normal operation, failure modes, and edge cases
3. **Mock external dependencies** using `unittest.mock.patch`
4. **Verify metadata tracking** to ensure tool calls are properly logged
5. **Test integration** with the SelfStudySynthesizer

## Dependencies

- `pytest`: Test framework
- `pytest-asyncio`: Async test support
- `unittest.mock`: Mocking framework (built-in)
- `numpy`: Array operations for logprobs
- `pydantic`: Configuration validation

The testing framework provides comprehensive coverage of tool integration scenarios while allowing testing without actual API calls or external dependencies.