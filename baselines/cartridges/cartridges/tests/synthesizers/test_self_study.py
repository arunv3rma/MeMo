"""Tests for SelfStudySynthesizer tool integration."""
import pytest
import asyncio
import numpy as np
from unittest.mock import patch

from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.tests.mocks import MockClient, MockTool, MockResource
from cartridges.data.tools import ToolOutput


class TestSelfStudySynthesizerToolIntegration:
    """Test suite for SelfStudySynthesizer tool integration."""
    
    @pytest.fixture
    def mock_client_config(self):
        """Create a mock client configuration."""
        return MockClient.Config(
            model_name="mock-model",
            responses=[
                "I need more information about the topic.",  # Bot A response
                "Based on the information provided, here's my analysis.",  # Bot B response
            ],
            response_tokens=[25, 35],
            top_logprobs_data={
                "num_input_tokens": 10,
                "token_ids": [1, 2, 3, 4, 5, 100, 101, 102, 103, 104],
                "top_logprobs": [[-1.0, -2.0, -3.0], [-1.5, -2.5, -3.5], [-1.2, -2.2, -3.2], 
                               [-1.8, -2.8, -3.8], [-1.1, -2.1, -3.1], [-1.9, -2.9, -3.9],
                               [-1.3, -2.3, -3.3], [-1.7, -2.7, -3.7], [-1.4, -2.4, -3.4],
                               [-1.6, -2.6, -3.6]],
                "top_ids": [[100, 200, 300], [101, 201, 301], [102, 202, 302],
                           [103, 203, 303], [104, 204, 304], [105, 205, 305],
                           [106, 206, 306], [107, 207, 307], [108, 208, 308],
                           [109, 209, 309]]
            }
        )
    
    @pytest.fixture
    def mock_tool_config(self):
        """Create a mock tool configuration."""
        return MockTool.Config(
            responses=[
                "Retrieved relevant information about the query.",
                "Found additional context for the discussion."
            ]
        )
    
    @pytest.fixture
    def mock_resource_config(self):
        """Create a mock resource configuration."""
        return MockResource.Config(
            context="This is a test context for the conversation.",
            prompts=[
                "What are the key concepts?",
                "How does this work in practice?"
            ]
        )
    
    @pytest.fixture
    def synthesizer_config(self, mock_client_config, mock_tool_config, mock_resource_config):
        """Create a SelfStudySynthesizer configuration with mocks."""
        return SelfStudySynthesizer.Config(
            client=mock_client_config,
            resources=[mock_resource_config],
            tools=[mock_tool_config],
            use_tools_a=True,
            use_tools_b=True,
            max_rounds=1,
            temperature_a=0.7,
            max_completion_tokens_a=100,
            temperature_b=0.0,
            max_completion_tokens_b=200,
            tokenizer="gpt2",  # Use a simple tokenizer for testing
            num_top_logprobs=3
        )
    
    @pytest.fixture
    def synthesizer(self, synthesizer_config):
        """Create a SelfStudySynthesizer instance."""
        return SelfStudySynthesizer(synthesizer_config)
    
    @pytest.mark.asyncio
    async def test_synthesizer_initialization(self, synthesizer):
        """Test that the synthesizer initializes correctly with tools."""
        assert synthesizer.client is not None
        assert isinstance(synthesizer.client, MockClient)
        assert len(synthesizer.tools) == 1
        assert "mock_tool" in synthesizer.tools
        assert len(synthesizer.resources) == 1
        assert isinstance(synthesizer.resources[0], MockResource)
    
    @pytest.mark.asyncio
    async def test_sample_convos_with_tools(self, synthesizer):
        """Test conversation sampling with tool usage enabled."""
        # Patch the tool call parser to return mock tool calls
        with patch('cartridges.synthesizers.self_study.MODEL_TO_TOOL_CALL_PARSER') as mock_parser_dict:
            from cartridges.data import ToolCall, FunctionCall
            from cartridges.data.tools import ToolOutput
            
            # Mock tool call parser
            def mock_parser(text):
                return [ToolCall(
                    function=FunctionCall(
                        name="mock_tool",
                        arguments={"query": "test query"}
                    )
                )]
            
            mock_parser_dict.__getitem__.return_value = mock_parser
            
            # Mock the tool template
            with patch('cartridges.synthesizers.self_study.MODEL_TO_TOOL_TEMPLATE') as mock_template_dict:
                mock_template_dict.__getitem__.return_value = "Available tools: {tools}"
                
                with patch('cartridges.synthesizers.self_study.render_tool_template') as mock_render:
                    mock_render.return_value = "Mock tool description"
                    
                    # Run the synthesizer
                    examples = await synthesizer.sample_convos(
                        batch_idx=0,
                        batch_size=2,
                        total_batches=1
                    )
                    
                    # Verify results
                    assert len(examples) == 2
                    for example in examples:
                        assert len(example.messages) > 0  # Should have messages
                        assert example.system_prompt is not None
                        assert example.metadata is not None
                        assert "tool_calls" in example.metadata
                        assert "seed_prompt" in example.metadata
                        assert "initial_system_prompt" in example.metadata
    
    @pytest.mark.asyncio
    async def test_tool_integration_without_tools(self, synthesizer_config):
        """Test synthesizer behavior when tools are disabled."""
        # Disable tools
        synthesizer_config.use_tools_a = False
        synthesizer_config.use_tools_b = False
        
        synthesizer = SelfStudySynthesizer(synthesizer_config)
        
        examples = await synthesizer.sample_convos(
            batch_idx=0,
            batch_size=1,
            total_batches=1
        )
        
        # Verify that examples are created without tool usage
        assert len(examples) == 1
        example = examples[0]
        assert example.metadata["tool_calls"] == []  # No tool calls should be made
    
    @pytest.mark.asyncio
    async def test_tool_failure_handling(self, synthesizer):
        """Test how the synthesizer handles tool failures."""
        # Configure tool to fail
        synthesizer.tools["mock_tool"].config.should_fail = True
        synthesizer.tools["mock_tool"].config.failure_message = "Tool execution failed"
        
        with patch('cartridges.synthesizers.self_study.MODEL_TO_TOOL_CALL_PARSER') as mock_parser_dict:
            from cartridges.data import ToolCall, FunctionCall
            
            def mock_parser(text):
                return [ToolCall(
                    function=FunctionCall(
                        name="mock_tool",
                        arguments={"query": "test query"}
                    )
                )]
            
            mock_parser_dict.__getitem__.return_value = mock_parser
            
            with patch('cartridges.synthesizers.self_study.MODEL_TO_TOOL_TEMPLATE') as mock_template_dict:
                mock_template_dict.__getitem__.return_value = "Available tools: {tools}"
                
                with patch('cartridges.synthesizers.self_study.render_tool_template') as mock_render:
                    mock_render.return_value = "Mock tool description"
                    
                    examples = await synthesizer.sample_convos(
                        batch_idx=0,
                        batch_size=1,
                        total_batches=1
                    )
                    
                    # Verify that the synthesizer handles tool failures gracefully
                    assert len(examples) == 1
                    example = examples[0]
                    tool_calls = example.metadata["tool_calls"]
                    assert len(tool_calls) >= 1  # Should have attempted tool calls
                    # Some tool calls should have failed
                    failed_calls = [call for call in tool_calls if not call["success"]]
                    assert len(failed_calls) > 0
    
    @pytest.mark.asyncio
    async def test_tool_call_parsing_error(self, synthesizer):
        """Test handling of tool call parsing errors."""
        with patch('cartridges.synthesizers.self_study.MODEL_TO_TOOL_CALL_PARSER') as mock_parser_dict:
            # Mock parser that raises an exception
            def failing_parser(text):
                raise ValueError("Failed to parse tool call")
            
            mock_parser_dict.__getitem__.return_value = failing_parser
            
            with patch('cartridges.synthesizers.self_study.MODEL_TO_TOOL_TEMPLATE') as mock_template_dict:
                mock_template_dict.__getitem__.return_value = "Available tools: {tools}"
                
                with patch('cartridges.synthesizers.self_study.render_tool_template') as mock_render:
                    mock_render.return_value = "Mock tool description"
                    
                    examples = await synthesizer.sample_convos(
                        batch_idx=0,
                        batch_size=1,
                        total_batches=1
                    )
                    
                    # Should handle parsing errors gracefully
                    assert len(examples) == 1
    
    @pytest.mark.asyncio
    async def test_multiple_tool_calls_per_round(self, synthesizer):
        """Test handling of multiple tool calls in a single round."""
        with patch('cartridges.synthesizers.self_study.MODEL_TO_TOOL_CALL_PARSER') as mock_parser_dict:
            from cartridges.data import ToolCall, FunctionCall
            
            def mock_parser(text):
                # Return multiple tool calls
                return [
                    ToolCall(
                        function=FunctionCall(
                            name="mock_tool",
                            arguments={"query": "first query"}
                        )
                    ),
                    ToolCall(
                        function=FunctionCall(
                            name="mock_tool",
                            arguments={"query": "second query"}
                        )
                    )
                ]
            
            mock_parser_dict.__getitem__.return_value = mock_parser
            
            with patch('cartridges.synthesizers.self_study.MODEL_TO_TOOL_TEMPLATE') as mock_template_dict:
                mock_template_dict.__getitem__.return_value = "Available tools: {tools}"
                
                with patch('cartridges.synthesizers.self_study.render_tool_template') as mock_render:
                    mock_render.return_value = "Mock tool description"
                    
                    examples = await synthesizer.sample_convos(
                        batch_idx=0,
                        batch_size=1,
                        total_batches=1
                    )
                    
                    # Verify multiple tool calls are handled
                    assert len(examples) == 1
                    example = examples[0]
                    tool_calls = example.metadata["tool_calls"]
                    # Should have multiple tool calls (from both bot A and bot B if both use tools)
                    assert len(tool_calls) >= 2
    
    def test_tool_responses_to_str(self, synthesizer):
        """Test the conversion of tool outputs to string format."""
        from cartridges.tests.mocks import MockToolInput
        
        tool_outputs = [
            ToolOutput(
                input=MockToolInput(query="test query 1"),
                success=True,
                error=None,
                response="Tool response 1"
            ),
            ToolOutput(
                input=MockToolInput(query="test query 2"),
                success=False,
                error="Tool failed",
                response=None
            ),
            ToolOutput(
                input=MockToolInput(query="test query 3"),
                success=True,
                error=None,
                response="Tool response 3"
            )
        ]
        
        result = synthesizer._tool_responses_to_str(tool_outputs)
        
        # Should only include successful tool calls
        assert "Tool response 1" in result
        assert "Tool response 3" in result
        assert "Tool failed" not in result
        assert "<tool_call>" in result
        assert "<tool_input>" in result
        assert "<tool_output>" in result
    
    @pytest.mark.asyncio
    async def test_client_interaction_tracking(self, synthesizer):
        """Test that client interactions are properly tracked."""
        with patch('cartridges.synthesizers.self_study.MODEL_TO_TOOL_CALL_PARSER') as mock_parser_dict:
            from cartridges.data import ToolCall, FunctionCall
            
            def mock_parser(text):
                return [ToolCall(
                    function=FunctionCall(
                        name="mock_tool",
                        arguments={"query": "test query"}
                    )
                )]
            
            mock_parser_dict.__getitem__.return_value = mock_parser
            
            with patch('cartridges.synthesizers.self_study.MODEL_TO_TOOL_TEMPLATE') as mock_template_dict:
                mock_template_dict.__getitem__.return_value = "Available tools: {tools}"
                
                with patch('cartridges.synthesizers.self_study.render_tool_template') as mock_render:
                    mock_render.return_value = "Mock tool description"
                    
                    initial_call_count = synthesizer.client.call_count
                    
                    await synthesizer.sample_convos(
                        batch_idx=0,
                        batch_size=1,
                        total_batches=1
                    )
                    
                    # Verify that the client was called multiple times
                    # (tool selection + bot A response + bot B response, potentially more with tools)
                    assert synthesizer.client.call_count > initial_call_count
                    assert len(synthesizer.client.call_history) > 0
    
    @pytest.mark.asyncio
    async def test_different_batch_sizes(self, synthesizer):
        """Test the synthesizer with different batch sizes."""
        batch_sizes = [1, 3, 5]
        
        for batch_size in batch_sizes:
            with patch('cartridges.synthesizers.self_study.MODEL_TO_TOOL_CALL_PARSER') as mock_parser_dict:
                from cartridges.data import ToolCall
                
                def mock_parser(text):
                    return [ToolCall(
                        function=ToolCall.Function(
                            name="mock_tool",
                            arguments={"query": "test query"}
                        )
                    )]
                
                mock_parser_dict.__getitem__.return_value = mock_parser
                
                with patch('cartridges.synthesizers.self_study.MODEL_TO_TOOL_TEMPLATE') as mock_template_dict:
                    mock_template_dict.__getitem__.return_value = "Available tools: {tools}"
                    
                    with patch('cartridges.synthesizers.self_study.render_tool_template') as mock_render:
                        mock_render.return_value = "Mock tool description"
                        
                        examples = await synthesizer.sample_convos(
                            batch_idx=0,
                            batch_size=batch_size,
                            total_batches=1
                        )
                        
                        # Should produce exactly the requested batch size
                        assert len(examples) == batch_size
                        
                        # Each example should have the required fields
                        for example in examples:
                            assert example.messages is not None
                            assert example.metadata is not None
                            assert "tool_calls" in example.metadata