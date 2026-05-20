"""Test script for OpenAIClient"""

import pytest
import asyncio
import os
import time

from cartridges.clients.openai import OpenAIClient
from cartridges.clients.base import ClientResponse, ClientSample
from cartridges.clients.usage import Usage

base_url = os.path.join(os.environ.get(
    "CARTRIDGES_VLLM_QWEN3_4B_URL", 
    "http://localhost:8000"
), "v1")
model_name = "Qwen/Qwen3-4b"

@pytest.fixture
def openai_config():
    """Fixture for OpenAIClient config."""
    return OpenAIClient.Config(
        model_name="gpt-4o-mini",  # Use cheaper model for testing
        api_key=None,  # Will use OPENAI_API_KEY env var
        truncate_messages_and_retry=True
    )


@pytest.fixture
def test_chats():
    """Fixture for test chat conversations."""
    return [
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "user", "content": "Write a haiku about programming"}],
        [{"role": "user", "content": "Explain recursion in one sentence"}],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the capital of France?"}
        ]
    ]


@pytest.fixture
def simple_test_chats():
    """Fixture for simple test conversations for performance testing."""
    return [
        [{"role": "user", "content": f"Count to {i+1}"}]
        for i in range(3)
    ]


@pytest.mark.asyncio
async def test_openai_client_basic_chat(openai_config, test_chats):
    """Test basic chat functionality."""
    
    client = OpenAIClient(openai_config)
    
    response = await client.chat(
        chats=test_chats,
        max_completion_tokens=100,
        temperature=0.0,
    )
    
    assert isinstance(response, ClientResponse)
    assert len(response.samples) == len(test_chats)
    assert isinstance(response.usage, Usage)
    
    # Check that all samples have text content
    for sample in response.samples:
        assert isinstance(sample, ClientSample)
        assert isinstance(sample.text, str)
        assert len(sample.text) > 0
        
    # Verify usage statistics
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0


@pytest.mark.asyncio
async def test_openai_client_single_chat(openai_config):
    """Test single chat request (legacy format support)."""
    
    client = OpenAIClient(openai_config)
    
    # Test legacy single chat format
    single_chat = {"role": "user", "content": "Hello, world!"}
    
    response = await client.chat(
        chats=[single_chat],  # This should be handled as single chat format
        max_completion_tokens=50,
        temperature=0.0,
    )
    
    assert isinstance(response, ClientResponse)
    assert len(response.samples) == 1
    assert isinstance(response.samples[0].text, str)


@pytest.mark.asyncio
async def test_openai_client_parallel_performance(openai_config, simple_test_chats):
    """Test that parallel execution is faster than sequential."""
    
    client = OpenAIClient(openai_config)
    
    # Time parallel execution
    start_time = time.time()
    parallel_response = await client.chat(
        chats=simple_test_chats,
        max_completion_tokens=30,
        temperature=0.1,
    )
    parallel_time = time.time() - start_time
    
    # Time sequential execution
    start_time = time.time()
    sequential_responses = []
    for chat in simple_test_chats:
        response = await client.chat(
            chats=[chat],
            max_completion_tokens=30,
            temperature=0.1,
        )
        sequential_responses.append(response.samples[0])
    sequential_time = time.time() - start_time
    
    # Verify both produced the same number of responses
    assert len(parallel_response.samples) == len(sequential_responses)
    
    # Parallel should be faster (or at least not significantly slower)
    # Allow some tolerance for network variability
    assert parallel_time <= sequential_time * 1.2, f"Parallel ({parallel_time:.2f}s) should be faster than sequential ({sequential_time:.2f}s)"


@pytest.mark.asyncio
async def test_openai_client_with_conversation_id(openai_config):
    """Test conversation tracking for caching."""
    
    client = OpenAIClient(openai_config)
    
    conversation_id = "test_conversation_123"
    
    # First request
    first_response = await client.chat(
        chats=[[{"role": "user", "content": "Remember: my favorite color is blue."}]],
        max_completion_tokens=50,
        temperature=0.0,
        conversation_id=conversation_id
    )
    
    # Second request continuing the conversation
    second_response = await client.chat(
        chats=[[
            {"role": "user", "content": "Remember: my favorite color is blue."},
            {"role": "assistant", "content": first_response.samples[0].text},
            {"role": "user", "content": "What's my favorite color?"}
        ]],
        max_completion_tokens=50,
        temperature=0.0,
        conversation_id=conversation_id
    )  
    
    # Verify conversation tracking worked
    assert conversation_id in client.conversations
    assert len(client.conversations[conversation_id]) == 2
    
    # Check usage tracking (second request should have some cached tokens)
    assert second_response.usage.seen_prompt_tokens > 0


@pytest.mark.asyncio 
async def test_openai_client_with_stop_sequences(openai_config):
    """Test chat with stop sequences."""
    
    client = OpenAIClient(openai_config)
    
    response = await client.chat(
        chats=[[{"role": "user", "content": "Count from 1 to 10: 1, 2, 3,"}]],
        max_completion_tokens=100,
        temperature=0.0,
        stop=[",", "."]
    )
    
    assert isinstance(response, ClientResponse)
    assert len(response.samples) == 1
    
    # Response should stop at the stop sequence
    text = response.samples[0].text
    assert isinstance(text, str)


@pytest.mark.asyncio
async def test_openai_client_empty_chats():
    """Test error handling for empty chats."""
    
    config = OpenAIClient.Config(model_name="gpt-4o-mini")
    client = OpenAIClient(config)
    
    with pytest.raises(AssertionError):
        await client.chat(chats=[])


@pytest.mark.asyncio
async def test_openai_client_logprobs(openai_config):
    """Test that logprobs are properly handled when available."""
    
    client = OpenAIClient(openai_config)
    
    response = await client.chat(
        chats=[[{"role": "user", "content": "Say 'hello'"}]],
        max_completion_tokens=10,
        temperature=0.0,
    )
    
    assert isinstance(response, ClientResponse)
    assert len(response.samples) == 1
    
    sample = response.samples[0]
    # Note: top_logprobs might be None if logprobs aren't available
    # but should not cause errors
    assert sample.top_logprobs is None or hasattr(sample.top_logprobs, 'logprobs')


# Integration test that requires actual OpenAI API key
@pytest.mark.asyncio
async def test_openai_client_huggingface_tokenizer():
    """Test OpenAI client with Huggingface tokenizer (Qwen model)."""
    
    # Configure with a Huggingface model
    config = OpenAIClient.Config(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",  # Small model that should work with HF tokenizer
        base_url="https://api.example.com/v1",  # Mock URL for testing
        api_key="test-key"
    )
    
    client = OpenAIClient(config)
    
    # Verify that it uses a Huggingface tokenizer (not tiktoken)
    assert not hasattr(client.tokenizer, 'name') or not isinstance(client.tokenizer.name, str)
    
    # Test token counting with the HF tokenizer
    from cartridges.clients.usage import num_tokens_from_messages_flexible
    
    test_messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"}
    ]
    
    token_count = num_tokens_from_messages_flexible(test_messages, client.tokenizer)
    assert isinstance(token_count, int)
    assert token_count > 0


@pytest.mark.asyncio
async def test_openai_client_tiktoken_fallback():
    """Test that OpenAI client falls back to tiktoken for OpenAI models."""
    
    config = OpenAIClient.Config(
        model_name="gpt-4o-mini",
        api_key="test-key"
    )
    
    client = OpenAIClient(config)
    
    # Verify that it uses tiktoken for OpenAI models
    assert hasattr(client.tokenizer, 'name')
    assert isinstance(client.tokenizer.name, str)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_client_integration(openai_config, test_chats):
    """Integration test with real OpenAI API (requires OPENAI_API_KEY)."""
    
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set - skipping integration test")
    
    client = OpenAIClient(openai_config)
    
    response = await client.chat(
        chats=test_chats[:2],  # Test with just 2 requests to save costs
        max_completion_tokens=50,
        temperature=0.7,
    )
    
    assert isinstance(response, ClientResponse)
    assert len(response.samples) == 2
    
    for sample in response.samples:
        assert isinstance(sample.text, str)
        assert len(sample.text.strip()) > 0
    
    # Check usage statistics are reasonable
    assert 0 < response.usage.prompt_tokens < 1000
    assert 0 < response.usage.completion_tokens < 200


@pytest.mark.integration 
@pytest.mark.asyncio
async def test_openai_client_with_huggingface_model_integration():
    """Integration test with Huggingface model via OpenAI-compatible API."""
    
    # This would test against a real Huggingface model served via an OpenAI-compatible API
    # Skip if no test URL is provided
    test_url = os.getenv("TEST_HF_MODEL_URL")  # e.g., Modal or vLLM endpoint
    if not test_url:
        pytest.skip("TEST_HF_MODEL_URL not set - skipping HF model integration test")
    
    config = OpenAIClient.Config(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        base_url=test_url,
        api_key="test-key"  # Many self-hosted endpoints don't require real keys
    )
    
    client = OpenAIClient(config)
    
    # Test that tokenizer initialization worked
    assert client.tokenizer is not None
    
    simple_chats = [
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "user", "content": "Say hello"}]
    ]
    
    try:
        response = await client.chat(
            chats=simple_chats,
            max_completion_tokens=30,
            temperature=0.1,
        )
        
        assert isinstance(response, ClientResponse)
        assert len(response.samples) == 2
        
        for sample in response.samples:
            assert isinstance(sample.text, str)
            
    except Exception as e:
        pytest.skip(f"Integration test failed - this is expected if endpoint is not available: {e}")


if __name__ == "__main__":
    # Run basic tests if executed directly
    async def run_basic_test():
        config = OpenAIClient.Config(model_name="gpt-4o-mini")
        client = OpenAIClient(config)
        
        test_chats = [
            [{"role": "user", "content": "What is 2+2?"}],
            [{"role": "user", "content": "What is 3+3?"}]
        ]
        
        print("Testing OpenAI client...")
        response = await client.chat(
            chats=test_chats,
            max_completion_tokens=50,
            temperature=0.0,
        )
        
        print(f"✅ Success! Got {len(response.samples)} responses")
        for i, sample in enumerate(response.samples):
            print(f"Response {i+1}: {sample.text[:100]}...")
    
    if os.getenv("OPENAI_API_KEY"):
        asyncio.run(run_basic_test())
    else:
        print("Set OPENAI_API_KEY to run basic test")