"""Test script for TokasaurusClient"""

import pytest
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.clients.base import ClientResponse, ClientSample
from cartridges.clients.usage import Usage

# NOTE: This test requires that we have a running Tokasaurus server. 
# Run `modal deploy infra/modal_deploy_tokasaurus.py` to deploy a server. 
# Then, paste the URL here.
URL = "https://fu-edh8vpwma2wnhmy8chmv8k.us-west.modal.direct/v1"

@pytest.fixture
def tokasaurus_config():
    """Fixture for TokasaurusClient config."""
    return TokasaurusClient.Config(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        url=URL,
        timeout=60,
        max_retries=3
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


@pytest.mark.asyncio
async def test_tokasaurus_client_basic_chat(tokasaurus_config, test_chats):
    """Test basic chat functionality."""
    
    with patch('cartridges.clients.tokasaurus.AutoTokenizer'):
        client = TokasaurusClient(tokasaurus_config)
        
        response = await client.chat(
            chats=test_chats,
            max_completion_tokens=100,
            temperature=0.0,
            top_logprobs=5
        )
        
        assert isinstance(response, ClientResponse)
        assert len(response.samples) == len(test_chats)

        for sample in response.samples:
            assert sample.top_logprobs is not None
        

