import pytest

from cartridges.models.llama.modeling_llama import FlexLlamaModel
from cartridges.models.llama.configuration_llama import LlamaConfig
from transformers import LlamaModel, LlamaConfig


def small_llama_config(num_layers):
    """Create a small Llama config for testing."""
    return LlamaConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=32,
        max_position_embeddings=256,
        use_cache=False,
    )


@pytest.mark.parametrize("num_layers", [1, 4])
@pytest.mark.parametrize("seq_lens", [
    [64],
    [128, 64, 256, 64],
])
def test_llama_no_cache_equivalence(num_layers, seq_lens):
    """Test FlexLlamaModel equivalence with LlamaModel without cache."""
    from cartridges.tests.models.common import test_output_no_cache_equivalence
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = small_llama_config(num_layers)
    model = FlexLlamaModel(config).to(device)
    ref_model = LlamaModel(config).to(device)
    
    test_output_no_cache_equivalence(seq_lens, model, ref_model)


@pytest.mark.parametrize("num_layers", [1, 4])
@pytest.mark.parametrize("seq_lens", [
    [64],
    [128, 64, 256, 64],
])
@pytest.mark.parametrize("cartridge_len", [1024])
def test_llama_with_cache_equivalence(num_layers, seq_lens, cartridge_len):
    """Test FlexLlamaModel equivalence with LlamaModel with cache and gradients."""
    from cartridges.tests.models.common import test_output_with_cache_equivalence
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = small_llama_config(num_layers)
    model = FlexLlamaModel(config).to(device)
    ref_model = LlamaModel(config).to(device)
    
    test_output_with_cache_equivalence(seq_lens, model, ref_model, config, cartridge_len)