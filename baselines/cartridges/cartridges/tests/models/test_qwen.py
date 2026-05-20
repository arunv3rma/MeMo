import pytest

from cartridges.models.qwen.modeling_qwen3 import FlexQwen3Model
from cartridges.models.qwen.configuration_qwen3 import Qwen3Config
from transformers import Qwen3Model, Qwen3Config


def small_qwen_config(num_layers):
    """Create a small Qwen3 config for testing."""
    return Qwen3Config(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=32,
        max_position_embeddings=256,
        use_cache=False,
        layer_types=["full_attention"] * num_layers,
    )


@pytest.mark.parametrize("num_layers", [1, 4])
@pytest.mark.parametrize("seq_lens", [
    [64],
    [128, 64, 256, 64],
])
def test_qwen_no_cache_equivalence(num_layers, seq_lens):
    """Test FlexQwen3Model equivalence with Qwen3Model without cache."""
    from cartridges.tests.models.common import test_output_no_cache_equivalence
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = small_qwen_config(num_layers)
    model = FlexQwen3Model(config).to(device)
    ref_model = Qwen3Model(config).to(device)

    test_output_no_cache_equivalence(seq_lens, model, ref_model)


@pytest.mark.parametrize("num_layers", [1, 4])
@pytest.mark.parametrize("seq_lens", [
    [64],
    [128, 64, 256, 64],
])
@pytest.mark.parametrize("cartridge_len", [1024])
def test_qwen_with_cache_equivalence(num_layers, seq_lens, cartridge_len):
    """Test FlexQwen3Model equivalence with Qwen3Model with cache and gradients."""
    from cartridges.tests.models.common import test_output_with_cache_equivalence
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = small_qwen_config(num_layers)
    model = FlexQwen3Model(config).to(device)
    ref_model = Qwen3Model(config).to(device)
    
    test_output_with_cache_equivalence(seq_lens, model, ref_model, config, cartridge_len)