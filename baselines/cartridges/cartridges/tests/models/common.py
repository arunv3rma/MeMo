from typing import List
import torch
import torch.nn as nn
from transformers import DynamicCache
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.initialization.random import KVFromRandomVectors

def _prepare_test_data(seq_lens: List[int]):
    """Prepare common test data for packed sequence testing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    batch_size = len(seq_lens)
    max_seq_len = max(seq_lens)
    total_seq_len = sum(seq_lens)
    
    seq_ids = torch.cat([
        torch.full((seq_len,), idx, dtype=torch.long, device=device) 
        for idx, seq_len in enumerate(seq_lens)
    ])
    input_ids = torch.randint(0, 1000, (total_seq_len,)).to(device)  # Flat input for Flex models
    position_ids = torch.cat([
        torch.arange(seq_len, device=device) for seq_len in seq_lens
    ])
    
    padded_input_ids = torch.full((batch_size, max_seq_len), 0, dtype=torch.long, device=device)
    start_idx = 0
    for i, seq_len in enumerate(seq_lens):
        padded_input_ids[i, :seq_len] = input_ids[start_idx:start_idx+seq_len]
        start_idx += seq_len
    
    return {
        'device': device,
        'batch_size': batch_size,
        'max_seq_len': max_seq_len,
        'total_seq_len': total_seq_len,
        'seq_ids': seq_ids,
        'input_ids': input_ids,
        'position_ids': position_ids,
        'padded_input_ids': padded_input_ids,
        'seq_lens': seq_lens
    }


def test_output_no_cache_equivalence(
    seq_lens: List[int],
    model: nn.Module,
    ref_model: nn.Module,
):
    """Test Flex model equivalence with reference model without cache."""
    data = _prepare_test_data(seq_lens)
    
    ref_model.load_state_dict(model.state_dict())
    
    out = model(data['input_ids'], seq_ids=data['seq_ids'], position_ids=data['position_ids']).last_hidden_state
    ref_out_padded = ref_model(data['padded_input_ids']).last_hidden_state
    
    ref_out = torch.cat([
        ref_out_padded[batch_idx, :seq_len] for batch_idx, seq_len in enumerate(data['seq_lens'])
    ], dim=0).unsqueeze(0)
    
    torch.testing.assert_close(out, ref_out, atol=1e-3, rtol=1e-3)


def test_output_with_cache_equivalence(
    seq_lens: List[int],
    model: nn.Module,
    ref_model: nn.Module,
    config,
    cartridge_len: int = 1024
):
    """Test Flex model equivalence with reference model with cache."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = _prepare_test_data(seq_lens)
    batch_size = data['batch_size']
    
    ref_model.load_state_dict(model.state_dict())

    rand_vecs = lambda: [
        torch.randn(
            1, config.num_key_value_heads, cartridge_len, config.head_dim,
            dtype=torch.bfloat16,
            device=device,
            requires_grad=True,
        )
        for _ in range(config.num_hidden_layers)
    ]
    keys = rand_vecs()
    values = rand_vecs()
    
    cache = TrainableCache(
        config=AttnConfig(
            n_layers=config.num_hidden_layers,
            n_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        ),
        init_keys=keys, 
        init_values=values
    )
    ref_cache = DynamicCache()
    for layer_idx in range(config.num_hidden_layers):
        ref_cache.update(
            key_states=keys[layer_idx].repeat(batch_size, 1, 1, 1),
            value_states=values[layer_idx].repeat(batch_size, 1, 1, 1),
            layer_idx=layer_idx
        )
    
    cache.to(data['device'])
    
    out = model(
        data['input_ids'], 
        seq_ids=data['seq_ids'], 
        position_ids=data['position_ids'], 
        use_cache=True, 
        past_key_values=cache
    ).last_hidden_state
    out.sum().backward()
    keys_grad = cache.trainable_keys[0].grad.clone()
    values_grad = cache.trainable_values[0].grad.clone()
    
    cache.zero_grad()
    cache.clear()
    
    ref_out_padded = ref_model(
        data['padded_input_ids'], 
        use_cache=True, 
        past_key_values=ref_cache
    ).last_hidden_state
    ref_out = torch.cat([
        ref_out_padded[batch_idx, :seq_len] for batch_idx, seq_len in enumerate(data['seq_lens'])
    ], dim=0).unsqueeze(0)
    ref_out.sum().backward()
    ref_keys_grad = keys[0].grad.clone()
    ref_values_grad = values[0].grad.clone()
    
    torch.testing.assert_close(out, ref_out, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(keys_grad, ref_keys_grad, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(values_grad, ref_values_grad, atol=1e-1, rtol=1e-2)
