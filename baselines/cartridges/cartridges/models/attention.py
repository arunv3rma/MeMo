import torch

from typing import Optional, Union, Literal
from torch.nn.attention.flex_attention import create_block_mask, flex_attention, BlockMask

from cartridges.cache import TrainableCache



# SE (07/21): `dynamic=False` is necessary to avoid a "PassManager::run failed" error
# when interacting with torch.amp.autocast during training. This is okay since we pack
# all sequences to the same length during training.
# SE (07/22): The `mode="max-autotune-no-cudagraphs"` gives a ~2x speedup on 
# backward running on a single A100.
flex_attention_train = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

# # SE (07/25): For generation, we need to use `dynamic=True` to avoid a "PassManager::run failed" error
flex_attention_generate = torch.compile(flex_attention, dynamic=True) 


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def create_block_mask_w_cache(
    cache: Optional[TrainableCache],
    seq_ids: torch.LongTensor, # [sum(seq_lens)]
    device: torch.device,
):
    cache_len = cache.num_tokens() if cache is not None else 0

    # Build the block mask
    # --- begin build block mask ---
    kv_seq_ids = seq_ids
    if cache_len > 0:
        kv_seq_ids = torch.cat([cache.seq_ids(), kv_seq_ids])

    def mask_func(_, _h, q_idx, kv_idx):
        return (kv_seq_ids[kv_idx] == -1) | ((seq_ids[q_idx] == kv_seq_ids[kv_idx]) & (q_idx + cache_len >= kv_idx))
    
    block_mask = create_block_mask(
        mask_func, B=1, H=1, Q_LEN=len(seq_ids), KV_LEN=len(seq_ids) + cache_len, 
        device=device,
        # _compile=True
    )
    return block_mask
    # --- end build block mask ---


def flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Union[torch.Tensor, "BlockMask"],
    scaling: Optional[float] = None,
    mode: Literal["train", "generate"] = "train",
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:

    if kwargs.get("dropout", 0.0) > 0:
        raise ValueError(
            "`flex_attention` does not support `dropout`. Please use it with inference"
            " only (`model.eval()`) or turn off the attention dropout in the respective config."
        )

    block_mask = None
    if isinstance(attention_mask, BlockMask):
        block_mask = attention_mask

    enable_gqa = True
    num_local_query_heads = query.shape[1]

    # SE (07/25): For grouped-query attention, to work, the `Number of shared query 
    # heads sharing the same KV head must be power of 2`
    # This is the case with 3.2-3B (24 qheads and 8 kvheads), so we need to repeat 
    # the key  and value and turn off GQA. :( 
    if not ((num_local_query_heads & (num_local_query_heads - 1)) == 0):
        key = repeat_kv(key, query.shape[1] // key.shape[1])
        value = repeat_kv(value, query.shape[1] // value.shape[1])
        enable_gqa = False

    kernel_options = kwargs.get("kernel_options", None)
    attn = flex_attention_train if mode == "train" else flex_attention_generate
    
    # SE (07/26): This helps to avoid recompiles, since during prefix tuning, the first
    # layer's query does not require grad.
    if key.requires_grad and not query.requires_grad:
        query.requires_grad = True

    attn_output = attn(
        query,
        key,
        value,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
        scale=scaling,
        kernel_options=kernel_options,
        return_lse=False,
    )    
    attn_output = attn_output.transpose(1, 2).contiguous()


    return attn_output

