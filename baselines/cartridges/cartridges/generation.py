from collections import defaultdict
import os
from typing import Any, Dict, List, Optional
from transformers import DynamicCache, AutoTokenizer
import torch
from tqdm import tqdm

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.utils import get_logger

logger = get_logger(__name__)


def flex_generate(
    model,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    seq_ids: torch.Tensor,
    position_ids: torch.Tensor,
    cache: Optional[TrainableCache] = None,
    stop_token_ids: Optional[List[int]] = None,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    show_progress: bool = False,
) -> Dict[int, List[int]]:
    """Autoregressive generation with FlexAttention (e.g. FlexLlamaModel, FlexQwen3Model).
    
    Args:
        model: The model to use for generation
        input_ids: (N,) tensor of input ids where N is the total number of tokens across 
            the sequences.
        seq_ids: (N,) tensor specifying the membership of each token to a sequence
        position_ids: (N,) tensor of position of a token within it's sequence
        stop_token_ids: By default, will use the end of text id from the tokenizer.
        tokenizer: tokenizer to use for decoding
        max_new_tokens: maximum number of new tokens to generate.
        temperature: temperature for sampling
        show_progress: whether to show a progress bar during generation
    
    This implementation relies on the PackedCache above.
    """
            
    device = input_ids.device
    model.eval()
    if stop_token_ids is None:
        stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    
    if cache is None:
        cache = TrainableCache(
            config=AttnConfig(
                n_layers=model.config.num_hidden_layers,
                n_heads=model.config.num_key_value_heads,
                head_dim=model.config.head_dim,
            ),
        )
        
    # Initialize generated sequences
    generated_tokens: Dict[int, List[int]] = defaultdict(list)
    
    # Current state
    current_input_ids = input_ids
    current_seq_ids = seq_ids
    current_position_ids = position_ids
    
    progress_range = tqdm(range(max_new_tokens), desc="Generating", disable=not show_progress, leave=False)
    for step in progress_range:
        # Forward pass - update cache with current seq_ids before the forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=current_input_ids,
                seq_ids=current_seq_ids,
                position_ids=current_position_ids,
                past_key_values=cache,
                use_cache=True,
                mode="generate",
            )
        
        # Get logits for the last token of each sequence
        logits = outputs.logits  # (1, seq_len, vocab_size)
        last_logits = logits[0, -len(current_input_ids):, :]  # Get logits for current tokens
        
        # Sample next tokens for each sequence
        next_tokens = []
        next_seq_ids = []
        next_position_ids = []
        
        # Group tokens by sequence
        seq_groups = {}
        for i, seq_id in enumerate(current_seq_ids):
            if seq_id.item() not in seq_groups:
                seq_groups[seq_id.item()] = []
            seq_groups[seq_id.item()].append(i)
        
        active_sequences = []
        
        for seq_id, token_indices in seq_groups.items():
            # Get the last token's logits for this sequence
            last_token_idx = token_indices[-1]
            token_logits = last_logits[last_token_idx]
            
            # Apply temperature
            if temperature > 0:
                token_logits = token_logits / temperature
                next_token = torch.multinomial(torch.softmax(token_logits, dim=-1), 1).item()
            else:
                next_token = token_logits.argmax().item()
            
            # Check if this sequence should continue
            if next_token not in stop_token_ids:
                next_tokens.append(next_token)
                next_seq_ids.append(seq_id)
                next_position_ids.append(current_position_ids[last_token_idx] + 1)
                generated_tokens[seq_id].append(next_token)
                active_sequences.append(seq_id)
        
        # If no sequences are active, break
        if not next_tokens:
            progress_range.close()
            break
        
        # Prepare inputs for next iteration
        current_input_ids = torch.tensor(next_tokens, device=device, dtype=torch.long)
        current_seq_ids = torch.tensor(next_seq_ids, device=device, dtype=torch.long)
        current_position_ids = torch.tensor(next_position_ids, device=device, dtype=torch.long)
        
    # SE (07/26): Very important to clear the cache after generation, otherwise, during
    # training, the keys and values from the last generation will be included
    # This issue is silent when training on a single GPU, but becomes apparent when
    # training on multiple GPUs. We get a crash on flex attention I guess because the 
    # cache sizes differ between GPUs.
    cache.clear()
    
    return generated_tokens
    
    

if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    from cartridges.utils.wandb import load_model_and_cache_from_wandb


    # Define command line argument parser
    parser = argparse.ArgumentParser(description="Select model type")
    parser.add_argument("--model", default="llama", help="Choose between 'llama' and 'qwen' models")
    args = parser.parse_args()

    # Import the appropriate model based on the command line argument
    if args.model == "llama":
        from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
        model_name = "meta-llama/Llama-3.2-3B-Instruct"
        model = FlexLlamaForCausalLM.from_pretrained(model_name).to("cuda").to(torch.bfloat16)
        cache = None
    elif args.model == "qwen":
        from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
        model_name = "Qwen/Qwen3-4B"
        model = FlexQwen3ForCausalLM.from_pretrained(model_name).to("cuda").to(torch.bfloat16)
        cache = None

    elif args.model.startswith("hazy-research"):
        cache_and_model = load_model_and_cache_from_wandb(
            wandb_run_id="hazy-research/cartridges/ehij7vlt",
            step=29,
        )
        model_name = cache_and_model.model.name_or_path
        cache = cache_and_model.cache.to("cuda").to(torch.bfloat16)
        model = cache_and_model.model.to("cuda").to(torch.bfloat16)
    else:
        raise ValueError(f"Model {args.model} not supported")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    convos = [
        [
            {"role": "user", "content": "What is the capital of the moon?"},
        ],
        [
            {"role": "user", "content": "Who is the patient?"},
        ],
        # [
        #     {"role": "user", "content": "Why is the sky blue?"},
        # ],
    ]

    input_ids, seq_ids, position_ids = [], [], []
    for idx, convo in enumerate(convos):
        curr_input_ids = tokenizer.apply_chat_template(
            convo, 
            tokenize=True, 
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")
        # Flatten the input_ids and create corresponding seq_ids and position_ids
        flat_input_ids = curr_input_ids.flatten()
        curr_seq_ids = torch.full((flat_input_ids.shape[0],), idx, dtype=torch.long, device="cuda")
        curr_position_ids = torch.arange(flat_input_ids.shape[0], device="cuda")
        
        input_ids.append(flat_input_ids)
        seq_ids.append(curr_seq_ids)
        position_ids.append(curr_position_ids)
    
    input_ids = torch.cat(input_ids, dim=0)
    seq_ids = torch.cat(seq_ids, dim=0)
    position_ids = torch.cat(position_ids, dim=0)

    print("Starting generation...")
    print(f"Input shapes: input_ids={input_ids.shape}, seq_ids={seq_ids.shape}, position_ids={position_ids.shape}")

    if cache is None:
        cache = TrainableCache(
            max_seq_len=1024,
            config=AttnConfig(
                n_layers=model.config.num_hidden_layers,
                n_heads=model.config.num_key_value_heads,
                head_dim=model.config.head_dim,
            ),
            device=str(input_ids.device),
        )
    
    output = flex_generate(
        model=model,
        input_ids=input_ids,
        seq_ids=seq_ids,
        position_ids=position_ids,
        tokenizer=tokenizer,
        max_new_tokens=128,  # Reduce for testing
        show_progress=True,
        cache=cache,
    )
    print("Generated tokens:", output)
    
    # Decode the output
    for seq_idx, tokens in output.items():
        if tokens:
            decoded = tokenizer.decode(tokens)
            print(decoded)
            print(f"Sequence {seq_idx}: {decoded}")