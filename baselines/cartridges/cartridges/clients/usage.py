from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import tiktoken

@dataclass
class Usage: 
    completion_tokens: int = 0
    prompt_tokens: int = 0

    # Some clients explicitly tell us whether or not we are using cached 
    # prompt tokens and being charged for less for them. This is distinct from 
    # seen_prompt_tokens since we don't know exactly how they determine if there's
    # a cache hit
    cached_prompt_tokens: int = 0
    
    # We keep track of the prompt tokens that have been seen in the 
    # conversation history.
    seen_prompt_tokens: int = 0

    @property
    def new_prompt_tokens(self) -> int:
        if self.seen_prompt_tokens is None:
            return self.prompt_tokens
        return self.prompt_tokens - self.seen_prompt_tokens
    
    @property
    def total_tokens(self) -> int:
        return self.completion_tokens + self.prompt_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            completion_tokens=self.completion_tokens + other.completion_tokens,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            cached_prompt_tokens=self.cached_prompt_tokens + other.cached_prompt_tokens,
            seen_prompt_tokens=self.seen_prompt_tokens + other.seen_prompt_tokens,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "completion_tokens": self.completion_tokens,
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "seen_prompt_tokens": self.seen_prompt_tokens,
            "new_prompt_tokens": self.new_prompt_tokens,
        }




def num_tokens_from_messages_openai(
    messages: List[Dict[str, str]], 
    encoding: tiktoken.Encoding,
    include_reply_prompt: bool = False,
):
    """Return the number of tokens used by a list of messages.
    Source: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """

    # NOTE: this may change in the future
    tokens_per_message = 3
    tokens_per_name = 1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    if include_reply_prompt:
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def num_tokens_from_messages_flexible(
    messages: List[Dict[str, str]], 
    tokenizer: Union[tiktoken.Encoding, Any],
    include_reply_prompt: bool = False,
):
    """Return the number of tokens used by a list of messages.
    
    Works with both tiktoken.Encoding and Huggingface tokenizers.
    """
    
    # Check if it's a tiktoken encoding
    if hasattr(tokenizer, 'encode') and hasattr(tokenizer, 'name'):
        # Use the original OpenAI counting logic for tiktoken
        return num_tokens_from_messages_openai(messages, tokenizer, include_reply_prompt)
    
    # Handle Huggingface tokenizers
    try:
        # Import here to avoid dependency issues if transformers isn't installed
        from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
        
        if isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            # For Huggingface tokenizers, we need to estimate the overhead
            # Different models have different chat templates and overhead
            tokens_per_message = 3  # Conservative estimate
            tokens_per_name = 1
            
            num_tokens = 0
            for message in messages:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    # Use the tokenizer's encode method
                    tokens = tokenizer.encode(value, add_special_tokens=False)
                    num_tokens += len(tokens)
                    if key == "name":
                        num_tokens += tokens_per_name
                        
            if include_reply_prompt:
                num_tokens += 3  # Rough estimate for reply prompt
                
            return num_tokens
        
    except ImportError:
        pass
    
    # Fallback: if we can't identify the tokenizer type, try to use it anyway
    # This handles custom tokenizers that implement an encode method
    if hasattr(tokenizer, 'encode'):
        tokens_per_message = 3
        tokens_per_name = 1
        
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                try:
                    tokens = tokenizer.encode(value)
                    # Handle different return types (list of ints, tensor, etc.)
                    if hasattr(tokens, '__len__'):
                        num_tokens += len(tokens)
                    else:
                        # Fallback to character-based estimation
                        num_tokens += len(value) // 4  # Rough estimate: 4 chars per token
                except Exception:
                    # Last resort: character-based estimation
                    num_tokens += len(value) // 4
                    
                if key == "name":
                    num_tokens += tokens_per_name
                    
        if include_reply_prompt:
            num_tokens += 3
            
        return num_tokens
    
    # Final fallback: character-based estimation
    total_chars = sum(len(str(message.get(key, ""))) for message in messages for key in message)
    return total_chars // 4  # Very rough estimate
