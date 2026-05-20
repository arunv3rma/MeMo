#!/usr/bin/env python3
"""
Interactive CLI for chatting with a trained cache model.
Usage: python -m cartridges.utils.chat <wandb_run_id>
Example: python -m cartridges.utils.chat hazy-research/cartridges/ehij7vlt
"""

import argparse
import os
import sys
import torch
import readline
from typing import List, Dict
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig
from cartridges.initialization.text import KVFromText
from cartridges.utils.wandb import load_model_and_cache_from_wandb
from cartridges.generation import flex_generate


class ChatSession:
    def __init__(self, model, tokenizer, cache):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache
        self.conversation_history: List[Dict[str, str]] = []
        self.input_history: List[str] = []
        
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})
    
    def undo_last_message(self):
        """Remove the last two messages (user and assistant)."""
        if len(self.conversation_history) >= 2:
            self.conversation_history = self.conversation_history[:-2]
            return True
        elif len(self.conversation_history) == 1:
            self.conversation_history = []
            return True
        return False
    
    def clear_conversation(self):
        """Clear the entire conversation history."""
        self.conversation_history = []
    
    def generate_response(self, user_input: str, enable_thinking: bool = False) -> str:
        """Generate a response to the user input."""
        # Add to input history for readline
        if user_input.strip() and user_input not in self.input_history:
            self.input_history.append(user_input)
            readline.add_history(user_input)
        
        # Add user message to history
        self.add_message("user", user_input)
        
        # Prepare the conversation for the tokenizer
        input_ids = self.tokenizer.apply_chat_template(
            self.conversation_history,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=enable_thinking,
        ).to("cuda")
        
        # Flatten and create seq_ids and position_ids for single conversation
        flat_input_ids = input_ids.flatten()
        seq_ids = torch.zeros(flat_input_ids.shape[0], dtype=torch.long, device="cuda")
        position_ids = torch.arange(flat_input_ids.shape[0], device="cuda")
        
        # Generate response
        output = flex_generate(
            model=self.model,
            input_ids=flat_input_ids,
            seq_ids=seq_ids,
            position_ids=position_ids,
            tokenizer=self.tokenizer,
            cache=self.cache,
            max_new_tokens=256,
            temperature=0.0,
            show_progress=True,
        )
        
        # Decode the response
        if 0 in output and output[0]:
            response_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            self.add_message("assistant", response_text)
            return response_text
        else:
            return "I'm sorry, I couldn't generate a response."


def main():
    parser = argparse.ArgumentParser(description="Chat with a trained cache model")
    parser.add_argument("wandb_run_id", help="WandB run ID (e.g., hazy-research/cartridges/ehij7vlt)")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable thinking")
    args = parser.parse_args()
    
    print(f"Loading model and cache from {args.wandb_run_id}...")
    
    try:
        # Load model and cache
        cache_and_model = load_model_and_cache_from_wandb(
            wandb_run_id=args.wandb_run_id,
        )
        
        model = cache_and_model.model.to("cuda").to(torch.bfloat16)
        cache = cache_and_model.cache.to("cuda").to(torch.bfloat16)
    
        tokenizer = AutoTokenizer.from_pretrained(cache_and_model.model.name_or_path)

        print("Model and cache loaded successfully!\n")
        
    except Exception as e:
        print(f"Error loading model and cache: {e}")
        sys.exit(1)
    
    # Initialize chat session
    chat = ChatSession(model, tokenizer, cache)
    
    # Configure readline for better input handling
    readline.set_startup_hook(None)
    
    print("=== Chat with Trained Cache ===")
    print("Commands:")
    print("  /undo  - Undo the last message exchange")
    print("  /clear - Clear the entire conversation")
    print("  /quit  - Exit the chat")
    print("  /help  - Show this help message")
    print("Arrow keys: ↑↓ for command history, ←→ for line editing")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input == "/quit":
                print("Goodbye!")
                break
            elif user_input == "/help":
                print("\nCommands:")
                print("  /undo  - Undo the last message exchange")
                print("  /clear - Clear the entire conversation")
                print("  /quit  - Exit the chat")
                print("  /help  - Show this help message")
                print("Arrow keys: ↑↓ for command history, ←→ for line editing")
                continue
            elif user_input == "/undo":
                if chat.undo_last_message():
                    print("Last message exchange undone.")
                else:
                    print("No messages to undo.")
                continue
            elif user_input == "/clear":
                chat.clear_conversation()
                print("Conversation cleared.")
                continue
            
            # Generate and display response
            print("Assistant: ", end="", flush=True)
            response = chat.generate_response(user_input, enable_thinking=args.enable_thinking)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError generating response: {e}")
            print("Type /help for available commands.")


if __name__ == "__main__":
    main()