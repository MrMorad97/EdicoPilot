import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Iterator, Optional
import time
import streamlit as st
import logging

class StreamingGPT2Chat:
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize the streaming GPT-2 model
        
        Args:
            model_name: Name of the GPT-2 model variant (e.g., "gpt2", "gpt2-medium")
        """
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model_name = model_name
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logging.error(f"Failed to initialize GPT-2 model: {e}")
            raise

    def generate_reply(self, prompt: str, max_length: int = 150) -> str:
        """
        Generate complete response (non-streaming)
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated response
            
        Returns:
            Generated text response
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_k=50
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            return "Error generating response"

    def stream_response(self, prompt: str, max_length: int = 150) -> Iterator[str]:
        """
        Generate token-by-token streaming response
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated response
            
        Yields:
            Tokens of the generated response one by one
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            generated = inputs["input_ids"].clone()
            
            for _ in range(max_length):
                with torch.no_grad():
                    outputs = self.model(generated)
                
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)
                
                decoded = self.tokenizer.decode(next_token, skip_special_tokens=True)
                
                if next_token == self.tokenizer.eos_token_id:
                    break
                    
                yield decoded
                time.sleep(0.03)  # Natural typing speed
                
        except Exception as e:
            logging.error(f"Streaming generation failed: {e}")
            yield "Error in streaming response"

class GPT2Chat:
    """Non-streaming GPT-2 chat interface"""
    def __init__(self, model_name: str = "gpt2"):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logging.error(f"Failed to initialize GPT2Chat: {e}")
            raise

    def generate_reply(self, prompt: str, max_length: int = 150) -> str:
        """Generate complete response"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            return "Error generating response"