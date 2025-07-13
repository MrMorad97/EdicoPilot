# FinetunedGPT2Chat.py

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Iterator
import time
import logging

class FinetunedGPT2Chat:
    def __init__(self, model_dir: str = "./fine_tuned_gpt2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_dir).to(self.device)

    def generate_reply(self, prompt: str, max_length: int = 150) -> str:
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
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            generated = inputs["input_ids"].clone()
            for _ in range(max_length):
                with torch.no_grad():
                    outputs = self.model(generated)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                decoded = self.tokenizer.decode(next_token, skip_special_tokens=True)
                if next_token == self.tokenizer.eos_token_id:
                    break
                yield decoded
                time.sleep(0.03)
        except Exception as e:
            logging.error(f"Streaming generation failed: {e}")
            yield "Error in streaming response"
