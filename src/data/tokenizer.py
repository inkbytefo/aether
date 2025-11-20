from transformers import AutoTokenizer
import torch

class Tokenizer:
    def __init__(self, model_name: str = "gpt2", max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        # GPT-2 does not have a pad token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    def encode(self, text: str) -> torch.Tensor:
        """Encodes text to tensor with padding/truncation."""
        return self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decodes tensor to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
