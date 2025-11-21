from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast
import torch
import os

class Tokenizer:
    def __init__(self, model_path: str = None, max_length: int = 512):
        self.max_length = max_length
        
        if model_path and os.path.exists(model_path):
            # Load custom trained tokenizer
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=model_path)
            self.tokenizer.pad_token = "<|endoftext|>"
            self.tokenizer.eos_token = "<|endoftext|>"
        else:
            # Fallback or initialization for training
            # We don't load GPT-2 by default anymore if we want custom
            pass

    def train(self, files, vocab_size=32000, save_path="tokenizer.json"):
        """Trains a BPE tokenizer on the provided files."""
        tokenizer = HFTokenizer(BPE())
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<|endoftext|>", "<|padding|>"],
            show_progress=True
        )
        
        tokenizer.train_from_iterator(files, trainer)
        tokenizer.save(save_path)
        
        # Reload as PreTrainedTokenizerFast for easy usage
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        self.tokenizer.pad_token = "<|endoftext|>"
        self.tokenizer.eos_token = "<|endoftext|>"
        print(f"Tokenizer saved to {save_path}")

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
        )['input_ids']

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decodes tensor to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
