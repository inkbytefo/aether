from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torch
from typing import Optional
from .tokenizer import Tokenizer

class TinyStoriesDataset(Dataset):
    def __init__(self, split: str = "train", tokenizer: Optional[Tokenizer] = None, max_length: int = 512):
        self.dataset = load_dataset("roneneldan/TinyStories", split=split)
        self.tokenizer = tokenizer if tokenizer else Tokenizer(max_length=max_length)
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        
        # Tokenize
        encodings = self.tokenizer.encode(text)
        input_ids = encodings['input_ids'].squeeze(0)
        attention_mask = encodings['attention_mask'].squeeze(0)
        
        # For Causal LM, labels are same as input_ids
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }

def create_dataloaders(
    batch_size: int = 32,
    max_length: int = 512,
    num_workers: int = 4
):
    tokenizer = Tokenizer(max_length=max_length)
    
    train_dataset = TinyStoriesDataset(split="train", tokenizer=tokenizer, max_length=max_length)
    val_dataset = TinyStoriesDataset(split="validation", tokenizer=tokenizer, max_length=max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, tokenizer
