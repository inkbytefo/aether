from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset
import torch
import json
import os
import glob
import numpy as np
from typing import Optional, List
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
        return self._process_text(text)

    def _process_text(self, text):
        encodings = self.tokenizer.encode(text)
        input_ids = encodings['input_ids'].squeeze(0)
        attention_mask = encodings['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }

class MixedDataset(Dataset):
    def __init__(self, data_paths: List[str], tokenizer: Optional[Tokenizer] = None, max_length: int = 512):
        self.tokenizer = tokenizer if tokenizer else Tokenizer(max_length=max_length)
        self.max_length = max_length
        self.samples = []
        
        for path in data_paths:
            if os.path.isdir(path):
                files = glob.glob(os.path.join(path, "*.jsonl"))
            else:
                files = [path]
                
            for f_path in files:
                if not os.path.exists(f_path):
                    print(f"Warning: File {f_path} not found.")
                    continue
                    
                with open(f_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if 'text' in data:
                                    self.samples.append(data['text'])
                            except json.JSONDecodeError:
                                pass
        
        print(f"Loaded {len(self.samples)} samples from {data_paths}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encodings = self.tokenizer.encode(text)
        input_ids = encodings['input_ids'].squeeze(0)
        attention_mask = encodings['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }

class BinaryDataset(Dataset):
    def __init__(self, path: str, max_length: int = 512):
        self.max_length = max_length
        # Load memory mapped file
        # We assume uint16 for tokens (vocab < 65535)
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.total_tokens = len(self.data)
        # Number of samples is total_tokens // max_length
        self.num_samples = self.total_tokens // self.max_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.max_length
        end = start + self.max_length
        
        # Read chunk
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        
        # Create inputs and targets (shifted by 1)
        # For causal LM: input is x[0:-1], label is x[1:]
        # But here we just return the chunk, model handles shifting or we do it here.
        # Standard practice: return input_ids and labels as same sequence, 
        # loss function handles shifting.
        
        return {
            'input_ids': chunk,
            'labels': chunk
        }

def create_dataloaders(
    config,
    tokenizer: Optional[Tokenizer] = None
):
    # Initialize tokenizer if not provided
    # Check if we have a custom tokenizer path in data config or default
    tokenizer_path = "data/phase1_tr/tokenizer.json"
    if tokenizer is None:
        if os.path.exists(tokenizer_path):
            tokenizer = Tokenizer(model_path=tokenizer_path, max_length=config.data.max_length)
        else:
            tokenizer = Tokenizer(max_length=config.data.max_length)
    
    if config.data.dataset_name == "roneneldan/TinyStories":
        train_dataset = TinyStoriesDataset(split="train", tokenizer=tokenizer, max_length=config.data.max_length)
        val_dataset = TinyStoriesDataset(split="validation", tokenizer=tokenizer, max_length=config.data.max_length)
    elif config.data.dataset_name == "phase1_tr":
        # Use BinaryDataset for our custom pre-processed data
        train_path = config.data.dataset_paths[0]
        val_path = config.data.dataset_paths[1]
        
        train_dataset = BinaryDataset(train_path, max_length=config.data.max_length)
        val_dataset = BinaryDataset(val_path, max_length=config.data.max_length)
    else:
        # Assume list of paths for MixedDataset
        # Config.data.dataset_name can be a list or string in yaml
        paths = config.data.dataset_paths if hasattr(config.data, 'dataset_paths') else []
        
        # Simple split for now: 90% train, 10% val
        full_dataset = MixedDataset(data_paths=paths, tokenizer=tokenizer, max_length=config.data.max_length)
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, tokenizer
