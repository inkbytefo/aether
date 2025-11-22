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
        encodings = self.tokenizer.encode(text, return_tensors=True, add_special_tokens=True)
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
        encodings = self.tokenizer.encode(text, return_tensors=True, add_special_tokens=True)
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
    if tokenizer is None:
        tokenizer_path = getattr(config.data, "tokenizer_path", "data/tokenizer.model")
        if os.path.exists(tokenizer_path):
            print(f"Loading tokenizer from {tokenizer_path}")
            tokenizer = Tokenizer(model_path=tokenizer_path)
        else:
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    
    # Detect data format: Binary (.bin files) or Legacy (dataset_paths)
    if hasattr(config.data, 'train_path') and config.data.train_path:
        # New binary format (from prepare_phase1_optimized.py)
        print(f"Using binary datasets: {config.data.train_path}, {config.data.val_path}")
        train_dataset = BinaryDataset(config.data.train_path, max_length=config.data.seq_length)
        val_dataset = BinaryDataset(config.data.val_path, max_length=config.data.seq_length)
    
    elif config.data.dataset_name == "roneneldan/TinyStories":
        train_dataset = TinyStoriesDataset(split="train", tokenizer=tokenizer, max_length=config.data.max_length)
        val_dataset = TinyStoriesDataset(split="validation", tokenizer=tokenizer, max_length=config.data.max_length)
    
    elif hasattr(config.data, 'dataset_paths') and config.data.dataset_paths:
        # Legacy format (JSONL)
        paths = config.data.dataset_paths
        full_dataset = MixedDataset(data_paths=paths, tokenizer=tokenizer, max_length=config.data.max_length)
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    else:
        raise ValueError("No valid data source found in config. Need train_path+val_path or dataset_name or dataset_paths")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True
    )
    
    return train_loader, val_loader, tokenizer
