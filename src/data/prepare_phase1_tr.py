import os
import sys
import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Add project root to sys.path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.tokenizer import Tokenizer

def prepare_phase1_tr():
    print("🚀 Starting Phase 1 Data Preparation (Turkish + English)...")
    
    # Paths
    os.makedirs("data/phase1_tr", exist_ok=True)
    tokenizer_path = "data/phase1_tr/tokenizer.json"
    train_bin_path = "data/phase1_tr/train.bin"
    val_bin_path = "data/phase1_tr/val.bin"
    
    # 1. Load Datasets
    print("📥 Loading Datasets...")
    # TinyStories for English foundation (simple narrative)
    ds_en = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    # OSCAR for Turkish foundation (web text)
    # Using streaming to avoid massive download
    try:
        ds_tr = load_dataset("oscar", "unshuffled_deduplicated_tr", split="train", streaming=True)
    except:
        print("⚠️ OSCAR not found or requires login. Falling back to 'wikipedia' (tr).")
        ds_tr = load_dataset("wikipedia", "20220301.tr", split="train", streaming=True)

    # 2. Train Tokenizer
    if not os.path.exists(tokenizer_path):
        print("Training Tokenizer...")
        # Create a generator to feed the tokenizer trainer
        def batch_iterator(batch_size=10000):
            batch = []
            # Mix EN and TR for tokenizer training
            iter_en = iter(ds_en)
            iter_tr = iter(ds_tr)
            
            for _ in range(batch_size):
                try:
                    text_en = next(iter_en)['text']
                    text_tr = next(iter_tr)['text']
                    batch.append(text_en)
                    batch.append(text_tr)
                except StopIteration:
                    break
            return batch

        # Train on a subset (e.g., 20k samples)
        training_corpus = batch_iterator(20000)
        
        tokenizer = Tokenizer()
        tokenizer.train(training_corpus, vocab_size=32000, save_path=tokenizer_path)
    else:
        print(f"Tokenizer found at {tokenizer_path}")
        tokenizer = Tokenizer(model_path=tokenizer_path)

    # 3. Tokenize and Save
    print("Tokenizing and Saving Data...")
    
    # We will create a binary file using numpy memmap
    # Let's process ~100MB of data for this phase to start quickly
    max_tokens = 100 * 1024 * 1024 # 100M tokens
    
    all_tokens = []
    
    iter_en = iter(ds_en)
    iter_tr = iter(ds_tr)
    
    pbar = tqdm(total=max_tokens)
    
    while len(all_tokens) < max_tokens:
        try:
            # Interleave 1 English, 1 Turkish to balance
            text_en = next(iter_en)['text']
            text_tr = next(iter_tr)['text']
            
            tokens_en = tokenizer.encode(text_en).squeeze().tolist()
            tokens_tr = tokenizer.encode(text_tr).squeeze().tolist()
            
            # Add EOS token
            tokens_en.append(tokenizer.eos_token_id)
            tokens_tr.append(tokenizer.eos_token_id)
            
            all_tokens.extend(tokens_en)
            all_tokens.extend(tokens_tr)
            
            pbar.update(len(tokens_en) + len(tokens_tr))
            
        except StopIteration:
            break
            
    pbar.close()
    
    # Split Train/Val
    n = len(all_tokens)
    train_data = all_tokens[:int(n*0.9)]
    val_data = all_tokens[int(n*0.9):]
    
    # Save to bin
    print(f"Saving {len(train_data)} training tokens...")
    train_ids = np.array(train_data, dtype=np.uint16)
    train_ids.tofile(train_bin_path)
    
    print(f"Saving {len(val_data)} validation tokens...")
    val_ids = np.array(val_data, dtype=np.uint16)
    val_ids.tofile(val_bin_path)
    
    print("✅ Data Preparation Complete!")

if __name__ == "__main__":
    prepare_phase1_tr()
