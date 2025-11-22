## Developer: inkbytefo
## Modified: 2025-11-22

"""
Optimized Phase 1 Data Preparation with Streaming + Memmap
Fixes RAM overflow by writing tokens to disk incrementally.
"""

import os
import sys
import sentencepiece as spm
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.data.tokenizer import Tokenizer


def prepare_phase1_tr_optimized():
    print("🚀 Starting Phase 1 Data Preparation (Optimized - Streaming)...")
    
    output_dir = "data/corpus_v1"
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer_model_prefix = os.path.join(output_dir, "tokenizer")
    tokenizer_path = tokenizer_model_prefix + ".model"
    train_bin_path = os.path.join(output_dir, "train.bin")
    val_bin_path = os.path.join(output_dir, "val.bin")
    
    # 1. Load Datasets (Streaming)
    print("📥 Loading Datasets (Streaming Mode)...")
    ds_en = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    try:
        print("Attempting to load 'wikimedia/wikipedia' (tr)...")
        ds_tr = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train", streaming=True)
    except Exception as e:
        print(f"⚠️ Wikipedia failed: {e}. Falling back to 'allenai/c4' (tr).")
        ds_tr = load_dataset("allenai/c4", "tr", split="train", streaming=True)
    
    # 2. Train Tokenizer (if needed)
    if not os.path.exists(tokenizer_path):
        print("Training Tokenizer...")
        temp_text_file = "temp_tokenizer_train.txt"
        with open(temp_text_file, "w", encoding="utf-8") as f:
            iter_en = iter(ds_en)
            iter_tr = iter(ds_tr)
            for _ in tqdm(range(25000), desc="Collecting tokenizer data"):
                try:
                    text_en = next(iter_en)['text']
                    text_tr = next(iter_tr)['text']
                    f.write(text_en + "\n")
                    f.write(text_tr + "\n")
                except StopIteration:
                    break
        
        spm.SentencePieceTrainer.train(
            input=temp_text_file,
            model_prefix=tokenizer_model_prefix,
            vocab_size=50257,
            model_type="unigram",
            character_coverage=0.9995,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3,
            pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>"
        )
        os.remove(temp_text_file)
        print(f"✅ Tokenizer trained and saved to {tokenizer_path}")
    else:
        print(f"Tokenizer found at {tokenizer_path}")
    
    tokenizer = Tokenizer(model_path=tokenizer_path)
    
    # 3. Tokenize with Memmap (no RAM overflow)
    print("Tokenizing and Saving Data (Memmap Streaming)...")
    
    max_tokens = 50_000_000   # 50M tokens target (T4 optimized - was 100M)
    chunk_size = 1_000_000    # Write 1M tokens at a time
    
    # Create temporary memmap file
    temp_memmap_path = os.path.join(output_dir, "temp_tokens.dat")
    
    # Estimate: we'll allocate max_tokens upfront, then trim later
    token_memmap = np.memmap(temp_memmap_path, dtype=np.uint16, mode='w+', shape=(max_tokens,))
    
    current_idx = 0
    iter_en = iter(ds_en)
    iter_tr = iter(ds_tr)
    
    pbar = tqdm(total=max_tokens, desc="Tokenizing (Memmap)")
    
    while current_idx < max_tokens:
        try:
            # Interleave 1 English, 1 Turkish
            text_en = next(iter_en)['text']
            text_tr = next(iter_tr)['text']
            
            # Encode
            enc_en = tokenizer.encode(text_en, return_tensors=False)
            ids_en = enc_en['input_ids']
            
            enc_tr = tokenizer.encode(text_tr, return_tensors=False)
            ids_tr = enc_tr['input_ids']
            
            # Flatten if batched
            if isinstance(ids_en[0], list): 
                ids_en = ids_en[0]
            if isinstance(ids_tr[0], list): 
                ids_tr = ids_tr[0]
            
            # Write to memmap
            n_en = len(ids_en)
            n_tr = len(ids_tr)
            total_new = n_en + n_tr
            
            if current_idx + total_new > max_tokens:
                # Reached limit
                remaining = max_tokens - current_idx
                combined = ids_en + ids_tr
                token_memmap[current_idx:current_idx + remaining] = combined[:remaining]
                current_idx = max_tokens
                break
            
            token_memmap[current_idx:current_idx + n_en] = ids_en
            current_idx += n_en
            token_memmap[current_idx:current_idx + n_tr] = ids_tr
            current_idx += n_tr
            
            pbar.update(total_new)
            
        except StopIteration:
            print("Dataset exhausted.")
            break
    
    pbar.close()
    
    # Trim memmap to actual size
    actual_tokens = current_idx
    print(f"Total tokens collected: {actual_tokens:,}")
    
    # Flush memmap
    token_memmap.flush()
    del token_memmap
    
    # Reload and trim
    all_tokens = np.memmap(temp_memmap_path, dtype=np.uint16, mode='r', shape=(actual_tokens,))
    
    # 4. Split Train/Val
    split_idx = int(actual_tokens * 0.9)
    
    print(f"Saving {split_idx:,} training tokens to {train_bin_path}...")
    train_data = np.array(all_tokens[:split_idx], dtype=np.uint16)
    train_data.tofile(train_bin_path)
    
    print(f"Saving {actual_tokens - split_idx:,} validation tokens to {val_bin_path}...")
    val_data = np.array(all_tokens[split_idx:], dtype=np.uint16)
    val_data.tofile(val_bin_path)
    
    # Cleanup
    os.remove(temp_memmap_path)
    
    print("✅ Data Preparation Complete! (Optimized)")


if __name__ == "__main__":
    prepare_phase1_tr_optimized()
