## Developer: inkbytefo
## Modified: 2025-11-21

import os
import sys
import argparse
import numpy as np
import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm
import re
from typing import List

# Add project root to sys.path to allow imports from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.train_tokenizer_sp import train_sentencepiece_tokenizer

def clean_text(text: str) -> str:
    """
    Basic cleaning for Turkish text.
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_save(
    input_file: str,
    tokenizer_model: str,
    output_dir: str,
    val_split: float = 0.1
):
    """
    Tokenize the raw corpus and save as binary numpy files (uint16).
    """
    print(f"🚀 Tokenizing {input_file}...")
    
    if not os.path.exists(tokenizer_model):
        raise FileNotFoundError(f"Tokenizer model not found at {tokenizer_model}")
        
    sp = spm.SentencePieceProcessor(model_file=tokenizer_model)
    
    token_ids = []
    
    # Read and tokenize
    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Tokenizing"):
            line = line.strip()
            if not line: continue
            
            # Encode
            ids = sp.encode(line)
            token_ids.extend(ids)
            # Add EOS token (SentencePiece usually has EOS=3)
            token_ids.append(sp.eos_id())

    print(f"📊 Total tokens: {len(token_ids)}")
    
    # Convert to numpy uint16 (max vocab 65535)
    arr = np.array(token_ids, dtype=np.uint16)
    
    # Split into Train/Val
    val_size = int(len(arr) * val_split)
    train_size = len(arr) - val_size
    
    train_arr = arr[:train_size]
    val_arr = arr[train_size:]
    
    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")
    
    print(f"💾 Saving train.bin ({len(train_arr)} tokens) to {train_path}...")
    train_arr.tofile(train_path)
    
    print(f"💾 Saving val.bin ({len(val_arr)} tokens) to {val_path}...")
    val_arr.tofile(val_path)
    
    print("✅ Binary datasets prepared.")

def prepare_corpus(output_dir: str, vocab_size: int = 50257):
    """
    Download and prepare Turkish corpus (Wiki + Code + Math).
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_file = os.path.join(output_dir, "raw_corpus.txt")
    
    print("🚀 Loading Datasets...")
    
    # 1. Turkish Wikipedia (60%)
    print("   - Downloading Turkish Wikipedia...")
    wiki = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train", streaming=True)
    
    # 2. Python Code (30%)
    print("   - Downloading Python Code (codeparrot/codeparrot-clean)...")
    code = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    
    # 3. Math (GSM8K) (10%)
    print("   - Downloading GSM8K (Math)...")
    math = load_dataset("gsm8k", "main", split="train", streaming=True)

    print(f"💾 Writing raw corpus to {raw_file}...")
    
    # We aim for ~100k lines total for this phase to be fast but representative.
    # Ratios: 60k Wiki, 30k Code, 10k Math
    
    with open(raw_file, "w", encoding="utf-8") as f:
        
        print("   - Processing Wikipedia (Target: 60k)...")
        count = 0
        for sample in tqdm(wiki):
            text = clean_text(sample['text'])
            if len(text) > 50:
                f.write(text + "\n")
                count += 1
            if count >= 60000: break
            
        print("   - Processing Code (Target: 30k)...")
        count = 0
        for sample in tqdm(code):
            text = sample['content']
            # Minimal cleaning for code to preserve structure
            if len(text.strip()) > 0:
                f.write(text + "\n")
                count += 1
            if count >= 30000: break
            
        print("   - Processing Math (Target: 10k)...")
        count = 0
        for sample in tqdm(math):
            text = sample['question'] + "\n" + sample['answer']
            text = clean_text(text)
            f.write(text + "\n")
            count += 1
            if count >= 10000: break
            
    print("✅ Raw corpus prepared.")
    
    # Train Tokenizer
    model_prefix = os.path.join(output_dir, "tokenizer")
    
    # Check if tokenizer already exists to save time, or force retrain
    # For Phase 1, we force retrain to ensure settings are correct.
    
    train_sentencepiece_tokenizer(
        input_files=[raw_file],
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=0.9995
    )
    
    print(f"🎉 Tokenizer trained and saved to {model_prefix}.model")
    
    # Tokenize and Save Binary
    tokenize_and_save(raw_file, f"{model_prefix}.model", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/corpus_v1")
    parser.add_argument("--vocab_size", type=int, default=50257)
    args = parser.parse_args()
    
    prepare_corpus(args.output_dir, args.vocab_size)
