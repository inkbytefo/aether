## Developer: inkbytefo
## Modified: 2025-11-21

import os
import os
import sys

# Add project root to sys.path to allow imports from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
from datasets import load_dataset
from tqdm import tqdm
import re
from src.data.train_tokenizer_sp import train_sentencepiece_tokenizer

def clean_text(text: str) -> str:
    """
    Basic cleaning for Turkish text.
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_corpus(output_dir: str, vocab_size: int = 50257):
    """
    Download and prepare Turkish corpus (Wiki + OSCAR + Code).
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_file = os.path.join(output_dir, "raw_corpus.txt")
    
    print("🚀 Loading Datasets...")
    
    # 1. Turkish Wikipedia (High Quality)
    print("   - Downloading Turkish Wikipedia...")
    # streaming=True to avoid massive RAM usage, but for tokenizer we need a file.
    # We'll take a subset for tokenizer training if full dataset is too big.
    # "wikipedia" script is deprecated, using "wikimedia/wikipedia"
    wiki = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train", streaming=True)
    
    # 2. Python Code (Reasoning)
    print("   - Downloading Python Code (codeparrot/codeparrot-clean)...")
    # codeparrot-clean is ungated and high quality.
    code = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    
    # 3. Math (GSM8K)
    print("   - Downloading GSM8K (Math)...")
    math = load_dataset("gsm8k", "main", split="train", streaming=True)

    print(f"💾 Writing raw corpus to {raw_file}...")
    
    with open(raw_file, "w", encoding="utf-8") as f:
        # Mix ratios: 60% TR, 30% Code, 10% Math
        # We'll approximate by counts.
        
        # Turkish Wiki (Target: ~60k lines for sample or full)
        # Let's take 100k samples total for tokenizer training to be fast but representative.
        # Real training would use terabytes.
        
        print("   - Processing Wikipedia...")
        count = 0
        for sample in tqdm(wiki):
            text = clean_text(sample['text'])
            if len(text) > 50:
                f.write(text + "\n")
                count += 1
            if count >= 60000: break
            
        print("   - Processing Code...")
        count = 0
        for sample in tqdm(code):
            # Code needs to be kept as is, mostly.
            text = sample['content']
            f.write(text + "\n")
            count += 1
            if count >= 30000: break
            
        print("   - Processing Math...")
        count = 0
        for sample in tqdm(math):
            text = sample['question'] + "\n" + sample['answer']
            f.write(text + "\n")
            count += 1
            if count >= 10000: break
            
    print("✅ Raw corpus prepared.")
    
    # Train Tokenizer
    model_prefix = os.path.join(output_dir, "tokenizer")
    train_sentencepiece_tokenizer(
        input_files=[raw_file],
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram"
    )
    
    print(f"🎉 Tokenizer trained and saved to {model_prefix}.model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/corpus_v1")
    parser.add_argument("--vocab_size", type=int, default=50257)
    args = parser.parse_args()
    
    prepare_corpus(args.output_dir, args.vocab_size)
