import os
import sys
import sentencepiece as spm
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Add project root to sys.path to allow importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.tokenizer import Tokenizer

def prepare_phase1_tr():
    print("🚀 Starting Phase 1 Data Preparation (Turkish + English)...")
    
    # Paths matching turkish_base.yaml
    output_dir = "data/corpus_v1"
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer_model_prefix = os.path.join(output_dir, "tokenizer")
    tokenizer_path = tokenizer_model_prefix + ".model"
    train_bin_path = os.path.join(output_dir, "train.bin")
    val_bin_path = os.path.join(output_dir, "val.bin")
    
    # 1. Load Datasets
    print("📥 Loading Datasets...")
    # TinyStories for English foundation (simple narrative)
    ds_en = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    # OSCAR or Wikipedia for Turkish foundation
    try:
        print("Attempting to load 'wikimedia/wikipedia' (tr)...")
        ds_tr = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train", streaming=True)
    except Exception as e:
        print(f"⚠️ Wikipedia failed: {e}. Falling back to 'allenai/c4' (tr).")
        ds_tr = load_dataset("allenai/c4", "tr", split="train", streaming=True)

    # 2. Train Tokenizer if needed
    if not os.path.exists(tokenizer_path):
        print("Training Tokenizer...")
        # Dump 50k samples to text file for training
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
        
        # Train SentencePiece
        # vocab_size=50257 to match GPT-2 / config
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

    # Load Tokenizer
    tokenizer = Tokenizer(model_path=tokenizer_path)

    # 3. Tokenize and Save
    print("Tokenizing and Saving Data...")
    
    # We will create a binary file using numpy memmap
    # Let's process ~100MB of data for this phase to start quickly
    max_tokens = 100 * 1024 * 1024 # 100M tokens
    
    all_tokens = []
    
    iter_en = iter(ds_en)
    iter_tr = iter(ds_tr)
    
    pbar = tqdm(total=max_tokens, desc="Tokenizing data")
    
    while len(all_tokens) < max_tokens:
        try:
            # Interleave 1 English, 1 Turkish to balance
            text_en = next(iter_en)['text']
            text_tr = next(iter_tr)['text']
            
            # Encode
            # Tokenizer.encode returns dict {'input_ids': ...}
            enc_en = tokenizer.encode(text_en, return_tensors=False)
            ids_en = enc_en['input_ids']
            
            enc_tr = tokenizer.encode(text_tr, return_tensors=False)
            ids_tr = enc_tr['input_ids']
            
            # Flatten if needed (batch size 1)
            if isinstance(ids_en[0], list): ids_en = ids_en[0]
            if isinstance(ids_tr[0], list): ids_tr = ids_tr[0]

            all_tokens.extend(ids_en)
            all_tokens.extend(ids_tr)
            
            pbar.update(len(ids_en) + len(ids_tr))
            
        except StopIteration:
            break
            
    pbar.close()
    
    # Split Train/Val
    n = len(all_tokens)
    split = int(n * 0.9)
    
    print(f"Saving {split} training tokens...")
    train_data = np.array(all_tokens[:split], dtype=np.uint16)
    train_data.tofile(train_bin_path)
    
    print(f"Saving {n - split} validation tokens...")
    val_data = np.array(all_tokens[split:], dtype=np.uint16)
    val_data.tofile(val_bin_path)
    
    print("✅ Data Preparation Complete!")

if __name__ == "__main__":
    prepare_phase1_tr()
