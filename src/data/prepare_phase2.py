import os
import sys
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.tokenizer import Tokenizer

def prepare_phase2():
    print("🚀 Starting Phase 2 Data Preparation (Mixed Curriculum)...")
    
    output_dir = "data/corpus_v2"
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer_path = "data/corpus_v1/tokenizer.model"
    train_bin_path = os.path.join(output_dir, "train.bin")
    val_bin_path = os.path.join(output_dir, "val.bin")
    
    if not os.path.exists(tokenizer_path):
        print(f"❌ Tokenizer not found at {tokenizer_path}. Run Phase 1 prep first.")
        return

    tokenizer = Tokenizer(model_path=tokenizer_path)
    print(f"✅ Tokenizer loaded (Vocab: {len(tokenizer)})")

    # Dataset Config
    # Ratios: TR (40%), Stories (20%), Code (25%), Math (15%)
    TOTAL_TOKENS = 200 * 1024 * 1024 # 200M tokens for Phase 2 start
    
    datasets_cfg = [
        {"name": "wikimedia/wikipedia", "subset": "20231101.tr", "split": "train", "ratio": 0.40, "key": "text"},
        {"name": "roneneldan/TinyStories", "subset": None, "split": "train", "ratio": 0.20, "key": "text"},
        {"name": "flytech/python-codes-25k", "subset": None, "split": "train", "ratio": 0.25, "key": "text"}, # Fallback code dataset
        {"name": "gsm8k", "subset": "main", "split": "train", "ratio": 0.15, "key": "question"} # Math questions
    ]
    
    # Load Iterators
    iterators = []
    print("📥 Loading Datasets...")
    for cfg in datasets_cfg:
        try:
            ds = load_dataset(cfg["name"], cfg["subset"], split=cfg["split"], streaming=True)
            iterators.append({
                "iter": iter(ds),
                "ratio": cfg["ratio"],
                "key": cfg["key"],
                "name": cfg["name"]
            })
            print(f"  ✅ Loaded {cfg['name']}")
        except Exception as e:
            print(f"  ⚠️ Failed to load {cfg['name']}: {e}")
            # Adjust ratios if one fails? For now just skip
            
    if not iterators:
        print("❌ No datasets loaded!")
        return

    # Tokenization Loop
    all_tokens = []
    pbar = tqdm(total=TOTAL_TOKENS, desc="Mixing & Tokenizing")
    
    while len(all_tokens) < TOTAL_TOKENS:
        # Probabilistic sampling based on ratios
        ratios = [it["ratio"] for it in iterators]
        # Normalize ratios in case some failed
        total_ratio = sum(ratios)
        probs = [r / total_ratio for r in ratios]
        
        # Pick a dataset
        chosen_idx = np.random.choice(len(iterators), p=probs)
        chosen = iterators[chosen_idx]
        
        try:
            item = next(chosen["iter"])
            text = item[chosen["key"]]
            
            # For GSM8K, append answer if available
            if chosen["name"] == "gsm8k":
                text += "\nAnswer: " + item["answer"]
            
            # Encode
            enc = tokenizer.encode(text, return_tensors=False)
            ids = enc['input_ids']
            if isinstance(ids[0], list): ids = ids[0] # Flatten
            
            all_tokens.extend(ids)
            pbar.update(len(ids))
            
        except StopIteration:
            # Restart iterator? Or just remove?
            # For streaming, StopIteration means end of stream. 
            # Let's try to reload or just ignore (dataset exhausted)
            print(f"  ⚠️ Dataset {chosen['name']} exhausted.")
            del iterators[chosen_idx]
            if not iterators:
                break
            
    pbar.close()
    
    # Save
    n = len(all_tokens)
    split = int(n * 0.95) # 95% train, 5% val for Phase 2
    
    print(f"Saving {split} training tokens...")
    train_data = np.array(all_tokens[:split], dtype=np.uint16)
    train_data.tofile(train_bin_path)
    
    print(f"Saving {n - split} validation tokens...")
    val_data = np.array(all_tokens[split:], dtype=np.uint16)
    val_data.tofile(val_bin_path)
    
    print(f"✅ Phase 2 Data Ready: {output_dir}")

if __name__ == "__main__":
    prepare_phase2()
