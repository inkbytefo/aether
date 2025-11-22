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
    
    if not iterators:
        print("❌ No datasets loaded!")
        return

    # Streaming setup
    train_file = open(train_bin_path, "wb")
    val_file = open(val_bin_path, "wb")
    
    split_idx = int(TOTAL_TOKENS * 0.95)
    current_tokens = 0
    buffer = []
    BUFFER_SIZE = 1000000 # 1M tokens buffer
    
    pbar = tqdm(total=TOTAL_TOKENS, desc="Mixing & Streaming")
    
    while current_tokens < TOTAL_TOKENS:
        # Probabilistic sampling
        ratios = [it["ratio"] for it in iterators]
        total_ratio = sum(ratios)
        probs = [r / total_ratio for r in ratios]
        
        chosen_idx = np.random.choice(len(iterators), p=probs)
        chosen = iterators[chosen_idx]
        
        try:
            item = next(chosen["iter"])
            text = item[chosen["key"]]
            
            if chosen["name"] == "gsm8k":
                text += "\nAnswer: " + item["answer"]
            
            enc = tokenizer.encode(text, return_tensors=False)
            ids = enc['input_ids']
            if isinstance(ids[0], list): ids = ids[0]
            
            buffer.extend(ids)
            
            # Flush buffer if full
            if len(buffer) >= BUFFER_SIZE:
                chunk = buffer[:BUFFER_SIZE]
                buffer = buffer[BUFFER_SIZE:]
                
                chunk_arr = np.array(chunk, dtype=np.uint16)
                
                if current_tokens < split_idx:
                    space_in_train = split_idx - current_tokens
                    if len(chunk) <= space_in_train:
                        train_file.write(chunk_arr.tobytes())
                    else:
                        train_part = chunk_arr[:space_in_train]
                        val_part = chunk_arr[space_in_train:]
                        train_file.write(train_part.tobytes())
                        val_file.write(val_part.tobytes())
                else:
                    val_file.write(chunk_arr.tobytes())
                
                current_tokens += len(chunk)
                pbar.update(len(chunk))
                
        except StopIteration:
            del iterators[chosen_idx]
            if not iterators:
                break
    
    # Flush remaining buffer
    if buffer:
        chunk_arr = np.array(buffer, dtype=np.uint16)
        if current_tokens < split_idx:
            space_in_train = split_idx - current_tokens
            if len(buffer) <= space_in_train:
                train_file.write(chunk_arr.tobytes())
            else:
                train_part = chunk_arr[:space_in_train]
                val_part = chunk_arr[space_in_train:]
                train_file.write(train_part.tobytes())
                val_file.write(val_part.tobytes())
        else:
            val_file.write(chunk_arr.tobytes())
        pbar.update(len(buffer))

    pbar.close()
    train_file.close()
    val_file.close()
    
    print(f"✅ Phase 2 Data Ready: {output_dir}")

if __name__ == "__main__":
    prepare_phase2()
