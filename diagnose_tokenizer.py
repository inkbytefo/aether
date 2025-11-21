## Developer: inkbytefo
## Modified: 2025-11-21

"""
Diagnostic script to check tokenizer and model compatibility.
"""

import torch
from src.utils.config import Config
from src.data.tokenizer import Tokenizer
from src.models.mamba import MambaLLM
import os

def main():
    print("=" * 60)
    print("AETHER Tokenizer/Model Diagnostic Tool")
    print("=" * 60)
    
    # Load Config
    config_path = "configs/phase1_tr.yaml"
    print(f"\n📋 Loading config: {config_path}")
    cfg = Config.load(config_path)
    print(f"   Model vocab_size: {cfg.model.vocab_size}")
    
    # Load Tokenizer
    tokenizer_path = "data/phase1_tr/tokenizer.json"
    print(f"\n🔤 Loading tokenizer: {tokenizer_path}")
    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer(model_path=tokenizer_path, max_length=cfg.data.max_length)
        print(f"   Tokenizer vocab_size: {len(tokenizer)}")
        print(f"   PAD token ID: {tokenizer.pad_token_id}")
        print(f"   BOS token ID: {tokenizer.bos_token_id}")
        print(f"   EOS token ID: {tokenizer.eos_token_id}")
        print(f"   UNK token ID: {tokenizer.unk_token_id}")
    else:
        print(f"   ❌ Tokenizer file not found!")
        return
    
    # Test encoding
    test_text = "selam"
    print(f"\n🧪 Testing encoding of: '{test_text}'")
    encodings = tokenizer.encode(test_text, return_tensors=True, add_special_tokens=True)
    input_ids = encodings['input_ids']
    print(f"   Shape: {input_ids.shape}")
    print(f"   Min ID: {input_ids.min().item()}")
    print(f"   Max ID: {input_ids.max().item()}")
    print(f"   First 20 token IDs: {input_ids[0][:20].tolist()}")
    
    # Check if any token IDs are out of bounds
    if input_ids.max().item() >= cfg.model.vocab_size:
        print(f"\n❌ PROBLEM DETECTED!")
        print(f"   Token IDs exceed model vocab_size ({cfg.model.vocab_size})")
        print(f"   Maximum token ID: {input_ids.max().item()}")
        print(f"   This will cause CUDA device-side assert errors!")
    else:
        print(f"\n✅ Token IDs are within valid range")
    
    # Model compatibility check
    print(f"\n🤖 Model Compatibility Check:")
    if cfg.model.vocab_size != len(tokenizer):
        print(f"   ❌ MISMATCH!")
        print(f"   Model expects: {cfg.model.vocab_size}")
        print(f"   Tokenizer has: {len(tokenizer)}")
        print(f"\n🔧 Fix:")
        print(f"   Option 1: Retrain tokenizer with vocab_size={cfg.model.vocab_size}")
        print(f"   Option 2: Update config to vocab_size={len(tokenizer)} and retrain model")
    else:
        print(f"   ✅ Vocab sizes match!")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
