import torch
import os
import sys
from src.utils.config import Config
from src.data.tokenizer import Tokenizer
from src.models.mamba import MambaLLM

def verify_remote():
    print("=== AETHER Phase 1 Remote Verification ===")
    
    # Paths
    config_path = "configs/phase1_tr.yaml"
    ckpt_path = "models/saved/aether_phase1.pt"
    tokenizer_path = "data/corpus_v1/tokenizer.model"
    
    # 1. File Existence Check
    print("\n[1/4] Checking Files...")
    files = [config_path, ckpt_path, tokenizer_path]
    missing = []
    for f in files:
        if os.path.exists(f):
            print(f"  ✅ Found: {f} ({os.path.getsize(f)/1024/1024:.2f} MB)")
        else:
            print(f"  ❌ MISSING: {f}")
            missing.append(f)
            
    if missing:
        print("❌ Critical files missing. Aborting.")
        return

    # 2. Config & Tokenizer Check
    print("\n[2/4] Checking Configuration...")
    try:
        cfg = Config.load(config_path)
        tokenizer = Tokenizer(tokenizer_path)
        
        print(f"  Config Vocab Size: {cfg.model.vocab_size}")
        print(f"  Tokenizer Vocab Size: {len(tokenizer)}")
        
        if cfg.model.vocab_size != len(tokenizer):
            print("  ⚠️ WARNING: Vocab size mismatch!")
        else:
            print("  ✅ Vocab sizes match.")
            
    except Exception as e:
        print(f"  ❌ Error loading config/tokenizer: {e}")
        return

    # 3. Model Loading
    print("\n[3/4] Loading Model...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")
        
        model = MambaLLM(cfg.model).to(device)
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        print("  ✅ Model loaded successfully.")
        print(f"  Parameter Count: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        return

    # 4. Inference Test
    print("\n[4/4] Running Inference Test...")
    try:
        model.eval()
        prompt = "Yapay zeka"
        input_ids = torch.tensor([tokenizer.encode(prompt)['input_ids']], device=device)
        
        with torch.no_grad():
            # Simple generation loop
            generated = input_ids
            for _ in range(20):
                outputs = model(generated)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                generated = torch.cat((generated, next_token), dim=1)
        
        decoded = tokenizer.decode(generated[0].tolist())
        print(f"  Prompt: {prompt}")
        print(f"  Output: {decoded}")
        print("  ✅ Inference successful.")
        
    except Exception as e:
        print(f"  ❌ Error during inference: {e}")

    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    verify_remote()
