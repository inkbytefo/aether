import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def verify_phase1():
    print("🔍 Verifying Phase 1 Components...")
    
    # 1. Test Config
    try:
        from src.utils.config import Config
        cfg = Config.load("configs/config.yaml")
        print("✅ Config loaded successfully.")
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return

    # 2. Test Tokenizer
    try:
        from src.data.tokenizer import Tokenizer
        tokenizer = Tokenizer()
        text = "Hello AETHER"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens['input_ids'][0])
        print(f"✅ Tokenizer works. '{text}' -> {tokens['input_ids'].shape} -> '{decoded}'")
    except Exception as e:
        print(f"❌ Tokenizer failed: {e}")

    # 3. Test Dataset (Mock)
    try:
        from src.data.dataset import TinyStoriesDataset
        # We won't load the full dataset here to save time/bandwidth if not present
        # Just checking import
        print("✅ Dataset module imported.")
    except Exception as e:
        print(f"❌ Dataset module failed: {e}")

    # 4. Test Model Architecture
    try:
        from src.models.mamba import MambaLLM
        model = MambaLLM(cfg.model)
        print(f"✅ MambaLLM initialized. Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    except ImportError:
        print("⚠️  Mamba-SSM not installed (Expected on local Windows). Skipping model init check.")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")

    print("\nPhase 1 Verification Complete (Local Check).")
    print("Run 'python train.py' on Cloud Cluster with CUDA to start training.")

if __name__ == "__main__":
    verify_phase1()
