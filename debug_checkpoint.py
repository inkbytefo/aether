import torch
import os
import sys
from src.utils.config import Config, ModelConfig
from src.models.mamba import MambaLLM
from src.models.plastic_mamba import PlasticMambaLLM

def check_checkpoint(config_path, checkpoint_path):
    print(f"--- Debugging Checkpoint: {checkpoint_path} ---")
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    # Load Config
    try:
        cfg = Config.load(config_path)
        print("Config loaded successfully.")
        print(f"Model Config: d_model={cfg.model.d_model}, n_layer={cfg.model.n_layer}, vocab_size={cfg.model.vocab_size}")
    except Exception as e:
        print(f"Failed to load config: {e}")
        return

    # Load Checkpoint
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        print(f"Checkpoint loaded. Keys: {len(state_dict.keys())}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    # Check for NaNs or Infs
    print("\nChecking for NaNs/Infs in weights...")
    has_nan = False
    for k, v in state_dict.items():
        if torch.isnan(v).any():
            print(f"❌ NaN found in {k}")
            has_nan = True
        if torch.isinf(v).any():
            print(f"❌ Inf found in {k}")
            has_nan = True
    
    if not has_nan:
        print("✅ No NaNs or Infs found in weights.")

    # Initialize Model to check compatibility
    print("\nInitializing Model...")
    try:
        if cfg.model.use_plasticity:
            print("Type: PlasticMambaLLM")
            model = PlasticMambaLLM(cfg.model)
        else:
            print("Type: MambaLLM")
            model = MambaLLM(cfg.model)
        
        print("Model initialized.")
        
        # Strict load
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"⚠️ Missing keys: {missing[:5]} ... ({len(missing)} total)")
        if unexpected:
            print(f"⚠️ Unexpected keys: {unexpected[:5]} ... ({len(unexpected)} total)")
        
        if not missing and not unexpected:
            print("✅ State dict matches model architecture perfectly.")
            
    except Exception as e:
        print(f"Failed to initialize model: {e}")

if __name__ == "__main__":
    # Default paths based on user request
    config_path = "configs/phase4.yaml"
    checkpoint_path = "models/saved/aether_phase4.pt"
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    if len(sys.argv) > 2:
        checkpoint_path = sys.argv[2]
        
    check_checkpoint(config_path, checkpoint_path)
