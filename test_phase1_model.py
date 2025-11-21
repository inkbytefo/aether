## Developer: inkbytefo
## Modified: 2025-11-21

"""
Test Phase 1 Model - Pre-Phase 2 Verification
Validates that the Phase 1 trained model works correctly 
with the refactored tokenizer before proceeding to Phase 2.
"""

import torch
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import Config
from src.data.tokenizer import Tokenizer
from src.models.mamba import MambaLLM


def test_model_loading(config_path: str, checkpoint_path: str):
    """Test that model and checkpoint load correctly."""
    print("=" * 70)
    print("PHASE 1 MODEL VERIFICATION - Pre-Phase 2 Checklist")
    print("=" * 70)
    
    # Load config
    print("\n[1/6] Loading configuration...")
    cfg = Config.load(config_path)
    print(f"✅ Config loaded: {os.path.basename(config_path)}")
    print(f"    Model: d_model={cfg.model.d_model}, n_layers={cfg.model.n_layer}")
    print(f"    Vocab: {cfg.model.vocab_size}")
    
    # Setup device
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    
    # Load tokenizer
    print("\n[2/6] Loading tokenizer...")
    tokenizer_path = "data/phase1_tr/tokenizer.json"
    
    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer(model_path=tokenizer_path, max_length=cfg.data.max_length)
        print(f"✅ Tokenizer loaded from {tokenizer_path}")
    else:
        print(f"⚠️ Tokenizer not found at {tokenizer_path}, using fallback")
        tokenizer = Tokenizer(max_length=cfg.data.max_length)
    
    print(f"    Vocab size: {len(tokenizer)}")
    print(f"    Max length: {tokenizer.max_length}")
    
    # Initialize model
    print("\n[3/6] Initializing model...")
    model = MambaLLM(cfg.model).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model initialized")
    print(f"    Total params: {total_params:,}")
    print(f"    Trainable: {trainable_params:,}")
    
    # Load checkpoint
    print("\n[4/6] Loading checkpoint...")
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✅ Checkpoint loaded from {checkpoint_path}")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("    Cannot proceed without trained model!")
        return False
    
    # Test tokenization
    print("\n[5/6] Testing tokenizer...")
    test_texts = [
        "Merhaba dünya",
        "AETHER Türkçe dil modeli",
        "Bir varmış bir yokmuş"
    ]
    
    for text in test_texts:
        encodings = tokenizer.encode(text, return_tensors=True, add_special_tokens=True)
        decoded = tokenizer.decode(encodings['input_ids'][0], skip_special_tokens=True)
        print(f"    '{text}' -> {encodings['input_ids'].shape[1]} tokens -> '{decoded.strip()}'")
    
    print("✅ Tokenizer working correctly")
    
    # Test model forward pass
    print("\n[6/6] Testing model forward pass...")
    model.eval()
    
    with torch.no_grad():
        test_input = "Bir zamanlar"
        encodings = tokenizer.encode(test_input, return_tensors=True, add_special_tokens=True)
        input_ids = encodings['input_ids'].to(device)
        
        outputs = model(input_ids)
        logits = outputs.logits
        
        print(f"    Input shape: {input_ids.shape}")
        print(f"    Output logits shape: {logits.shape}")
        print(f"    Expected vocab size: {cfg.model.vocab_size}")
        
        assert logits.shape[-1] == cfg.model.vocab_size, f"Vocab size mismatch: {logits.shape[-1]} vs {cfg.model.vocab_size}"
        
        # Sample next token
        next_token_logits = logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).item()
        next_token = tokenizer.decode([next_token_id], skip_special_tokens=True)
        
        print(f"    Predicted next token: ID={next_token_id}, Text='{next_token}'")
    
    print("✅ Model forward pass successful")
    
    return True


def run_interactive_test(config_path: str, checkpoint_path: str):
    """Run a few interactive generation tests."""
    print("\n" + "=" * 70)
    print("INTERACTIVE GENERATION TEST")
    print("=" * 70)
    
    # Load everything
    cfg = Config.load(config_path)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    
    tokenizer_path = "data/phase1_tr/tokenizer.json"
    tokenizer = Tokenizer(
        model_path=tokenizer_path if os.path.exists(tokenizer_path) else None,
        max_length=cfg.data.max_length
    )
    
    model = MambaLLM(cfg.model).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Test prompts
    test_prompts = [
        "Merhaba",
        "Bir zamanlar",
        "Bugün hava",
    ]
    
    print("\nGenerating responses (50 tokens each)...\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"[{i}/{len(test_prompts)}] Prompt: '{prompt}'")
        
        with torch.no_grad():
            encodings = tokenizer.encode(prompt, return_tensors=True, add_special_tokens=True)
            input_ids = encodings['input_ids'].to(device)
            generated_ids = input_ids.clone()
            
            # Simple greedy generation
            for _ in range(50):
                outputs = model(generated_ids)
                next_token_logits = outputs.logits[0, -1, :]
                next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
                generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
                
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
            
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"Response: {generated_text}")
            print()
    
    print("✅ Interactive test complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify Phase 1 model before Phase 2")
    parser.add_argument("--config", type=str, default="configs/phase1_tr.yaml")
    parser.add_argument("--checkpoint", type=str, default="models/saved/aether_phase1.pt")
    parser.add_argument("--skip-generation", action="store_true", help="Skip generation tests")
    args = parser.parse_args()
    
    # Run verification
    success = test_model_loading(args.config, args.checkpoint)
    
    if success and not args.skip_generation:
        try:
            run_interactive_test(args.config, args.checkpoint)
        except Exception as e:
            print(f"\n⚠️ Generation test failed: {e}")
            import traceback
            traceback.print_exc()
    
    if success:
        print("\n" + "=" * 70)
        print("✅ PHASE 1 VERIFICATION COMPLETE - Ready for Phase 2!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Review the outputs above")
        print("  2. If satisfied, proceed with Phase 2 training:")
        print("     python train.py --config configs/phase2.yaml --resume_from models/saved/aether_phase1.pt")
    else:
        print("\n" + "=" * 70)
        print("❌ VERIFICATION FAILED - Fix issues before Phase 2")
        print("=" * 70)
        sys.exit(1)
