import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.plasticity import HebbianMemory

def test_hebbian_consistency():
    print("Testing HebbianMemory consistency...")
    
    dim = 64
    seq_len = 10
    batch_size = 1
    
    model = HebbianMemory(dim)
    model.eval()
    
    # Random input
    x = torch.randn(batch_size, seq_len, dim)
    
    # 1. Full Sequence Pass
    print("Running full sequence pass...")
    out_full, state_full = model(x)
    
    # 2. Step-by-Step Pass
    print("Running step-by-step pass...")
    state = None
    outputs_step = []
    
    for t in range(seq_len):
        x_t = x[:, t:t+1, :] # (B, 1, D)
        
        # We use inference_params to simulate the new flow, or just pass state explicitly
        # The new implementation supports passing state directly.
        out_t, state = model(x_t, state=state)
        outputs_step.append(out_t)
        
    out_step = torch.cat(outputs_step, dim=1)
    
    # 3. Compare
    print("\nComparing outputs...")
    diff_out = torch.abs(out_full - out_step).max().item()
    diff_state = torch.abs(state_full - state).max().item()
    
    print(f"Max Output Difference: {diff_out}")
    print(f"Max State Difference: {diff_state}")
    
    if diff_out < 1e-5 and diff_state < 1e-5:
        print("✅ TEST PASSED: Outputs and states match.")
    else:
        print("❌ TEST FAILED: Significant difference found.")

if __name__ == "__main__":
    test_hebbian_consistency()
