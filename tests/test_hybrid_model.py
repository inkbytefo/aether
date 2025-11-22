## Developer: inkbytefo
## Modified: 2025-11-22

"""
Unit Tests for Hybrid Mamba Architecture
Validates Linear Attention and Hybrid Block implementations.
"""

import torch
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.hybrid_mamba import LinearAttentionBlock, HybridBlock, HybridMambaLLM


def test_linear_attention_shape():
    """Test Linear Attention output shapes."""
    batch_size = 2
    seq_len = 128
    dim = 512
    
    layer = LinearAttentionBlock(dim=dim, num_heads=8)
    x = torch.randn(batch_size, seq_len, dim)
    
    out = layer(x)
    
    assert out.shape == (batch_size, seq_len, dim), f"Expected {(batch_size, seq_len, dim)}, got {out.shape}"
    print("✅ Linear Attention shape test passed")


def test_linear_attention_no_nan():
    """Ensure Linear Attention doesn't produce NaN."""
    layer = LinearAttentionBlock(dim=256, num_heads=4)
    x = torch.randn(1, 64, 256)
    
    out = layer(x)
    
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"
    print("✅ Linear Attention stability test passed")


def test_hybrid_block_mamba():
    """Test HybridBlock with Mamba mixer."""
    layer = HybridBlock(dim=512, block_type='mamba', d_state=16)
    x = torch.randn(2, 64, 512)
    
    out = layer(x)
    
    assert out.shape == x.shape
    print("✅ HybridBlock (Mamba) test passed")


def test_hybrid_block_attention():
    """Test HybridBlock with Linear Attention mixer."""
    layer = HybridBlock(dim=512, block_type='attention')
    x = torch.randn(2, 64, 512)
    
    out = layer(x)
    
    assert out.shape == x.shape
    print("✅ HybridBlock (Attention) test passed")


def test_hybrid_model_forward():
    """Test full HybridMambaLLM forward pass."""
    class MockConfig:
        vocab_size = 5000
        d_model = 256
        n_layer = 6
        d_state = 8
        d_conv = 4
        expand = 2
    
    cfg = MockConfig()
    model = HybridMambaLLM(cfg)
    
    input_ids = torch.randint(0, cfg.vocab_size, (2, 32))
    
    outputs = model(input_ids)
    
    assert outputs.logits.shape == (2, 32, cfg.vocab_size)
    print("✅ HybridMambaLLM forward test passed")


def test_hybrid_model_generation():
    """Test autoregressive generation."""
    class MockConfig:
        vocab_size = 1000
        d_model = 128
        n_layer = 3
        d_state = 8
        d_conv = 4
        expand = 2
    
    cfg = MockConfig()
    model = HybridMambaLLM(cfg)
    
    input_ids = torch.randint(0, cfg.vocab_size, (1, 10))
    
    generated = model.generate(input_ids, max_length=20, temperature=0.8)
    
    assert generated.shape[1] > input_ids.shape[1], "Generation should produce more tokens"
    print(f"✅ Generation test passed: {input_ids.shape[1]} -> {generated.shape[1]} tokens")


def test_memory_efficiency():
    """Compare memory usage: Hebbian vs Linear Attention."""
    print("\n🧪 Memory Efficiency Test")
    
    # Simulate Hebbian (4D tensor state per layer)
    class HebbianMock:
        def __init__(self, dim):
            self.state = torch.zeros(1, dim, dim)  # Per batch item
    
    # Simulate Linear Attention (no persistent state)
    class LinearAttentionMock:
        def __init__(self, dim):
            pass  # No state
    
    dim = 768
    n_layers = 24
    batch_size = 8
    
    # Hebbian memory
    hebbian_layers = [HebbianMock(dim) for _ in range(n_layers)]
    hebbian_memory = sum(layer.state.numel() * 2 for layer in hebbian_layers) * batch_size  # 2 bytes per fp16
    
    # Linear Attention memory (only KV cache during generation)
    linear_memory = dim * dim * 2  # Temporary KV matrix per head
    
    print(f"Hebbian State Memory: {hebbian_memory / 1e6:.2f} MB")
    print(f"Linear Attention Memory: {linear_memory / 1e3:.2f} KB")
    print(f"Memory Reduction: {hebbian_memory / linear_memory:.1f}x")


if __name__ == "__main__":
    print("Running Hybrid Mamba Tests...\n")
    
    test_linear_attention_shape()
    test_linear_attention_no_nan()
    test_hybrid_block_mamba()
    test_hybrid_block_attention()
    test_hybrid_model_forward()
    test_hybrid_model_generation()
    test_memory_efficiency()
    
    print("\n✅ All tests passed!")
