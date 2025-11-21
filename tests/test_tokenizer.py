## Developer: inkbytefo
## Modified: 2025-11-21

"""
Tokenizer Unit Tests for AETHER Phase 1/2
Tests encoding, decoding, special tokens, and edge cases.
"""

import pytest
import torch
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.tokenizer import Tokenizer


class TestTokenizer:
    """Test suite for Tokenizer class."""
    
    def test_char_fallback_initialization(self):
        """Test that character-level fallback works when no model is provided."""
        tokenizer = Tokenizer(model_path=None, max_length=128)
        assert tokenizer._char_fallback is True
        assert tokenizer.vocab_size > 0
        print(f"✅ Char fallback vocab size: {tokenizer.vocab_size}")
    
    def test_special_tokens(self):
        """Test that special tokens are properly configured."""
        tokenizer = Tokenizer(max_length=128)
        
        assert tokenizer.pad_token_id is not None
        assert tokenizer.eos_token_id is not None
        assert tokenizer.bos_token_id is not None
        assert tokenizer.unk_token_id is not None
        
        # All should be different
        ids = {tokenizer.pad_token_id, tokenizer.eos_token_id, 
               tokenizer.bos_token_id, tokenizer.unk_token_id}
        assert len(ids) == 4, "Special tokens should have unique IDs"
        print(f"✅ Special tokens: PAD={tokenizer.pad_token_id}, EOS={tokenizer.eos_token_id}, BOS={tokenizer.bos_token_id}, UNK={tokenizer.unk_token_id}")
    
    def test_encode_single_text(self):
        """Test encoding a single text."""
        tokenizer = Tokenizer(max_length=128)
        text = "Merhaba dünya"
        
        encodings = tokenizer.encode(text, return_tensors=True)
        
        assert 'input_ids' in encodings
        assert 'attention_mask' in encodings
        assert encodings['input_ids'].shape[0] == 1  # batch size
        assert encodings['input_ids'].shape[1] == 128  # max_length
        assert encodings['attention_mask'].shape == encodings['input_ids'].shape
        print(f"✅ Single text encoded: {encodings['input_ids'].shape}")
    
    def test_encode_batch(self):
        """Test encoding multiple texts."""
        tokenizer = Tokenizer(max_length=128)
        texts = ["Merhaba dünya", "AETHER projesi", "Türkçe dil modeli"]
        
        encodings = tokenizer.encode(texts, return_tensors=True)
        
        assert encodings['input_ids'].shape[0] == 3  # batch size
        assert encodings['input_ids'].shape[1] == 128  # max_length
        print(f"✅ Batch encoded: {encodings['input_ids'].shape}")
    
    def test_encode_decode_roundtrip(self):
        """Test that encoding and decoding returns original text."""
        tokenizer = Tokenizer(max_length=128)
        original_text = "Bu bir test cümlesidir."
        
        # Encode
        encodings = tokenizer.encode(original_text, return_tensors=True, add_special_tokens=False)
        
        # Decode
        decoded_text = tokenizer.decode(encodings['input_ids'][0], skip_special_tokens=True)
        
        # Should be similar (might have whitespace differences in char mode)
        assert original_text.replace(" ", "") in decoded_text.replace(" ", "")
        print(f"✅ Roundtrip: '{original_text}' -> '{decoded_text}'")
    
    def test_turkish_characters(self):
        """Test Turkish-specific characters."""
        tokenizer = Tokenizer(max_length=128)
        text = "çÇğĞıİöÖşŞüÜ"  # Turkish special chars
        
        encodings = tokenizer.encode(text, return_tensors=True, add_special_tokens=False)
        decoded = tokenizer.decode(encodings['input_ids'][0], skip_special_tokens=True)
        
        # All Turkish chars should be preserved
        for char in text:
            assert char in decoded, f"Turkish char '{char}' lost in encoding"
        print(f"✅ Turkish characters preserved: {text}")
    
    def test_max_length_truncation(self):
        """Test that texts are truncated to max_length."""
        max_len = 32
        tokenizer = Tokenizer(max_length=max_len)
        long_text = "a" * 1000  # Very long text
        
        encodings = tokenizer.encode(long_text, return_tensors=True)
        
        assert encodings['input_ids'].shape[1] == max_len
        print(f"✅ Truncation works: {long_text[:50]}... -> {max_len} tokens")
    
    def test_padding(self):
        """Test that short texts are padded."""
        tokenizer = Tokenizer(max_length=128)
        short_text = "hi"
        
        encodings = tokenizer.encode(short_text, return_tensors=True)
        
        # Should have padding tokens
        assert (encodings['input_ids'] == tokenizer.pad_token_id).sum() > 0
        
        # Attention mask should be 0 for padding
        assert (encodings['attention_mask'] == 0).sum() > 0
        print(f"✅ Padding applied: {(encodings['input_ids'] == tokenizer.pad_token_id).sum()} pad tokens")
    
    def test_empty_text(self):
        """Test encoding empty text."""
        tokenizer = Tokenizer(max_length=128)
        
        encodings = tokenizer.encode("", return_tensors=True, add_special_tokens=True)
        
        # Should only contain special tokens and padding
        assert encodings['input_ids'].shape[1] == 128
        print("✅ Empty text handled")
    
    def test_special_tokens_in_decode(self):
        """Test that special tokens can be skipped in decoding."""
        tokenizer = Tokenizer(max_length=128)
        text = "test"
        
        encodings = tokenizer.encode(text, return_tensors=True, add_special_tokens=True)
        
        # Decode with special tokens
        decoded_with = tokenizer.decode(encodings['input_ids'][0], skip_special_tokens=False)
        
        # Decode without special tokens
        decoded_without = tokenizer.decode(encodings['input_ids'][0], skip_special_tokens=True)
        
        # Without special tokens should be shorter or equal
        assert len(decoded_without) <= len(decoded_with)
        print(f"✅ Special token skipping: WITH='{decoded_with[:50]}' WITHOUT='{decoded_without[:50]}'")


def test_phase1_config_compatibility():
    """Test that tokenizer works with Phase 1 configuration."""
    # This matches phase1_tr.yaml config
    tokenizer = Tokenizer(max_length=512, vocab_size=32000)
    
    assert tokenizer.max_length == 512
    assert tokenizer.vocab_size > 0
    print(f"✅ Phase 1 config compatible: max_length=512, vocab_size={tokenizer.vocab_size}")


def test_dataset_integration():
    """Test tokenizer integration with dataset expectations."""
    tokenizer = Tokenizer(max_length=512)
    
    # Simulate dataset usage
    sample_text = "AETHER projesi Türkçe dil modeli geliştiriyor."
    
    encodings = tokenizer.encode(sample_text, return_tensors=True)
    
    # Dataset expects these keys
    assert 'input_ids' in encodings
    assert 'attention_mask' in encodings
    
    # Should be torch tensors
    assert isinstance(encodings['input_ids'], torch.Tensor)
    assert isinstance(encodings['attention_mask'], torch.Tensor)
    
    # Should be long dtype for embedding
    assert encodings['input_ids'].dtype == torch.long
    
    print("✅ Dataset integration verified")


if __name__ == "__main__":
    print("=" * 60)
    print("AETHER Tokenizer Test Suite - Phase 1/2 Compatibility")
    print("=" * 60)
    
    # Run tests manually
    test_suite = TestTokenizer()
    
    try:
        test_suite.test_char_fallback_initialization()
        test_suite.test_special_tokens()
        test_suite.test_encode_single_text()
        test_suite.test_encode_batch()
        test_suite.test_encode_decode_roundtrip()
        test_suite.test_turkish_characters()
        test_suite.test_max_length_truncation()
        test_suite.test_padding()
        test_suite.test_empty_text()
        test_suite.test_special_tokens_in_decode()
        
        test_phase1_config_compatibility()
        test_dataset_integration()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Tokenizer is Phase 2 ready!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
