## Developer: inkbytefo
## Modified: 2025-11-21

# CUDA Device-Side Assert Fix

## Problem

```
RuntimeError: Triton Error [CUDA]: device-side assert triggered
```

**Root Cause:** Special tokens (PAD, BOS, EOS, UNK) were assigned token IDs **32000-32003**, which are **out of bounds** for a model with `vocab_size=32000` (valid range: 0-31999).

## Why This Happened

During tokenizer training with `BpeTrainer`, special tokens should have been placed at indices 0-3. However, HuggingFace's `add_special_tokens()` method was called AFTER training, appending them beyond the vocabulary boundary.

## Solution

### Option 1: Retrain Tokenizer (RECOMMENDED)

```bash
python retrain_tokenizer.py
```

This will:
- Train a new tokenizer with 32000 vocab size
- **Correctly** place special tokens at indices 0-3
- Validate the result
- Save to `data/phase1_tr/tokenizer.json`

After retraining, you'll need to retrain your model from scratch since the token mappings will change.

### Option 2: Temporary Workaround (CURRENT STATE)

The code now includes:

1. **Validation on load** ([`tokenizer.py:82-107`](file:///c:/Users/tpoyr/OneDrive/Desktop/AETHER/src/data/tokenizer.py#L82-L107))
   - Detects out-of-bounds special token IDs
   - Warns user to retrain

2. **Token ID clamping** ([`tokenizer.py:189-198`](file:///c:/Users/tpoyr/OneDrive/Desktop/AETHER/src/data/tokenizer.py#L189-L198))
   - Clamps any token ID >= vocab_size to valid range
   - Prevents CUDA errors but **corrupts input semantics**

3. **Compatibility check** ([`inference.py:118-129`](file:///c:/Users/tpoyr/OneDrive/Desktop/AETHER/inference.py#L118-L129))
   - Compares model vs tokenizer vocab_size
   - Exits early if mismatch detected

## Current Status

✅ **Inference runs without crashing**  
⚠️ **Output quality is poor** (clamping corrupts special tokens)  
🔧 **Action Required:** Retrain tokenizer for proper fix

## Diagnostic Tool

```bash
python diagnose_tokenizer.py
```

Shows:
- Actual vocab size
- Special token IDs
- Sample encoding test
- Validates token ID ranges

## Expected Output After Retraining

```
PAD token: '<pad>' -> ID 0
UNK token: '<unk>' -> ID 1
BOS token: '<s>' -> ID 2
EOS token: '</s>' -> ID 3
```

## Files Modified

1. [`src/data/tokenizer.py`](file:///c:/Users/tpoyr/OneDrive/Desktop/AETHER/src/data/tokenizer.py)
   - Added vocab_size sync in `_load_pretrained()`
   - Added validation in `_configure_special_tokens()`
   - Added token ID clamping in `encode()`
   - Fixed `__len__()` to return actual vocab size

2. [`inference.py`](file:///c:/Users/tpoyr/OneDrive/Desktop/AETHER/inference.py)
   - Added vocab_size compatibility check

3. [`retrain_tokenizer.py`](file:///c:/Users/tpoyr/OneDrive/Desktop/AETHER/retrain_tokenizer.py) [NEW]
   - Standalone script to retrain tokenizer correctly

4. [`diagnose_tokenizer.py`](file:///c:/Users/tpoyr/OneDrive/Desktop/AETHER/diagnose_tokenizer.py) [NEW]
   - Diagnostic utility for tokenizer issues
