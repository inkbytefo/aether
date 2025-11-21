## Developer: inkbytefo
## Modified: 2025-11-21

"""
Retrain tokenizer with CORRECT special token placement.
This fixes the CUDA device-side assert error caused by out-of-bounds token IDs.
"""

import os
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from tqdm import tqdm

def main():
    print("=" * 70)
    print("AETHER Tokenizer Retraining - Special Token Fix")
    print("=" * 70)
    
    # Configuration
    VOCAB_SIZE = 32000
    OUTPUT_PATH = "data/phase1_tr/tokenizer.json"
    SAMPLE_SIZE = 50000  # Number of samples to train on
    
    # Special tokens (will be assigned to IDs 0-3)
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    
    print(f"\n📋 Configuration:")
    print(f"   Vocab Size: {VOCAB_SIZE}")
    print(f"   Output: {OUTPUT_PATH}")
    print(f"   Training Samples: {SAMPLE_SIZE}")
    
    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Initialize BPE tokenizer
    print(f"\n🔧 Initializing BPE tokenizer...")
    tokenizer = HFTokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()
    
    # Configure trainer with special tokens FIRST
    # This ensures they get IDs 0, 1, 2, 3
    print(f"   Special tokens will be placed at indices 0-3")
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=[PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN],
        show_progress=True,
        min_frequency=2,
        initial_alphabet=ByteLevel.alphabet()
    )
    
    # Load training data
    print(f"\n📥 Loading training data...")
    print(f"   Source 1: TinyStories (English)")
    print(f"   Source 2: Wikipedia Turkish")
    
    try:
        ds_en = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        ds_tr = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train", streaming=True)
    except Exception as e:
        print(f"   ⚠️ Error loading datasets: {e}")
        print(f"   Using fallback datasets...")
        ds_en = load_dataset("roneneldan/TinyStories", split="train[:5000]")
        ds_tr = load_dataset("sil-ai/turkish-proverbs", split="train")
    
    # Collect training texts
    print(f"\n📝 Collecting {SAMPLE_SIZE} training samples...")
    training_texts = []
    
    if hasattr(ds_en, '__iter__'):
        iter_en = iter(ds_en)
        iter_tr = iter(ds_tr)
        
        pbar = tqdm(total=SAMPLE_SIZE)
        for i in range(SAMPLE_SIZE // 2):
            try:
                # Get English sample
                sample_en = next(iter_en)
                text_en = sample_en.get('text', '') or sample_en.get('story', '')
                if text_en:
                    training_texts.append(text_en)
                    pbar.update(1)
                
                # Get Turkish sample
                sample_tr = next(iter_tr)
                text_tr = sample_tr.get('text', '') or sample_tr.get('proverb', '')
                if text_tr:
                    training_texts.append(text_tr)
                    pbar.update(1)
            except StopIteration:
                break
        pbar.close()
    
    print(f"   Collected {len(training_texts)} texts")
    
    # Train tokenizer
    print(f"\n🚀 Training tokenizer...")
    tokenizer.train_from_iterator(training_texts, trainer=trainer)
    
    # Save
    print(f"\n💾 Saving tokenizer to {OUTPUT_PATH}...")
    tokenizer.save(OUTPUT_PATH)
    
    # Verify
    print(f"\n✅ Verifying tokenizer...")
    loaded_tokenizer = PreTrainedTokenizerFast(tokenizer_file=OUTPUT_PATH)
    
    print(f"   Vocab size: {loaded_tokenizer.vocab_size}")
    print(f"   PAD token: '{loaded_tokenizer.pad_token}' -> ID {loaded_tokenizer.pad_token_id}")
    print(f"   UNK token: '{loaded_tokenizer.unk_token}' -> ID {loaded_tokenizer.unk_token_id}")
    print(f"   BOS token: '{loaded_tokenizer.bos_token}' -> ID {loaded_tokenizer.bos_token_id}")
    print(f"   EOS token: '{loaded_tokenizer.eos_token}' -> ID {loaded_tokenizer.eos_token_id}")
    
    # Test encoding
    test_text = "Merhaba dünya! Hello world!"
    encodings = loaded_tokenizer(test_text, return_tensors='pt')
    input_ids = encodings['input_ids']
    max_id = input_ids.max().item()
    
    print(f"\n🧪 Test encoding: '{test_text}'")
    print(f"   Token IDs: {input_ids[0][:20].tolist()}")
    print(f"   Max ID: {max_id}")
    
    if max_id >= VOCAB_SIZE:
        print(f"   ❌ ERROR: Max ID {max_id} >= vocab_size {VOCAB_SIZE}")
    else:
        print(f"   ✅ All token IDs within valid range!")
    
    print("\n" + "=" * 70)
    print("✅ Tokenizer retraining complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
