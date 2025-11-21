## Developer: inkbytefo
## Modified: 2025-11-21

import sentencepiece as spm
import os
import glob
from typing import List, Optional

def train_sentencepiece_tokenizer(
    input_files: List[str],
    model_prefix: str = "tokenizer",
    vocab_size: int = 64000,
    character_coverage: float = 0.9995,
    model_type: str = "unigram"
):
    """
    Train a SentencePiece tokenizer.
    
    Args:
        input_files: List of paths to text files to train on.
        model_prefix: Prefix for the output model files.
        vocab_size: Target vocabulary size.
        character_coverage: Amount of characters covered by the model. 
                            Turkish has specific chars, so 0.9995 or 1.0 is good.
        model_type: 'unigram' or 'bpe'. Unigram is preferred for agglutinative languages.
    """
    
    if not input_files:
        raise ValueError("No input files provided for tokenizer training.")
        
    # Join files with comma for sentencepiece
    input_argument = ",".join(input_files)
    
    print(f"🚀 Training SentencePiece ({model_type}) model...")
    print(f"   Vocab Size: {vocab_size}")
    print(f"   Input Files: {len(input_files)} files")
    
    # Explicitly define special tokens to ensure ID consistency
    # 0: <pad>, 1: <unk>, 2: <s>, 3: </s>
    # SentencePiece maps unk/bos/eos by default. We need to ensure pad is 0.
    # We use --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3
    
    cmd = [
        f"--input={input_argument}",
        f"--model_prefix={model_prefix}",
        f"--vocab_size={vocab_size}",
        f"--character_coverage={character_coverage}",
        f"--model_type={model_type}",
        f"--pad_id=0",
        f"--unk_id=1",
        f"--bos_id=2",
        f"--eos_id=3",
        f"--pad_piece=<pad>",
        f"--unk_piece=<unk>",
        f"--bos_piece=<s>",
        f"--eos_piece=</s>",
        # user_defined_symbols can be added here if needed
    ]
    
    # If input is too small for vocab_size, SP might error. 
    # We'll let it run and catch error if needed, but usually it just warns.
    
    spm.SentencePieceTrainer.Train(" ".join(cmd))
    
    # Correct approach for SP training with specific IDs:
    # We will let SP handle unk, bos, eos. We will add <pad> as a user defined symbol.
    # BUT, we want specific IDs.
    
    # Actually, simpler approach:
    # Don't force IDs in training, just define symbols.
    # unk_id=0, bos_id=1, eos_id=2, pad_id=-1 (by default)
    # We want to shift them.
    
    # Let's try this:
    # unk_id=1, bos_id=2, eos_id=3, pad_id=0
    # And we need to tell SP that ID 0 is <pad>.
    # SP doesn't support pad_id argument in train() directly in all versions or it might conflict.
    
    # Let's look at the error: "<unk> must not be defined with..."
    # It means I passed <unk> in user_defined_symbols AND it's a control symbol.
    
    # Fix: Only pass <pad> in user_defined_symbols.
    user_defined_symbols = ["<pad>"]
    
    # Train command
    spm.SentencePieceTrainer.train(
        input=input_argument,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        user_defined_symbols=user_defined_symbols,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        normalization_rule_name='nmt_nfkc_cf' 
    )
    
    print(f"✅ Tokenizer training complete. Saved to {model_prefix}.model and {model_prefix}.vocab")

if __name__ == "__main__":
    # Example usage
    # You would typically run this with a list of globbed files
    # files = glob.glob("data/raw/*.txt")
    # train_sentencepiece_tokenizer(files, model_prefix="data/tokenizer_sp", vocab_size=64000)
    pass
