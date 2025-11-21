## Developer: inkbytefo
## Modified: 2025-11-21

from typing import Optional, Union, List, Dict, Any
import torch
import os
import sentencepiece as spm

class Tokenizer:
    """
    SentencePiece-based tokenizer wrapper for AETHER project.
    Optimized for Turkish agglutinative morphology using Unigram language model.
    """
    
    # Special tokens (IDs match SentencePiece training defaults)
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    
    def __init__(
        self, 
        model_path: str, 
        max_length: int = 2048
    ):
        """
        Initialize tokenizer.
        
        Args:
            model_path: Path to .model file trained by SentencePiece
            max_length: Maximum sequence length
        """
        self.max_length = max_length
        self.sp = spm.SentencePieceProcessor()
        
        if os.path.exists(model_path):
            self.sp.load(model_path)
            self.vocab_size = self.sp.get_piece_size()
            print(f"✅ Loaded SentencePiece model from {model_path}")
            print(f"   Vocab size: {self.vocab_size}")
        else:
            raise FileNotFoundError(f"❌ Tokenizer model not found at {model_path}")
            
    def encode(
        self, 
        text: Union[str, List[str]], 
        add_special_tokens: bool = True,
        return_tensors: bool = True
    ) -> Union[Dict[str, torch.Tensor], List[List[int]]]:
        """
        Encode text to token IDs.
        
        Args:
            text: Single text or batch of texts
            add_special_tokens: Whether to add BOS/EOS tokens
            return_tensors: If True, return torch tensors, else return lists
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors (if return_tensors=True)
            or List of List of ints (if return_tensors=False)
        """
        if isinstance(text, str):
            text = [text]
            
        batch_ids = []
        batch_masks = []
        
        for t in text:
            # SentencePiece encode
            ids = self.sp.encode_as_ids(t)
            
            if add_special_tokens:
                ids = [self.BOS_ID] + ids + [self.EOS_ID]
                
            # Truncate
            if len(ids) > self.max_length:
                ids = ids[:self.max_length]
                # Ensure EOS is present if we truncated and special tokens were requested
                if add_special_tokens:
                    ids[-1] = self.EOS_ID
            
            # Pad
            mask = [1] * len(ids)
            padding_len = self.max_length - len(ids)
            if padding_len > 0:
                ids = ids + [self.PAD_ID] * padding_len
                mask = mask + [0] * padding_len
                
            batch_ids.append(ids)
            batch_masks.append(mask)
            
        if return_tensors:
            return {
                'input_ids': torch.tensor(batch_ids, dtype=torch.long),
                'attention_mask': torch.tensor(batch_masks, dtype=torch.long)
            }
        else:
            # If not returning tensors, we might want just the IDs without padding for some use cases,
            # but for consistency with the previous API, let's return the padded lists.
            # Or better, let's return the raw lists if return_tensors is False, 
            # but the previous implementation returned a dict-like object or similar structure.
            # Let's stick to the signature: Union[Dict[str, torch.Tensor], List[int]]
            # Wait, the type hint says List[int] but logic implies List[List[int]] for batch.
            # The previous implementation returned `encodings` object from HF or a dict.
            # Let's return the dict structure even if not tensors, for compatibility.
            return {
                'input_ids': batch_ids,
                'attention_mask': batch_masks
            }

    def decode(self, token_ids: Union[torch.Tensor, List[int]], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        """
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() > 1:
                # Batch decode
                return [self.decode(seq, skip_special_tokens) for seq in token_ids]
            token_ids = token_ids.tolist()
            
        if skip_special_tokens:
            # Filter out special tokens manually before decoding
            # SentencePiece might handle this but explicit is better
            token_ids = [
                tid for tid in token_ids 
                if tid not in [self.PAD_ID, self.BOS_ID, self.EOS_ID, self.UNK_ID]
            ]
            
        return self.sp.decode_ids(token_ids)

    @property
    def pad_token_id(self) -> int:
        return self.PAD_ID
    
    @property
    def eos_token_id(self) -> int:
        return self.EOS_ID
    
    @property
    def bos_token_id(self) -> int:
        return self.BOS_ID
    
    @property
    def unk_token_id(self) -> int:
        return self.UNK_ID
    
    def __len__(self) -> int:
        return self.vocab_size
