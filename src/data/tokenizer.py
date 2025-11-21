## Developer: inkbytefo
## Modified: 2025-11-21

from typing import Optional, Union, List, Dict, Any
from pathlib import Path
import torch
import os

try:
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from transformers import PreTrainedTokenizerFast
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️ HuggingFace tokenizers not available. Install with: pip install tokenizers transformers")


class Tokenizer:
    """
    Production-grade tokenizer wrapper for AETHER project.
    
    Supports:
    - Custom BPE tokenizer training
    - Loading pre-trained tokenizers
    - Fallback to character-level tokenization
    - Turkish language optimization
    """
    
    # Special tokens
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    
    def __init__(
        self, 
        model_path: Optional[str] = None, 
        max_length: int = 512,
        vocab_size: int = 32000
    ):
        """
        Initialize tokenizer.
        
        Args:
            model_path: Path to pre-trained tokenizer.json file
            max_length: Maximum sequence length
            vocab_size: Vocabulary size (used during training)
        """
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.tokenizer: Optional[PreTrainedTokenizerFast] = None
        self._char_fallback = False
        
        if not HF_AVAILABLE:
            self._init_char_fallback()
            return
        
        if model_path and os.path.exists(model_path):
            self._load_pretrained(model_path)
        else:
            if model_path:
                print(f"⚠️ Tokenizer not found at {model_path}")
            self._init_char_fallback()
    
    def _load_pretrained(self, model_path: str) -> None:
        """Load pre-trained tokenizer from file."""
        try:
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=model_path)
            self._configure_special_tokens()
            # Update vocab_size to match loaded tokenizer
            self.vocab_size = self.tokenizer.vocab_size
            print(f"✅ Loaded tokenizer from {model_path}")
            print(f"   Vocab size: {self.tokenizer.vocab_size}")
        except Exception as e:
            print(f"❌ Failed to load tokenizer: {e}")
            self._init_char_fallback()
    
    def _configure_special_tokens(self) -> None:
        """Configure special tokens for the tokenizer."""
        if not self.tokenizer:
            return
            
        special_tokens = {
            'pad_token': self.PAD_TOKEN,
            'unk_token': self.UNK_TOKEN,
            'bos_token': self.BOS_TOKEN,
            'eos_token': self.EOS_TOKEN,
        }
        
        # Only add tokens that don't exist
        tokens_to_add = {}
        for key, token in special_tokens.items():
            if getattr(self.tokenizer, key, None) is None:
                tokens_to_add[key] = token
        
        if tokens_to_add:
            self.tokenizer.add_special_tokens(tokens_to_add)
        
        # Set padding side and truncation
        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'right'
        self.tokenizer.model_max_length = self.max_length
    
    def _init_char_fallback(self) -> None:
        """Initialize simple character-level fallback tokenizer."""
        self._char_fallback = True
        # Turkish + English alphabet + digits + common symbols
        self._char_vocab = list("abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ0123456789 .,!?:;-'\"()[]{}/@#$%^&*+=_~`<>\n\t")
        self._char_vocab = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN] + self._char_vocab
        self._char_to_id = {c: i for i, c in enumerate(self._char_vocab)}
        self._id_to_char = {i: c for i, c in enumerate(self._char_vocab)}
        self.vocab_size = len(self._char_vocab)
        print(f"⚠️ Using character-level fallback tokenizer (vocab_size={self.vocab_size})")
    
    def train(
        self, 
        files: List[str], 
        output_path: str,
        vocab_size: Optional[int] = None
    ) -> None:
        """
        Train a new BPE tokenizer on given files.
        
        Args:
            files: List of text file paths
            output_path: Where to save tokenizer.json
            vocab_size: Vocabulary size (defaults to self.vocab_size)
        """
        if not HF_AVAILABLE:
            raise RuntimeError("Cannot train tokenizer without HuggingFace tokenizers library")
        
        vocab_size = vocab_size or self.vocab_size
        
        # Initialize BPE tokenizer
        tokenizer = HFTokenizer(BPE(unk_token=self.UNK_TOKEN))
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.decoder = ByteLevelDecoder()
        
        # Configure trainer
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN],
            show_progress=True,
            min_frequency=2
        )
        
        # Train
        print(f"🚀 Training tokenizer on {len(files)} files...")
        tokenizer.train(files, trainer)
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tokenizer.save(output_path)
        print(f"✅ Tokenizer saved to {output_path}")
        
        # Load it
        self._load_pretrained(output_path)
    
    def encode(
        self, 
        text: Union[str, List[str]], 
        add_special_tokens: bool = True,
        return_tensors: bool = True
    ) -> Union[Dict[str, torch.Tensor], List[int]]:
        """
        Encode text to token IDs.
        
        Args:
            text: Single text or batch of texts
            add_special_tokens: Whether to add BOS/EOS tokens
            return_tensors: If True, return torch tensors, else return lists
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask' tensors
        """
        if self._char_fallback:
            return self._encode_char(text, add_special_tokens, return_tensors)
        
        # Use HuggingFace tokenizer
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt' if return_tensors else None,
            add_special_tokens=add_special_tokens
        )
        
        # Validate token IDs are within vocab range
        if return_tensors:
            input_ids = encodings['input_ids']
            max_id = input_ids.max().item()
            if max_id >= self.vocab_size:
                print(f"⚠️ WARNING: Token ID {max_id} exceeds vocab_size {self.vocab_size}")
                print(f"   This will cause CUDA errors. Clamping to valid range.")
                encodings['input_ids'] = torch.clamp(input_ids, 0, self.vocab_size - 1)
        
        return encodings
    
    def _encode_char(
        self, 
        text: Union[str, List[str]], 
        add_special_tokens: bool,
        return_tensors: bool
    ) -> Dict[str, torch.Tensor]:
        """Character-level encoding fallback."""
        if isinstance(text, str):
            text = [text]
        
        batch_ids = []
        batch_masks = []
        
        for t in text:
            ids = []
            if add_special_tokens:
                ids.append(self._char_to_id[self.BOS_TOKEN])
            
            for char in t[:self.max_length - (2 if add_special_tokens else 0)]:
                ids.append(self._char_to_id.get(char, self._char_to_id[self.UNK_TOKEN]))
            
            if add_special_tokens:
                ids.append(self._char_to_id[self.EOS_TOKEN])
            
            # Padding
            attention_mask = [1] * len(ids)
            while len(ids) < self.max_length:
                ids.append(self._char_to_id[self.PAD_TOKEN])
                attention_mask.append(0)
            
            batch_ids.append(ids)
            batch_masks.append(attention_mask)
        
        if return_tensors:
            return {
                'input_ids': torch.tensor(batch_ids, dtype=torch.long),
                'attention_mask': torch.tensor(batch_masks, dtype=torch.long)
            }
        else:
            return {
                'input_ids': batch_ids,
                'attention_mask': batch_masks
            }
    
    def decode(self, token_ids: Union[torch.Tensor, List[int]], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs (tensor or list)
            skip_special_tokens: Whether to remove special tokens
            
        Returns:
            Decoded text string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
        
        if self._char_fallback:
            return self._decode_char(token_ids, skip_special_tokens)
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def _decode_char(self, token_ids: List[int], skip_special_tokens: bool) -> str:
        """Character-level decoding fallback."""
        chars = []
        special_ids = {
            self._char_to_id[self.PAD_TOKEN],
            self._char_to_id[self.BOS_TOKEN],
            self._char_to_id[self.EOS_TOKEN],
            self._char_to_id[self.UNK_TOKEN]
        }
        
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            chars.append(self._id_to_char.get(tid, ''))
        
        return ''.join(chars)
    
    @property
    def pad_token_id(self) -> int:
        """Get PAD token ID."""
        if self._char_fallback:
            return self._char_to_id[self.PAD_TOKEN]
        return self.tokenizer.pad_token_id
    
    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID."""
        if self._char_fallback:
            return self._char_to_id[self.EOS_TOKEN]
        return self.tokenizer.eos_token_id
    
    @property
    def bos_token_id(self) -> int:
        """Get BOS token ID."""
        if self._char_fallback:
            return self._char_to_id[self.BOS_TOKEN]
        return self.tokenizer.bos_token_id
    
    @property
    def unk_token_id(self) -> int:
        """Get UNK token ID."""
        if self._char_fallback:
            return self._char_to_id[self.UNK_TOKEN]
        return self.tokenizer.unk_token_id
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        if self._char_fallback:
            return self.vocab_size
        return self.tokenizer.vocab_size if self.tokenizer else self.vocab_size
