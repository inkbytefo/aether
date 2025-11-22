## Developer: inkbytefo
## Modified: 2025-11-22

import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    d_model: int
    n_layer: int
    vocab_size: int
    
    # SSM parameters (Mamba)
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    
    # Hybrid architecture
    attention_interval: Optional[int] = None  # If set, creates Hybrid Mamba-Attention
    
    # Legacy parameters (backward compatibility)
    ssm_cfg: Optional[dict] = None
    use_plasticity: bool = False
    hebbian_cfg: Optional[dict] = None
    
    # Model setup
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8

@dataclass
class TrainingConfig:
    learning_rate: float
    max_steps: int
    device: str
    
    # Optimizer settings
    optimizer: str = "AdamW"
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    
    # Scheduler settings
    scheduler: str = "cosine"
    warmup_steps: int = 1000
    min_lr: float = 1e-5
    
    # Batch settings
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    
    # Training control
    max_grad_norm: float = 1.0
    seed: int = 42
    sequence_len: int = 2048
    save_steps: int = 1000
    eval_steps: int = 100

@dataclass
class DataConfig:
    tokenizer_path: str
    
    # Binary data files (new format)
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    seq_length: int = 2048
    batch_size: int = 8
    
    # Legacy HuggingFace datasets
    dataset_name: Optional[str] = None
    max_length: Optional[int] = None
    dataset_paths: Optional[list] = None
    train_split: float = 0.9
    num_workers: int = 4

class Config:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        self.model = ModelConfig(**cfg['model'])
        self.training = TrainingConfig(**cfg['training'])
        self.data = DataConfig(**cfg['data'])

    @staticmethod
    def load(path: str = "configs/config.yaml") -> 'Config':
        return Config(path)
