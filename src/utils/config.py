import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    d_model: int
    n_layer: int
    vocab_size: int
    ssm_cfg: dict
    use_plasticity: bool = False
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8

@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    max_steps: int
    seed: int
    device: str
    sequence_len: int = 2048
    min_lr: float = 1e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

@dataclass
class DataConfig:
    dataset_name: str
    max_length: int
    dataset_paths: Optional[list] = None
    tokenizer_path: str = "data/tokenizer.model"
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
