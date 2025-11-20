import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    d_model: int
    n_layer: int
    vocab_size: int
    ssm_cfg: dict

@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    max_steps: int
    seed: int
    device: str

@dataclass
class DataConfig:
    dataset_name: str
    max_length: int

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
