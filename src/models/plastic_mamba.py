import torch
import torch.nn as nn
from mamba_ssm import Mamba
# from mamba_ssm.modules.mamba_simple import Block # Unused and causing error
from .plasticity import HebbianMemory
from ..utils.config import ModelConfig

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm
except ImportError:
    RMSNorm = nn.RMSNorm

class PlasticMambaBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx # Store layer_idx for state management
        
        # Standard Mamba Block (Mixer + Norm)
        # We use the Block class from mamba_ssm which encapsulates Mixer and Norm
        # But we need to inject Hebbian Memory. 
        # So we will manually construct: Norm -> Mixer + Norm -> Hebbian -> Residual
        
        self.norm = RMSNorm(config.d_model)
        self.mixer = Mamba(
            d_model=config.d_model,
            d_state=config.ssm_cfg.get('d_state', 16),
            d_conv=config.ssm_cfg.get('d_conv', 4),
            expand=config.ssm_cfg.get('expand', 2),
            layer_idx=layer_idx
        )
        
        # Hebbian Memory Layer
        # We add this every N layers or just once? Let's add it to every block for now
        # or maybe make it optional per block.
        self.hebbian = HebbianMemory(config.d_model)
        self.norm_hebbian = RMSNorm(config.d_model)
        
    def forward(self, x, inference_params=None):
        # Standard Mamba Path
        residual = x
        x = self.norm(x)
        x = self.mixer(x, inference_params=inference_params)
        x = residual + x
        
        # Hebbian Path
        residual = x
        x = self.norm_hebbian(x)
        
        # Manage Hebbian State
        hebbian_state = None
        if inference_params is not None:
            hebbian_state = inference_params.get(f"hebbian_{self.layer_idx}", None)
            
        # We pass state explicitly. We do NOT pass inference_params to hebbian() 
        # to avoid it writing to a global "hebbian_state" key.
        x, new_state = self.hebbian(x, state=hebbian_state) 
        
        if inference_params is not None:
            inference_params[f"hebbian_{self.layer_idx}"] = new_state
            
        x = residual + x
        
        return x

class PlasticMambaLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        self.layers = nn.ModuleList([
            PlasticMambaBlock(config, layer_idx=i)
            for i in range(config.n_layer)
        ])
        
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
        
    def forward(self, input_ids, inference_params=None, num_last_tokens=0):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x, inference_params=inference_params)
            
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        from collections import namedtuple
        Output = namedtuple('Output', ['logits'])
        return Output(logits=logits)
