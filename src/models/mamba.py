import torch
import torch.nn as nn
from mamba_ssm import MambaLMHeadModel
from ..utils.config import ModelConfig

class MambaLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Initialize Mamba Backbone
        # MambaLMHeadModel expects: d_model, n_layer, vocab_size
        # We can pass other kwargs for SSM config if needed, but MambaLMHeadModel 
        # usually takes a specific config object or kwargs. 
        # For simplicity in this wrapper, we assume standard initialization.
        
        self.backbone = MambaLMHeadModel(
            d_model=config.d_model,
            n_layer=config.n_layer,
            vocab_size=config.vocab_size,
            ssm_cfg=config.ssm_cfg,
            device=None, # Will be moved to device later
            dtype=torch.float32, # Default to float32
        )

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0):
        """
        Forward pass for Mamba.
        Args:
            input_ids: (batch, seq_len)
            position_ids: Not used in Mamba usually, but kept for compatibility
            inference_params: For fast inference
            num_last_tokens: For optimization
        """
        return self.backbone(
            input_ids=input_ids, 
            inference_params=inference_params,
            num_last_tokens=num_last_tokens
        )

    def generate(self, input_ids, max_length=100, **kwargs):
        """
        Simple generation wrapper.
        """
        return self.backbone.generate(input_ids, max_length=max_length, **kwargs)
