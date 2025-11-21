import torch
import torch.nn as nn
from mamba_ssm import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from ..utils.config import ModelConfig

class MambaLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Create MambaConfig
        mamba_config = MambaConfig(
            d_model=config.d_model,
            n_layer=config.n_layer,
            vocab_size=config.vocab_size,
            ssm_cfg=config.ssm_cfg,
        )
        
        # Initialize Mamba Backbone
        self.backbone = MambaLMHeadModel(mamba_config)
        
        # Note: We rely on Mamba's implicit state for position awareness.
        # Explicit positional embeddings (RoPE/Learned) are difficult to inject 
        # into the pre-packaged MambaLMHeadModel without significant overriding.
        # For Phase 1, the SSM state is sufficient for Turkish morphology.


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
