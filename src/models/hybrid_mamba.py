## Developer: inkbytefo
## Modified: 2025-11-22

"""
Hybrid Mamba-Attention Architecture (Jamba-style)
Alternates Mamba SSM blocks with Linear Attention for associative memory.
"""

import torch
import torch.nn as nn

# Conditional import - only needed for Mamba blocks
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.models.mixer_seq_simple import _init_weights
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("⚠️ mamba_ssm not installed. Mamba blocks will not be available.")


class LinearAttentionBlock(nn.Module):
    """
    Linear Attention with Fast Weights (Hebbian-like Associative Memory)
    
    Uses: out = softmax(Q @ K^T) @ V approximated as (Q @ (K^T @ V))
    This is O(L*D^2) instead of O(L^2*D), enabling efficient long-context.
    
    Implements "Fast Weights" concept without manual recurrence.
    """
    def __init__(self, dim, num_heads=8, drop_rate=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(drop_rate)
        
    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            out: (B, L, D)
        """
        B, L, D = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, D_h)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Linear Attention: O(L * D^2)
        # Instead of: Attn = softmax(Q @ K^T) @ V -> O(L^2 * D)
        # We use: Out = Q @ (K^T @ V) -> O(L * D^2)
        
        # Normalize K and Q (replace softmax with feature maps)
        # For simplicity, use ELU+1 as feature map (guarantees positivity)
        Q = torch.nn.functional.elu(Q) + 1  # (B, H, L, D_h)
        K = torch.nn.functional.elu(K) + 1
        
        # Compute KV: (K^T @ V)
        KV = torch.einsum('bhld,bhle->bhde', K, V)  # (B, H, D_h, D_h)
        
        # Compute QKV: Q @ (K^T @ V)
        out = torch.einsum('bhld,bhde->bhle', Q, KV)  # (B, H, L, D_h)
        
        # Normalize by sum of keys (to mimic softmax normalization)
        K_sum = K.sum(dim=2, keepdim=True)  # (B, H, 1, D_h)
        out = out / (torch.einsum('bhld,bhld->bhl', Q, K_sum).unsqueeze(-1) + 1e-6)
        
        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        
        return self.out_proj(self.dropout(out))


class HybridBlock(nn.Module):
    """
    A single layer that can be either Mamba or Linear Attention.
    """
    def __init__(self, dim, block_type='mamba', d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.block_type = block_type
        
        if block_type == 'mamba':
            if not MAMBA_AVAILABLE:
                raise ImportError("mamba_ssm is required for Mamba blocks. Install with: pip install mamba-ssm")
            self.mixer = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        elif block_type == 'attention':
            self.mixer = LinearAttentionBlock(dim=dim, num_heads=8)
        else:
            raise ValueError(f"Unknown block_type: {block_type}")
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, inference_params=None):
        """
        Args:
            x: (B, L, D)
            inference_params: For Mamba state management
        """
        residual = x
        x = self.norm(x)
        
        if self.block_type == 'mamba':
            x = self.mixer(x, inference_params=inference_params)
        else:
            x = self.mixer(x)  # Linear Attention doesn't need inference_params
        
        return residual + x


class HybridMambaLLM(nn.Module):
    """
    Hybrid Mamba-Attention Language Model (Jamba-style).
    
    Architecture:
    - Alternates Mamba and Linear Attention blocks.
    - Mamba: Efficient sequential processing (SSM)
    - Linear Attention: Associative memory (Fast Weights)
    
    Config Pattern:
    - Every 3rd layer is Linear Attention, rest are Mamba.
    - Example (12 layers): M M A M M A M M A M M A
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Build Hybrid Layers
        self.layers = nn.ModuleList()
        for i in range(config.n_layer):
            # Every 3rd layer is Linear Attention
            block_type = 'attention' if (i + 1) % 3 == 0 else 'mamba'
            
            self.layers.append(HybridBlock(
                dim=config.d_model,
                block_type=block_type,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
            ))
        
        # Final Norm
        self.norm_f = nn.LayerNorm(config.d_model)
        
        # LM Head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight Tying
        self.lm_head.weight = self.embedding.weight
        
        # Initialize weights (custom function to avoid signature mismatch)
        def init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        self.apply(init_weights)
        
    def forward(self, input_ids, inference_params=None):
        """
        Args:
            input_ids: (B, L)
            inference_params: Optional state dict for inference
        
        Returns:
            outputs with .logits attribute
        """
        x = self.embedding(input_ids)  # (B, L, D)
        
        for layer in self.layers:
            x = layer(x, inference_params=inference_params)
        
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        # Return in same format as MambaLMHeadModel
        class Output:
            def __init__(self, logits):
                self.logits = logits
        
        return Output(logits)
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50):
        """
        Autoregressive generation with state reuse.
        """
        self.eval()
        inference_params = {}
        
        with torch.no_grad():
            for _ in range(max_length):
                # Only pass the last token if we have state
                if len(inference_params) > 0:
                    current_input = input_ids[:, -1:]
                else:
                    current_input = input_ids
                
                outputs = self(current_input, inference_params=inference_params)
                next_token_logits = outputs.logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if EOS token
                if next_token.item() == 3:  # Assuming EOS = 3
                    break
        
        return input_ids
