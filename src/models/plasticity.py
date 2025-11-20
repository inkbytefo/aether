import torch
import torch.nn as nn
import torch.nn.functional as F

class HebbianMemory(nn.Module):
    """
    A layer that implements Hebbian Learning (Fast Weights).
    It maintains a dynamic weight matrix 'A' that updates during the forward pass.
    
    Update Rule:
    A_{t+1} = \lambda * A_t + \eta * (x_t @ y_t^T)
    
    Output:
    out = x @ W_static + (x @ A_t) @ W_read
    """
    def __init__(self, dim, learning_rate=0.1, decay_rate=0.9):
        super().__init__()
        self.dim = dim
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        
        # Static weights to transform input before memory interaction
        self.w_key = nn.Linear(dim, dim)
        self.w_value = nn.Linear(dim, dim)
        self.w_query = nn.Linear(dim, dim)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x, state=None):
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            state: Previous memory state (batch, dim, dim)
        """
        batch_size, seq_len, dim = x.shape
        
        # Initialize state if not provided
        if state is None:
            state = torch.zeros(batch_size, dim, dim, device=x.device)
            
        # We need to process sequentially to update state step-by-step
        # For efficiency in training, we could use a custom CUDA kernel or linear attention formulation
        # But for this prototype, we'll use a loop or a simplified associative scan if possible.
        # For now, let's implement the loop version for clarity and correctness.
        
        outputs = []
        current_state = state
        
        k = self.w_key(x)   # (B, L, D)
        v = self.w_value(x) # (B, L, D)
        q = self.w_query(x) # (B, L, D)
        
        for t in range(seq_len):
            k_t = k[:, t, :].unsqueeze(2) # (B, D, 1)
            v_t = v[:, t, :].unsqueeze(2) # (B, D, 1)
            q_t = q[:, t, :].unsqueeze(1) # (B, 1, D)
            
            # Read from memory
            # memory_out = q_t @ current_state
            memory_out = torch.bmm(q_t, current_state).squeeze(1) # (B, D)
            
            outputs.append(memory_out)
            
            # Update memory (Hebbian Rule)
            # A_new = lambda * A + eta * (k @ v.T)
            update = torch.bmm(k_t, v_t.transpose(1, 2)) # (B, D, D)
            current_state = self.decay_rate * current_state + self.learning_rate * update
            
        outputs = torch.stack(outputs, dim=1) # (B, L, D)
        
        # Final projection
        return self.out_proj(outputs), current_state
