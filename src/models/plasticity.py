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
        
    def forward(self, x, state=None, inference_params=None):
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
            state: Previous memory state (batch, dim, dim) - DEPRECATED, use inference_params
            inference_params: Dictionary to store state during generation
        """
        batch_size, seq_len, dim = x.shape
        
        # Handle inference_params for stateful generation
        if inference_params is not None:
            # Check if we have a stored state for this layer
            # We need a unique key for this layer. For now, let's assume the caller manages this
            # or we use a simple key if this is the only Hebbian layer.
            # Ideally, PlasticMambaBlock should pass the specific state for this layer.
            # But here we receive 'state' directly if passed explicitly, or we look into inference_params.
            
            # If state is passed explicitly, use it (legacy support or direct control)
            if state is None:
                # Try to retrieve from inference_params
                # We expect inference_params to be a dict where we can store/retrieve our state
                # Key convention: "hebbian_state"
                state = inference_params.get("hebbian_state", None)

        # Initialize state if not provided
        if state is None:
            state = torch.zeros(batch_size, dim, dim, device=x.device)
            
        # Optimization: If seq_len is 1 and we have state, we can do a fast update
        if seq_len == 1:
            k = self.w_key(x)   # (B, 1, D)
            v = self.w_value(x) # (B, 1, D)
            q = self.w_query(x) # (B, 1, D)
            
            # Read from memory
            # q: (B, 1, D), state: (B, D, D) -> (B, 1, D)
            memory_out = torch.bmm(q, state) # (B, 1, D)
            
            # Update memory
            # k: (B, 1, D), v: (B, 1, D) -> (B, D, D)
            update = torch.bmm(k.transpose(1, 2), v) # Note: k is (B, 1, D), so k.T is (B, D, 1). Wait.
            # Original rule: A_{t+1} = lambda * A + eta * (x_t @ y_t^T)
            # Here x_t is k_t, y_t is v_t.
            # k_t: (B, D, 1) in loop. Here k is (B, 1, D).
            # So we need k.transpose(1, 2) @ v ? No.
            # In loop: k_t = k[:, t, :].unsqueeze(2) -> (B, D, 1)
            #          v_t = v[:, t, :].unsqueeze(2) -> (B, D, 1)
            #          update = k_t @ v_t.T -> (B, D, 1) @ (B, 1, D) -> (B, D, D)
            
            k_t = k.transpose(1, 2) # (B, D, 1)
            v_t = v # (B, 1, D) - already in row vector form if we consider v_t^T
            
            # Let's match the loop logic exactly:
            # update = torch.bmm(k_t, v_t.transpose(1, 2)) 
            # Wait, in loop v_t was (B, D, 1).
            # Here v is (B, 1, D). So v.transpose(1, 2) is (B, D, 1).
            # So k_t @ v_t.T is (B, D, 1) @ (B, 1, D) -> (B, D, D). Correct.
            
            update = torch.bmm(k_t, v) 
            
            new_state = self.decay_rate * state + self.learning_rate * update
            
            # Save state back to inference_params if present
            if inference_params is not None:
                inference_params["hebbian_state"] = new_state
                
            out = self.out_proj(memory_out)
            return out, new_state


        # Sequential processing for training (or first step of inference)
        # OPTIMIZED: Vectorized update instead of loop
        k = self.w_key(x)   # (B, L, D)
        v = self.w_value(x) # (B, L, D)
        q = self.w_query(x) # (B, L, D)
        
        # Compute all outer products at once: k_t ⊗ v_t^T for all t
        # k: (B, L, D), v: (B, L, D) -> updates: (B, L, D, D)
        updates = torch.einsum('bld,ble->blde', k, v)  # (B, L, D, D)
        
        # Apply learning rate
        updates = self.learning_rate * updates  # (B, L, D, D)
        
        # Compute cumulative states with decay
        # A_t = λ^t * A_0 + Σ_{i=0}^{t-1} λ^{t-i-1} * η * update_i
        # We'll use a custom scan operation
        current_state = state
        states = self._compute_cumulative_states(updates, current_state)  # (B, L, D, D)
        
        # Read from memory: q_t @ A_t for all t
        # q: (B, L, D), states: (B, L, D, D) -> memory_out: (B, L, D)
        memory_out = torch.einsum('bld,blde->ble', q, states)  # (B, L, D)
        
        # Final state for next iteration
        final_state = states[:, -1, :, :]  # (B, D, D)
        
        # If inference_params is present, save the final state
        if inference_params is not None:
            inference_params["hebbian_state"] = final_state
            
        return self.out_proj(memory_out), final_state
    
    def _compute_cumulative_states(self, updates, initial_state):
        """
        Compute cumulative Hebbian states with decay.
        A_t = λ * A_{t-1} + update_t
        
        Args:
            updates: (B, L, D, D) - per-timestep updates
            initial_state: (B, D, D) - initial memory state
        
        Returns:
            states: (B, L, D, D) - memory state at each timestep
        """
        batch_size, seq_len, dim, _ = updates.shape
        device = updates.device
        
        # Preallocate states tensor
        states = torch.zeros(batch_size, seq_len, dim, dim, device=device)
        
        # Use a more efficient scan with decay powers
        # Instead of loop, we can use cumsum with decay factors
        # But for now, use a simple loop (still faster than the original)
        # TODO: Replace with custom CUDA kernel for maximum speed
        
        current = initial_state
        for t in range(seq_len):
            current = self.decay_rate * current + updates[:, t, :, :]
            states[:, t, :, :] = current
        
        return states

