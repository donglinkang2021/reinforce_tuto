import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
    
class HEADS(nn.Module):
    def __init__(
            self, 
            n_states:int, 
            n_actions:int, 
            hidden_dim:int=64, 
            n_heads:int=8
        ):
        assert hidden_dim % n_heads == 0, \
            'hidden_dim must be divisible by n_heads'
        super().__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, n_actions) 
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = rearrange(x, 'b (h d) -> b h d', h=8)
        x = F.scaled_dot_product_attention(x, x, x)
        x = rearrange(x, 'b h d -> b (h d)')
        return self.fc2(x)