import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, n_states:int, n_actions:int, hidden_dim:int=128):
        super().__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) 
        self.fc3 = nn.Linear(hidden_dim, n_actions) 
        
    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        return self.fc3(x)