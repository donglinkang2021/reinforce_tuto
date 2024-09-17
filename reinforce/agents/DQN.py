import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from reinforce.utils.replay import ReplayBuffer
from reinforce.utils.batch import Batch
from reinforce.utils.epsilon import Epsilon

class DQN:
    def __init__(
            self, 
            model:nn.Module, 
            n_actions:int,
            epsilon:Epsilon,
            gamma:float = 0.99,
            lr:float = 1e-3,
            batch_size:int = 32,
            buffer_size:int = 300,
            device:str = 'cpu',
        ) -> None:
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device(device) 
        self.policy_net = model.to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr)
        self.memory = ReplayBuffer(buffer_size)
        self.epsilon = epsilon
        self.loss = None

    def choose_action(self, state:np.ndarray) -> int:
        if random.random() > self.epsilon.value:
            action = self.predict(state)
        else:
            action = random.randrange(self.n_actions)
        return action
    
    @torch.no_grad()
    def predict(self, state:np.ndarray) -> int:
        state = torch.tensor(state).to(self.device, torch.float32).unsqueeze(dim=0)
        q_values:torch.Tensor = self.policy_net(state)
        action = q_values.argmax(dim=-1).item()
        return action
    
    def update(self):
        if len(self.memory) < self.batch_size: 
            return
        batch = Batch(self.memory.sample(self.batch_size), device=self.device)
        # calculate the Q value of the current state (s_t, a)
        q_values:torch.Tensor = self.policy_net(batch.state)  # (batch_size, n_actions)
        q_values = q_values.gather(dim=-1, index=batch.action)  # (batch_size, 1)
        # calculate the Q value of the next state (s_t_, a), just use max value
        next_q_values:torch.Tensor = self.policy_net(batch.next_state) # (batch_size, n_actions)
        next_q_values = next_q_values.max(dim=-1)[0].detach() # (batch_size, )
        # calculate the expected Q value, for the terminal state, done_batch[0]=1, 
        # the corresponding expected_q_value is equal to reward
        expected_q_values = batch.reward + self.gamma * next_q_values * (1 - batch.done)
        loss = F.mse_loss(q_values, expected_q_values.unsqueeze(1))
        # update the model
        self.optimizer.zero_grad()  
        loss.backward()
        self.loss = loss.item() # record the loss
        # avoid gradient explosion
        for param in self.policy_net.parameters():  
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step() 

    def save(self, path:str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path:str) -> None:
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))