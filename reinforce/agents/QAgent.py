import math
import numpy as np
import json
from collections import defaultdict
from reinforce.utils.epsilon import Epsilon

class QAgent(object):
    def __init__(
            self, 
            state_dim:int, 
            action_dim:int, 
            epsilon:Epsilon,
            gamma:float,
            lr:float
        ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim 
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.Q_table  = defaultdict(lambda: np.zeros(action_dim)) 

    def save(self, path:str) -> None:
        Q_table = {}
        for key, value in self.Q_table.items():
            Q_table[key] = value.tolist()     
        with open(path, 'w') as f:
            json.dump(Q_table, f)

    def load(self, path:str) -> None:
        Q_table = {}
        with open(path, 'r') as f:
            Q_table = json.load(f)
        for key, value in Q_table.items():
            self.Q_table[key] = np.array(value)

    def choose_action(self, state:int) -> int:
        if np.random.uniform(0, 1) > self.epsilon.value:
            action = self.predict(state)    
        else:
            action = np.random.choice(self.action_dim)
        return action
    
    def predict(self, state:int) -> int:
        action = self.Q_table[str(state)].argmax()
        return action
    
    def update(
            self, 
            state:int, 
            action:int, 
            reward:float, 
            next_state:int, 
            done:bool
        ): 
        raise NotImplementedError
    

class QLearning(QAgent):
    def __init__(self, state_dim, action_dim, epsilon, gamma, lr):
        super(QLearning, self).__init__(state_dim, action_dim, epsilon, gamma, lr)
    
    def update(self, state, action, reward, next_state, done):
        Q_now = self.Q_table[str(state)][action]
        if done:
            Q_target = reward
        else:
            # greedy policy
            # next_action = self.predict(next_state) 
            Q_target = reward + self.gamma * self.Q_table[str(next_state)].max() 
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_now)



class Sarsa(QAgent):
    def __init__(self, state_dim, action_dim, epsilon, gamma, lr):
        super(Sarsa, self).__init__(state_dim, action_dim, epsilon, gamma, lr)
    
    def update(self, state, action, reward, next_state, done):
        Q_now = self.Q_table[str(state)][action]
        if done:
            Q_target = reward
        else:
            # epsilon greedy policy
            next_action = self.choose_action(next_state) 
            Q_target = reward + self.gamma * self.Q_table[str(next_state)][next_action]
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_now)