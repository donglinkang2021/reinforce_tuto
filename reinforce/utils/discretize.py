import numpy as np
import gymnasium as gym
from typing import Tuple

class Discretizer:
    def __init__(self, env:gym.Env, num_parts:int):
        """
        Discretize the continuous state space into discrete states.
        """
        lows = env.observation_space.low
        highs = env.observation_space.high
        self.num_parts = num_parts
        self.spaces = [np.linspace(low, high, num_parts) for low, high in zip(lows, highs)]
        self.space_dim = len(self.spaces)

    def state2id(self, state:Tuple) -> int:
        idx = 0
        for i, (s, sp) in enumerate(zip(state, self.spaces)):
            idx += np.digitize(s, sp) * self.num_parts ** (self.space_dim - i - 1)
        return idx
        
if __name__ == '__main__':
    env = gym.make("MountainCar-v0")
    # env = gym.make("CartPole-v1")
    discretizer = Discretizer(env, num_parts=2)
    state, info = env.reset()
    state_id = discretizer.state2id(state)
    print(state, state_id)
    env.close()

# python code/MountainCar/utils.py