from collections import deque
import random
from typing import Tuple

class ReplayBuffer(object):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

    def push(self, transition: Tuple) -> None:
        """(state, action, reward, next_state, done)"""
        self.buffer.append(transition)

    def sample(self, batch_size: int, sequential: bool = False) -> zip:
        batch_size = min(batch_size, len(self.buffer))
        if sequential:
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
            return zip(*batch)
        else:
            batch = random.sample(self.buffer, batch_size)
            return zip(*batch)
        
    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)
    
if __name__ == '__main__':
    buffer = ReplayBuffer(capacity=100)
    import gymnasium as gym
    env = gym.make("CartPole-v1", render_mode="rgb_array") # for train, rgb_array
    state, info = env.reset()
    for _ in range(1000):
        action = 1 if state[2] + state[3] > 0 else 0
        next_state, reward, terminated, truncated, info = env.step(action)
        buffer.push((state, action, reward, next_state, terminated))
        state = next_state
    print(len(buffer))
    batch_transitions = buffer.sample(10)
    print(type(batch_transitions))
    for state, action, reward, next_state, done in zip(*batch_transitions):
        print(state, action, reward, next_state, done)
    buffer.clear()
    print(len(buffer))

# python reinforce/utils/replay.py