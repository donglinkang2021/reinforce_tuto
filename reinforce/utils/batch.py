import torch
import numpy as np

class Batch:
    def __init__(self, transition:zip, device=torch.device, dtype=torch.float32) -> None:
        state, action, reward, next_state, done = transition
        self.state = torch.tensor(np.array(state)).to(device, dtype) # (batch_size, state_dim)
        self.action = torch.tensor(action).long().unsqueeze(-1).to(device) # (batch_size, 1)
        self.reward = torch.tensor(reward).to(device, dtype) # (batch_size,)
        self.next_state = torch.tensor(np.array(next_state)).to(device, dtype) # (batch_size, state_dim)
        self.done = torch.tensor(done).to(device, dtype) # (batch_size,)

    def __len__(self) -> int:
        return len(self.state)
    
    def __repr__(self) -> str:
        return f"Batch(state.shape={self.state.shape}, action.shape={self.action.shape}, reward.shape={self.reward.shape}, next_state.shape={self.next_state.shape}, done.shape={self.done.shape})"

if __name__ == '__main__':
    from .replay import ReplayBuffer
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
    batch = Batch(batch_transitions, device=torch.device('cpu'))
    print(batch)

# python -m reinforce.utils.batch