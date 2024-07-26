import gymnasium as gym
from config import *
from agent import QLearning 
from pathlib import Path
import numpy as np
output_dir = 'output/mountaincar/Qlearning'
Path(output_dir).mkdir(exist_ok=True, parents=True)

# train
env = gym.make(id=id_name, render_mode="human")

num_parts = 20
pos_low, vel_low = env.observation_space.low
pos_high, vel_high = env.observation_space.high
pos_space = np.linspace(pos_low, pos_high, num_parts)
vel_space = np.linspace(vel_low, vel_high, num_parts)

def get_state_id(state:tuple) -> int:
    pos, vel = state
    pos_idx = np.digitize(pos, pos_space)
    vel_idx = np.digitize(vel, vel_space)
    return pos_idx * num_parts + vel_idx

agent = QLearning(
    state_dim = num_parts * num_parts,
    action_dim = env.action_space.n,
    gamma = gamma,
    lr = lr,
)

agent.load_Q_table(f'{output_dir}/Q_table.json')
state, info = env.reset()
state = get_state_id(state)
while True:
    action = agent.predict(state) 
    next_state, reward, terminated, truncated, info = env.step(action)
    next_state = get_state_id(next_state)
    state = next_state
    if terminated:
        break
env.close()