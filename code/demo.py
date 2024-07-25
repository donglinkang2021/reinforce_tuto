import gymnasium as gym
from utils import generate_random_map
from config import *

map_desc = generate_random_map(
    size = map_size, 
    p = frozen_prob, 
    seed = map_seed
)

env = gym.make(
    id = id_name, 
    desc = map_desc,
    is_slippery = is_slippery, 
    render_mode = render_mode
)

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()