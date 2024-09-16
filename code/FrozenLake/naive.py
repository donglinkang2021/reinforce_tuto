import gymnasium as gym
from utils import generate_random_map
from config import *

map_desc = generate_random_map(
    size = map_size, 
    p = frozen_prob, 
    seed = map_seed
)

from astar import astar, path2table
path = astar(map_desc, (0, 0), (map_size-1, map_size-1))
assert path is not None, "No path found!"
table = path2table(path)

env = gym.make(
    id = id_name, 
    desc = map_desc,
    is_slippery = is_slippery, 
    render_mode = 'human' # for non-interactive mode, training
)

state, info = env.reset()
while True:
    action = table.get((state // map_size, state % map_size), 0)
    next_state, reward, terminated, truncated, info = env.step(action)
    state = next_state
    if terminated:
        break
env.close()

# python code/FrozenLake/naive.py