import gymnasium as gym
from utils import generate_random_map
from config import *
from agent import QLearning 
output_dir = 'output/frozenlake/Qlearning'
map_desc = generate_random_map(
    size = map_size, 
    p = frozen_prob, 
    seed = map_seed
)
env = gym.make(
    id = id_name, 
    desc = map_desc,
    is_slippery = is_slippery, 
    render_mode = 'human' # for interactive mode, testing
)
agent = QLearning(
    state_dim = env.observation_space.n,
    action_dim = env.action_space.n,
    gamma = gamma,
    lr = lr,
)
agent.load_Q_table(f'{output_dir}/Q_table.json')
state, info = env.reset()
while True:
    action = agent.predict(state) 
    next_state, reward, terminated, truncated, info = env.step(action)
    state = next_state
    if terminated:
        break
env.close()