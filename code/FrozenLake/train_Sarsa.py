import gymnasium as gym
from utils import generate_random_map
from config import *
from reinforce.agents.QAgent import Sarsa
from reinforce.utils.plot import plot_rewards
from tqdm import tqdm
from pathlib import Path
output_dir = 'output/frozenlake/Sarsa'
Path(output_dir).mkdir(exist_ok=True, parents=True)
map_desc = generate_random_map(
    size = map_size, 
    p = frozen_prob, 
    seed = map_seed
)

# train
env = gym.make(
    id = id_name, 
    desc = map_desc,
    is_slippery = is_slippery, 
    render_mode = 'ansi' # for non-interactive mode, training
)
agent = Sarsa(
    state_dim = env.observation_space.n,
    action_dim = env.action_space.n,
    gamma = gamma,
    lr = lr,
)
pbar = tqdm(
    total=max_episodes,
    desc='Training',
    dynamic_ncols=True,
)
ep_rewards = []
for epoch in range(max_episodes):
    ep_reward = 0
    state, info = env.reset()
    while True:
        action = agent.choose_action(state) 
        next_state, reward, terminated, truncated, info = env.step(action)
        if reward == 0:
            if terminated:
                reward = -1
            else:
                reward = -0.01
        agent.update(
            state = state, 
            action = action, 
            reward = reward, 
            next_state = next_state, 
            done = terminated
        )
        state = next_state
        ep_reward += reward
        if terminated:
            break
    ep_rewards.append(ep_reward)
    pbar.update(1)
pbar.close()
env.close()
agent.save_Q_table(f'{output_dir}/Q_table.json')
plot_rewards(ep_rewards, f'{output_dir}/rewards_curve.png')

# test
env = gym.make(
    id = id_name, 
    desc = map_desc,
    is_slippery = is_slippery, 
    render_mode = 'human' # for interactive mode, testing
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