import gymnasium as gym
from config import *
from reinforce.utils.discretize import Discretizer
from reinforce.agents.QAgent import QLearning, Sarsa, QAgent
from reinforce.utils.plot import plot_rewards
from tqdm import tqdm
import numpy as np

def train(output_dir:str, agent:QAgent):
    env = gym.make(id=id_name, render_mode="rgb_array")
    discretizer = Discretizer(env, num_parts)
    pbar = tqdm(
        total=max_episodes,
        desc='Training',
        dynamic_ncols=True,
    )
    ep_rewards = []
    epsilon = 1
    epsilon_decay = 0.999
    for epoch in range(max_episodes):
        ep_reward = 0
        state, info = env.reset()
        state = discretizer.state2id(state)
        while True:
            if np.random.uniform(0, 1) > epsilon:
                action = agent.predict(state) 
            else:
                action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = discretizer.state2id(next_state)
            agent.update(
                state = state, 
                action = action, 
                reward = reward, 
                next_state = next_state, 
                done = terminated
            )
            state = next_state
            ep_reward += reward
            if terminated or ep_reward > 2000:
                break
        ep_rewards.append(ep_reward)
        epsilon = epsilon * epsilon_decay
        pbar.set_postfix(Epoch=epoch, Reward=ep_reward, Epsilon=epsilon) 
        pbar.update(1)
    pbar.close()
    env.close()
    agent.save_Q_table(f'{output_dir}/Q_table.json')
    plot_rewards(ep_rewards, f'{output_dir}/rewards_curve.png')

# test
def test(output_dir:str, agent:QAgent):
    env = gym.make(id=id_name, render_mode="human")
    discretizer = Discretizer(env, num_parts)
    agent.load_Q_table(f'{output_dir}/Q_table.json')
    state, info = env.reset()
    state = discretizer.state2id(state)
    while True:
        action = agent.predict(state) 
        next_state, reward, terminated, truncated, info = env.step(action)
        state = discretizer.state2id(next_state)
        if terminated or truncated:
            break
    env.close()

if __name__ == '__main__':
    from pathlib import Path
    QClass:QAgent = QLearning

    output_dir = f'output/cartpole/{QClass.__name__}'
    Path(output_dir).mkdir(exist_ok=True, parents=True)    
    agent = QClass(
        state_dim = num_parts ** 4,
        action_dim = 2,
        gamma = gamma, lr = lr,
    )

    # train(output_dir, agent)
    test(output_dir, agent)

# python code/CartPole/run_QAgent.py