import gymnasium as gym
from reinforce.utils.epsilon import Epsilon
from reinforce.agents.DQN import DQN
from reinforce.utils.plot import plot_rewards
from tqdm import tqdm

def train(id_name:str, output_dir:str, agent:DQN, max_episodes:int):
    env = gym.make(id=id_name, render_mode="rgb_array")
    pbar = tqdm(
        total=max_episodes,
        desc='Training',
        dynamic_ncols=True,
    )
    ep_rewards = []
    max_reward = -100000
    for epoch in range(max_episodes):
        ep_reward = 0
        state, info = env.reset()
        for _ in range(2000):
            action = agent.choose_action(state) 
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.memory.push((state, action, reward, next_state, terminated))
            state = next_state
            ep_reward += reward
            if terminated:
                break
        agent.update()
        agent.epsilon.update()
        ep_rewards.append(ep_reward)
        pbar.set_postfix(
            Epoch=epoch, Reward=ep_reward, 
            Loss=agent.loss, Epsilon=agent.epsilon()
        ) 
        pbar.update(1)

        if epoch > 200 and ep_reward > max_reward:
            max_reward = ep_reward
            agent.save(f'{output_dir}/DQN_model.pth')
    pbar.close()
    env.close()
    plot_rewards(ep_rewards, f'{output_dir}/rewards_curve.png')

def test(id_name:str, output_dir:str, agent:DQN):
    env = gym.make(id=id_name, render_mode="human")
    agent.load(f'{output_dir}/DQN_model.pth')
    state, info = env.reset()
    while True:
        action = agent.predict(state) 
        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state
        if terminated:
            break
    env.close()

if __name__ == '__main__':
    import torch
    from pathlib import Path
    from reinforce.models.mlp import MLP
    from reinforce.models.heads import HEADS


    QClass:DQN = DQN
    modelClass = MLP
    output_dir = f'output/mountaincar/{QClass.__name__}/{modelClass.__name__}'
    Path(output_dir).mkdir(exist_ok=True, parents=True)    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    from config_dqn import *
    epsilon = Epsilon(start, end, decay)
    model = modelClass(n_states, n_actions, n_hidden)
    agent = QClass(
        model = model,
        n_actions = n_actions,
        epsilon = epsilon,
        gamma = gamma, lr = lr,
        batch_size = batch_size, 
        buffer_size = buffer_size,
        device = device
    )
    train(id_name, output_dir, agent, max_episodes)
    test(id_name, output_dir, agent)

# python code/MountainCar/run_DQN.py