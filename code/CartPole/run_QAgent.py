import gymnasium as gym
from config import *
from reinforce.utils.discretize import Discretizer
from reinforce.agents.QAgent import QAgent
from reinforce.utils.epsilon import Epsilon
from reinforce.utils.plot import plot_rewards
from tqdm import tqdm

def train(output_dir:str, agent:QAgent):
    env = gym.make(id=id_name, render_mode="rgb_array")
    discretizer = Discretizer(env, num_parts)
    pbar = tqdm(
        total=max_episodes,
        desc='Training',
        dynamic_ncols=True,
    )
    ep_rewards = []
    max_reward = 0
    for epoch in range(max_episodes):
        ep_reward = 0
        state, info = env.reset()
        state = discretizer.state2id(state)
        while True:
            action = agent.choose_action(state)
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
            if terminated or ep_reward > 5000:
                break
        agent.epsilon.update()
        ep_rewards.append(ep_reward)
        pbar.set_postfix(Epoch=epoch, Reward=ep_reward, Epsilon=agent.epsilon()) 
        pbar.update(1)

        if ep_reward > 5000:
            agent.save(f'{output_dir}/Q_table.json')
            break

        if epoch > 200 and ep_reward > max_reward:
            max_reward = ep_reward
            agent.save(f'{output_dir}/Q_table.json')

    pbar.close()
    env.close()
    plot_rewards(ep_rewards, f'{output_dir}/rewards_curve.png')

# test
def test(output_dir:str, agent:QAgent):
    env = gym.make(id=id_name, render_mode="human")
    discretizer = Discretizer(env, num_parts)
    agent.load(f'{output_dir}/Q_table.json')
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
    from reinforce.agents.QAgent import QLearning, Sarsa
    QClass:QAgent = Sarsa

    output_dir = f'output/cartpole/{QClass.__name__}'
    Path(output_dir).mkdir(exist_ok=True, parents=True)    

    epsilon = Epsilon(start, end, decay)
    agent = QClass(
        state_dim = num_parts ** 4,
        action_dim = 2, epsilon = epsilon,
        gamma = gamma, lr = lr,
    )

    train(output_dir, agent)
    test(output_dir, agent)

# python code/CartPole/run_QAgent.py