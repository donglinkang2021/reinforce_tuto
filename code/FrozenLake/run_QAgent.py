import gymnasium as gym
from utils import generate_random_map
from config import *
from reinforce.agents.QAgent import QLearning, Sarsa, QAgent
from reinforce.utils.plot import plot_rewards
from tqdm import tqdm
from pathlib import Path
map_desc = generate_random_map(
    size = map_size, 
    p = frozen_prob, 
    seed = map_seed
)

# train
def train(output_dir:str, agent:QAgent):
    env = gym.make(
        id = id_name, 
        desc = map_desc,
        is_slippery = is_slippery, 
        render_mode = 'ansi' # for non-interactive mode, training
    )
    agent = QLearning(
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
            if isinstance(agent, QLearning):
                action = agent.predict(state)
            else:
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
        pbar.set_postfix(Epoch=epoch, Reward=ep_reward)
        pbar.update(1)
    pbar.close()
    env.close()
    agent.save(f'{output_dir}/Q_table.json')
    plot_rewards(ep_rewards, f'{output_dir}/rewards_curve.png')

# test
def test(output_dir:str, agent:QAgent):
    env = gym.make(
        id = id_name, 
        desc = map_desc,
        is_slippery = is_slippery, 
        render_mode = 'human' # for interactive mode, testing
    )
    agent.load(f'{output_dir}/Q_table.json')
    state, info = env.reset()
    while True:
        action = agent.predict(state) 
        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state
        if terminated:
            break
    env.close()


if __name__ == '__main__':
    from pathlib import Path
    QClass:QAgent = Sarsa
    # QClass:QAgent = QLearning

    output_dir = f'output/frozenlake/{QClass.__name__}'
    Path(output_dir).mkdir(exist_ok=True, parents=True)    

    from reinforce.utils.epsilon import Epsilon
    epsilon = Epsilon()
    agent = QClass(
        state_dim = map_size * map_size,
        action_dim = 4, epsilon = epsilon,
        gamma = gamma, lr = lr,
    )

    # train(output_dir, agent)
    test(output_dir, agent)

# python code/FrozenLake/run_QAgent.py