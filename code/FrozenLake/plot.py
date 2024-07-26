import matplotlib.pyplot as plt
from typing import List

def plot_rewards_curve(ep_rewards: List[float], save_path:str) -> None:
    plt.figure(figsize=(10, 5))
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.plot(ep_rewards, label='rewards')
    plt.legend()
    plt.title('rewards curve')
    plt.savefig(save_path)
    print(f"Rewards curve saved at {save_path}")
