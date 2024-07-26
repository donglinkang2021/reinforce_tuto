import gymnasium as gym
env = gym.make("MountainCar-v0", render_mode="human") # for train, rgb_array
observation, info = env.reset()

for _ in range(1000):
    position, velocity = observation
    action = 2 if velocity > 0 else 0
    # action = env.action_space.sample()  
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()