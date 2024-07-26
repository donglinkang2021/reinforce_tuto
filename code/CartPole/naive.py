import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="human") # for train, rgb_array
observation, info = env.reset()

for _ in range(1000):
    position, velocity, angle, angular_velocity = observation
    action = 1 if angle + angular_velocity > 0 else 0
    # action = env.action_space.sample()  
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
        print(f"terminated: {terminated}, truncated: {truncated}")

env.close()