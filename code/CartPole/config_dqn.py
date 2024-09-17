id_name = 'CartPole-v1'

# model
n_states = 4
n_actions = 2
n_hidden = 64

# agent
gamma = 0.95
lr = 0.01

# epsilon
start = 0.95
end = 0.01
decay = 100

# train
max_episodes = 1500
batch_size = 32

# replay buffer
buffer_size = 50000