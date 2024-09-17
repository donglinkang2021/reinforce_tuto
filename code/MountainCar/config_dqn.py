id_name = 'MountainCar-v0'

# model
n_states = 2
n_actions = 3
n_hidden = 64

# agent
gamma = 0.95
lr = 0.01

# epsilon
start = 1
end = 0.01
decay = 1000

# train
max_episodes = 1000
batch_size = 32

# replay buffer
buffer_size = 5000