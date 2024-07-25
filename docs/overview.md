# Overview

> Snippet

- Q-Learning Fomula

```python
q[state, action] = q[state, action] + alpha * (reward + gamma * np.max(q[new_state, :]) - q[state, action])
```

- alpha: learning rate
- gamma: discount factor

- DQL Fomula

```python
q[state, action] = reward if new_state is TERMINAL else reward + gamma * np.max(q[new_state, :])
```