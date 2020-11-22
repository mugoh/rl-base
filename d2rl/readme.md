# D2RL: DEEP DENSE ARCHITECTURES IN REINFORCEMENT LEARNING

- https://arxiv.org/abs/2010.09163

> findings reveal that current methods **benefit significantly from dense connections and deeper networks**, across a suite of manipulation and locomotion tasks, for both proprioceptive and image-based observations.


The dense network is created by concatenating the output of each layer with
the input to the network. Here is an intuitive example taken from the paper.

```python

# Sample state, action from the replay buffer
    state, action = replay_buffer.sample()
# Feed state, action into the first linear layer of a Q-network
    q_input = concatenate(state, action)
    h = MLP(q_input)
# Concatenate the hidden representation with the input
    h = concatenate(h, q_input)
# Feed the concatenated representation into the second layer
    h = MLP(h)
```

