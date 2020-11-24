## Leave no Trace: Learning to Reset for Safe and Autonomous Reinforcement Learning

- url: https://arxiv.org/pdf/1711.06782


### Overview
Reinforcement Learning requires repeated attempts and enviroment resets but not all tasks are easily reversible without human intervention.

This work learns a forward and a reset policy. The reset policy resets the environment for a subsequent attempt and determines when the forward policy is about to enter a non-reversible state.


The reset policy is implemented as a Q-function. The algorithm is however implemented using DDPG
