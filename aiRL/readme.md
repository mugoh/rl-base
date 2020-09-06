# AIRL

#### Implementation of the Paper [Learning Robust Rewards with Adversarial inverse Reinforcement Learning](https://arxiv.org/abs/1710.11248)

##### Tranining details and hyperparameters used in paper
- Policy samples mixed from previous 20 iterations
    > *From the paper*:
    > IRL methods commonly learn rewards which explain behavior locally for the current policy, because the reward can ”forget” the signal that it gave to an earlier policy.
    > This makes rewards obtained at the end of training difficult to optimize from scratch, as they overfit to samples from the current iteration. To somewhat migitate this effect, we mix policy samples from the previous 20 iterations
of training as negatives when training the discriminator.
- Entropy regularizer of 0.1
- Function Approximators
 - *g*: Linear function approximator
 - *h*: 2 layer ReLu
 - **pi**: 2 layer ReLu Gaussian policy
- All approximators have 32 units

