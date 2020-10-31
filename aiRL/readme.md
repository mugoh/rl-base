# Adversarial Inverse Reinforcement Learning [ AIRL ]

### Implementation of the Paper [Learning Robust Rewards with Adversarial inverse Reinforcement Learning](https://arxiv.org/abs/1710.11248)

This implementation is attached to this [ blog post ](https://mugoh.github.io/mug-log/adversarial-inverse-rl/) - for a detailed guide on the implementation of this work

#### Selected tranining details and hyperparameters used
- Policy samples mixed from previous 20 iterations


 *From the paper*:

  > IRL methods commonly learn rewards which explain behavior locally for the current policy, because the reward can ”forget” the signal that it gave to an earlier policy.
  >
  > This makes rewards obtained at the end of training difficult to optimize from scratch, as they overfit to samples from the current iteration.
  >
  > To somewhat migitate this effect, we mix policy samples from the previous 20 iterations of training as negatives when training the discriminator.


- Entropy regularizer of 0.1

    The policy gradient algorithm used in both expert data collection and inverse Reiforcement Learning(iRL) is PPO.
    With aiRL, [Soft PPO](https://arxiv.org/abs/1912.01557) is used, which maximizes on the entropy during policy updates. This encourages exploratory behaviour.
    The entropy regularizer is a scalar that determines how much entropy is incorporated in a policy update


- Function Approximators
  - *g*: Linear function approximator
  - *h*: 2 layer ReLu
  - **pi**: 2 layer ReLu Gaussian policy
- All approximators have 32 units


### 1. Setup 
This work is build on Python3.6

1. Clone the repository `git clone https://github.com/mugoh/rl-base.git`
2. Install the project depencies

 a) Using Pip

  - [Create and activate a virtual env](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#installing-virtualenv)
  - Install the dependencies `pip install -r requirements.txt`


 b) Using conda
 
  - [Setup and activate](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
  - Install requirements 

      ```
        while read requirement; do conda install --yes $requirement || pip install $requirement; done < requirements.txt
      ```



### 2.  Running adversarial inverse RL

#### 1. Collect the expert demonstrations.
 **The `expert_policy/` PPO implementation is devoted to help with this**

 ##### a) Using already collected demos
   There is a file of already saved expert demonstrations for the `HalfCheetah-v2 `env. Download it from [ here ](https://drive.google.com/file/d/1eZa6uXpJhmzKyChrI-zCTHGsI62Xy7VL/view?usp=drivesdk)


 ##### b) Alternatively, collect your own demostrations 

   - To collect the demonstation with the default hyperparameters and settings, run:

     ```$
        python3 expert_policy/collect_data.py
     ```

   This will collect data from the final 25 epochs of the expert data policy


  - For a list of tunable options and hyperparams, see the help menu by running:

      ```$
          python3 expert_policy/collect_data.py --help
      ```


   > The above command will collect and save expert demonstrations in your current directory. The file will have the naming format

   ```$
       expert_data_{gym_env_name}_{YMD:HMS timestamp}_{number_of_demo_trajectories_saved}[.npz]
   ```



 ---



#### 2. Runnning inverse RL

 - The airl implementation entry script is `main.py`.

  - All algorithm hyperparams, Neural Network architecutures(not accessible via the CLI) can be hand-modfied in that script


   **The Default network architechtures** in `main.py` are:

   Actor and critic

 ```python3
     # ac: policy & value function

     ac_args = {'hidden_size': [64, 64], 'size': 2}
 ```

   Discriminator

  ```python3
 
    # Discriminator approximators g() and h()

    disc_args = {
        'g_args': {
            'hidden_layers': [32, 1],
            'size': 1,
            'activation': nn.Identity
        },
        'h_args': {
            'hidden_layers': [32, 32, 1],
            'size': 2,
            'activation': nn.LeakyReLU
        }
    }

 ```
 - To do inverse RL on the default hyperparams and algoritm settings, just run the command:
     ```
        python3 main.py -demos path_to_saved_expert_demos.npz
     ```
     _The expert demos must be in numpy format and accessible with `np.loadz`_

 - It's important to ensure the `--env` option used in expert data collection is the same used for iRL.

    It's possible to make a mistake and use the demonstations on a different env from that which they were collected


#### 3. Visualizing outputs from runs

- Tensorboard logs are saved in `airl/data/` and can be viewed by starting a tensorboard server

    `tensorboard --logdir=data/`

 ##### 3.1 Examples of program outputs
  - Expert Policy

   ![ expert demo performance](https://github.com/mugoh/rl-base/blob/master/aiRL/halfcheetah%20data/iRL_expert_demo.png) 

  - Expert Policy vs iRL based policy

  ![iRL vs stdPPO](https://github.com/mugoh/rl-base/blob/master/aiRL/halfcheetah%20data/Figure_1_irl_vs_stdPPO.png)

    Your results may vary

   Here's an example of a different aiRL run

  - (b) Expert Policy vs iRL based policy

   ![iRL_learned_vs_demo_returns](https://github.com/mugoh/rl-base/blob/master/aiRL/halfcheetah%20data/iRL_learned_vs_demo_returns.png)



