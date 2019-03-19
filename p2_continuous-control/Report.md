[//]: # (Image References)

[image1]: https://github.com/sayandev/deep-reinforcement-learning/blob/master/p2_continuous-control/unity_reacher_ppo_agent.gif "Trained Agent"
[image2]: https://github.com/sayandev/deep-reinforcement-learning/blob/master/p2_continuous-control/score.png "Score"
[image3]: https://github.com/sayandev/deep-reinforcement-learning/blob/master/p2_continuous-control/ddpg_pseudo_algo.png "Pseudo Algo"
[image4]: https://github.com/sayandev/deep-reinforcement-learning/blob/master/p2_continuous-control/ddpg_critic_loss.png "ddpg_critic_loss"
[image5]: https://github.com/sayandev/deep-reinforcement-learning/blob/master/p2_continuous-control/ddpg_actor_loss.png "ddpg_actor_loss"
[image5]: https://github.com/sayandev/deep-reinforcement-learning/blob/master/p2_continuous-control/ddpg_actor_loss.png "ddpg_actor_loss"
[image6]: https://github.com/sayandev/deep-reinforcement-learning/blob/master/p2_continuous-control/TrainedAgentUnityScreenshot.png "TrainedAgentUnityScreenshot"

# Project report

### Project's goal

In this environment called Reacher, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal the agent is to maintain its position at the target location for as many time steps as possible. Additional information can be found [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher).

![Trained Agent][image1]

The observation space consists of Each action is a vector with four numbers:

- *State space* => 33 dimensional continuous vector, consisting of position, rotation, velocity, and angular velocities of the arm.
- *Action space* => 4 dimentional continuous vector, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.
- *Solution criteria* => the environment is considered as solved when the agent gets an average score of +30 over 100 consecutive episodes (averaged over all agents in case of multiagent environment).


### Algorithm

This project is an extension of the earlier project in applying [Deep Q Network](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) to solve single agent navigation envuiornment, however this project has a more complex enviornment with continious action spaces and involved multiple agents. [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971) been used to solve this project utilzing some [refiniment](https://blogs.unity3d.com/2018/09/11/ml-agents-toolkit-v0-5-new-resources-for-ai-researchers-available-now/). 

DDPG algorithm is summarized below: 

![Pseudo Algo][image3]

DDPG is a model free policy based learning algorithm in which the agent can learn competitive policies for all of the tasks using low-dimensional observations (e.g. cartesian coordinates or joint angles) using the same hyper-parameters and network structure without knowing the domain dynamic information. Thus the same algorithm can be applied across domains which is quite an accomplishmet with respect to traditional planning algorithms. While DQN solves problems with high-dimensional observation spaces, it can only handle discrete and low-dimensional action spaces. Many tasks of interest, most notably physical control tasks, have continuous (real valued) and high dimensional action spaces. DQN cannot be straightforwardly applied to continuous domains since it relies on a finding the action that maximizes the action-value function, which in the continuous valued case requires an iterative optimization process at every step. DDPG combine the actor-critic approach with insights from the recent success of [Deep Q Network](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf); where the **critic** learns the value function alike DQN and uses it to determine the Actor's policy based model should change. Critic use neural network for Q-value function approximation as state -> action mapping with the following loss function minimised:
![ddpg_critic_loss][image4]

The actor contributes in continious action space without need for extra layer of optimization procedures require in value based function while the critic provides the actor with the performance metrics. Actor use neural network for determenistic policy approximation as state -> argmax_Q mapping with the following loss function minimised:
![ddpg_actor_loss][image5]


### Model Architecture

20 individual DDPG agents corresponding to 20 agents in the enviornment and a single replay buffer shared by all the 20 DDPG agents. Later switched to a single DDPG agent with one replay buffer that has experiences collected from all 20 DDPG agents. Which results is to train a single brain rather than 20 brains which will be definitely quicker and having a single source of decision making will result in better performance with time. This scheme results in robust and better performance. However, in distributed enviornemnt training the brains sepeartely might be more sclabale.

**Actor model :** Neural network with 2 hidden layers with 400 and 300 hidden nodes respectively; _tanh_ activation function been used in the last layer which performs state -> action mapping. Batch Normalization regularization technique been used for mini batch training.

**Critic model :** Similar to Actor model except the final layer is fully connected hich performs state -> argmax_Q  mapping.

**Hyper Parameter :** The actual configuration of the hyperparameters is:

- Learning Rate: 1e-3 (in both Actor & Critic) soft update of target
- Batch Size: 1024 with max time step of 1000 in each episode.
- Replay Buffer: 1e6
- Gamma: 0.99 with weight decay set to 0

### Project Implemenation

- Continuous_Control_Solution.ipynb - main file where the program execution starts.
- ddpg_agent.py - agent class implementation.
- models.py - neural network implementations (PyTorch)

#### Result

The DDPG agent took 102 Episodes to achieve the average rewards of 30.13 in approximately 1 hour 45 minutes. 
![Score][image2]

In the screenshot of the trained DDPG agents we can see thay are pretty well in tracing the moving target location for all 20 arms.

![TrainedAgentUnityScreenshot][image6] 


### Future Work

- Use a deeper network.
- Tst the implementation using raw pixels.
- Use some form of curiosity to [favorize exploration](https://arxiv.org/abs/1808.04355)
- Trying out Proximal Policy Optimisation
- The DDPG algorithm can be improved and robust by applying [Prioritized experience replay](https://ieeexplore.ieee.org/document/8122622).
