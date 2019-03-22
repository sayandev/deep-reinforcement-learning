[//]: # (Image References)

[image1]: https://github.com/sayandev/deep-reinforcement-learning/blob/master/p3_collab-compet/tennis.gif "Trained Agent"
[image2]: https://github.com/sayandev/deep-reinforcement-learning/blob/master/p3_collab-compet/score.png "Score"
[image3]: https://github.com/sayandev/deep-reinforcement-learning/blob/master/p3_collab-compet/ddpg_pseudo_algo.png "Pseudo Algo"
[image4]: https://github.com/sayandev/deep-reinforcement-learning/blob/master/p3_collab-compet/ddpg_critic_loss.png "ddpg_critic_loss"
[image5]: https://github.com/sayandev/deep-reinforcement-learning/blob/master/p3_collab-compet/ddpg_actor_loss.png "ddpg_actor_loss"
[image6]: https://github.com/sayandev/deep-reinforcement-learning/blob/master/p3_collab-compet/TrainedAgent1.png "TrainedAgentUnityScreenshot1"
[image7]: https://github.com/sayandev/deep-reinforcement-learning/blob/master/p3_collab-compet/TrainedAgent2.png "TrainedAgentUnityScreenshot2"
[image8]: https://github.com/sayandev/deep-reinforcement-learning/blob/master/p3_collab-compet/TrainedAgent3.png "TrainedAgentUnityScreenshot3"

# Project report

### Project's goal
In this environment, two agents control rackets collabortate and compete to play tennis. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play, as such playing as long as possible in an episode. Additional information can be found [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis).

![Trained Agent][image1]

The observation space consists of each action is a vector with eight variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, both agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).


### Algorithm

This project is an extension of the earlier project implemented in this course of [Learning Continious Control in Deep Reinforcement Learning](https://github.com/sayandev/deep-reinforcement-learning/blob/master/p2_continuous-control/Report.md), however this project has a more complex competitive environment involving two tennis players competing with each other. [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971) been used to solve this project utilizing  some [refinements](https://blogs.unity3d.com/2018/09/11/ml-agents-toolkit-v0-5-new-resources-for-ai-researchers-available-now/). 


Deep Deterministic Policy Gradient (DDPG) algorithm is summarized below: 

![Pseudo Algo][image3]

DDPG is a model free policy based learning algorithm in which the agent can learn competitive policies for all of the tasks using low-dimensional observations (e.g. Cartesian coordinates or joint angles) using the same hyper-parameters and network structure without knowing the domain dynamics information. Thus the same algorithm can be applied across domains which is quite an accomplishment with respect to traditional planning algorithms. While DQN solves problems with high-dimensional observation spaces, it can only handle discrete and low-dimensional action spaces. Many tasks of interest, most notably physical control tasks, have continuous (real valued) and high dimensional action spaces. DQN cannot be straightforwardly applied to continuous domains since it relies on a finding the action that maximizes the action-value function, which in the continuous valued case requires an iterative optimization process at every step. DDPG combine the actor-critic approach with insights from the recent success of [Deep Q Network](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf); where the **critic** learns the value function alike DQN and uses it to determine the Actor's policy based model should change. Critic use neural network for Q-value function approximation as state -> action mapping by minimizing the following loss function:
![ddpg_critic_loss][image4]

The actor contributes in continuous action space without need for extra layer of optimization procedures require in value based function while the critic provides the actor with the performance metrics. Actor use neural network for deterministic policy approximation as state -> argmax_Q mapping by minimizing the following loss function:
![ddpg_actor_loss][image5]


### Model Architecture

Following the [competitive enviornment](https://deepmind.com/blog/alphago-zero-learning-scratch/) used in the application of Deep Reinforcement Learning the DDPG agents adopted from the [previous project](https://github.com/sayandev/deep-reinforcement-learning/blob/master/p2_continuous-control/Report.md) is incorporated in the competitive Tennis environment of 2 players using a single Brain shared by the DDPG agent. The DDPG agent collect experiences from both Tennis players with the sahred replay buffer. The Neural network archityecture and hyper paramaters been modified for efficient training phase explained below. 

**Actor model:** Neural network with 2 hidden layers with 512 and 256 hidden nodes respectively; _tanh_ activation function been used in the last layer which performs state -> action mapping. Batch Normalization regularization technique been used for mini batch training.

**Critic model:** Similar to Actor model except the final layer is fully connected which performs state -> argmax_Q  mapping. To avoid overfitting and making the learning proces efficient a drop out layer (with 0.2 probability) been added before the output layer.

**Hyper Parameter :** The actual configuration of the hyperparameters is:

- Learning Rate: Actor=> 1e-4 & Critic=> 3e-4 with soft update of target => 2e-1.
- Batch Size: 512 with max time step of 2000 in each episode.
- Replay Buffer: 1e5
- Gamma: 0.99 with weight decay set to 0

### Project Implemenation

- Continuous_Control_Solution.ipynb - This Jupyter notebooks allows to train the agent and test the performace.
- ddpg_agent.py - agent class implementation.
- models.py - neural network implementations (PyTorch).

#### Result

The DDPG agent took 564 Episodes to achieve the average max rewards of 0.5 score in approximately 45 minutes using the GPU instance provided in the Udacity course workspace. As ecident in the plot scores fluctuate withon 300 and 500 episode however the average Max score increses gradually. Once the agent adapted the new technique the score increased significatnty after 500 episodes and achieve the goal quickly.

![Score][image2]

In the screenshot of the trained DDPG agents we can see thay performing pretty well in the competitive environment.

![TrainedAgent1][image6] 

![TrainedAgent2][image7] 

![TrainedAgent3][image8] 


### Ideas for Future Work

- Use a deeper network.
- Incorporate some form of curiosity to [favorize exploration](https://arxiv.org/abs/1808.04355).
- Trying out [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) as it performs comparably if not better than state-of-the-art approaches while being much simpler to implement and tune which may converge faster and shall be able to handle more complicated multi-agent in a more complex environment tasks.
- The DDPG algorithm can be improved (robust) by applying [Prioritized experience replay](https://ieeexplore.ieee.org/document/8122622) using a special data structure [Sum Tree](https://github.com/rlcode/per)
