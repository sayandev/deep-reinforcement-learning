[//]: # (Image References)

[image1]: https://github.com/sayandev/deep-reinforcement-learning/blob/master/p2_continuous-control/unity_reacher_ppo_agent.gif "Trained Agent"
[image2]: https://github.com/sayandev/deep-reinforcement-learning/blob/master/p2_continuous-control/score.png "Score"
[image3]: https://github.com/sayandev/deep-reinforcement-learning/blob/master/p2_continuous-control/ddpg_pseudo_algo.png "Pseudo Algo"

# Project report

### Project's goal

In this environment called Reacher, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal the agent is to maintain its position at the target location for as many time steps as possible. Additional information can be found [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher).

![Trained Agent][image1]

The observation space consists of Each action is a vector with four numbers,

- *State space* => 33 dimensional continuous vector, consisting of position, rotation, velocity, and angular velocities of the arm.
- *Action space* => 4 dimentional continuous vector, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.
- *Solution criteria* => the environment is considered as solved when the agent gets an average score of +30 over 100 consecutive episodes (averaged over all agents in case of multiagent environment).
