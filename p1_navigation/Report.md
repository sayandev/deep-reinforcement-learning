[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[image2]: https://github.com/sayandev/deep-reinforcement-learning/blob/master/p1_navigation/score.png "Score"
[image3]: https://github.com/sayandev/deep-reinforcement-learning/blob/master/p1_navigation/score_report.PNG "Score Report"
[image4]: https://github.com/sayandev/deep-reinforcement-learning/blob/master/p1_navigation/DQN_pseudo_algo.png "Pseudo Algo"

# Project report

### Project's goal

In this project, the goal is to train an agent to navigate a virtual world and collect as many yellow bananas as possible while avoiding blue bananas.

![Trained Agent][image1]


### Algorithm

In this project the task is solved utilizing a Value Based method; Deep Q-Networks. Deep Q Learning consist of 2 approaches :
- A Reinforcement Learning method called [Q Learning](https://en.wikipedia.org/wiki/Q-learning) (SARSA max)
- A Deep Neural Network to learn a [Q-table](https://www.youtube.com/watch?time_continue=94&v=WQgdnzzhSLM) approximation (action-values)

This includes the two following major training improvements by Deepmind and described in their Nature publication : ["Human-level control through deep reinforcement learning (2015)"](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)

- Experience Replay
- Fixed Q Targets

The steps of the algorithm (screenshot is taken from the [Deep Reinforcement Learning Nanodegree course](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)) are listed below.

![Pseudo Algo][image4]

### Learning algorithm

The learning algorithm is implemented following the vanilla Deep Q Learning as described in published paper. As an input the vector of state is used instead of an image so convolutional neural nework is replaced with deep neural network. The deep neural network has following layers:

- Fully connected layer - input: 37 (State Size) output: 128
- Fully connected layer - input: 128 output 64
- Fully connected layer - input: 64 output: (Action Size)

### Code implementation
The code used here is derived from the tutorial from the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893), and has been slightly adjusted for being used with the banana environment.

The code consist of :

- navigation_solution.ipynb: This Jupyter notebooks allows to train the agent and test the performace. 
- model_qnet.py: In this python file, a PyTorch QNetwork class is implemented. 
- agent_dqn.py: In this python file, a DQN agent and a Replay Buffer memory used by the DQN agent is build.


### DQN parameters and results

The Neural Networks use the Adam optimizer and the following parameters values:

- BUFFER_SIZE = int(1e5)  # replay buffer size
- BATCH_SIZE = 64         # minibatch size
- GAMMA = 0.99            # discount factor
- TAU = 1e-3              # for soft update of target parameters
- LR = 5e-4               # learning rate 
- UPDATE_EVERY = 4        # how often to update the network
- Maximum steps per episode: 1000

Agent select next action based on Epsilon Greedy. At probability epsilon, agent select at random from action space. The value of epsilon is=>
- Starting epsilion: 1.0
- Ending epsilion: 0.01
- Epsilion decay rate: 0.999


### Results

Given the chosen architecture and parameters, the results :

![Score][image2]
![Score Report][image3]



### Ideas for future work

- Learning from pixels

In the [Deep Reinforcement Learning Nanodegree course](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893), a possible extension of this project been suggested, where the agent will be trained directly from the environment's observed raw pixels instead of using the environment's internal states (37 dimensions). To achieve that a Convolutional Neural Network would be added at the input of the network in order to process the raw pixels values following image preprocessing such as rescaling the image size, converting RGB to gray scale etc.

- Double DQN

The vanila flavour Q-learning algorithm is known to overestimate action values under certain conditions. It was not previously known whether, in practice, such overestimations are common, whether they harm performance, and whether they can generally be prevented. In [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) paper, the authors answer all these questions affirmatively. In particular, they first show that the recent DQN algorithm, which combines Q-learning with a deep neural network, suffers from substantial overestimations in some games in the Atari 2600 domain. Then it has been shown that the idea behind the Double Q-learning algorithm, which was introduced in a tabular setting, can be generalized to work with large-scale function approximation. The authors' propose a specific adaptation to the DQN algorithm and show that the resulting algorithm not only reduces the observed overestimations, as hypothesized, but that this also leads to much better performance on several games.

- Dueling DQN

In recent years there have been many successes of using deep representations in reinforcement learning. Still, many of these applications use conventional architectures, such as convolutional networks, LSTMs, or auto-encoders. In [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) paper, the authors presented a new neural network architecture for model-free reinforcement learning. The proposed dueling network represents two separate estimators: one for the state value function and one the other for the state-dependent action advantage function. The main benefit of this factoring is to generalize learning across actions without imposing any change to the underlying reinforcement learning algorithm. The results show that this architecture leads to better policy evaluation in the presence of many similar-valued actions. Moreover, the dueling architecture enables the proposed RL agent to outperform the state-of-the-art on the Atari 2600 domain.

- Prioritized experience replay

Experience replay lets online reinforcement learning agents remember and reuse experiences from the past. In prior work, experience transitions were uniformly sampled from a replay memory. However, [Prioritized Experience Replay
](https://arxiv.org/abs/1511.05952) approach simply replays transitions at the same frequency that they were originally experienced, regardless of their significance. In [Prioritized Experience Replay
](https://arxiv.org/abs/1511.05952) paper the authors develop a framework for prioritizing experience, so as to replay important transitions more frequently, and therefore learn more efficiently. The authors use prioritized experience replay in Deep Q-Networks (DQN), a reinforcement learning algorithm that achieved human-level performance across many Atari games. DQN with prioritized experience replay achieves a new state-of-the-art, outperforming DQN with uniform replay on 41 out of 49 games.


- [Extensive hyperparameter optimization](https://medium.com/@mikkokotila/a-comprehensive-list-of-hyperparameter-optimization-tuning-solutions-88e067f19d9)

Grid search of hyper-parameter, Q-value model.



