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

!Pseudo Algo][image4]

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

- BUFFER_SIZE = int(1e5)  # replay buffer size
- BATCH_SIZE = 64         # minibatch size
- GAMMA = 0.99            # discount factor
- TAU = 1e-3              # for soft update of target parameters
- LR = 5e-4               # learning rate 
- UPDATE_EVERY = 4        # how often to update the network
- Maximum steps per episode: 1000
- Starting epsilion: 1.0
- Ending epsilion: 0.01
- Epsilion decay rate: 0.999

### Results

![Score][image2]
![Score Report][image3]



### Ideas for future work
- Extensive hyperparameter optimization
- Double DQN
- Dueling DQN
- Prioritized experience replay
- Learning from pixels


