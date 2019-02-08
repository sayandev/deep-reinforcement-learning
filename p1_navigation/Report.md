[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[image2]: https://github.com/sayandev/deep-reinforcement-learning/blob/master/p1_navigation/score.png "Score"

# Project report

### Project's goal

In this project, the goal is to train an agent to navigate a virtual world and collect as many yellow bananas as possible while avoiding blue bananas.

![Trained Agent][image1]

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

Episode 100	Average Score: 0.19
Episode 200	Average Score: 0.93
Episode 300	Average Score: 1.44
Episode 400	Average Score: 2.35
Episode 500	Average Score: 3.55
Episode 600	Average Score: 4.53
Episode 700	Average Score: 5.42
Episode 800	Average Score: 6.37
Episode 900	Average Score: 6.95
Episode 1000	Average Score: 7.60
Episode 1100	Average Score: 7.58
Episode 1200	Average Score: 8.40
Episode 1300	Average Score: 8.81
Episode 1400	Average Score: 9.85
Episode 1500	Average Score: 10.53
Episode 1600	Average Score: 10.00
Episode 1700	Average Score: 11.01
Episode 1800	Average Score: 11.82
Episode 1900	Average Score: 11.28
Episode 2000	Average Score: 11.97
Episode 2100	Average Score: 12.54
Episode 2163	Average Score: 13.04
Environment solved in 2063 episodes!	Average Score: 13.04

### Ideas for future work
- Extensive hyperparameter optimization
- Double DQN
- Dueling DQN
- Prioritized experience replay
- Learning from pixels


