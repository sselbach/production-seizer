
# Solving the Halite Challenge with a Deep-Q network


## Introduction

For our final project in the course of “Implementing ANN with TensorFlow” we decided to revive the challenge proposed by TWO SIGMA in 2016. The challenge was composed of the provided game “[Halite](https://2016.halite.io/index.html)” and its corresponding rule set. The competition then started in November 2016 and finalized 3 month later, in February 2017. The send in application were a mix of different approaches, consisting of C#, python, java scripts etc. But also some Machine Learning based bots. In our growing interest in self learning systems we formulated our idea to challenge the competition with an ANN based bot. In this report we will show our different approaches, our successful as well as our not so successful ones.

### The Rules

Halite is played on a rectangular grid, where the size correlates with the number of players (2-6). The goal is to seize the means of production. Each player starts on a specific tile of the grid, all other tiles are considered unowned. The tiles come in different qualities. A tile produces or enhances one drone per turn with the strength of its quality value. Drones have the ability to move in cityblock distance or to stay at their momentary tile. An unowned tile can be converted into an owned one by being occupied by a drone. The strength of the drones is capped to the value of 255. Tiles owned by foreign drones can be conquered by overwhelming through higher strength. The strength values get subtracted. When all enemy drones and tiles are conquered, the game ends. The map is continuous, go to far left and you will appear on the right (bottom and top the same).

![](halite_gif.gif)

## Our Goal

The Goal of our final project was to train a Reinforcement Learning Deep Q Network to play this game in a sufficient efficient way.

### Whats a Deep-Q network (DQN)?

A DQN is a reinforcment learning(RL) based convolutional neural network(CNN). RL is about training an agent to interact with its environment to achieve a certain goal. To achieve said goals the agent has to decide on an action <img src="https://render.githubusercontent.com/render/math?math=a"> which then leads to certain states <img src="https://render.githubusercontent.com/render/math?math=s">. These actions can impact the reward in a positive or negative way. The agent's purpose is, to maximize the reward in each episode. An episode is anything between the first state and the terminal state. We reinforce the agent to learn to perform the most rewarding action by experience. How rewarding an action can be is not obvious in most scenarios. Therefore a Markov decision process is initiated to save every action to each state. To allocate a reward <img src="https://render.githubusercontent.com/render/math?math=Q"> to a state we use the Q-function: <img src="https://render.githubusercontent.com/render/math?math=Q(s,a)= r(s,a) + \gamma  max  Q(s',a)">
Gamma here is the discount factor which controls the contribution of rewards further in the future. Wheras <img src="https://render.githubusercontent.com/render/math?math=s'"> is the future state. We select an action using the epsilon-greedy policy. With the probability epsilon, we select a random action a and with probability 1-epsilon, we select an action that has a maximum Q-value, such as <img src="https://render.githubusercontent.com/render/math?math=a = argmax(Q(s,a,w))">. We perform this action and move to the next state <img src="https://render.githubusercontent.com/render/math?math=s'">, while also storing this choice in our replay buffer as <img src="https://render.githubusercontent.com/render/math?math=(s,a,r,s')">. In deep Q-learning, we use a neural network to approximate the Q-value function. The state is given as the input and the Q-value of all possible actions is generated as the output. The Loss is just the squared difference between target Q and predicted Q (mean-squared-error). Perform gradient descent with respect to our actual network parameters in order to minimize this loss.  Repeat these steps for M number of episodes.



## Our Approach
<img src="https://render.githubusercontent.com/render/math?math=\gamma Q(s',a) = -1">
<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">

### Groundwork
