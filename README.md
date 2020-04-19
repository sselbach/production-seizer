# Solving the Halite Challenge with a Deep-Q Network


## Introduction

For our final project in the course “Implementing ANNs with TensorFlow” we decided to revive a challenge posed by TWO SIGMA in 2016. The challenge consisted of the game “[Halite](https://2016.halite.io/index.html)”, a game specifically made to be played by bots. The competition then started in November 2016 and finalized 3 month later, in February 2017. The sent in solutions were mostly traditional "old-fashioned" AI programs using rule sets and search algorithms. However, also some Machine Learning based bots participated with some success. In our growing interest in self learning systems we set out to try to challenge the competition with an ANN based bot, or at the very least learn basic tactics of the game through reinforcement learning. In this report we will show our different approaches, our successful as well as our not so successful ones.

### The Rules

Halite is played on a rectangular, toroidal grid, where the size correlates with the number of players (2-6). Each player starts on a specific tile of the grid, all other tiles are considered unowned. The starting situation is symmetric, i.e. no player has an advantage at the start. Tiles come in different qualities. A tile produces one drone per turn with the *strength* of its *production* value. Drones have the ability to move one step in a cardinal direction or to stay still at their momentary tile. Turns happen simultaneously, similar to the game *Diplomacy*. If a drone chooses to remain in place, its strength grows by the production value of the tile it is on. If two or more allied drones move onto the same tile, they merge into a single, stronger drone with the combined strength value. Strength is *capped* at 255. If a drone moves onto or next to a tile not currently owned by the player, a fight ensues. The strength of participating parties get subtracted, and the winner gains control of the tile. For a more precise statement of the fighting rules, please see the official [Halite rules](https://2016.halite.io/rules_game.html). The goal is to seize control of the entire grid, or if that fails to control the most territory after a predefined number of turns.

[Replay of a game of Halite played by some of the top bots](https://2016.halite.io/game.html?replay=ar1487297318-1684222918.hlt)

Use arrow keys and space bar to control the flow of the replay

<iframe src="https://2016.halite.io/game.html?replay=ar1487297318-1684222918.hlt" width="800" height="600">

</iframe>


## Background
We decided to take the classic approach of Deep Q Learning as first presented in the paper by *DeepMind*, *[Playing Atari games with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)*.

In their paper, Mnih et al. used a Deep-Q neural network to play different Atari games like: Pong, Breakout, Space Invaders, Seaquest and Beam Rider, using an almost raw video feed from the screen as input. The agent was able to learn to play these games despite the huge state space, which we thought was a good feature to have for playing halite. The results they got were more than okay. In six of the seven tried games they outperformed other bot approaches. Even though Humans still played better (except for Seaquest). But what is a DQN?

### What is a Deep-Q network (DQN)?


A DQN is a deep neural network used in a Q learning agent to approximate its Q function. 

$Q(s, a)$ tells the agent what *expected cumulative future reward* it will get by performing action $a$ in state $s$. In regular Q learning, the agent will first perform more or less random actions and receive a certain reward for that. After performing an action, it immediately updates the corresponding $Q(s, a)$ entry in its *Q table* according to the following formula:

$$ Q(s, a) \leftarrow r(s, a) + \gamma \; Q(s', a) $$

where $r(s, a)$ is the *immediate reward* given by the environment and $\gamma \; Q(s', a)$ is the discounted expected future reward.

The problem with using a *table* for the Q function is however that the state space can potentially be huge, making only ever updating single values impractical. The solution is of course to use a neural network as a function approximator instead, but this adds another level of complexity: We cannot just *make* a neural network output a desired value, it has to be trained.

To weave the training process into the reinforcement learning loop, enter the *replay buffer*. Instead of updating the Q function directly after taking an action, a $(state, \;action, \;reward, \;state')$ tuple memorized in a large buffer. Then, after each RL training step a random mini-batch of experiences is drawn from the buffer and a single network optimization step is performed, using standard techniques like Adam or SGD.

What is left to discuss is how the agent selects which actions to take, as there is a tradeoff to make: Always performing random actions makes it incredibly unlikely to get to advanced states in the environment, for which then nothing will be learned. On the other hand, always performing the action the agent thinks is best in a given state has the potential to get it stuck in a local optimum while interesting strategies may remain untried. The solution is to a parameter $\epsilon$ that determines the "greediness" of the agent. Usually, it will decay over time but will never get too close to 0 either.

## Our Approach

While DQN solves the problem of the large state space, Halite also has a huge action space (5 actions per drone for potentially hundreds of drones). To work around this issue, we chose to make the simplifying assumption that the drones can make good decisions independent of one another. With that, we can have a *state* be a window centered around the drone that is making a decision. Consequently, for a full *turn* we need to evaluate the network once for each drone, and we also get one $(s, \; a, \; r, \; s')$ tuple per drone per turn.

Another possible approach would have been to just use an RL approach that can deal with big/continuous action spaces like some *Actor Critic* methods, but that would have been beyond the scope of this project.

### Code Structure
We used the starter package from the [Halite Challenge](https://2016.halite.io/downloads.html) to get going. This provides us with a game environment$^*$ and templates for the competing bots. The bots are able to communicate with the game environment via stdout. When playing they can get the game map from the environment and may send a list of moves back for the squares that they own in the game map which the game environment incorporates and uses to update the map. 
All code can be found on github in  the [working_branch](https://github.com/sselbach/production-seizer) branch.

$^*$ *See Appendix for more information on technical struggles with this*
### Overview of important modules:

Name | Function
------ | ------
[rl_bot.py](https://github.com/sselbach/production-seizer/blob/working_branch/rl_bot.py) | Interface between the game environment and the DQN.
[window.py](https://github.com/sselbach/production-seizer/blob/working_branch/window.py) | Handles the near surroundings (windows) for the single owned squares of the bot. Used as input for DQN.
[reward.py](https://github.com/sselbach/production-seizer/blob/working_branch/reward.py) | Contains variations of reward functions used for training the DQN.
[replay_buffer.py](https://github.com/sselbach/production-seizer/blob/working_branch/replay_buffer.py)| Saving trajectories for training the DQN with experience replay.
[dqn.py](https://github.com/sselbach/production-seizer/blob/working_branch/dqn.py)| Contains the definition of the network architecture of our DQN and the training procedure. It also handles loading and saving of models.
[hyperparameters.py](https://github.com/sselbach/production-seizer/blob/working_branch/hyperparameters.py)| Specifies all the hyperparameters used, general ones as well as ones specific for training eg. model saving directory and learning rate.
[trainings_manager.py](https://github.com/sselbach/production-seizer/blob/working_branch/trainings_manager.py) | Takes current epsilon value and training steps across episodes.

### Training procedure outline:

* **Outer loop: episodes**
1. Loading the most current model (loading all trainable parameters) in the model saving directory. If this directory is empty, initialize random parameters by making a forward pass with random input.
2. Start the game and interact with the game environment. [rl_bot.py](https://github.com/sselbach/production-seizer/blob/working_branch/rl_bot.py)
    * **Inner loop: steps**
    1. Get all the windows for squares owned (old state)
   2. Pass windows through the DQN to get an action for each owned square or choose a random action with the probability of Epsilon
   3. Sent moves to game environment
   4. Get the now states from the game environment
   5. Compute the rewards
   6. Save old states, actions, rewards and new states to the replay buffer
   7. Trainings step:
        1. Sample a batch from the replay buffer (batch size dependen on the parameters set)
        2. Get current estimates for old states
        3. Calculate new estimates with values of new states, actions, and rewards.
		4. Calculate loss between old estimates and new estimates.
		5. Optimize Network using Gradient Descent
		6. Add loss and reward from that batch to our data handler.
3. If a game ends, save the current model in the specified directory, aswell as the replay buffer and the trainings manager and produce a plot monitoring the training process in that episode.

## Results

### Architecture construction

#### Global vs local evaluation

For solving the challenge we thought about two different approaches. One approach was to consider the whole game map as one state and output a 30 * 30 * 5 cube that corresponds to the q-values of each individual square for all actions. Sadly this did not work, we assume because the model could not generalize on such a large data structure. 
Therefore we came up with a more local approach. Instead of passing the whole game map as a state through the network, we use the network to estimate the q-value of only one owned square by using the relevant square as well as neighbors up to a specified distance in our parameters file as the input. This means that the distance parameter creates a trade-off between accuracy of our estimates (since small distance means less data to base estimation on) and computational solvability by the network (more iput -> more difficult to train).
Furthermore we lose the ability for squares to work together which adds another level of uncertainty that is not easy for the network to circumvent.
The [window.py](https://github.com/sselbach/production-seizer/blob/working_branch/window.py) file contains all relevant functions needed for our local approach.

```python
def get_owned_squares(game_map, id):
    """
    Returns all currently owned squares by id that have a strength > 0
    """

    # Run through gamemap and check if condition is met
    # add correct squares to lists
    owned_squares = []
    for y in range(30):
        for x in range(30):
            current = game_map.contents[y][x]
            if(current.owner == id and current.strength != 0):
                owned_squares.append(current)

    return owned_squares

def prepare_for_input(game_map, squares, id):
    """
    Converts owned squares into states that can be used by the network
    """

    # Initialize array with correct shape
    states = np.zeros((len(squares), NEIGHBORS * 2))

    for i in range(len(squares)):

        # Get neighbors with distance defined in hyperparameters from the owned square
        n = game_map.neighbors(squares[i], DISTANCE, True)
        j = 0
        # Run through all neighbors add add strength and production part to corresponding array slice
        for current_n in n:

            states[i, j] = current_n.strength  if current_n.owner == id else -current_n.strength
            states[i, j + NEIGHBORS] = current_n.production if current_n.owner == id else -current_n.production

            j += 1
    return states

def get_targets(game_map, squares, actions):
    """
    Return new squares after applying corresponding actions
    """
    targets = []

    # Apply action to each individual square and add to list
    for i in range(len(squares)):
        new_square = game_map.get_target(squares[i], actions[i])
        targets.append(new_square)

    return targets

```

#### Network structure

The structure we ended up with is rather simple. We only have two Dense layers with 8 units each using a leaky RELU. In the output layer we have 5 units for our 5 different actions a square can take. In the output layer we are not using an activation function. We used the Adam optimizer and the huber loss for training. For more information see: [dqn.py](https://github.com/sselbach/production-seizer/blob/working_branch/dqn.py)

```python
def __init__(self):
        super().__init__()

        self.dense1 = tf.keras.layers.Dense(units=8, activation=tf.nn.leaky_relu)

        self.dense2 = tf.keras.layers.Dense(units=8, activation=tf.nn.leaky_relu)

        self.output_layer = tf.keras.layers.Dense(units=5, activation=None)
        
        self.optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

        self.loss_function = tf.keras.losses.Huber()
```

#### Reward function

Like our network our [reward function](https://github.com/sselbach/production-seizer/blob/working_branch/reward.py) works on a local level. We came up with this reward function that compares the relevant squares old position with the positions it would be after moving in the old game map and the new one. Then we distinguish between 3 different things. Either the square lost to another square in battle, meaning the square at its new position is owned by another party, in which case we grant a negative reward to that square or the square wins against an enemy or neutral square, meaning its old target position was not owned by it but its new target position is. If none of these two cases apply we just grant the square a small negative reward. This reward function is by far not optimized but creating an optimal version is very difficult, due to reasons mentioned above.

```python
def reward(owned_squares, old_targets, new_targets, id):
    """
    Calculates reward of all owned squares in regards to party id
    """

    rewards = []
    
    # Iterate over all squares
    for i in range(len(owned_squares)):
        s = owned_squares[i]
        o = old_targets[i]
        n = new_targets[i]
        
        # Get reward for relevant situation
        if(n.owner != id):
            rewards.append(-5)

        elif(o.owner != id and n.owner == id):
            rewards.append(10)

        else:
            rewards.append(-0.001)

    return rewards
```

### Training

We trained the network for around 500 episodes. This plot below shows the average loss and reward in each episode.

![hier steht ihre werbung](https://github.com/sselbach/production-seizer/blob/working_branch/results/data_training/Training_483.0.png?raw=true)
![hier nicht](https://github.com/sselbach/production-seizer/blob/working_branch/results/final_replay.PNG?raw=true)

What you can see in the replays [(result folder)](https://github.com/sselbach/production-seizer/tree/working_branch/results) that instead of always going in one direction, the bot learns to wait with moving squares before they are strong enough to win against an enemy square. Important to note is that the final bot has an epsilon of 0 therefore never making random choices instead of 0.8 in the beginning of training.

#### Parameters

These are the parameters we used. We decay epsilon after every episode during training and the variable NEIGHBORS is used to determine the size of the input for our neural network.

Parameter | Value
------|-----
EPSILON_START | 0.8
EPSILON_DECAY | 0.98
EPSILON_END | 0.1
GAMMA | 0.99
BATCH_SIZE | 256
LEARNING_RATE | 0.001
BUFFER_SIZE | 100000
DISTANCE | 3
NEIGHBORS | 2 * DISTANCE * DISTANCE + 2 * DISTANCE + 1

## Conclusion

Although, as seen in the replays, the bot does learn something and also is able to play the game, it is by far not optimal yet. First of all the bot struggles with spawnpoints where the strength of the neutral squares is quite high, since an owned square would  rather like to suicide than to wait until it can defeat the neutral square. This can possible be circumvented by making the reward function more sophisticated. Furthermore there is no synchronisation between squares so sometimes it might happen that two squares make good moves individually but after they are executed they make the global state worse. 
Lastly we did not train or fight against winning bots from the original challenge since these are most definitely better because they are handcrafted and therefore optimized or trained on already existing bots.

However, given that the bot new nothing about the game to begin with, we are impressed with how much it was able to learn. In that sense, the approach was a success and has potential to grow a lot better with further improvements.

## Usage

**Note: We currently only support Ubuntu 64-bit. Other Linux distros may work, but Windows definitely won't.**

### Setting up a conda environments
In order to run the code we need a tensorflow environment with some extra packages. For that see the requirements.txt ifle. If you have a cuda enabled GPU you can choose to run tensorflow on your GPU by choosing requirements_gpu.txt. You then need to add the key word "gpu" when running the scripts.

```console
(base) username@dev:~/production-seizer$ conda create --name <env_name> --file requirements.txt
(base) username@dev:~/production-seizer$ conda actvate <env_name> # activates the environment
(<env_name>) username@dev:~/production-seizer$ conda deactivate # deactivates environment
(base) username@dev:~/production-seizer$
```
### Modified Halite Binary

The repository contains a file `halite_mod`, which is the main game binary. Because we have modified and recompiled it, it may not work on your system. In that case, you have to **compile halite from source**.

For that, open a terminal, navigate to the `halite_source` directory inside the repository, and run `make`. This should produce a binary called `halite`, which you can use to replace the `halite_mod` one that is currently at the repo's top level.

### Running Training
To run the training script please create directory for saving models and one for saving the results. Then adjust the following parameters in the hyperparameters.py:

```python
MODEL_PATH = "<model_dir>/"
WRITER_DIRECTORY = '<result_dir>/'
```
Execute the training shell script in the terminal:
```console
(<env_name>) usr@dev:~/production-seizer$ ./run_training.sh  # if usinig cuda add: gpu
```
In the "production-seizer" folder .hlt files will be created which one can uploaded to [a visualizer](https://2016.halite.io/local_visualizer.html) to see a replay of the game.
In the result directory a .csv will be created containing loss and reward for each time step and also averages over one episode which are also automatically plotted and saved. 

### Running Final Bot
If you wanna see one game of our bot playing against itself just execute the `run_final_game.sh`. It uses the model saved in the folder specified in the hyperparameters. After the game ended it generates a replay file that you can visualize [here](https://2016.halite.io/local_visualizer.html).


## Appendix

### Makeshift RL Environment

While the Halite challenge was designed with automated players in mind, there are still some technical challenges to overcome when developing a reinforcement learning based agent for this game.

Reinforcement learning requires a well defined *environment* that the agent can request the current state from and send actions to. When training an RL agent, typically the outermost instance that is run by the user is a program that contains the training loop and the agent itself. The environment runs in the background and is being told by the training loop when to perform an action, when to reset to an initial state etc. This is a very natural setup for reinforcement learning and is for example the way that the OpenAI Gym environments are used.

Halite, being a multiplayer game, had to choose a different architecture: Here the `halite` binary is the master process and starts the individual bots as sub-processes. It contains the game loop and requests actions from the bots at each turn. Crucially, the halite process *ends* at the end of a game. This presents a problem, as you typically need to train for many episodes to get any sort of sensible results.

To solve this issue with as little modification to the halite binary as possible, we opted to stick to halite's setup, and having the bot save the training progress, as well as the replay buffer, to disk at the end of each game and load the latest save at the start of the next. However, even this required a very small modification of the halite binary, replacing a SIGKILL signal with a SIGTERM one, enabling the bot to catch the event of the game ending. Please see the usage section below on how to get the modified halite version.