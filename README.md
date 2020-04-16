# Solving the Halite Challenge with a Deep-Q network


## Introduction

For our final project in the course of “Implementing ANN with TensorFlow” we decided to revive the challenge proposed by TWO SIGMA in 2016. The challenge was composed of the provided game “[Halite](https://2016.halite.io/index.html)” and its corresponding rule set. The competition then started in November 2016 and finalized 3 month later, in February 2017. The send in application were a mix of different approaches, consisting of C#, python, java scripts etc. But also some Machine Learning based bots. In our growing interest in self learning systems we formulated our idea to challenge the competition with an ANN based bot. In this report we will show our different approaches, our successful as well as our not so successful ones.

### The Rules

Halite is played on a rectangular grid, where the size correlates with the number of players (2-6). The goal is to seize the means of production. Each player starts on a specific tile of the grid, all other tiles are considered unowned. The tiles come in different qualities. A tile produces or enhances one drone per turn with the strength of its quality value. Drones have the ability to move in cityblock distance or to stay at their momentary tile. An unowned tile can be converted into an owned one by being occupied by a drone. The strength of the drones is capped to the value of 255. Tiles owned by foreign drones can be conquered by overwhelming through higher strength. The strength values get subtracted. When all enemy drones and tiles are conquered, the game ends. The map is continuous, go to far left and you will appear on the right (bottom and top the same).

![](halite_gif.gif)
