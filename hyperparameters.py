"""
The hyperparameters used by all different parts of our algorithms
"""

# General
MODEL_PATH = "models/current/"
WRITER_DIRECTORY = 'experiments/test/'

# Model
EPSILON_START = 0.8
EPSILON_DECAY = 0.98
EPSILON_END = 0.1
GAMMA = 0.99
BATCH_SIZE = 256
LEARNING_RATE = 0.001
BUFFER_SIZE = 100000
DISTANCE = 3
NEIGHBORS = 2 * DISTANCE * DISTANCE + 2 * DISTANCE + 1
