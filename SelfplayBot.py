import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square
import random
from dqn import DQN
import sys
from hyperparameters import *
import window
import tensorflow as tf
import logging
LOG_FILENAME = 'debug_self.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)
logging.debug("self play says hello")
if 'gpu' in sys.argv:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

# default on simple conv
key = 'simple_conv'

if 'simple_conv' in sys.argv:
    key = 'simple_conv'
if 'simple_no_conv' in sys.argv:
    key = 'simple_no_conv'
if 'res_net' in sys.argv:
    key = 'res_net'

myID, game_map = hlt.get_init()
hlt.send_init("SelfplayBot")
logging.debug(key)
model = DQN(key)
logging.debug("model init")
model.load_random(MODEL_PATH)


while True:
    owned_squares, current_states = window.get_windows(game_map.contents)

    moves = model.get_action(current_states)
    moves = moves.numpy().tolist()
    moves = [Move(square, move) for square, move in zip(owned_squares, moves)]

    hlt.send_frame(moves)
    game_map.get_frame()
