import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square
import random
from dqn200 import DQN
import sys
from hyperparameters import *
import window
import tensorflow as tf
import logging
import config
from config import key


myID, game_map = hlt.get_init()
hlt.send_init("SelfplayBot")
model = DQN()
model.load_random(MODEL_PATH)

logging.debug("SelfplayBot "  + str(myID))


while True:
    owned_squares = window.get_owned_squares(game_map, myID)

    old_states = window.prepare_for_input(game_map, owned_squares, myID, DISTANCE)

    directions = model.get_actions(old_states, epsilon=False)

    moves = [Move(square, move) for square, move in zip(owned_squares, directions)]

    hlt.send_frame(moves)
    game_map.get_frame()
