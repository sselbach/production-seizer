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
from datetime import datetime

LOG_FILENAME = 'debug_sefplay.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)
logging.warning(f"starting new episode at {datetime.now().strftime('%d-%m-%Y_%I-%M-%S_%p')}")


myID, game_map = hlt.get_init()
hlt.send_init("SelfplayBot")
model = DQN()
model.load_last(MODEL_PATH)

while True:
    owned_squares = window.get_owned_squares(game_map, myID)

    old_states = window.prepare_for_input(game_map, owned_squares, myID, DISTANCE)

    directions = model.get_actions(old_states, epsilon=False)

    logging.debug(directions)

    moves = [Move(square, move) for square, move in zip(owned_squares, directions)]

    hlt.send_frame(moves)
    game_map.get_frame()
