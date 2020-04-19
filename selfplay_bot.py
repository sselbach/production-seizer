import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square
import random
from dqn import DQN
import sys
from hyperparameters import MODEL_PATH
import window
import tensorflow as tf
import logging
import config
from datetime import datetime

LOG_FILENAME = 'debug_selfplay.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)
logging.warning(f"starting new episode at {datetime.now().strftime('%d-%m-%Y_%I-%M-%S_%p')}")


myID, game_map = hlt.get_init()
hlt.send_init("SelfplayBot")

model = DQN()
model.load_last(MODEL_PATH)

game_map.get_frame()

while True:
    owned_squares = window.get_owned_squares(game_map, myID)

    old_states = window.prepare_for_input(game_map, owned_squares, myID)

    directions = model.get_actions(old_states, 0.1)

    moves = [Move(square, move) for square, move in zip(owned_squares, directions)]

    hlt.send_frame(moves)
    game_map.get_frame()
