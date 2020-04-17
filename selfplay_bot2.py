import tensorflow as tf

import hlt
from hlt import Move, Square

from hyperparameters import BATCH_SIZE, MODEL_PATH, EPSILON_END, EPSILON_DECAY
import window
import reward as reward_functions

import sys
import logging
import signal
import random
from datetime import datetime

import config
from config import key

from dqn_conv import DQN

LOG_FILENAME = 'debug_selfplay.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)
logging.warning(f"starting new episode at {datetime.now().strftime('%d-%m-%Y_%I-%M-%S_%p')}")

myID, game_map = hlt.get_init()
hlt.send_init("ProductionSeizer")
logging.debug("sent init message to game")
logging.debug(myID)


## INIT MODEL

model = DQN()
logging.debug("initialized model")
model.load_last(MODEL_PATH)
logging.debug("loaded latest model")


## START MAIN LOOP
while True:

    game_map.get_frame()

    current_state = window.prepare_for_input_conv(game_map, myID)

    action_matrix = model.get_action_matrix(current_state, tm.content["epsilon"])

    moves = [Move(square, action_matrix[square.y, square.x] if square.strength > 0 else STILL) for square in game_map if square.owner == myID]

    hlt.send_frame(moves)
