import tensorflow as tf

import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square

from dqn import DQN
from replay_buffer import ReplayBuffer
from hyperparameters import *
import window
import reward

import sys
import logging
import signal
import random
from datetime import datetime

import config
from config import key

LOG_FILENAME = 'debug.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)
logging.warning(f"starting new episode at {datetime.now().strftime('%d-%m-%Y_%I-%M-%S_%p')}")

myID, game_map = hlt.get_init()
hlt.send_init("ProductionSeizer")
logging.debug("sent init message to game")
logging.debug(key)

r = ReplayBuffer(100000)
r.load_from_file()

## INIT MODEL

proto = DQN(key)
logging.debug("initialized model")
proto.load_last(MODEL_PATH)
logging.debug("loaded latest model")

def termination_handler(signal, frame):
    logging.debug(f"finished episode at {datetime.now().strftime('%d-%m-%Y_%I-%M-%S_%p')}")
    logging.shutdown()

    r.save_to_file()
    proto.save(MODEL_PATH)
    sys.exit(0)

signal.signal(signal.SIGTERM, termination_handler)


## START MAIN LOOP
while True:
    owned_squares, current_states = window.get_windows(game_map.contents, myID)

    moves = proto.get_action(current_states)
    moves = moves.numpy().tolist()
    moves = [Move(square, move) for square, move in zip(owned_squares, moves)]

    hlt.send_frame(moves)
    game_map.get_frame()

    new_states = window.get_windows_for_squares(game_map.contents, owned_squares)

    #rewards = [reward.reward(s) for s in new_states]

    rewards = [reward.reward2(current_states[i], new_states[i]) for i in range(len(owned_squares))]

    logging.debug(rewards)

    tuples = zip(current_states, moves, rewards, new_states)

    r.add_tuples(tuples)

    if len(r) >= BATCH_SIZE:
        proto.train(r.get_batch(BATCH_SIZE))
