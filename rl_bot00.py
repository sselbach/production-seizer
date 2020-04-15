import tensorflow as tf

import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square

from dqn200 import DQN
from replay_buffer2 import ReplayBuffer
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

from progress import Writer

LOG_FILENAME = 'debug.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)
logging.warning(f"starting new episode at {datetime.now().strftime('%d-%m-%Y_%I-%M-%S_%p')}")

writer = Writer("data", "HelloWorld")

myID, game_map = hlt.get_init()
hlt.send_init("ProductionSeizer")
logging.debug("sent init message to game")
logging.debug(key)
logging.debug(myID)

r = ReplayBuffer()
r.load_from_file()

## INIT MODEL

model = DQN()
logging.debug("initialized model")
model.load_last(MODEL_PATH)
logging.debug("loaded latest model")

def termination_handler(signal, frame):
    logging.debug(f"finished episode at {datetime.now().strftime('%d-%m-%Y_%I-%M-%S_%p')}")
    logging.shutdown()
    #writer.save_progress(0, np.mean(np.array(losses)), np.mean(np.array(rewardl)))
    writer.plot_progress(False)
    r.save_to_file()
    model.save(MODEL_PATH)
    sys.exit(0)

signal.signal(signal.SIGTERM, termination_handler)

#owned_squares, current_states = window.get_windows(game_map.contents, myID)

#logging.debug("RLBot "  + str(myID))

timestep = 0

#losses = []

#rewardl = []

## START MAIN LOOP
while True:

    owned_squares = window.get_owned_squares(game_map, myID)

    old_states = window.prepare_for_input(game_map, owned_squares, myID, DISTANCE)

    directions = model.get_actions(old_states)

    old_targets = window.get_targets(game_map, owned_squares, directions)

    moves = [Move(square, direction) for square, direction in zip(owned_squares, directions)]

    hlt.send_frame(moves)
    game_map.get_frame()

    new_targets = window.get_targets(game_map, owned_squares, directions)

    new_states = window.prepare_for_input(game_map, new_targets, myID, DISTANCE)

    rewards = reward.reward4(owned_squares, old_targets, new_targets, myID)

    #logging.debug(rewards)

    for i in range(len(owned_squares)):

        r.add(old_states[i], directions[i], rewards[i], new_states[i])

    if len(r) >= BATCH_SIZE:
        batch = r.get_batch(BATCH_SIZE)
        #logging.debug(batch["new_states"].shape)
        loss, rewar = model.train(batch)

        #losses.append(loss)
        #rewardl.append(rewar)

        writer.save_progress(timestep, loss, rewar)

    if(timestep % 10 == 0):
        logging.debug(model.trainable_variables[0])

    timestep += 1
    #EPSILON *= 0.99
