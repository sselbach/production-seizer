import tensorflow as tf

import hlt
from hlt import Move, Square, STILL

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

from progress import Writer
from trainings_manager import TrainingsManager
from replay_buffer import ReplayBuffer
from dqn_conv import DQN

LOG_FILENAME = 'debug.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)
logging.warning(f"starting new episode at {datetime.now().strftime('%d-%m-%Y_%I-%M-%S_%p')}")

writer = Writer("data", "HelloWorld")

myID, game_map = hlt.get_init()
hlt.send_init("ProductionSeizer")
logging.debug("sent init message to game")
logging.debug(myID)

buffer = ReplayBuffer()

tm = TrainingsManager()


## INIT MODEL

model = DQN()
logging.debug("init ialized model")
model.load_last(MODEL_PATH)
logging.debug("loaded latest model")



def termination_handler(signal, frame):

    tm.content["episodes"] += 1

    if(tm.content["epsilon"] > EPSILON_END):

        tm.content["epsilon"] *= EPSILON_DECAY

    tm.save()

    logging.debug("saved manager")

    buffer.save()

    logging.debug("saved buffer")

    model.save(MODEL_PATH)

    writer.plot_progress()

    logging.debug(f"finished episode at {datetime.now().strftime('%d-%m-%Y_%I-%M-%S_%p')}")
    logging.shutdown()

    sys.exit(0)

signal.signal(signal.SIGTERM, termination_handler)

old_state = None


## START MAIN LOOP
while True:

    game_map.get_frame()

    current_state = window.prepare_for_input_conv(game_map, myID)

    logging.debug(current_state.shape)

    if(old_state is not None):

        reward = reward_functions.reward_global(old_state, current_state)

        buffer.add(old_state, action_matrix, reward, current_state, 0)


    if len(buffer) >= BATCH_SIZE:
        batch = buffer.get_batch(BATCH_SIZE)

        loss, rewar = model.train(batch)

        writer.save_progress(tm.content["timesteps"], loss, rewar)

    action_matrix = model.get_action_matrix(current_state, tm.content["epsilon"])

    moves = [Move(square, action_matrix[square.y, square.x] if square.strength > 0 else STILL) for square in game_map if square.owner == myID]

    hlt.send_frame(moves)

    old_state = current_state

    tm.content["timesteps"] += 1
