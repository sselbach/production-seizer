import tensorflow as tf

import hlt
from hlt import Move, Square

from hyperparameters import BATCH_SIZE, MODEL_PATH, EPSILON_END, EPSILON_DECAY
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
from trainings_manager import TrainingsManager
from replay_buffer import ReplayBuffer
from dqn import DQN

LOG_FILENAME = 'debug.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)
logging.warning(f"starting new episode at {datetime.now().strftime('%d-%m-%Y_%I-%M-%S_%p')}")

writer = Writer("data", "HelloWorld")

myID, game_map = hlt.get_init()
hlt.send_init("ProductionSeizer")
logging.debug("sent init message to game")
logging.debug(myID)

r = ReplayBuffer()

tm = TrainingsManager()


## INIT MODEL

model = DQN()
logging.debug("initialized model")
model.load_last(MODEL_PATH)
logging.debug("loaded latest model")


l_list = []
r_list = []

def termination_handler(signal, frame):
    logging.debug(f"finished episode at {datetime.now().strftime('%d-%m-%Y_%I-%M-%S_%p')}")
    logging.shutdown()

    tm.content["episodes"] += 1

    if(tm.content["epsilon"] > EPSILON_END):

        tm.content["epsilon"] *= EPSILON_DECAY

    tm.save()

    writer.plot_progress(True)
    r.save()
    model.save(MODEL_PATH)
    sys.exit(0)

signal.signal(signal.SIGTERM, termination_handler)

game_map.get_frame()


## START MAIN LOOP
while True:

    owned_squares = window.get_owned_squares(game_map, myID)

    old_states = window.prepare_for_input(game_map, owned_squares, myID)

    directions = model.get_actions(old_states, tm.content["epsilon"])

    old_targets = window.get_targets(game_map, owned_squares, directions)

    moves = [Move(square, direction) for square, direction in zip(owned_squares, directions)]

    hlt.send_frame(moves)
    game_map.get_frame()

    new_targets = window.get_targets(game_map, owned_squares, directions)

    done = [int(t.owner == id) for t in new_targets]

    new_states = window.prepare_for_input(game_map, new_targets, myID)

    rewards = reward.reward(owned_squares, old_targets, new_targets, myID)

    #logging.debug(rewards)

    for i in range(len(owned_squares)):

        r.add(old_states[i], directions[i], rewards[i], new_states[i], done[i])

    if len(r) >= BATCH_SIZE:
        batch = r.get_batch(BATCH_SIZE)

        loss, rewar = model.train(batch)

        writer.save_progress(tm.content["timesteps"], loss, rewar)

    #if(timestep % 10 == 0):
        #logging.debug(model.trainable_variables[0])

    tm.content["timesteps"] += 1
