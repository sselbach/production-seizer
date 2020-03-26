import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square
import random
from dqn import DQN
import sys
from hyperparameters import *
import window
import tensorflow as tf
import logging
import config
from config import key



myID, game_map = hlt.get_init()
hlt.send_init("SelfplayBot")
model = DQN(key)
model.load_random(MODEL_PATH)


while True:
    owned_squares, current_states = window.get_windows(game_map.contents)

    moves = model.get_action(current_states)
    moves = moves.numpy().tolist()
    moves = [Move(square, move) for square, move in zip(owned_squares, moves)]

    hlt.send_frame(moves)
    game_map.get_frame()
