import numpy as np
import logging
from hyperparameters import NEIGHBORS, DISTANCE

def get_owned_squares(game_map, id):
    owned_squares = []
    for row in range(30):
        for col in range(30):
            current = game_map.contents[row][col]
            if(current.owner == id):
                owned_squares.append(current)

    return owned_squares

def prepare_for_input(game_map, squares, id):
    states = np.zeros((len(squares), 2, NEIGHBORS))

    for i in range(len(squares)):
        #logging.debug(squares[i])
        n = game_map.neighbors(squares[i], DISTANCE, True)
        j = 0
        for current_n in n:
            #logging.debug(current_n)
            states[i, 0, j] = current_n.strength if current_n.owner == id else -current_n.strength
            states[i, 1, j] = current_n.production if current_n.owner == id else -current_n.production

            j += 1
        #logging.debug("BLUB")
    return states

def get_targets(game_map, squares, actions):

    targets = []

    for i in range(len(squares)):
        new_square = game_map.get_target(squares[i], actions[i])
        targets.append(new_square)

    return targets
