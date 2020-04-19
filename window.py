import numpy as np
import logging
from hyperparameters import NEIGHBORS, DISTANCE

def get_owned_squares(game_map, id):
    """
    Returns all currently owned squares by id that have a strength > 0
    """

    # Run through gamemap and check if condition is met
    # add correct squares to lists
    owned_squares = []
    for y in range(30):
        for x in range(30):
            current = game_map.contents[y][x]
            if(current.owner == id and current.strength != 0):
                owned_squares.append(current)

    return owned_squares

def prepare_for_input(game_map, squares, id):
    """
    Converts owned squares into states that can be used by the network
    """

    # Initialize array with correct shape
    states = np.zeros((len(squares), NEIGHBORS * 2))

    for i in range(len(squares)):

        # Get neighbors with distance defined in hyperparameters from the owned square
        n = game_map.neighbors(squares[i], DISTANCE, True)
        j = 0
        # Run through all neighbors add add strength and production part to corresponding array slice
        for current_n in n:

            #logging.debug(current_n)
            states[i, j] = current_n.strength  if current_n.owner == id else -current_n.strength
            states[i, j + NEIGHBORS] = current_n.production if current_n.owner == id else -current_n.production

            j += 1
        #logging.debug("")
    return states

def get_targets(game_map, squares, actions):
    """
    Return new squares after applying corresponding actions
    """
    targets = []

    # Apply action to each individual square and add to list
    for i in range(len(squares)):
        new_square = game_map.get_target(squares[i], actions[i])
        targets.append(new_square)

    return targets
