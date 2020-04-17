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
    states = np.zeros((len(squares), 2, NEIGHBORS))

    for i in range(len(squares)):

        # Get neighbors with distance defined in hyperparameters from the owned square
        n = game_map.neighbors(squares[i], DISTANCE, True)
        j = 0
        # Run through all neighbors add add strength and production part to corresponding array slice
        for current_n in n:

            d = game_map.get_distance(squares[i], current_n)

            div = d if d != 0 else 1

            #logging.debug(current_n)
            states[i, 0, j] = current_n.strength / div if current_n.owner == id else -current_n.strength / div
            states[i, 1, j] = current_n.production / div if current_n.owner == id else -current_n.production / div

            j += 1
        #logging.debug("")
    return states

def prepare_for_input_conv(game_map, id):

    state = np.zeros((30, 30, 3))

    for y in range(30):
        for x in range(30):
            current = game_map.contents[y][x]

            state[y,x,0] = current.strength
            state[y,x,2] = current.production

            if(current.owner == id):
                state[y, x, 1] = 1
            elif(current.owner != 0):
                state[y, x, 1] = -1
            else:
                state[y, x, 1] = 0

    state[:,:,0] /= 255
    state[:,:,0] -= 0.5
    state[:,:,2] /= 17
    state[:,:,2] -= 0.5

    return state



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
