import numpy as np
from hyperparameters import PRODUCTION_TRADEOFF
from hlt import NORTH, EAST, SOUTH, WEST, STILL

def reward(window, production_factor=PRODUCTION_TRADEOFF):
    """Reward == (own strength - enemy strength) + k * (own prod. - enemy prod.)"""

    return (np.sum(window[:, :, 0]) - np.sum(window[:, :, 1])) \
        +  production_factor * (np.sum(window[:, :, 3]) - np.sum(window[:, :, 4]))

def reward4(owned_squares, old_targets, new_targets, id):

    rewards = []

    for i in range(len(owned_squares)):

        s = owned_squares[i]
        o = old_targets[i]
        n = new_targets[i]

        if(n.owner != id):
            rewards.append(-s.strength)

        elif(o.owner == 0 and n.owner == id):
            rewards.append(n.production + s.strength)

        elif(s.x == n.x and s.y == n.y):
            rewards.append(1)

        else:
            rewards.append(-1)

    return rewards
