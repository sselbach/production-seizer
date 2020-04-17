import logging
import numpy as np

def reward(owned_squares, old_targets, new_targets, id):

    rewards = []

    for i in range(len(owned_squares)):

        s = owned_squares[i]
        o = old_targets[i]
        n = new_targets[i]

        if(n.owner != id):
            rewards.append(-5)

        elif(o.owner == 0 and n.owner == id):
            rewards.append(10)

        else:
            rewards.append(-0.001)

    logging.debug(rewards)

    return rewards

def reward_global(old_state, new_state):

    difference = new_state - old_state

    owner_layer = difference[:,:,1]

    strength = difference[:,:,0] * owner_layer

    production = difference[:,:,2] * owner_layer

    return np.sum(strength) + np.sum(production)
