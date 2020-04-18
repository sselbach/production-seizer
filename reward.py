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

        elif(o.owner != id and n.owner == id):
            rewards.append(10)

        else:
            rewards.append(-0.001)

    logging.debug(rewards)

    return rewards

def reward_global(old_state, new_state):

    #logging.debug("REWARDS")

    mask1 = old_state[:,:,1] == 1
    mask2 = new_state[:,:,1] == 1

    #logging.debug(mask1.shape)

    old_strength = old_state[:,:,0] * mask1
    new_strength = new_state[:,:,0] * mask1

    old_production = old_state[:,:,2] * mask1
    new_production = new_state[:,:,2] * mask1

    #return np.sum(strength) + 2 * np.sum(production)
    return np.sum(new_strength) - np.sum(old_strength) + 3 * (np.sum(new_production) - np.sum(old_production))

    #change = np.sum(mask2) - np.sum(mask1)

    #if(change == 0):
        #return -0.001
    #else:
        #return change

def reward_terminal(state):
    owner_layer = state[:,:,1]

    winner = np.sum(owner_layer)

    if(winner > 0):
        return 100

    else:
        return -100
