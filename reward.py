import numpy as np
from hyperparameters import PRODUCTION_TRADEOFF

def reward(window, production_factor=PRODUCTION_TRADEOFF):
    """Reward == (own strength - enemy strength) + k * (own prod. - enemy prod.)"""

    return (np.sum(window[:, :, 0]) - np.sum(window[:, :, 1])) \
        +  production_factor * (np.sum(window[:, :, 3]) - np.sum(window[:, :, 4]))

def reward2(old_state, new_state, production_factor=PRODUCTION_TRADEOFF):
    difference = new_state - old_state

    return (np.sum(difference[:, :, 0]) - np.sum(difference[:, :, 1])) \
        +  production_factor * (np.sum(difference[:, :, 3]) - np.sum(difference[:, :, 4]))
