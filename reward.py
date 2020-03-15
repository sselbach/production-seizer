import numpy as np
from hyperparameters import PRODUCTION_TRADEOFF

def reward(window, production_factor=PRODUCTION_TRADEOFF):
    """Reward == (own strength - enemy strength) + k * (own prod. - enemy prod.)"""

    return (np.sum(window[:, :, 0]) - np.sum(window[:, :, 1])) \
        +  production_factor * (np.sum(window[:, :, 3]) - np.sum(window[:, :, 4]))
