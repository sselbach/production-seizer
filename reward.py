import numpy as np

def reward(window, production_factor=1):
    """Reward == (own strength - enemy strength) + k * (own prod. - enemy prod.)"""

    return (np.sum(window[:, :, 0]) - np.sum(window[:, :, 1])) \
        +  production_factor * (np.sum(window[:, :, 3]) - np.sum(window[:, :, 4]))
