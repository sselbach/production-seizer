import numpy as np
import pickle
import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL
from hyperparameters import *

class ReplayBuffer:
    """
    The Replay Buffer used to train the DQN
    """

    def __init__(self):
        self.count = 0
        self.buffer = {
        "new_states": [],
        "rewards": [],
        "actions": [],
        "old_states" : []
        }

    def __len__(self):
        return self.count

    def save_to_file(self):
        """
        Saves the buffer to a file
        """
        with open("buffer.pickle", "wb") as file:
            pickle.dump(self.buffer, file)

    def load_from_file(self):
        """
        Loads the buffer from a file.
        If file doesn't exist creates it.
        """
        try:
            with open("buffer.pickle", "rb") as file:
                self.buffer = pickle.load(file)
                self.count = len(self.buffer["new_states"])
        except FileNotFoundError:
            open("buffer.pickle", "a").close()
        except EOFError:
            return

    def add(self, old_state, action, reward, new_state):
        """
        Adds a SARS tuple to the buffer
        """
        self.buffer["old_states"].append(old_state)
        self.buffer["new_states"].append(new_state)
        self.buffer["actions"].append(action)
        self.buffer["rewards"].append(reward)

        if(self.count == BUFFER_SIZE):
            self.buffer["old_states"].pop(0)
            self.buffer["new_states"].pop(0)
            self.buffer["actions"].pop(0)
            self.buffer["rewards"].pop(0)

        else:
            self.count += 1


    def get_batch(self, k):
        """
        Returns a batch of size k from the buffer
        """
        assert self.count >= k, "Trying to get batch although Buffer has not enough elements yet."

        sample = np.random.choice(self.count, k, replace=False)

        batch = {
        "new_states": np.array(self.buffer["new_states"])[sample],
        "rewards": np.array(self.buffer["rewards"])[sample],
        "actions": np.array(self.buffer["actions"])[sample],
        "old_states" : np.array(self.buffer["old_states"])[sample]
        }

        return batch
