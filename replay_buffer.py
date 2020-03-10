import random
import pickle
import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL

class ReplayBuffer:
    """
    The Replay Buffer used to train the DQN
    """

    def __init__(self, buffer_size):
        self.size = buffer_size
        self.buffer = []

    def count(self):
        """
        Returns how many elements are in the Buffer
        """
        return len(self.buffer)

    def save_to_file(self):
        """
        Saves the buffer to a file
        """
        with open("buffer", "wb") as file:
            pickle.dump(self.buffer, file)

    def load_from_file(self):
        """
        Loads the buffer from a file.
        If file doesn't exist creates it.
        """
        try:
            with open("buffer.pickle", "rb") as file:
                self.buffer = pickle.load(file)
        except FileNotFoundError:
            open("buffer.pickle", "a").close()

    def add_tuple(self, old_state, action, reward, new_state, terminal):
        """
        Adds a SARS tuple to the buffer
        """
        self.buffer.append((old_state, action, reward, new_state, terminal))

        if(self.count() > self.size):
            self.buffer.pop(0)

    def add_tuples(self, tuple_list):
        """
        Adds multiple tuples to the buffer
        """
        self.buffer.extend(tuple_list)

        for _ in range(self.count() - self.size):
            self.buffer.pop(0)

    def get_batch(self, k):
        """
        Returns a batch of size k from the buffer
        """
        assert self.count() >= k, "Trying to get batch although Buffer has not enough elements yet."

        sample = random.sample(self.buffer, k=k)

        return sample
