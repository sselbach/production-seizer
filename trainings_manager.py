from hyperparameters import EPSILON_START
import pickle

class TrainingsManager:

    def __init__(self):
        try:
            with open("manager.pickle", "rb") as file:
                self.content = pickle.load(file)
        except FileNotFoundError:
            open("manager.pickle", "a").close()
            self.content = {
            "epsilon" : EPSILON_START,
            "timesteps" : 0,
            "episodes" : 0
            }
        except EOFError:
            return

    def save(self):
        with open("manager.pickle", "wb") as file:
            pickle.dump(self.content, file)
