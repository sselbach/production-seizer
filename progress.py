import csv
from hyperparameters import *
import pandas as pd
import matplotlib.pyplot as plt

class Writer(object):
    """Create csv for keeping track of learning progress"""

    def __init__(self, filename, model_name):
        self.filename = WRITER_DIRECTORY + filename + '.csv'
        self.model_name = model_name

        try:
            open(self.filename, "r")
        except FileNotFoundError:
            with open(self.filename, 'w', newline='') as file:
                self.writer = csv.writer(file)
                self.writer.writerow(["step", "loss", "reward"])
        except EOFError:
            return




    def save_progress(self, step, loss, reward):
        with open(self.filename, 'a', newline='') as file:
            self.writer = csv.writer(file)
            self.writer.writerow([step, loss, reward])

    def plot_progress(self, rewards=False):
        df = pd.read_csv(self.filename)
        train_step = df["step"].to_numpy()[-1]
        fig_name = WRITER_DIRECTORY + self.model_name + '_' + str(train_step) + '.png'
        if rewards:
            df = df[["loss", "reward"]]
            df.cumsum()
        else:
            df = df["loss"]
        plt.figure()
        df.plot()
        plt.legend(loc='best')
        plt.title(self.model_name)
        plt.xlabel("training steps")
        plt.savefig(fig_name)

if __name__=="__main__":
    # testeng
    w = Writer("try_1", "model_1")
    for i in range(500):
        w.save_progress(i,100-i,2*i)
        if i%100==0:
            w.plot_progress(True)
