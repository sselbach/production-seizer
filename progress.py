import csv
from hyperparameters import *
import pandas as pd
import matplotlib.pyplot as plt
import logging
import numpy as np



class Writer(object):
    """
    Create csv for keeping track of learning progress.
    On default, a running average (combined current with last average) is plottet indicated by the "running" flag.
    Plotting simple averages is also possible.
    """

    def __init__(self, filename, model_name, running=True):
        self.filename = WRITER_DIRECTORY + filename + '.csv'
        self.model_name = model_name
        self.count = 1
        self.episodes = 0
        self.loss_agg = 0
        self.rewar_agg = 0
        self.fieldnames = ["step", "loss", "reward", "episode", "avg loss", "avg reward"]

        # if file already exists keep writing to it
        try:
            open(self.filename, "r")
            df = pd.read_csv(self.filename)
            self.episodes = np.asarray(df["episode"].dropna())[-1]
            # if we want to plot smooth averages, set the current average to the last one
            if running:
                self.loss_agg = np.asarray(df["avg loss"].dropna())[-1]
                self.rewar_agg = np.asarray(df["avg reward"].dropna())[-1]
                self.count+=1

        # else create new csv
        except FileNotFoundError:
            with open(self.filename, 'w', newline='') as file:
                self.writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                self.writer.writeheader()
        except EOFError:
            return





    def save_progress(self, step, loss, reward):
        """
        Write progress to csv.
        Accumulate loss and reward in each step to compute averages.
        Number of steps before termination not neccesarily the same each time,
        thus counting is neccesary.
        """
        with open(self.filename, 'a', newline='') as file:
            self.writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            self.writer.writerow({"step": step, "loss": loss, "reward": reward})
        self.loss_agg += loss
        self.rewar_agg += reward
        self.count+=1



    def plot_progress(self, rewards=True, average=True):
        """
        Plotting training progress and saving average over episode.
        """

        df = pd.read_csv(self.filename)
        train_step = df["step"].to_numpy()[-1]
        episodes = df["episode"].to_numpy()[-1]


        if average:
            fig_name = WRITER_DIRECTORY + self.model_name + '_' + str(self.episodes) + '.png'
            self.plot_progress_sub(fig_name, "episode", "avg reward", "avg loss", rewards)
        else:
            fig_name = WRITER_DIRECTORY + self.model_name + '_' + str(train_step) + '.png'
            self.plot_progress_sub(fig_name, "steps", "reward", "loss", running)

        # if plotting then episode over, save average over episode and reset
        self.episodes+=1
        with open(self.filename, 'a', newline='') as file:
            self.writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            self.writer.writerow({"episode": self.episodes, "avg loss": (self.loss_agg/self.count), "avg reward": (self.rewar_agg/self.count)})
        self.count = 1
        self.loss_agg = 0
        self.rewar_agg = 0

    def plot_progress_sub(self, fig_name, stepstr, rewarstr, lossstr, rewards):
        """
        Subroutine for creating matplotlib plots.
        """
        df = pd.read_csv(self.filename)
        if rewards:
            df_loss = df[lossstr].dropna()
            df_reward = df[rewarstr].dropna()
            df_reward = df_reward.dropna()
            fig, ax1 = plt.subplots()
            ax1.set_xlabel(stepstr)
            ax1.set_ylabel(lossstr, color='b')
            if stepstr !="steps":
                df_x = df[stepstr].dropna()
                ax1.plot(df_x, df_loss, 'b')
            else: ax1.plot(df_loss, 'b')
            ax2 = ax1.twinx()
            ax2.set_ylabel(rewarstr, color='r')
            if stepstr!="steps":
                ax2.plot(df_x, df_reward, 'r')
            else: ax2.plot(df_reward, 'r')

        else:
            df = df[lossstr]
            pl.figure()
            plt.plot()
        plt.title(self.model_name)
        plt.savefig(fig_name)


if __name__=="__main__":
    # testeng
    w = Writer("try_runnung", "model_running", running=True)
    for i in range(500):
        w.save_progress(i,100-i,2*i)
        if i%100==0:
            w.plot_progress()
