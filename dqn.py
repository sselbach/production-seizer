import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.layers import Layer
from tensorflow.keras import  Model

import numpy as np

from hyperparameters import LEARNING_RATE, NEIGHBORS, DISTANCE, GAMMA
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square

import random
import sys
import glob
import logging
from datetime import datetime

class DQN(Model):
    """
    Implementation of DQN with a simple Neural Network
    """

    def __init__(self):
        super().__init__()

        self.dense1 = tf.keras.layers.Dense(units=8, activation=tf.nn.leaky_relu)

        self.dense2 = tf.keras.layers.Dense(units=8, activation=tf.nn.leaky_relu)

        self.output_layer = tf.keras.layers.Dense(units=5, activation=None)

        self.optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

        self.loss_function = tf.keras.losses.Huber()


    def call(self, x):

        o = self.dense1(x)

        o = self.dense2(o)

        o = self.output_layer(o)

        return o


    def get_actions(self, states, epsilon=0):
        """
        Returns valid actions for all states, using an epsilon-greedy approach
        """

        # Get q-values for states
        q_values = self(states)

        #logging.debug(q_values)

        # Get the action that has the highest q-value
        actions = tf.math.argmax(q_values, axis=-1).numpy()

        # Generate which actions should be randomized using epsilon
        randoms = np.random.binomial(n=1, p=epsilon, size=actions.shape)

        # Get random actions
        random_actions = np.random.choice(a=[0, 1, 2, 3, 4], size=actions.shape, p=[0.2, 0.2, 0.2, 0.2, 0.2])

        # Make actions random for states that should be randomized
        actions = actions * np.logical_not(randoms) + randoms * random_actions

        return actions

    def load_random(self, model_dir):
        """
        Loads one of the last n models at random.
        """
        list_of_files = glob.glob(model_dir+'*')

        # if model directory is empty make a dummy forward pass
        if len(list_of_files) <=  1:
            input_map = tf.random.normal(shape=(1,2 * NEIGHBORS))
            self.get_actions(input_map)
            self.save_weights(model_dir + 'random_initialization')

        # if less then N models. just choose on randomly
        if len(list_of_files) <= N:
            while True:
                random_file = random.choice(os.listdir(model_dir))
                # avoid loading checkpoint
                if (random_file != 'checkpoint'):
                    break
        # if more than N models, choose randomly from last N
        else:
            checkpoint = True
            while checkpoint:
                list_of_files = sorted(os.listdir(model_dir), key=lambda f: os.path.getmtime(model_dir + f))
                random_n = random.randint(0, N)
                random_file = list_of_files[-random_n]
                # avoid checkpoint
                if (random_file!='checkpoint'):
                    checkpoint=False

        random_file = random_file.rsplit('.', 1)[0]
        logging.debug(random_file)
        self.load_weights(model_dir + random_file)


    def train(self, batch):
        """
        Trains the model on one batch
        """

        with tf.GradientTape() as tape:

            action_indices = np.append(np.arange(batch["actions"].shape[0]).reshape(-1,1), batch["actions"].reshape(-1,1), axis = 1)

            # Get q-values for old states
            old_state_values = tf.gather_nd(self(batch["old_states"]), action_indices)

            logging.debug(old_state_values.shape)

            not_terminal_mask = np.where(batch["done"])

            not_terminal_states = batch["new_states"][not_terminal_mask]

            # Get q-values for new states that are not terminal
            next_state_values = np.zeros(batch["new_states"].shape[0])

            next_state_values[not_terminal_mask] = tf.reduce_max(self(not_terminal_states), axis = 1)

            expected_values = batch["rewards"] + GAMMA * next_state_values

            # Calculate loss
            loss = self.loss_function(expected_values, old_state_values)

            # Get means for rewards for later plotting
            #loss_mean = tf.reduce_mean(loss, axis=-1).numpy()
            reward_mean = tf.reduce_mean(expected_values, axis=-1).numpy()

            # Calculate gradients
            gradients = tape.gradient(loss, self.trainable_variables)

        # Train model
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss.numpy(), reward_mean

    def save(self, model_dir):
        """
        Save weights of current configuration.
        """
        self.save_weights(model_dir + datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p"))

    def load_last(self, model_dir):
        """
        Loads the latest model.
        """
        list_of_files = glob.glob(model_dir+'*') # * means all if need specific format then *.csv

        if len(list_of_files) <=  1:
            input_map = tf.random.normal(shape=(1, 2 * NEIGHBORS))
            self.get_actions(input_map)
            self.save_weights(model_dir + 'random_initialization')

        list_of_files = sorted(os.listdir(model_dir), key=lambda f: os.path.getmtime(model_dir + f))
        latest_file = list_of_files[-1]
        latest_file = latest_file.rsplit('.', 1)[0]

        if latest_file == 'checkpoint':
            latest_file = list_of_files[-2]
            latest_file = latest_file.rsplit('.', 1)[0]

        self.load_weights(model_dir + latest_file)
