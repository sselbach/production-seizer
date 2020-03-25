import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.layers import Layer
from tensorflow.keras import  Model

import numpy as np

from hyperparameters import *
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square

import random
import sys
import glob
import logging
from datetime import datetime


class DQN(Model):
    def __init__(self, key):

        """
        Initialize Model
        different kind of Models
        """
        super().__init__()
        if (key == "simple_conv"):
            self.init_simple_conv()
        elif (key == "simple_no_conv"):
            logging.debug("in key if ")
            self.init_simple_no_conv()
            logging.debug("initialized")

        self.loss_function = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()


    # input x:(MAP_SIZE_x, MAP_SIZE_y, CHANNELS) map
    # output: (5,1) rewards
    def call(self, x):
        """
        Forward pass.
        """
        for layer in self._layers:
            x = layer(x)

        return x

    def get_action(self, x):
        """
        Do a forward pass and return tensor of chosen actions.
        """
        # shape : (batch_size, 5)

        rewards = self.call(x)

        # pick position of actions (number between 0 and 4)

        # Changed: don't make one random decision for the entire batch, instead decide on individual basis

        randoms = np.random.uniform(0, 1, size=x.shape[0])

        actions = []
        greedy_actions = tf.math.argmax(rewards, axis=-1)

        for i, r in enumerate(randoms):
            if r < EPSILON:
                actions.append(np.random.randint(0, 5))
            else:
                actions.append(greedy_actions[i])

        return tf.convert_to_tensor(actions)

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
            input_map = tf.random.normal(shape=(BATCH_SIZE,MAP_SIZE_x,MAP_SIZE_y,CHANNELS))
            self.get_action(input_map)
            self.save_weights(model_dir + 'random_initialization')

        list_of_files = sorted(os.listdir(model_dir), key=lambda f: os.path.getmtime(model_dir + f))
        latest_file = list_of_files[-1]
        latest_file = latest_file.rsplit('.', 1)[0]

        if latest_file == 'checkpoint':
            latest_file = list_of_files[-2]
            latest_file = latest_file.rsplit('.', 1)[0]

        self.load_weights(model_dir + latest_file)

    def load_random(self, model_dir):
        """
        Loads one model at random.
        """
        while True:
            random_file = random.choice(os.listdir(model_dir))
            if (random_file != 'checkpoint'):
                break

        random_file = random_file.rsplit('.', 1)[0]
        self.load_weights(model_dir + random_file)


    def train(self, batch):
        # n times (old state, action, reward, new_state)
        # bacth = list of last n tuples from replay ReplayBuffer
        # get old STATES
        old_states, actions, rewards, new_states = zip(*batch)
        # pass through models to get q value
        moves = [action.direction for action in actions]

        with tf.GradientTape() as tape:
            q_values_old = self.call(tf.convert_to_tensor(list(old_states)))
            q_target = q_values_old.numpy()
            # get new STATES
            # pass new_states through models -> (batch_size, 5)
            q_values_new = self.call(tf.convert_to_tensor(list(new_states))).numpy()
            # pick maximum -> (batch_size)
            max_q = np.max(q_values_new, axis=-1)
            # compute y vector (batch_size)

            for i, action in enumerate(moves):
                q_target[i, action] = rewards[i] + GAMMA * max_q[i]

            loss = self.loss_function(q_values_old, q_target)

            gradients = tape.gradient(loss, self.trainable_variables)

            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def init_simple_conv(self):

        self._layers = [
            tf.keras.layers.Conv2D(
                input_shape=(MAP_SIZE_x, MAP_SIZE_y, CHANNELS),
                filters=FILTERS, kernel_size=KERNEL_SIZE, padding='valid', activation=tf.nn.relu
            ),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=10, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=5)
        ]

    def init_simple_no_conv(self):
        logging.debug("in init")
        self._layers = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=10, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=5)
        ]
