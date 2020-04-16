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

    def __init__(self):
        super().__init__()

        self.production_layer = tf.keras.layers.Dense(units=4, activation=tf.keras.activations.sigmoid)

        self.strength_layer = tf.keras.layers.Dense(units=4, activation=tf.keras.activations.sigmoid)

        self.dense1 = tf.keras.layers.Dense(units=8, activation=tf.keras.activations.relu)

        self.dense2 = tf.keras.layers.Dense(units=8, activation=tf.keras.activations.relu)

        self.output_layer = tf.keras.layers.Dense(units=5, activation=None)

        self.optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

        self.loss_function = tf.keras.losses.Huber()


    def call(self, x):

        strength = x[:,0,:]

        production = x[:,1,:]

        s_o = self.strength_layer(strength)
        p_o = self.production_layer(production)

        stack = tf.concat([s_o, p_o], -1)

        #stack = tf.concat([strength, production], -1)

        o = self.dense1(stack)

        o = self.dense2(o)

        o = self.output_layer(o)

        return o


    def get_actions(self, states, epsilon=0):

        q_values = self(states)

        #logging.debug(q_values)

        actions = tf.math.argmax(q_values, axis=-1).numpy()

        #mask = states[:, 0, STILL] == 0

        #logging.debug(mask)
        #logging.debug(states[:, 0, 4])

        randoms = np.random.binomial(n=1, p=epsilon, size=actions.shape)
        logging.debug(epsilon)
            #random_actions = np.random.randint(0, 5, actions.shape)

        random_actions = np.random.choice(a=[0, 1, 2, 3, 4], size=actions.shape, p=[0.1, 0.1, 0.1, 0.1, 0.6])

        actions = actions * np.logical_not(randoms) + randoms * random_actions

        #actions = actions * np.logical_not(mask) + STILL * np.int_(mask)
        #logging.debug(actions)
        #logging.debug(mask)
        return actions

    def load_random(self, model_dir):
        """
        Loads one of the last n models at random.
        """
        list_of_files = glob.glob(model_dir+'*')

        # if model directory is empty make a dummy forward pass
        if len(list_of_files) <=  1:
            input_map = tf.random.normal(shape=(1,2,NEIGHBORS))
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
        #old_states, actions, rewards, new_states = zip(*batch)

        #moves = np.array([action.direction for action in actions])

        #actions = np.array(actions)

        with tf.GradientTape() as tape:

            output = self.call(batch["new_states"])

            y = batch["rewards"] + GAMMA * tf.reduce_max(output, axis = 1)

            q_values = self.call(batch["old_states"])

            action_indices = np.append(np.arange(batch["actions"].shape[0]).reshape(-1,1), batch["actions"].reshape(-1, 1), axis = 1)

            q_values = tf.gather_nd(q_values, action_indices)

            #loss = self.loss_function(y, q_values)
            #loss = tf.square(y - q_values)
            loss = self.loss_function(y, q_values)

            #loss_mean = tf.reduce_mean(loss, axis=-1).numpy()
            reward_mean = tf.reduce_mean(y, axis=-1).numpy()

            gradients = tape.gradient(loss, self.trainable_variables)

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
            input_map = tf.random.normal(shape=(1, 2, NEIGHBORS))
            self.get_actions(input_map)
            self.save_weights(model_dir + 'random_initialization')

        list_of_files = sorted(os.listdir(model_dir), key=lambda f: os.path.getmtime(model_dir + f))
        latest_file = list_of_files[-1]
        latest_file = latest_file.rsplit('.', 1)[0]

        if latest_file == 'checkpoint':
            latest_file = list_of_files[-2]
            latest_file = latest_file.rsplit('.', 1)[0]

        self.load_weights(model_dir + latest_file)
