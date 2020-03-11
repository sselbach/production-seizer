import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import  Model
import numpy as np
from hyperparameters import *
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square
import random
import sys
import glob
import os
from datetime import datetime

# if running cuda type gpu while excecuting in the terminal
if 'gpu' in sys.argv:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

class ProtoSeizer(Model):
    def __init__(self):

        """
        Initialize Model
        """
        super(ProtoSeizer, self).__init__()
        # 1 conv 3x3 kernel, no padding
        self.conv = tf.keras.layers.Conv2D(input_shape=(MAP_SIZE_x, MAP_SIZE_y, CHANNELS), filters=FILTERS, kernel_size=KERNEL_SIZE, padding='valid', activation= tf.nn.relu)
        # flatten
        self.flatten = tf.keras.layers.Flatten()
        # dense with 5 output neurons, no activation
        self.dense = tf.keras.layers.Dense(units=5)


    # input x:(MAP_SIZE_x, MAP_SIZE_y, CHANNELS) map
    # output: (5,1) rewards
    def call(self, x):
        """
        Forward pass.
        """
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

    def get_action(self, x):
        """
        Do a forward pass and return tensor of chosen actions.
        """
        # shape : (batch_size, 5)
        rewards = self.call(x)
        # pick position of actions (number between 0 and 4)
        if random.random()<EPSILON:
            # make random choice with probability of epsilon
            action = tf.random.uniform(shape=tf.shape(x)[0], maxval=len(rewards), minval=0, dtype=tf.int32)
        else:
            # else pick action with maximumm reward
            action = tf.math.argmax(rewards, axis=-1)
        # return action
        return action

    def save(self, model_dir):
        """
        Save weights of current configuration.
        """
        self.save_weights(model_dir +  datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p"))

    def load_last(self, model_dir):
        """
        Loads the latest model.
        """
        list_of_files = glob.glob(model_dir+'*') # * means all if need specific format then *.csv
        if len(list_of_files) <=  1:
            input_map = tf.random.normal(shape=(BATCH_SIZE,MAP_SIZE_x,MAP_SIZE_y,CHANNELS))
            self.get_action(input_map)
            self.save_weights(model_dir + 'random_initialization')
        os.chdir(model_dir)
        list_of_files = sorted(os.listdir(os.getcwd()), key=os.path.getmtime)
        latest_file = list_of_files[-1]
        latest_file = latest_file.rsplit('.',1)[0]
        self.load_weights(latest_file)

    def load_random(self, model_dir):
        """
        Loads one model at random.
        """
        while True:
            random_file = random.choice(os.listdir(model_dir))
            if (random_file!='checkpoint'):
                break
        random_file = random_file.rsplit('.',1)[0]
        self.load_weights(model_dir+random_file)


if __name__ == "__main__":
    model = ProtoSeizer()
    print("initialized")
    # create fake input (1,7,7,6)
    #for i in range(5):
    #    input_map = tf.random.normal(shape=(BATCH_SIZE,MAP_SIZE_x,MAP_SIZE_y,CHANNELS))
    #    actions = model.get_action(input_map)
    #    model.save('models/prototest/')
    model.load_last('models/prototest/')
    print("loaded")
