import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square
import random
import window0
import signal
from proto import ProtoSeizer
from replay_buffer import ReplayBuffer
import tensorflow as tf
import sys
import logging
LOG_FILENAME = 'example.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)

logging.warning('This message should go to the log file')

#logfile.log("did sth")
#if 'gpu' in sys.argv:
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

myID, game_map = hlt.get_init()
hlt.send_init("Prototype2")
logging.debug("initialized")
r = ReplayBuffer(1000)
r.load_from_file()

## INIT MODEL

proto = ProtoSeizer()
logging.debug("initialized model")
proto.load_last("models/prototest/")
logging.debug("loaded last model")

def termination_handler(signal, frame):
    r.save_to_file()
    proto.save("models/")
    sys.exit(0)

signal.signal(signal.SIGTERM, termination_handler)

logging.debug("hello")
## START MAIN LOOP
while True:

    logging.debug("BLA2")

    owned_squares, current_states = window0.get_windows(game_map.contents)

    logging.debug("BLA2")
    logging.debug(current_states.shape)
    moves = proto.get_action(current_states)

    moves = moves.numpy().tolist()

    moves = [Move(square, move) for square, move in zip(owned_squares, moves)]

    logging.debug(moves)

    hlt.send_frame(moves)
    logging.debug("sent frames")
    game_map.get_frame()
    logging.debug("got frames")
    new_states = window0.get_windows_for_squares(game_map.contents, owned_squares)
    logging.debug("new statst")
    tuples = zip(current_states, moves, new_states)

    r.add_tuples(tuples)
    logging.debug("added tuples")
