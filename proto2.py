import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square
import random



import window
import signal

from proto import ProtoSeizer

from replay_buffer import ReplayBuffer


myID, game_map = hlt.get_init()
hlt.send_init("Prototype 2")

r = ReplayBuffer(1000)
r.load_from_file()

## INIT MODEL

proto = ProtoSeizer()
proto.load_last("models/")


def termination_handler(signal, frame):
    r.save_to_file()
    proto.save("models/")

signal.signal(signal.SIGTERM, termination_handler)

## START MAIN LOOP
while True:

    owned_squares, current_states = window.get_windows(game_map.contents)

    moves = proto.get_action(current_states)

    hlt.send_frame(list(moves))

    game_map.get_frame()

    new_states = window.get_windows_for_squares(game_map.contents, owned_squares)

    tuples = zip(current_states, list(moves), new_states)

    r.add_tuples(tuples)
