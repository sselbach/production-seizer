import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square

import window
import signal

from proto import ProtoSeizer

from replay_buffer import ReplayBuffer

logfile = open("logfile.log", "a+")

myID, game_map = hlt.get_init()
hlt.send_init("Prototype 2")

r = ReplayBuffer(1000)
r.load_from_file()

## INIT MODEL

logfile.write("BLA")

proto = ProtoSeizer()
proto.load_last("models/")

logfile.write("LOADED MODEL")

def termination_handler(signal, frame):
    r.save_to_file()
    proto.save("models/")
    logfile.write("CLOSED")
    logfile.close()

signal.signal(signal.SIGTERM, termination_handler)

logfile.write("HELLO")

## START MAIN LOOP
while True:

    logfile.write("BLA2")

    owned_squares, current_states = window.get_windows(game_map.contents)

    moves = proto.get_action(game_map.contents)

    logfile.write(moves)

    hlt.send_frame(list(moves))

    game_map.get_frame()

    new_states = window.get_windows_for_squares(game_map.contents, owned_squares)

    tuples = zip(current_states, list(moves), new_states)

    r.add_tuples(tuples)
