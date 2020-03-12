#!/bin/bash

if hash python3 2>/dev/null; then
    ./halite_mod -d "30 30" -t "python3 proto2.py" "python3 RandomBot.py"
else
    ./halite_mod -d "30 30" -t "python proto2.py" "python RandomBot.py"
fi
