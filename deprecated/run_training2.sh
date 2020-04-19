#!/bin/bash

for episode in {1..100}
do

  ./halite_mod -d "30 30" -t "python rl_bot2.py $1 $2" "python selfplay_bot2.py"

done
