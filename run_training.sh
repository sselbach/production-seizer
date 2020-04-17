#!/bin/bash

for episode in {1..1000}
do

  ./halite_mod -d "30 30" -t "python rl_bot.py $1 $2" "python selfplay_bot.py"

done
