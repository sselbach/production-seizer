#!/bin/bash

for episode in {1..100}
do

  ./halite_mod -d "30 30" -t -s 2168 "python rl_bot.py $1 $2" "python selfplay_bot.py"

done
