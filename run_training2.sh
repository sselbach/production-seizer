#!/bin/bash

for episode in {1..10}
do

  ./halite_mod -d "30 30" -t "python rl_bot2.py $1 $2" "python random_bot.py"

done
