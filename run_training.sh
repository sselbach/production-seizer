#!/bin/bash

rm buffer.pickle

for episode in {1..1000}
do

  ./halite_mod -d "30 30" -t "python rl_bot00.py $1 $2" "python SelfplayBot.py"

done
