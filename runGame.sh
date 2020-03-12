#!/bin/bash

for episode in {1..1000}
do

  ./halite_mod -d "30 30" -t "python proto2.py $1" "python RandomBot.py"

done
