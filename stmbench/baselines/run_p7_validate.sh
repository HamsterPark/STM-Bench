#!/bin/bash
# cluster job: drags for the carry reader, seeds 0-39, 60 drags each, eight processes of five seeds
set -e
mkdir -p ~/stmbench/data/p7
for k in 0 1 2 3 4 5 6 7; do
  python -m stmbench.baselines.p7_validate ~/stmbench/data/p7/drags_$k.pkl $((k*5)) 5 60 > ~/stmbench/data/p7/drags_$k.log 2>&1 &
done
wait
cat ~/stmbench/data/p7/drags_*.log
