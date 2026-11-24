#!/bin/bash
# cluster job: (1) drags for the carry reader, seeds 0-39, 60 drags each, eight processes of five seeds;
# (2) the baseline on development seeds 0-29, eight processes. Output under ~/stmbench/data/p7b
set -e
D=~/stmbench/data/p7b
mkdir -p $D
for k in 0 1 2 3 4 5 6 7; do
  python -m stmbench.baselines.p7_validate $D/drags_$k.pkl $((k*5)) 5 60 > $D/drags_$k.log 2>&1 &
done
wait
cat $D/drags_*.log | tail -8
python -m stmbench.cli baseline --paper P7 --seeds 30 --first-seed 0 --jobs 8 --json $D/dev_0_29.json
