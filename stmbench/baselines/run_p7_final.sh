#!/bin/bash
# cluster job, one commit: (1) drags for an out-of-sample check of the carry reader (seeds 100-139, 60 each),
# (2) the baseline on development seeds 0-29, (3) the baseline on the held-out seeds 451-480.
# Output under ~/stmbench/data/p7c; each stage is eight processes.
D=~/stmbench/data/p7c
mkdir -p $D
for k in 0 1 2 3 4 5 6 7; do
  python -m stmbench.baselines.p7_validate $D/drags_$k.pkl $((100 + k*5)) 5 60 > $D/drags_$k.log 2>&1 &
done
wait
python -m stmbench.cli baseline --paper P7 --seeds 30 --first-seed 0 --jobs 8 --json $D/dev_0_29.json > $D/dev.log 2>&1
python -m stmbench.cli baseline --paper P7 --seeds 30 --first-seed 451 --jobs 8 --json $D/heldout_451_480.json > $D/heldout.log 2>&1
echo finished > $D/DONE
