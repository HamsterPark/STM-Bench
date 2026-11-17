#!/bin/bash
# cluster job, one commit: the baseline on development seeds 0-29 (regression) and the fresh held-out seeds 481-510.
# Output under ~/stmbench/data/p7d.
D=~/stmbench/data/p7d
mkdir -p $D
python -m stmbench.cli baseline --paper P7 --seeds 30 --first-seed 0 --jobs 8 --json $D/dev_0_29.json > $D/dev.log 2>&1
python -m stmbench.cli baseline --paper P7 --seeds 30 --first-seed 481 --jobs 8 --json $D/heldout_481_510.json > $D/heldout.log 2>&1
echo finished > $D/DONE
