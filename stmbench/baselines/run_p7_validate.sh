#!/bin/bash
# cluster job: drags for the carry reader, seeds 0-39, 60 drags each (run from the code directory)
set -e
mkdir -p ~/stmbench/data/p7
python -m stmbench.baselines.p7_validate ~/stmbench/data/p7/drags_0_39.pkl 0 40 60
