#!/bin/bash
# The P7 calibration as one cluster job (submit.sh re-joins its arguments through ssh, so the
# command lives in a script): seeds 0-29, eight processes, the calibrated protocol (fit design,
# 2 mV, 32 drags per atom; the precision-cost curve cuts the series shorter).
# Usage: p7_calibrate_job.sh <tag> [seeds=30]. Output: ~/stmbench/data/p7/<tag>.json
set -u
TAG=${1:-final}; SEEDS=${2:-30}
mkdir -p "$HOME/stmbench/data/p7"
python -m stmbench.reference.p7_calibrate --seeds "$SEEDS" --jobs 8 --n-drags 32 --manip-v 0.002 --design fit \
    --out "$HOME/stmbench/data/p7/$TAG.json"
