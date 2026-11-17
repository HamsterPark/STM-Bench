#!/bin/bash
# The P7 calibration as one cluster job (submit.sh re-joins its arguments through ssh, so the
# command lives in a script): seeds 0-29, eight processes, 32 drags per atom, three variants one
# after the other. Usage: p7_calibrate_job.sh <tag> [seeds=30]
# Output: ~/stmbench/data/p7/<tag>-{stair2mV,fit2mV,stair5mV}.json
set -u
TAG=${1:-cal}; SEEDS=${2:-30}
mkdir -p "$HOME/stmbench/data/p7"
run() { python -m stmbench.reference.p7_calibrate --seeds "$SEEDS" --jobs 8 --n-drags 32 --out "$HOME/stmbench/data/p7/$TAG-$1.json" "${@:2}"; }
run stair2mV --manip-v 0.002 --design stair
run fit2mV --manip-v 0.002 --design fit
run stair5mV --manip-v 0.005 --design stair
