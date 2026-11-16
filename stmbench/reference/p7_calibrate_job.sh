#!/bin/bash
# The P7 calibration as one cluster job (submit.sh re-joins its arguments through ssh, so the
# command lives in a script). Usage: p7_calibrate_job.sh <tag> [seeds=30] [n_drags=32] [jobs=8] [extra args]
# Output: ~/stmbench/data/p7/<tag>.json
set -u
TAG=${1:-cal}; SEEDS=${2:-30}; NDRAGS=${3:-32}; JOBS=${4:-8}
shift 4 2>/dev/null || shift $#
mkdir -p "$HOME/stmbench/data/p7"
python -m stmbench.reference.p7_calibrate --seeds "$SEEDS" --jobs "$JOBS" --n-drags "$NDRAGS" \
    --out "$HOME/stmbench/data/p7/$TAG.json" "$@"
