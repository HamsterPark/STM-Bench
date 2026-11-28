#!/bin/bash
# The P1 baseline's validation as one cluster job (submit.sh re-joins its arguments through ssh, so the
# command lives in a script): eight processes. Usage: p1_validate_job.sh <tag> [seeds=30] [first_seed=400].
# Output: ~/stmbench/p1-base/<tag>/p1.json
set -u
TAG=${1:-heldout}; SEEDS=${2:-30}; FIRST=${3:-400}
mkdir -p "$HOME/stmbench/p1-base/$TAG"
python -m stmbench.cli baseline --paper P1 --seeds "$SEEDS" --first-seed "$FIRST" --jobs 8 \
    --json "$HOME/stmbench/p1-base/$TAG/p1.json"
