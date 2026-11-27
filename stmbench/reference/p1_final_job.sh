#!/bin/bash
# The published P1 reference run and the chance audit as one cluster job: the reference on seeds 0-59
# (0-29 published, 30-59 a stability check), then the chance audit with its hold-out.
# Usage: p1_final_job.sh <tag>. Output: ~/stmbench/p1-ref/<tag>/{p1.json,audit.json}
set -u
TAG=${1:-final}
mkdir -p "$HOME/stmbench/p1-ref/$TAG"
python -m stmbench.cli reference --paper P1 --seeds 60 --first-seed 0 --jobs 8 \
    --json "$HOME/stmbench/p1-ref/$TAG/p1.json"
python -m stmbench.cli audit chance --scenario P1_barth1990_au111 --seeds 2000 --holdout --jobs 8 \
    --json "$HOME/stmbench/p1-ref/$TAG/audit.json"
