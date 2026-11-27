#!/usr/bin/env python3
"""Write ``gt.json`` of this task: the scenario and the seed of its hidden world, and the grader image.

The seed is drawn once by the task's author and never changes with a solver's output; this script
reads only the seeds file and writes only gt.json beside itself. The task id is the name of the
directory this script is in.

    python make_gt.py --seeds seeds.json --grader-image stmbench-grader:<12 hex digits of the commit>

``seeds.json`` maps a task id to ``{"scenario": "<scenario id>", "seed": <int>, ...}``
(other keys are ignored); the mapping may also sit under a ``"tasks"`` key.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", required=True, type=Path)
    ap.add_argument("--grader-image", required=True)
    ap.add_argument("--task-id", default=HERE.name)
    a = ap.parse_args()
    table = json.loads(a.seeds.read_text(encoding="utf-8"))
    table = table.get("tasks", table)
    if a.task_id not in table:
        print(f"make_gt: {a.seeds.name} has no entry for task {a.task_id!r}", file=sys.stderr)
        return 1
    entry = table[a.task_id]
    gt = {"scenario": str(entry["scenario"]), "seed": int(entry["seed"]), "grader_image": a.grader_image}
    (HERE / "gt.json").write_text(json.dumps(gt, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {HERE.name}/gt.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
