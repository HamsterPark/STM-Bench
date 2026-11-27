#!/usr/bin/env python3
"""Write the guest image's private ``tasks.json`` from the ``gt.json`` of every task of a data tree.

    python make_tasks_json.py <data root holding stm/> <output file>

The launcher in the guest (``deploy/osworld/guest/stm_launcher.py``) reads ``{task_id: {"scenario", "seed"}}`` and
refuses any other keys. The file is made from the same ``gt.json`` files the grader reads, so the two cannot
disagree. The seeds go only into the output file (mode 0600); the script prints task ids and counts, never a seed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    root, dest = Path(sys.argv[1]), Path(sys.argv[2])
    tasks = {}
    for gt in sorted((root / "stm" / "private" / "reference_private").glob("*/gt.json")):
        d = json.loads(gt.read_text(encoding="utf-8"))
        tasks[gt.parent.name] = {"scenario": str(d["scenario"]), "seed": int(d["seed"])}
    if not tasks:
        print("make_tasks_json: no gt.json found", file=sys.stderr)
        return 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(tasks, f)
    seeds = [t["seed"] for t in tasks.values()]
    print(f"wrote {dest.name}: {len(tasks)} tasks ({', '.join(sorted(tasks))}), {len(set(seeds))} distinct seeds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
