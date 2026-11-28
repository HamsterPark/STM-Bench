#!/usr/bin/env python3
"""Write the guest image's private ``tasks.json`` from the ``gt.json`` of every task of a data tree.

    python make_tasks_json.py <data root holding stm/> <output file> [--families p3,p3r,p4]

The launcher in the guest (``deploy/osworld/guest/stm_launcher.py``) reads ``{task_id: {"scenario", "seed"}}`` and
refuses any other keys. The file is made from the same ``gt.json`` files the grader reads, so the two cannot
disagree. ``--families`` names the selection the tree was built for (default: all seven); the tree must hold exactly
those tasks. The seeds go only into the output file (mode 0600); the script prints task ids and counts, never a seed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import families as F  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("dest", type=Path)
    ap.add_argument("--families", help="the family selection the tree was built for (default: all seven)")
    a = ap.parse_args()
    keys = F.parse_families(a.families)
    want = sorted(t for t in F.task_ids() if F.family_of(t) in keys)
    tasks = {}
    for gt in sorted((a.root / "stm" / "private" / "reference_private").glob("*/gt.json")):
        d = json.loads(gt.read_text(encoding="utf-8"))
        tasks[gt.parent.name] = {"scenario": str(d["scenario"]), "seed": int(d["seed"])}
    if sorted(tasks) != want:
        print(f"make_tasks_json: the tree holds {len(tasks)} tasks, the selection {keys} has {len(want)}: "
              f"missing {sorted(set(want) - set(tasks))}, extra {sorted(set(tasks) - set(want))}", file=sys.stderr)
        return 1
    a.dest.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(a.dest, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(tasks, f)
    seeds = [t["seed"] for t in tasks.values()]
    print(f"wrote {a.dest.name}: {len(tasks)} tasks ({', '.join(sorted(tasks))}), {len(set(seeds))} distinct seeds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
