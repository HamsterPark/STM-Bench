#!/usr/bin/env python3
"""The MANIFEST fragment of the stm domain: every file of ``<root>/stm`` with its size, sha256 and mtime.

    python make_fragment.py <data root holding stm/> <output file> [--families p3,p3r,p4]

``build_tree.py`` writes the same fragment when it builds; run this on a finished tree (for instance after the real
fixtures were added) to write it again. ``--families`` names the selection the tree was built for (default: all
seven): the tree must hold exactly those tasks and no file of another family. It prints the number of tasks and
files, nothing else.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import families as F  # noqa: E402


def manifest(root: Path, keys: list[str] | None = None) -> dict:
    """Every file of ``<root>/stm``. With ``keys`` (family keys), a file of another family is an error (the tree was
    not built for this selection); files that belong to no family are kept."""
    out = {}
    for p in sorted((root / "stm").rglob("*"), key=lambda q: q.relative_to(root).as_posix()):
        if p.is_file() and "__pycache__" not in p.parts:
            rel = p.relative_to(root).as_posix()
            if keys is not None and F.family_of_path(rel) not in (None, *keys):
                raise SystemExit(f"make_fragment: {rel} belongs to a family outside the selection {keys}")
            out[rel] = {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                        "mtime": int(p.stat().st_mtime)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("dest", type=Path)
    ap.add_argument("--families", help="the family selection the tree was built for (default: all seven)")
    a = ap.parse_args()
    keys = F.parse_families(a.families)
    m = manifest(a.root, keys)
    tasks = [k for k in m if k.startswith("stm/tasks/")]
    want = [t for t in F.task_ids() if F.family_of(t) in keys]
    if sorted(Path(k).stem for k in tasks) != sorted(want):
        print(f"make_fragment: the tree holds {len(tasks)} tasks, the selection {keys} has {len(want)}", file=sys.stderr)
        return 1
    a.dest.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
    print(f"{a.dest.name}: {len(tasks)} tasks, {len(m)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
