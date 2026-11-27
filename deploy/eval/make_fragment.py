#!/usr/bin/env python3
"""The MANIFEST fragment of the stm domain: every file of ``<root>/stm`` with its size, sha256 and mtime.

    python make_fragment.py <data root holding stm/> <output file>

``build_tree.py`` writes the same fragment when it builds; run this on a finished tree (for instance after the real
fixtures were added) to write it again. It prints the number of tasks and files, nothing else.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def manifest(root: Path) -> dict:
    out = {}
    for p in sorted((root / "stm").rglob("*"), key=lambda q: q.relative_to(root).as_posix()):
        if p.is_file() and "__pycache__" not in p.parts:
            out[p.relative_to(root).as_posix()] = {"bytes": p.stat().st_size,
                                                   "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                                   "mtime": int(p.stat().st_mtime)}
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    root, dest = Path(sys.argv[1]), Path(sys.argv[2])
    m = manifest(root)
    tasks = sum(1 for k in m if k.startswith("stm/tasks/"))
    dest.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
    print(f"{dest.name}: {tasks} tasks, {len(m)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
