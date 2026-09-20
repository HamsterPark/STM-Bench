"""Export STM-Bench's checked-in controller command registry.

This module has no external-client discovery path. The registry is maintained
alongside the simulator and preserves the legacy-compatible wire formats required
by its implemented commands. That preservation is a compatibility decision, not
a claim that this module independently created the protocol or settles provenance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stmsim.wire.spec import SPEC_PATH, load_spec


def export(path: Path | str) -> Path:
    """Write a deterministic copy of the local registry to *path*."""
    destination = Path(path)
    destination.write_text(json.dumps(load_spec(), indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="write a copy of the local registry here")
    args = parser.parse_args(argv)
    if args.out is None:
        print(SPEC_PATH)
        return
    print(export(args.out))


if __name__ == "__main__":
    main()
