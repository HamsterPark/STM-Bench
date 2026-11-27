#!/usr/bin/env python3
"""Grader tests of the stm tasks of P4 (single-atom positioning), carried over from the delivered suite.

PASS: the reference solution's export; the reported place moved by 0.3 nm (the card allows 0.6); an episode that never
reached ``finish``; a digest hash differing in its last bits. FAIL: the damaged logs of ``stm_cases.common_cases``;
the reported place 0.9 nm off, at the atom's place before the move (-4 nm), and moved by the whole move (+4 nm); a
report citing a frame taken before the atom moved.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stm_cases as sc  # noqa: E402
from stm_cases import FAIL, PASS  # noqa: E402

CLAIM = "target_site"
MOVE_NM = 4.0                          # the move the task asks for, along +x


def moved_place(e: sc.Export, dx_nm: float) -> sc.Export:
    """The reported place moved by dx along x (the value field holds x, so it moves too)."""
    m = e.copy()
    rs = sc.reports(m, CLAIM)
    assert rs, "the fixture holds no report of the claim"
    for r in rs:
        a = r["args"]
        a["x_nm"] = float(a["x_nm"]) + dx_nm
        a["value"] = float(a["value"]) + dx_nm
    return m


def cites_first_frame(e: sc.Export) -> sc.Export:
    m = e.copy()
    f = next(d["frame"] for d in m.digests if "frame" in d)
    for r in sc.reports(m, CLAIM):
        r["args"]["frame"] = f
        r["args"]["evidence"] = [f]
    return m


def cases(task, tasks, fx):
    e = fx[task]
    out = sc.common_cases(task, tasks, fx)
    out += [
        ("reported place moved by 0.3 nm (within 0.6)", PASS, moved_place(e, 0.3)),
        ("reported place 0.9 nm off (outside 0.6)", FAIL, moved_place(e, 0.9)),
        ("reported the atom's place before the move", FAIL, moved_place(e, -MOVE_NM)),
        ("reported place moved by the whole move", FAIL, moved_place(e, MOVE_NM)),
        ("report cites a frame taken before the move", FAIL, cites_first_frame(e)),
    ]
    return out


if __name__ == "__main__":
    sys.exit(sc.run_suite("P4", "stm_eval_p4_atom_[0-9][0-9]", cases))
