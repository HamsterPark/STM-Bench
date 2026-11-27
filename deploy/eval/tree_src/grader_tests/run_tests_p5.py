#!/usr/bin/env python3
"""Grader tests of the stm tasks of P5 (force spectroscopy of an Fe adatom with a qPlus sensor).

The correct submission is the export of an episode the privileged reference solution (stmbench/reference/p5.py)
played through the instrument (``fixtures/<id>/export/``, see reference_export.py). It must PASS with score 1.0, and so
must: each reported value moved by half of its relative tolerance (15 % F_min, 8 % decay length, 20 % binding energy;
the tolerances are read from the rule card), in either direction where the reference's own error leaves it inside;
an episode that never reached ``finish``; a digest hash that differs in its last bits. Must FAIL: the damaged logs of
``stm_cases.common_cases``, each value 2.5 tolerances off, values citing only the curves taken over the atom, only
the background curves, or no curve.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stm_cases as sc  # noqa: E402
from stm_cases import FAIL, PASS  # noqa: E402

CLAIMS = {"f_min_pn": 0.15, "f_decay_pm": 0.08, "e_bind_mev": 0.20}      # defaults if the card cannot be parsed


def curves(e: sc.Export) -> tuple[list[str], list[str]]:
    """(atom curves, background curves) of the log: the Z sweeps, told apart by how close the sweep started (the
    reference takes the curves over the atom closer than the background ones)."""
    zs = [(r["params"].get("z_offset_nm"), r["result"]["spectrum"]) for r in e.records
          if r.get("kind") == "act" and r.get("name") == "zspec" and isinstance(r.get("result"), dict)
          and r["result"].get("spectrum")]
    assert zs, "the fixture holds no Z sweep"
    near = min(z for z, _ in zs)
    return [n for z, n in zs if z == near], [n for z, n in zs if z != near]


def cases(task, tasks, fx):
    e = fx[task]
    tols = {c: sc.card_rel_tol(task, c, d) for c, d in CLAIMS.items()}
    atom, bare = curves(e)
    out = sc.common_cases(task, tasks, fx)
    out += sc.rel_shift_cases(e, task, tols)
    for c, t in tols.items():
        out.append((f"{c} 2.5 tolerances off (+{2.5 * t:.0%})", FAIL, sc.scaled(e, c, 1 + 2.5 * t)))
    both = list(tols)
    out += [
        ("values cite only the curves over the atom (no background)", FAIL, sc.set_evidence(e, both, atom)),
        ("values cite only background curves (none over the atom)", FAIL, sc.set_evidence(e, both, bare)),
        ("values cite no curve", FAIL, sc.set_evidence(e, both, [])),
    ]
    return out


if __name__ == "__main__":
    sys.exit(sc.run_suite("P5", "stm_eval_p5_force_[0-9][0-9]", cases))
