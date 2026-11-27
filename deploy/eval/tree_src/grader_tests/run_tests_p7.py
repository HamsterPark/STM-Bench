#!/usr/bin/env python3
"""Grader tests of the stm tasks of P7 (the pull threshold of two atoms).

The correct submission is the export of an episode the privileged reference solution (stmbench/reference/p7.py)
played through the instrument (``fixtures/<id>/export/``). It must PASS with score 1.0, and so must: each threshold
moved by 5 % (half of the 10 % the rule card allows), in either direction where the reference's own error leaves it
inside; an episode that never reached ``finish``; a digest hash that differs in its last bits. Must FAIL: the damaged
logs of ``stm_cases.common_cases``, a threshold 25 % off, a series cut to 3 drags (the card wants 8), a threshold
citing no drag, both thresholds citing the first atom's drags (the second must be another atom), and the first
threshold citing the second atom's drags (only where the reference's two thresholds differ by more than 25 %, so
that the wrong atom's drags cannot bracket the value by chance).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stm_cases as sc  # noqa: E402
from stm_cases import FAIL, PASS  # noqa: E402

C1, C2 = "r_pull_1_kohm", "r_pull_2_kohm"
TOL = 0.10                                   # default if the card cannot be parsed


def truths(task: str):
    """The two atoms' true thresholds as the fixture's reference run recorded them (privileged), or None."""
    p = sc.FIXTURES / task / "reference_run.json"
    if not p.is_file():
        return None
    atoms = json.loads(p.read_text(encoding="utf-8")).get("atoms") or []
    t = [a.get("truth_kohm") for a in atoms]
    return t if len(t) == 2 and all(isinstance(x, (int, float)) and x > 0 for x in t) else None


def cases(task, tasks, fx):
    e = fx[task]
    tol = sc.card_rel_tol(task, C1, TOL)
    out = sc.common_cases(task, tasks, fx)
    out += sc.rel_shift_cases(e, task, {C1: tol, C2: tol})
    out += [(f"{c} 25 % off", FAIL, sc.scaled(e, c, 1.25)) for c in (C1, C2)]
    ev1, ev2 = sc.evidence_of(e, C1), sc.evidence_of(e, C2)
    out += [
        (f"{C1} cites 3 drags only (rule card: at least 8)", FAIL, sc.set_evidence(e, [C1], ev1[:3])),
        ("both thresholds cite no drag", FAIL, sc.set_evidence(e, [C1, C2], [])),
        ("the second threshold cites the first atom's drags (must be another atom)", FAIL,
         sc.set_evidence(e, [C2], ev1)),
    ]
    t = truths(task)
    if t and abs(t[0] / t[1] - 1) > 0.25:
        out.append(("the first threshold cites the second atom's drags", FAIL, sc.set_evidence(e, [C1], ev2)))
    return out


if __name__ == "__main__":
    sys.exit(sc.run_suite("P7", "stm_eval_p7_pull_[0-9][0-9]", cases))
