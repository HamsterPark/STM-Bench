#!/usr/bin/env python3
"""Grader tests of the stm tasks of P1 (the herringbone reconstruction of Au(111)). HOOK: written, not yet run.

SKIPPED: no reference solution yet (stmbench/reference/p1.py does not exist at the commit this tree is built for), so
there are no fixtures. When it exists, set ``REFERENCE_EXISTS = True``; the cases below then run as for the other
families. They follow the scenario's claims (period within 0.15 nm; three orientations within 6 degrees modulo 180,
the two domains at least 30 degrees apart) and are untested: the shape of the reference's reports (position and frame
fields of the orientation claims) is assumed to follow P4's.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stm_cases as sc  # noqa: E402
from stm_cases import FAIL  # noqa: E402

REFERENCE_EXISTS = False
PERIOD, A, B = "stripe_period_nm", "domain_a_orientation_deg", "domain_b_orientation_deg"
ANGLES = ("stripe_orientation_deg", A, B)


def cases(task, tasks, fx):
    e = fx[task]
    ptol = sc.card_abs_tol(task, PERIOD, "nm", 0.15)
    atol = sc.card_abs_tol(task, A, "deg", 6.0)
    out = sc.common_cases(task, tasks, fx)
    out += sc.abs_shift_cases(e, task, (PERIOD,), ptol, "nm")
    out += sc.abs_shift_cases(e, task, ANGLES, atol, "deg", together=False)
    out += [
        (f"period {ptol + 0.35:.2g} nm off (outside {ptol:g})", FAIL, sc.shifted(e, PERIOD, ptol + 0.35)),
        (f"domain A orientation 25 degrees off (outside {atol:g})", FAIL, sc.shifted(e, A, 25.0)),
        ("both domains reported with the same orientation (must differ by 30 degrees)", FAIL,
         sc.shifted(e, B, sc.reported(e, A) - sc.reported(e, B))),
        ("period and orientations cite no frame", FAIL, sc.set_evidence(e, (PERIOD, *ANGLES), [])),
    ]
    return out


if __name__ == "__main__":
    sys.exit(sc.run_suite("P1", "stm_eval_p1_herringbone_[0-9][0-9]", cases, reference=REFERENCE_EXISTS))
