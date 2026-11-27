#!/usr/bin/env python3
"""Grader tests of the stm tasks of P2 (standing waves on Cu(111)). HOOK: written, not yet run.

SKIPPED: no reference solution yet (stmbench/reference/p2.py does not exist at the commit this tree is built for), so
there are no fixtures. When it exists, set ``REFERENCE_EXISTS = True``. The cases follow the scenario's claims (band
edge within 15 meV, effective mass within 2 %, evidence: spectra near a scatterer or dI/dV maps) and are untested.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stm_cases as sc  # noqa: E402
from stm_cases import FAIL  # noqa: E402

REFERENCE_EXISTS = False
E0, M = "e0_mev", "m_eff"


def cases(task, tasks, fx):
    e = fx[task]
    etol = sc.card_abs_tol(task, E0, "meV", 15.0)
    mtol = sc.card_rel_tol(task, M, 0.02)
    out = sc.common_cases(task, tasks, fx)
    out += sc.abs_shift_cases(e, task, (E0,), etol, "meV")
    out += sc.rel_shift_cases(e, task, {M: mtol})
    out += [
        (f"band edge 40 meV off (outside {etol:g})", FAIL, sc.shifted(e, E0, 40.0)),
        (f"effective mass {2.5 * mtol:.0%} off (outside {mtol:.0%})", FAIL, sc.scaled(e, M, 1 + 2.5 * mtol)),
        ("both values cite no spectrum or map", FAIL, sc.set_evidence(e, (E0, M), [])),
        ("both values cite a single spectrum (card: at least 8, or 4 maps)", FAIL,
         sc.set_evidence(e, (E0, M), sc.evidence_of(e, E0)[:1])),
    ]
    return out


if __name__ == "__main__":
    sys.exit(sc.run_suite("P2", "stm_eval_p2_waves_[0-9][0-9]", cases, reference=REFERENCE_EXISTS))
