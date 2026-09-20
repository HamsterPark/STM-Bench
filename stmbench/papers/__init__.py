"""Scripted baselines for the paper families — the solvability gate, not a leaderboard entry.

Before a paper scenario can ask a model to reproduce a measurement, somebody has to show the
measurement is reachable with the tools on offer inside the stated budget. That is what these
scripts are for: mode C runs one per family over ten seeds, and the spread of
(baseline answer − truth) is what the claim tolerances are then derived from.

Two rules keep the gate honest:

* a baseline may only use what the instrument shows it — MAST skills through
  ``host.context()`` and the files they leave in ``session/``. It must never read
  ``world.truth()``; ``tests/test_papers.py`` checks that by inspection.
* it reports through :class:`stmbench.harness.results.ResultSink`, so the fold the judge sees
  is the same shape an LLM episode produces.
"""
from __future__ import annotations

from typing import Callable

from .p1_barth1990 import run_p1
from .p2_standing_waves import run_p2
from .p3_corral import run_p3
from .p4_atom_positioning import run_p4
from .p5_qplus_force import run_p5

#: family → (host, scenario, seed, extra) → skill_result
PAPER_BASELINES: dict[str, Callable] = {
    "P1": run_p1,
    "P2": run_p2,
    "P3": run_p3,
    "P4": run_p4,
    "P5": run_p5,
}

__all__ = ["PAPER_BASELINES", "run_p1", "run_p2", "run_p3", "run_p4", "run_p5"]
