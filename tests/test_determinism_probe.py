"""The same seed and the same actions should give the same measurement on any machine.

They do not yet: stmsim's only time source is the wall clock (stmsim/physics/clock.py) and
sim time is wall time × time_scale, so a slower or busier host lets more instrument time pass
between two actions — more drift, other tip events, other noise. This probe pins that down so
that the planned virtual instrument clock (sim time advanced by the actions themselves) has a
test to turn green; when it lands, the probe runs on that clock and the xfail goes.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from stmsim.physics.scanner import SIG_Z
from stmsim.scenario import Scenario

SCEN = Path(__file__).resolve().parent.parent / "stmbench" / "trackB" / "scenarios"


def _frame_after(delay_s: float, tmp_path: Path, tag: str) -> np.ndarray:
    sc = Scenario.load(SCEN / "P4_atom_positioning_cu111.yaml")
    w = sc.build_world(0, session_dir=tmp_path / tag)
    st = w.scan
    st.w = st.h = 10e-9
    st.nx = st.ny = 32
    time.sleep(delay_s)                  # a slower or busier machine between the same two actions
    w.scan_start()
    return np.array(w.frame.data[SIG_Z][0], copy=True)


@pytest.mark.xfail(strict=True, reason="sim time follows the wall clock; the virtual instrument "
                                       "clock (benchmark plan M2) is what makes this pass")
def test_the_same_actions_give_the_same_frame_on_a_slower_machine(tmp_path):
    fast = _frame_after(0.0, tmp_path, "fast")
    slow = _frame_after(0.05, tmp_path, "slow")
    assert np.array_equal(fast, slow)
