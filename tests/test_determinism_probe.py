"""The same seed and the same actions should give the same measurement on any machine.

On the host clock they do not: there sim time is wall time × time_scale, so a slower or
busier host lets more instrument time pass between two actions — more drift, other tip
events, other noise. That is the clock the wire-protocol clients (MAST) run on, and the
strict xfail below keeps the limitation on record. A world built on the virtual instrument
clock (``build_world(..., virtual_clock=True)``, which the benchmark instrument uses) moves
only when an action advances it, and gives the same frame whatever the host does.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from stmsim.physics.scanner import SIG_Z
from stmsim.scenario import Scenario

SCEN = Path(__file__).resolve().parent.parent / "stmbench" / "trackB" / "scenarios"


def _frame_after(delay_s: float, tmp_path: Path, tag: str, *, virtual: bool = False) -> np.ndarray:
    sc = Scenario.load(SCEN / "P4_atom_positioning_cu111.yaml")
    w = sc.build_world(0, session_dir=tmp_path / tag, virtual_clock=virtual)
    st = w.scan
    st.w = st.h = 10e-9
    st.nx = st.ny = 32
    time.sleep(delay_s)                  # a slower or busier machine between the same two actions
    w.scan_start()
    if virtual:
        w.clock.advance(st.frame_time_s)
        assert not w.scan_running()
    return np.array(w.frame.data[SIG_Z][0], copy=True)


@pytest.mark.xfail(strict=True, reason="on the host clock sim time follows the wall clock")
def test_on_the_host_clock_a_slower_machine_changes_the_frame(tmp_path):
    fast = _frame_after(0.0, tmp_path, "fast")
    slow = _frame_after(0.05, tmp_path, "slow")
    assert np.array_equal(fast, slow)


def test_on_the_virtual_clock_the_same_actions_give_the_same_frame(tmp_path):
    fast = _frame_after(0.0, tmp_path, "vfast", virtual=True)
    slow = _frame_after(0.05, tmp_path, "vslow", virtual=True)
    assert np.array_equal(fast, slow)
