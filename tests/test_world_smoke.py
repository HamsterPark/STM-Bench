"""World + dispatcher: the local command registry, the current at the setpoint, truth and
events, piezo travel, and the preamplifier table."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from stmsim.modules import build_dispatcher
from stmsim.physics.rig import RigProfile
from stmsim.physics.world import World
from stmsim.wire.spec import commands


def _approached_world(tmp_path: Path, seed: int = 1, time_scale: float = 50.0) -> World:
    w = World(rig=RigProfile.load("reference-stm"), seed=seed, session_dir=tmp_path / "session",
              time_scale=time_scale)
    w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
    w.withdrawn = False
    w.zctrl_set(True)
    return w


def test_dispatcher_covers_the_local_registry():
    w = World(rig=RigProfile.load("reference-stm"), seed=0)
    d = build_dispatcher(w)
    # Every locally registered command for a loaded module has a simulator handler.
    loaded = w.rig.modules_loaded
    missing = [command for command, spec in commands().items()
               if spec.module in loaded and d.handler_for(command) is None]
    assert not missing, f"registered commands without handlers: {missing}"


def test_world_current_matches_setpoint_when_feedback_on(tmp_path):
    w = _approached_world(tmp_path)
    w.set_bias(0.1)
    w.set_setpoint(100e-12)
    time.sleep(0.7)  # let the setpoint transient finish
    i = np.array([w.current_now() for _ in range(200)])
    assert abs(np.median(np.abs(i)) - 100e-12) / 100e-12 < 0.15
    assert not w.withdrawn
    z = w.z_now()
    assert w.zctrl.limit_low_m < z < w.zctrl.limit_high_m


def test_truth_and_events(tmp_path):
    w = _approached_world(tmp_path)
    t = w.truth()
    assert t["tip"]["n_apex"] == 1 and not t["withdrawn"]
    w.withdraw()
    assert w.truth()["withdrawn"] and w.events[-1]["kind"] == "withdraw"


def test_piezo_limits_are_volts_so_the_scanner_has_travel(tmp_path):
    """``Piezo.XYZLimitsGet`` returns DAC volts, not metres.

    Control software works out a reachable half range as ``calibration × min(10 V, |limit|)``. Reporting
    the limits in metres makes that product ~1e-13 m, so the scanner looks like it has no
    travel at all and every ``ConfigureScan`` is refused as out of range — which is how this
    was found. With volts the half range comes out at exactly ``Piezo.RangeGet / 2``.
    """
    from stmsim.modules import build_dispatcher

    w = _approached_world(tmp_path)
    d = build_dispatcher(w)
    enabled, xlo, xhi, ylo, yhi, zlo, zhi = d.handler_for("Piezo_XYZLimitsGet")()
    rx, ry, rz = d.handler_for("Piezo_RangeGet")()
    cx, cy, cz = d.handler_for("Piezo_CalibrGet")()
    assert enabled == 1
    assert (xlo, xhi, ylo, yhi) == (-10.0, 10.0, -10.0, 10.0)
    assert cx * min(10.0, abs(xlo), abs(xhi)) == pytest.approx(rx / 2, rel=1e-9)
    assert cy * min(10.0, abs(ylo), abs(yhi)) == pytest.approx(ry / 2, rel=1e-9)
    # Z carries the controller's own limits, converted through the same calibration
    assert cz * zhi == pytest.approx(w.zctrl.limit_high_m, rel=1e-9)
    assert cz * zlo == pytest.approx(w.zctrl.limit_low_m, rel=1e-9)
    assert -10.0 <= zlo < zhi <= 10.0


def test_preamp_gain_table_round_trips_and_the_widest_range_is_10_uA():
    """Gain index i is a full scale of 10 V / 10^(6+i); index 0 (1E6 V/A) is the widest
    range the preamp switches to, which is what the harness registers as the preamp's
    full scale (control software clamps the setpoint ceiling to that fact)."""
    from stmsim.physics.junction import Preamp

    p = Preamp()
    assert p.max_full_scale_a == pytest.approx(10e-6)
    assert p.full_scale_for(3) == pytest.approx(10e-9) and p.full_scale_for(5) == pytest.approx(100e-12)
    for i in range(6):
        assert Preamp.index_for(p.full_scale_for(i)) == i
