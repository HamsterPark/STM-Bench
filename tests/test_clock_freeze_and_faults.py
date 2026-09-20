"""Clock pause/resume and the immediate thermal-drift fault (no MAST needed)."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from stmsim.physics.clock import Clock
from stmsim.scenario import Scenario

SCEN = Path(__file__).resolve().parents[1] / "stmbench" / "trackB" / "scenarios"


# ────────────────────────── Clock.pause / resume ──────────────────────────

def test_pause_freezes_wall_sim_and_slow_and_resume_is_continuous():
    c = Clock(time_scale=20.0, slow_scale=3.0)
    c.advance_sim(100.0)
    time.sleep(0.05)
    assert not c.is_paused
    c.pause()
    assert c.is_paused
    w0, s0, sl0 = c.wall(), c.sim(), c.slow()
    time.sleep(0.15)
    # frozen: not a single reading moves
    assert c.wall() == w0
    assert c.sim() == s0
    assert c.slow() == sl0
    assert c.paused_total_s >= 0.15
    c.resume()
    assert not c.is_paused
    # continuous: right after resume the clock is where it stopped, not 0.15 s later
    w1 = c.wall()
    assert w1 - w0 < 0.05
    assert c.paused_total_s >= 0.15
    time.sleep(0.1)
    # and it runs again, with the sim / slow scaling intact
    w2 = c.wall()
    assert 0.08 < w2 - w0 < 0.2
    # scaling intact — compare while frozen so wall/sim/slow are read at one instant
    with c.paused():
        w2 = c.wall()
        assert c.sim() == pytest.approx(w2 * 20.0 + 100.0)
        assert c.slow() == pytest.approx(w2 * 3.0 + 100.0)
        # advance_sim semantics unchanged: offsets both virtual clocks, not wall
        c.advance_sim(50.0)
        assert c.wall() == w2
        assert c.sim() == pytest.approx(w2 * 20.0 + 150.0)
        assert c.slow() == pytest.approx(w2 * 3.0 + 150.0)


def test_nested_pause_needs_matching_resumes():
    c = Clock()
    c.pause()
    c.pause()
    w0 = c.wall()
    time.sleep(0.05)
    c.resume()                       # one resume: still paused
    assert c.is_paused
    time.sleep(0.05)
    assert c.wall() == w0
    c.resume()                       # second resume: runs again
    assert not c.is_paused
    assert c.paused_total_s >= 0.1
    time.sleep(0.05)
    assert 0.03 < c.wall() - w0 < 0.15
    c.resume()                       # unmatched resume is a no-op
    assert not c.is_paused


def test_paused_context_manager_resumes_on_exit_even_on_error():
    c = Clock()
    with c.paused():
        assert c.is_paused
        w0 = c.wall()
        time.sleep(0.05)
        assert c.wall() == w0
    assert not c.is_paused
    with pytest.raises(RuntimeError):
        with c.paused():
            raise RuntimeError("boom")
    assert not c.is_paused
    # nesting through the context manager too
    c.pause()
    with c.paused():
        assert c.is_paused
    assert c.is_paused
    c.resume()
    assert not c.is_paused


# ────────────────────────── immediate thermal drift (B8) ──────────────────────────

def test_b8_immediate_drift_pins_z_at_limit_within_two_seconds_wall(tmp_path):
    sc = Scenario.load(SCEN / "B8_z_at_limit_drift.yaml")
    fault = next(f for f in sc.faults if f["kind"] == "thermal_drift_to_limit")
    assert fault["params"].get("immediate") is True
    w = sc.build_world(seed=3, session_dir=tmp_path / "s")
    sched = sc.scheduler(w)
    assert abs(w.z_now()) < 50e-9
    assert not w.z_at_limit()
    t_start = time.perf_counter()
    w.clock.advance_sim(100.0)       # past at_sim_s (offset 21600 + 100 > 21660)
    sched.tick()
    assert any(e["kind"] == "fault" and e["fault"] == "thermal_drift_to_limit" for e in w.events)
    # pinned NOW, with no creep left running and no wall time spent waiting for it
    assert w.z_at_limit()
    assert w.z_now() == pytest.approx(w.zctrl.limit_high_m)
    assert abs(w.current_now()) > 10 * w.zctrl.setpoint_a
    assert w._coarse_creep_v == 0.0
    assert time.perf_counter() - t_start < 2.0
    assert w.clock.wall() < 2.0


def test_immediate_drift_jump_is_at_least_1p2_times_the_fine_z_span(tmp_path):
    sc = Scenario.load(SCEN / "B8_z_at_limit_drift.yaml")
    w = sc.build_world(seed=3, session_dir=tmp_path / "s")
    sched = sc.scheduler(w)
    gap0 = w.coarse.coarse_gap_m
    w.clock.advance_sim(100.0)
    sched.tick()
    span = w.zctrl.limit_high_m - w.zctrl.limit_low_m
    assert gap0 - w.coarse.coarse_gap_m >= 1.2 * span - 1e-15
