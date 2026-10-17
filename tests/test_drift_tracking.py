"""Thermal drift, and the instrument's own way of dealing with it.

A real STM drifts: the lab rig walks about a nanometre a minute at 4 K, and any measurement
that takes an hour — a spectroscopy line, a corral repair, a force curve on a chosen atom —
has the sample slide out from under the tip while it runs. Turning that down in the scenarios
would delete the capability worth testing; what the benchmark asks instead is that the agent
notice it and track it.

The instrument offers the means (``Piezo.DriftCompSet``) and the operator drives it (a
measured drift, programmed). These tests pin that the loop closes:
compensation programmed from a measurement stops the motion, the wrong sign makes it worse
rather than doing nothing, and the ramp runs out of piezo eventually.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stmsim.modules import build_dispatcher
from stmsim.physics.rig import RigProfile
from stmsim.physics.world import World


def _world(tmp_path: Path, seed: int = 0) -> World:
    w = World(rig=RigProfile.load("reference-stm"), seed=seed,
              session_dir=tmp_path / "session", time_scale=1.0)
    w._creep = []                       # creep is a separate effect; these tests are about drift
    return w


def _offset(w: World, t: float) -> np.ndarray:
    return np.array(w.drift_offset(t))


def _apparent_velocity(w: World, t0: float, span: float = 600.0) -> np.ndarray:
    """What a two-frame drift measurement reports: how fast features move in the scan frame.

    A feature fixed on the sample sits at scan coordinates ``sample − offset``, so its apparent
    velocity is the negative of the offset's."""
    return -(_offset(w, t0 + span) - _offset(w, t0)) / span


def test_the_rig_drifts_about_a_nanometre_a_minute(tmp_path):
    """The number the scenarios are asked to cope with, from the profile's own calibration."""
    speeds = []
    for seed in range(8):
        v = _world(tmp_path / f"s{seed}", seed).drift_v_m_per_s
        speeds.append(float(np.hypot(v[0], v[1])) * 60 * 1e9)
    assert 0.3 < float(np.median(speeds)) < 3.0, speeds


def test_compensation_programmed_from_a_measurement_stops_the_motion(tmp_path):
    """Type in what was measured and the sample stops moving. This is the whole contract:
    if it did not hold, an agent that did everything right would gain nothing."""
    w = _world(tmp_path)
    t0 = w.clock.sim()
    v = _apparent_velocity(w, t0)
    assert np.hypot(v[0], v[1]) > 0

    build_dispatcher(w).handler_for("Piezo_DriftCompSet")(1, float(v[0]), float(v[1]),
                                                          float(v[2]), 0.9)
    frozen = _offset(w, w.clock.sim())
    for later in (600.0, 3600.0):
        moved = np.abs(_offset(w, w.clock.sim() + later) - frozen)
        assert moved.max() < 1e-12, (later, moved)


def test_compensation_freezes_the_drift_it_does_not_undo_it(tmp_path):
    """An hour of drift that already happened stays happened. Anything else would let an agent
    recover a measurement it took while the sample was walking."""
    w = _world(tmp_path)
    t0 = w.clock.sim()
    w.clock.advance_sim(1800.0)
    before = _offset(w, w.clock.sim())
    assert np.hypot(before[0], before[1]) > 1e-9      # it really did move

    v = _apparent_velocity(w, t0)
    w.set_drift_comp(True, float(v[0]), float(v[1]), float(v[2]))
    after = _offset(w, w.clock.sim() + 1800.0)
    assert np.allclose(after, before, atol=1e-12)


def test_the_wrong_sign_doubles_the_drift_instead_of_doing_nothing(tmp_path):
    """A sign error has to be visible. If it merely did nothing, "I set the compensation" and
    "the compensation works" would look the same, and the verification step would be pointless."""
    w = _world(tmp_path)
    t0 = w.clock.sim()
    v = _apparent_velocity(w, t0)
    plain = np.hypot(*(_offset(w, t0 + 600.0) - _offset(w, t0))[:2])

    w.set_drift_comp(True, -float(v[0]), -float(v[1]), -float(v[2]))
    t1 = w.clock.sim()
    wrong = np.hypot(*(_offset(w, t1 + 600.0) - _offset(w, t1))[:2])
    assert wrong == pytest.approx(2 * plain, rel=0.02), (wrong, plain)


def test_switching_it_off_leaves_the_piezo_where_the_ramp_took_it(tmp_path):
    w = _world(tmp_path)
    t0 = w.clock.sim()
    v = _apparent_velocity(w, t0)
    w.set_drift_comp(True, float(v[0]), float(v[1]), float(v[2]))
    w.clock.advance_sim(1200.0)
    held = _offset(w, w.clock.sim())

    w.set_drift_comp(False, 0.0, 0.0, 0.0)
    assert np.allclose(_offset(w, w.clock.sim()), held, atol=1e-12)
    # and the drift resumes from there, rather than jumping back
    resumed = _offset(w, w.clock.sim() + 600.0) - held
    assert np.hypot(resumed[0], resumed[1]) == pytest.approx(
        np.hypot(*w.drift_v_m_per_s[:2]) * 600.0, rel=0.02)


def test_the_ramp_runs_out_of_piezo(tmp_path):
    """The compensation is a piezo offset, and the scanner is finite. Past the saturation limit
    the sample walks away again — which is why a long campaign still needs re-registration."""
    w = _world(tmp_path)
    rx, ry = w.rig.xy_range_m
    w.set_drift_comp(True, 1e-6, 0.0, 0.0, sat=0.5)        # 1 um/s: saturates almost at once
    accum = w._comp_accum(w.clock.slow() + 100.0)
    assert np.hypot(accum[0], accum[1]) == pytest.approx(0.5 * min(rx, ry) / 2, rel=1e-6)


def test_the_panel_reads_back_what_was_set(tmp_path):
    w = _world(tmp_path)
    d = build_dispatcher(w)
    d.handler_for("Piezo_DriftCompSet")(1, 1.5e-11, -2.5e-11, 0.0, 0.8)
    on, vx, vy, vz, *_rest = d.handler_for("Piezo_DriftCompGet")()
    assert on == 1
    assert (vx, vy, vz) == pytest.approx((1.5e-11, -2.5e-11, 0.0))
    assert w.drift_comp_on and w.drift_comp_v[0] == pytest.approx(1.5e-11)


def test_no_scenario_turns_the_drift_down_to_make_itself_passable(tmp_path):
    """``drift_scale`` exists for scenarios that model a different machine, not as a way to
    make a hard measurement easy. A paper that needs an hour of stability has to earn it with
    drift compensation, because that is the skill being tested."""
    import yaml

    scen = Path(__file__).resolve().parent.parent / "stmbench" / "trackB" / "scenarios"
    offenders = []
    for f in sorted(scen.glob("P*.yaml")):
        ini = (yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get("initial") or {}
        scale = ini.get("drift_scale")
        if scale is not None and float(scale) < 1.0:
            offenders.append(f"{f.name}: drift_scale={scale}")
    assert not offenders, offenders


def test_old_moves_keep_their_creep():
    """Every move creeps on, x(t) = D·γ·ln(1 + t/τ), and the creep of all of them adds up.
    Only the last 20 moves used to be kept, and an old one dropped out of the sum with all its
    creep at once: after a tip was conditioned 150 nm away and back, the sample jumped by
    ±10 nm between frames as the out-and-back pair slid across the window's edge (the scripted
    P4 baseline, seeds 77 and 138). Here 30 short moves follow such a pair; none may move the sample."""
    from stmbench.instrument import Instrument

    inst = Instrument("P4_atom_positioning_cu111", 0)
    w = inst.world
    w.drift_v_m_per_s = np.zeros(3)
    w._creep.clear()
    inst.act("move_tip", x_nm=0.0, y_nm=150.0)
    inst.act("wait", seconds=60.0)
    inst.act("move_tip", x_nm=0.0, y_nm=0.0)
    worst = 0.0
    for k in range(30):
        inst.act("wait", seconds=30.0)
        before = np.array(w.drift_offset(w.clock.sim()))
        inst.act("move_tip", x_nm=6.0 if k % 2 == 0 else 0.0, y_nm=0.0)
        after = np.array(w.drift_offset(w.clock.sim()))
        worst = max(worst, float(np.hypot(*(after - before)[:2])) * 1e9)
    assert worst < 0.05, worst
