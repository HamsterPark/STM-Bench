"""Scenario loading, fault scheduling and truth criteria (no MAST needed)."""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pytest

from stmbench.trackB.truth_criteria import CRITERIA, judge
from stmsim.faults import Fault, FaultScheduler
from stmsim.physics.surface import Feature
from stmsim.scenario import Scenario
from stmsim.wire.server import Dispatcher, WireServer

SCEN = Path(__file__).resolve().parent.parent / "stmbench" / "trackB" / "scenarios"


@pytest.mark.parametrize("name", sorted(p.name for p in SCEN.glob("*.yaml")))
def test_every_scenario_builds_a_world(name, tmp_path):
    sc = Scenario.load(SCEN / name)
    w = sc.build_world(seed=1, session_dir=tmp_path / "s")
    t = w.truth()
    assert t["tip"]["radius_nm"] > 0
    assert sc.success["kind"] in CRITERIA          # the registry is the source of truth
    assert sc.family and sc.task.strip() and sc.budget["sim_hours"] > 0
    for f in sc.faults:
        assert f["kind"] and (f.get("at_sim_s") is not None or f.get("when"))
    # a real (unstarted) wire server: comms faults write its latency/drop knobs, and the
    # fault library reports a comms kind as "unknown" when there is no server to write to
    srv = WireServer(Dispatcher(), ports=[])
    sched = sc.scheduler(w, server=srv)
    sched.tick()   # nothing due at t=0 except unconditional ones
    for f in sched.faults:
        assert f.fired == (f.at_sim_s is not None and f.at_sim_s <= w.clock.sim())
    if sc.family == "B9":
        # the >5 s reply must already be armed before the agent's first STS (see B9 notes)
        assert srv.latency_for("BiasSpectr.Start") > 5.0 and srv.latency_for("Bias.Get") == 0.0
    # the truth is what the judge sees: every criterion must at least evaluate on a fresh world
    judge(sc.success["kind"], {**t, "events_tail": w.events[-50:]}, sc.success.get("thresholds"))


_HIDDEN_TRUTH_WORDS = ("真值", "隐藏", "apex_sigma", "radius_nm", "lambda", "contamination", "immediate")


@pytest.mark.parametrize("name", sorted(p.name for p in SCEN.glob("*.yaml")))
def test_task_text_does_not_leak_the_hidden_truth(name):
    """The task is what the agent reads; everything the scenario hides belongs in ``notes``."""
    from stmsim.calibrate.working_points import _MATERIALS   # the repo's own material matcher

    sc = Scenario.load(SCEN / name)
    task = sc.task
    low = task.lower()
    for word in _HIDDEN_TRUTH_WORDS:
        assert word.lower() not in low, f"{name}: task text contains {word!r}"
    for f in sc.faults:
        assert f["kind"] not in task, f"{name}: task text names the fault kind {f['kind']!r}"
    real = str((sc.initial or {}).get("material") or sc.material or "")
    if real:
        short = {n for n, pat in _MATERIALS if re.search(pat, real.lower())}
        named = {n for n, pat in _MATERIALS if re.search(pat, low)}
        if named - short:    # the task poses as a different material: the real one must stay hidden
            assert real not in task, f"{name}: task names the real material {real!r}"
            assert not (named & short), f"{name}: task names the real material {short}"


def test_flat_region_truth_and_criterion(tmp_path):
    sc = Scenario.load(SCEN / "B3_find_flat_terrace.yaml")
    w = sc.build_world(seed=5, session_dir=tmp_path / "s")
    t = w.truth()
    assert t["flat_window_nm"] in w.FLAT_WINDOW_LADDER_NM or t["flat_window_nm"] == 0.0
    # a single-terrace site: remove every step edge -> the 200 nm ladder top is reached
    site = w.surface.site
    site.boundaries = np.array([], dtype=float)
    site.phi_blobs.clear()
    t = w.truth()
    assert t["flat_window_nm"] == 200.0
    assert judge("flat_region_found", t, sc.success.get("thresholds")).success
    # operator damage inside the window shrinks it below 50 nm
    sx, sy = w.sample_xy()
    w.surface.add_feature(Feature(sx + 15e-9, sy, 1e-9, 3e-9, kind="crater"))
    t = w.truth()
    assert t["flat_window_nm"] < 50.0
    v = judge("flat_region_found", t, sc.success.get("thresholds"))
    assert not v.success and not v.details["checks"]["flat_window"]
    # a step edge 20 nm from the tip bounds the window at 30 nm
    site.features.clear()
    site.boundaries = np.array([20e-9])
    site.wander_amp = 0.0
    site.step_angle = 0.0          # step edge along y at x = +20 nm
    w.tip_x = w.tip_y = 0.0
    w._creep.clear()
    w.drift_v_m_per_s = np.zeros(3)
    assert w.truth()["flat_window_nm"] == 30.0


def test_sts_event_and_criterion(tmp_path):
    sc = Scenario.load(SCEN / "B6_sts_clean.yaml")
    w = sc.build_world(seed=6, session_dir=tmp_path / "s")
    w.surface.site.phi_blobs.clear()          # judge the criterion, not the seed's contamination
    t = w.truth()
    assert t["n_sts"] == 0 and t["sts_last"] is None
    v = judge("sts_acquired", t)
    assert not v.success and not v.details["checks"]["sts_taken"]
    w.sts_curve(-1.0, 1.0, 200)
    t = w.truth()
    assert w.events[-1]["kind"] == "sts" and w.events[-1]["jump"] is False
    assert t["n_sts"] == 1 and t["sts_last"]["n"] == 200
    assert judge("sts_acquired", t).success, judge("sts_acquired", t).details
    # a sweep that never reaches the surface-state onset is not the spectrum asked for
    w.sts_curve(0.0, 1.0, 200)
    assert not judge("sts_acquired", w.truth()).details["checks"]["covers_onset"]
    # degenerate sweep
    w.sts_curve(-1.0, 1.0, 4)
    assert not judge("sts_acquired", w.truth()).details["checks"]["enough_points"]
    # a tip change during the sweep is flagged on the event (forced: λ high, rng seeded)
    w.tip.lambda_per_s = 0.05
    jumps = []
    for _ in range(12):
        w.sts_curve(-1.0, 1.0, 200)
        jumps.append(w.events[-1]["jump"])
    assert any(jumps)
    t = w.truth()
    if t["sts_last"]["jump"]:
        assert not judge("sts_acquired", t).details["checks"]["no_tip_jump"]


def test_watch_survived_criterion_and_z_limit_truth(tmp_path):
    sc = Scenario.load(SCEN / "B7_long_watch.yaml")
    w = sc.build_world(seed=7, session_dir=tmp_path / "s")
    kinds = [f["kind"] for f in sc.faults]
    assert "tip_change_burst" in kinds and "thermal_drift_to_limit" in kinds
    t = w.truth()
    assert not t["z_at_limit"] and t["n_crash"] == 0
    v = judge("watch_survived", t, sc.success.get("thresholds"))
    assert not v.success and not v.details["checks"]["frames_saved"]
    assert v.details["checks"]["tunnelling"] and v.details["checks"]["z_in_range"]
    # enough frames saved -> pass
    w.frames_saved.extend([f"f{i}.sxm" for i in range(6)])
    assert judge("watch_survived", w.truth(), sc.success.get("thresholds")).success
    # Z pinned at the +limit (the drift signature) is not "survived", even though flags say tunnelling
    w.zctrl.limits_enabled = True
    w.coarse.coarse_gap_m = w.surface_height_here() - 400e-9     # sample crept into the tip
    t = w.truth()
    assert t["zctrl_on"] and not t["withdrawn"] and t["z_at_limit"]
    v = judge("watch_survived", t, sc.success.get("thresholds"))
    assert not v.success and not v.details["checks"]["z_in_range"]
    # a crash counts against the watch
    w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
    w._crash("test", severity=1.0)
    v = judge("watch_survived", w.truth(), sc.success.get("thresholds"))
    assert not v.details["checks"]["no_crash"] and w.truth()["n_crash"] == 1


def test_blunt_scenario_fails_and_repaired_tip_passes(tmp_path):
    sc = Scenario.load(SCEN / "B5_repair_blunt.yaml")
    w = sc.build_world(seed=2, session_dir=tmp_path / "s")
    v = judge("tip_repaired", w.truth())
    assert not v.success and not v.details["checks"]["sharp"]
    w.tip.radius_m = 0.8e-9
    w.tip.apex_sigma_m = 0.06e-9       # sharpness is judged on the apex, not the radius
    w.tip.lambda_per_s = 1e-4
    w.tip.metastable = False
    v = judge("tip_repaired", w.truth())
    assert v.success, v.details
    # a blunt mesoscopic tip that happens to end in one clean atom still images atoms:
    w.tip.radius_m = 6e-9
    assert judge("tip_repaired", w.truth()).details["checks"]["sharp"]
    # ...and a fine radius with a smeared apex does not
    w.tip.radius_m = 0.8e-9
    w.tip.apex_sigma_m = 0.3e-9
    assert not judge("tip_repaired", w.truth()).details["checks"]["sharp"]


def test_thermal_drift_fault_pins_z_at_limit(tmp_path):
    sc = Scenario.load(SCEN / "B8_z_at_limit_drift.yaml")
    w = sc.build_world(seed=3, session_dir=tmp_path / "s")
    w.clock.time_scale = 5000.0
    w.clock.slow_scale = 5000.0     # this test accelerates the slow physics too
    sched = sc.scheduler(w)
    z0 = w.z_now()
    assert abs(z0) < 50e-9
    # advance sim time past the fault and let the creep eat the range
    w.clock.advance_sim(100.0)
    sched.tick()
    assert any(e["kind"] == "fault" for e in w.events)
    time.sleep(0.25)       # 5000× → 1250 s sim → 0.3 nm/s × 1250 = 375 nm of creep
    z = w.z_now()
    i = abs(w.current_now())
    assert z == pytest.approx(w.zctrl.limit_high_m)
    assert i > 10 * w.zctrl.setpoint_a        # the "10 nA at +169.5 nm" signature


def test_tip_damage_and_comms_faults(tmp_path):
    sc = Scenario.load(SCEN / "B5_repair_multi.yaml")
    w = sc.build_world(seed=4, session_dir=tmp_path / "s")
    sched = FaultScheduler(w, [Fault(kind="tip_damage", params={"mode": "low_phi", "phi_ev": 0.8}, at_sim_s=0)])
    sched.tick()
    assert w.tip.phi_ev == 0.8 and w.phi_junction() < 3.0
    f = Fault(kind="feedback_oscillation", params={"i_gain_factor": 100}, when="on_command:Scan.Action")
    sched2 = FaultScheduler(w, [f])
    sched2.tick("Bias.Get")
    assert not f.fired
    sched2.tick("Scan.Action")
    assert f.fired and not w.loop_params().stable


def test_honeypot_scoring_caps_passivity():
    t = {"tip": {"multi": False, "radius_nm": 1, "lambda_per_s": 1e-4, "metastable": False, "dead": False},
         "phi_junction_ev": 4.0, "withdrawn": False, "zctrl_on": True, "ghost_contrast": 0.0}
    assert judge("honeypot", t, wrong_actions=0, progress=False).partial == 0.5
    assert judge("honeypot", t, wrong_actions=2, progress=True).partial == 0.5
    assert judge("honeypot", t, wrong_actions=0, progress=True).success
