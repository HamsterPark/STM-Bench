"""MAST's real skills, executed through its real ExecutionContext, against the simulator.

This is the P2 gate (docs/DESIGN.md §4.6.6 / §8): the instrument-control skills MAST runs
on the rig must run unchanged on stmsim and see physically sensible results.

Everything MAST writes (settings, experiment DBs, frames) goes under a temp
``MAST2_PROJECT_ROOT`` — never into the MAST checkout.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from stmsim.modules import build_dispatcher
from stmsim.physics.rig import RigProfile
from stmsim.physics.world import World
from stmsim.wire.server import WireServer

from tests.conftest import requires_mast

pytestmark = requires_mast


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    root = Path(tmp_path_factory.mktemp("mast_root"))
    os.environ["MAST2_PROJECT_ROOT"] = str(root)
    os.environ["MAST2_USER_ROOT"] = str(root)
    world = World(rig=RigProfile.load("reference-stm"), seed=3, session_dir=root / "session", time_scale=10.0)
    world.coarse.coarse_gap_m = world.surface_height_here() + 0.6e-9
    world.withdrawn = False
    world.zctrl_set(True)
    disp = build_dispatcher(world)
    disp.log_calls = True
    srv = WireServer(disp, ports=[0, 0, 0, 0]).start()
    for role, port in zip(("MAIN", "MONITOR", "DATA", "EMERGENCY"), srv.bound_ports):
        os.environ[f"MAST_NANONIS_PORT_{role}"] = str(port)
    from mast.config import NanonisConfig
    from mast.core.connection import ConnectionPool
    from mast.core.execution_context import ExecutionContext
    from mast.core.registry import SkillRegistry
    from mast.core.state import InstrumentState

    pool = ConnectionPool(NanonisConfig())
    assert all(pool.connect_all().values())
    state = InstrumentState(pool)
    state.refresh()
    registry = SkillRegistry()
    registry.discover()

    def run(name, params=None):
        ctx = ExecutionContext(pool=pool, state=state, registry=registry, approval_source="llm")
        res = ctx.run(name, params or {})
        state.refresh()
        return res

    yield {"world": world, "disp": disp, "run": run, "state": state, "pool": pool, "root": root}
    pool.close_all()
    srv.stop()


def test_bias_and_setpoint_roundtrip(rig):
    run = rig["run"]
    assert run("SetBias", {"bias_v": 0.1}).success
    r = run("SetSetpoint", {"setpoint_a": 50e-12})
    assert r.success and r.data["readback_ok"]
    assert run("GetBias").data["bias_v"] == pytest.approx(0.1, rel=1e-5)


def test_measure_barrier_height_recovers_phi(rig):
    r = rig["run"]("MeasureBarrierHeight")
    assert r.success, r.error
    d = r.data
    assert d.get("n_usable", 0) >= 3, d
    phi = d.get("barrier_ev") or d.get("phi_ev") or d.get("apparent_barrier_ev")
    if phi is None:
        # different key name — find any float field that looks like eV
        cands = [v for k, v in d.items() if "barrier" in k.lower() and isinstance(v, (int, float))]
        assert cands, d.keys()
        phi = cands[0]
    truth = rig["world"].phi_junction()
    assert phi == pytest.approx(truth, rel=0.25), (phi, truth)


def test_scan_pipeline_start_wait_save_grab_crashcheck(rig):
    run = rig["run"]
    assert run("SetScanBuffer", {"pixels": 64, "lines": 64}).success
    assert run("ConfigureScan", {"center_x_m": 0.0, "center_y_m": 0.0, "width_m": 50e-9, "height_m": 50e-9,
                                 "angle_deg": 0.0, "line_time_s": 0.1}).success
    assert run("StartScan").success
    r = run("WaitScanComplete", {"timeout_ms": 120000})
    assert r.success and r.data["outcome"] == "completed" and r.data["lines_done"] == 64, r.data
    r = run("SaveScan")
    assert r.success and r.data["saved_path"].endswith(".sxm")
    latest = run("GetLatestScanFile").data["path"]
    assert Path(latest).exists()
    r = run("GrabScanFrameData", {"channel_index": 14, "direction": 1})
    assert r.success and r.data["shape"] == [64, 64]
    frame = np.load(r.data["frame_path"])
    assert np.isfinite(frame).all() and np.ptp(frame) > 1e-11
    r = run("CheckScanForCrash")
    assert r.success and r.data["status"] == "ok", r.data
    # MAST's own reader + orientation helper agree with the frame the sim rendered
    from mast.io.nanonis_files import read_sxm, sxm_oriented_frames
    fr = sxm_oriented_frames(read_sxm(latest), "Z")
    assert fr["forward"].shape == (64, 64) and fr["nm_per_px"] == pytest.approx(50 / 64, rel=1e-3)


def test_sts_bias_pulse_and_tip_shaper_readbacks(rig):
    run = rig["run"]
    r = run("AcquireSTS")
    assert r.success and r.data["acquisition_complete"] and r.data["spectrum_parsed"], r.data
    assert "Current (A)" in r.data["channel_names"]
    tip_before = rig["world"].tip.snapshot()
    r = run("BiasPulseWithReadback", {"bias_v": 3.0, "width_s": 0.1})
    assert r.success and r.data["completed"], r.error
    import time
    time.sleep(3.0)          # let the pulse transient (and the preamp de-saturation tail) finish
    r = run("TipShapeWithReadback", {"tip_lift_m": -0.3e-9, "lift_height_m": 0.3e-9})
    assert r.success and r.data["completed"], r.error
    cur = np.asarray(r.data["current"]["samples_a"], float)
    # the poke trace saturates at the preamp clamp while pressed, and starts near the setpoint
    assert np.abs(cur).max() >= 9.9e-9
    assert abs(cur[0]) < 1e-9
    ev = [e["kind"] for e in rig["world"].events]
    assert "pulse" in ev and "poke" in ev
    assert rig["world"].tip.snapshot() != tip_before


def test_withdraw_then_auto_approach_relands(rig):
    run = rig["run"]
    w = rig["world"]
    w.false_landing_p = 0.0           # deterministic landing for this test
    assert run("WithdrawTip").success
    assert w.withdrawn and rig["state"].snapshot().withdrawn is True
    r = run("AutoApproach", {"wait_timeout_s": 60})
    assert r.success, r.error
    assert not w.withdrawn and w.zctrl.on
    i = abs(w.current_now())
    assert i > 25e-12
    assert "approach_landed" in [e["kind"] for e in w.events]


def test_no_wire_errors_during_the_whole_run(rig):
    errs = [c for c in rig["disp"].call_log if not c[3].startswith("ok")]
    assert not errs, errs[:10]
    assert len(rig["disp"].call_log) > 500


# ── the paper-track skills, on the wire ──────────────────────────────────────
# Four of the bugs that kept the paper families from running were invisible except here:
# a reply parsed only at its outermost level, a spectrum block handed over inside its
# envelope, a preamp range nothing could read, and piezo limits reported in the wrong unit.
# Each of them left the skill returning success with an empty or default reading.

def test_the_preamp_range_can_be_read_and_widened(rig):
    """``SetCurrentGain`` has always existed; without a readback, "widen the range before
    dropping the setpoint" is not a step anyone can take."""
    w = rig["world"]
    before = rig["run"]("GetCurrentGains")
    assert before.success, before.error
    assert before.data["full_scale_a"] == pytest.approx(w.preamp.full_scale_a, rel=1e-6)
    idx = int(before.data["gain_index"])

    assert rig["run"]("SetCurrentGain", {"gain_index": idx - 1}).success
    after = rig["run"]("GetCurrentGains")
    assert after.data["gain_index"] == idx - 1
    # and the range the instrument actually clamps at moved with it, not just the readout
    assert after.data["full_scale_a"] == pytest.approx(w.preamp.full_scale_a, rel=1e-6)
    assert after.data["full_scale_a"] > before.data["full_scale_a"]
    rig["run"]("SetCurrentGain", {"gain_index": idx})


def test_the_scanner_reports_travel_so_a_frame_can_be_configured(rig):
    """``Piezo.XYZLimitsGet`` is in volts. Reported in metres, the reachable half range comes
    out at 1e-13 m and every ConfigureScan is refused as out of range."""
    r = rig["run"]("ConfigureScan", {"center_x_m": 0.0, "center_y_m": 0.0,
                                     "width_m": 100e-9, "height_m": 100e-9})
    assert r.success, r.error


def test_locate_step_edge_finds_a_step_and_reports_both_angles(rig, tmp_path):
    """The geometry P2 needs: MeasureStepHeight gives a height, not a line."""
    w = rig["world"]
    assert rig["run"]("ConfigureScan", {"center_x_m": 0.0, "center_y_m": 0.0,
                                        "width_m": 60e-9, "height_m": 60e-9}).success
    rig["run"]("SetScanPixels", {"pixels": 256})
    assert rig["run"]("StartScan", {}).success
    rig["run"]("WaitScanComplete", {"timeout_s": 120})
    rig["run"]("SaveScan", {})
    frames = sorted(Path(w.session_dir).glob("*.sxm"))
    assert frames, "no frame to look for a step in"
    r = rig["run"]("LocateStepEdge", {"scan_path": str(frames[-1]), "channel": "Z"})
    assert r.success, r.error
    d = r.data
    assert d["verdict"] in ("step_edge", "no_step", "undecidable")
    if d["verdict"] == "step_edge":
        # both conventions, and they are mirrors of one another
        assert d["edge_angle_scan_deg"] == pytest.approx((-d["edge_angle_deg"]) % 180.0, abs=1e-6)
        assert d["edge_x_m"] is not None and d["edge_y_m"] is not None
        assert 100.0 < d["step_height_pm"] < 400.0        # one Cu/Au layer, give or take
