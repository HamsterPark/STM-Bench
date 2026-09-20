"""MAST's composite skills end to end on the simulator, hosted by ``RuntimeHost``.

The P2 gate (docs/DESIGN.md §4.6.6 item 6) asks for each instrument composite to run
unchanged on stmsim with at least one success and one expected failure. ``test_mast_skills_e2e``
covers the primitives and the small composites through a bare ``ExecutionContext``; this
module covers the ones that need the full ``CoreRuntime`` (tip registry, scan-map scope,
marker sink — see ``stmbench.harness.runtime_host``):

* ``PreScanCheck`` — a verdict from the *file* path, the borrowed scan frame handed back,
  plus the live-buffer caveat that made MAST prefer the file (unscanned rows read as 0,
  not NaN, so a NaN gate cannot see an incomplete buffer);
* ``AchieveAtomicResolution`` — success on a sharp tip; a clean, tip-untouched verdict
  on a blunt one;
* ``ForgeAuTip`` — a smoke bounded by its own site/round caps (< 60 s wall), and the
  full minimum budget run marked ``slow`` (MAST's ``time_budget_h`` floor is 0.1 h and it
  is measured with ``time.time()``, so the sim's ``time_scale`` cannot shorten it).

``RelocateCoarseXY`` is exercised in ``test_runtime_host`` (marker sink / coord epoch).

Two tips, both with λ = 0 so nothing changes under our feet, and ``apex_radius_ref_m``
pinned so the simulator never resamples the apex smearing:

* sharp — radius 1 nm, apex σ 0.06 nm (< σ* = 0.09 nm; the B4 scenario's ``good_tip``);
* blunt — radius 8 nm, apex σ 0.5 nm (the ceiling): the lattice cannot reach the image.

Everything MAST writes goes under the host's temp ``MAST2_PROJECT_ROOT``.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest

from stmsim.physics.rig import RigProfile
from stmsim.physics.tip import Apex, Tip
from stmsim.physics.world import World

from tests.conftest import requires_mast, slow

pytestmark = requires_mast

SEED = 3
TIME_SCALE = 20.0
SHARP = dict(radius_nm=1.0, apex_sigma_nm=0.06)
BLUNT = dict(radius_nm=8.0, apex_sigma_nm=0.5)

# the operator's working point for an atomic frame on Au(111): 20 mV, 500 pA, 5 nm,
# 256 px, 0.1 s/line — the same point trackB.derive_thresholds calibrated σ* at
ATOMIC_WP = dict(bias_v=0.02, setpoint_a=500e-12, size_m=5e-9, pixels=256, line_time_s=0.1)


def _world(root: Path, *, radius_nm: float, apex_sigma_nm: float) -> World:
    tip = Tip(material="W", form="qplus", radius_m=radius_nm * 1e-9,
              rng=np.random.default_rng(SEED + 1), apexes=[Apex(0.0, 0.0, 0.0, 1.0)],
              lambda_per_s=0.0)
    tip.apex_sigma_m = apex_sigma_nm * 1e-9
    tip.apex_radius_ref_m = tip.radius_m            # pin: no resampling
    w = World(rig=RigProfile.load("reference-stm"), seed=SEED, session_dir=root / "session",
              time_scale=TIME_SCALE, tip=tip, adsorbate_density_per_um2=2.0)
    w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
    w.withdrawn = False
    w.zctrl_set(True)
    w.transients.clear()
    w.achievable_z_tip()
    return w


@pytest.fixture
def host_factory(tmp_path):
    """Build a world + started RuntimeHost; every host is stopped at teardown."""
    from stmbench.harness.runtime_host import RuntimeHost

    hosts = []

    def make(tip: dict, name: str = "h"):
        w = _world(tmp_path / name, **tip)
        host = RuntimeHost(w, tmp_path / name / "out")
        host.start()
        hosts.append(host)
        return host

    yield make
    for h in hosts:
        h.stop()


def _wire_errors(host) -> list:
    return [c for c in host.dispatcher.call_log
            if not c[3].startswith("ok") and "NeedModule" not in c[3]]


def _set_atomic_working_point(host) -> tuple[float, float]:
    """What an operator does before asking for an atomic frame: bias, setpoint, a 5 nm
    frame on a step-free spot, 256 px, 0.1 s/line. Through MAST's own skills, over the wire."""
    from stmbench.trackB.derive_thresholds import _flat_centre

    w = host.world
    cx, cy = _flat_centre(w, ATOMIC_WP["size_m"] * 1e9)
    for name, params in (
        ("SetBias", {"bias_v": ATOMIC_WP["bias_v"]}),
        ("SetSetpoint", {"setpoint_a": ATOMIC_WP["setpoint_a"]}),
        ("SetScanBuffer", {"pixels": ATOMIC_WP["pixels"], "lines": ATOMIC_WP["pixels"]}),
        ("ConfigureScan", {"center_x_m": cx, "center_y_m": cy, "width_m": ATOMIC_WP["size_m"],
                           "height_m": ATOMIC_WP["size_m"], "angle_deg": 0.0,
                           "line_time_s": ATOMIC_WP["line_time_s"]}),
    ):
        r = host.run_skill(name, params)
        assert r.success, (name, r.error)
    return cx, cy


# ────────────────────────────── PreScanCheck ──────────────────────────────

def test_prescan_check_judges_from_the_saved_file_and_hands_the_frame_back(host_factory):
    host = host_factory(SHARP, "prescan")
    w = host.world
    before = (w.scan.cx, w.scan.cy, w.scan.w, w.scan.h)
    r = host.run_skill("PreScanCheck", {"center_x_m": 0.0, "center_y_m": 0.0, "width_m": 50e-9,
                                        "pixels": 64, "line_time_s": 0.1})
    assert r.success, r.error
    d = r.data
    # a verdict, not an abstention: a sharp single-apex tip retraces its own line
    assert d["tip_ready"] is True and d["recommendation"] == "none", d
    sim = d["similarity"]
    assert sim is not None and math.isfinite(sim) and sim >= 0.8, d
    assert d["trace_retrace_correlation"] == sim
    assert "read_failure" not in d and "abstain_reason" not in d and "safe_mode_raw" not in d
    # the number came from the .sxm on disk, not from the live buffer — the only path that
    # can produce a tip verdict since 2026-08-14 (the buffer path is inconclusive-only)
    src = str(d["quality_source"])
    assert src.startswith("sxm:"), src
    assert Path(src[len("sxm:"):]).exists()
    # "borrow the frame, hand it back": the 50 nm × 2.5 nm strip is not left behind
    after = (w.scan.cx, w.scan.cy, w.scan.w, w.scan.h)
    assert after == pytest.approx(before, rel=1e-6, abs=1e-12), (before, after)
    # and the scan is stopped, with the fact declared for the next skill's precondition
    assert not w.scan_running()
    assert d.get("_verified_state") == {"scan_running": False}
    assert not _wire_errors(host), _wire_errors(host)[:5]


def test_live_buffer_unscanned_rows_are_zero_not_nan(host_factory):
    """The caveat behind PreScanCheck's file-first rule: a frame grabbed from the live
    buffer mid-scan has finite zeros where nothing was scanned yet, so a NaN-row gate
    (which is how the file path detects an incomplete frame) passes it as complete.
    The saved .sxm of the same partial frame carries NaN rows — that is the gate's input."""
    host = host_factory(SHARP, "buffer")
    w = host.world
    assert host.run_skill("SetScanBuffer", {"pixels": 64, "lines": 64}).success
    assert host.run_skill("ConfigureScan", {"center_x_m": 0.0, "center_y_m": 0.0, "width_m": 50e-9,
                                            "height_m": 50e-9, "angle_deg": 0.0,
                                            "line_time_s": 0.5}).success
    assert host.run_skill("StartScan").success
    time.sleep(0.6)                   # ≈ 12 sim-seconds at 20×: a dozen of the 64 lines
    try:
        r = host.run_skill("GrabScanFrameData", {"channel_index": 14, "direction": 1})
        assert r.success, r.error
        fr = np.load(r.data["frame_path"])
        assert fr.shape == (64, 64)
        assert np.isfinite(fr).all(), "live buffer must not carry NaN"
        zero_rows = np.all(fr == 0, axis=1)
        assert 0 < int(zero_rows.sum()) < 64, int(zero_rows.sum())
        assert np.ptp(fr[~zero_rows]) > 1e-12    # the scanned rows hold terrain
    finally:
        assert host.run_skill("StopScan").success
    assert not w.scan_running()
    # the same partial frame, saved: the unscanned rows are NaN in the file
    assert host.run_skill("SaveScan").success
    latest = host.run_skill("GetLatestScanFile").data["path"]
    from mast.io.nanonis_files import read_sxm, sxm_oriented_frames
    saved = sxm_oriented_frames(read_sxm(latest), "Z")["forward"]
    nan_rows = np.all(np.isnan(saved), axis=1)
    assert 0 < int(nan_rows.sum()) < 64, int(nan_rows.sum())
    assert not _wire_errors(host), _wire_errors(host)[:5]


# ────────────────────────────── AchieveAtomicResolution ──────────────────────────────

def test_achieve_atomic_resolution_on_a_sharp_tip(host_factory):
    host = host_factory(SHARP, "atomic_sharp")
    w = host.world
    _set_atomic_working_point(host)
    t0 = time.perf_counter()
    r = host.run_skill("AchieveAtomicResolution", {"total_budget_min": 5.0, "allow_relocate": False})
    elapsed = time.perf_counter() - t0
    assert r.success, r.error
    d = r.data
    assert d["achieved"] is True and d["verified"] == "atomic_resolved", d
    assert d["rungs_used"] == ["r1"], d["rungs"]           # the cheapest rung, no tip work
    r1 = d["rungs"][0]
    assert r1["outcome"] == "found" and r1["attempts"] == 1 and r1["n_undetermined"] == 0, r1
    assert d["angular_concentration"] > 60.0                 # ScanUntilAtomic's concentration_min
    path = Path(d["final_path"])
    assert path.exists() and path.suffix == ".sxm"
    assert not d["aborted"]
    # the tip was never touched: no pulse / poke, snapshot unchanged
    assert not any(e["kind"] in ("pulse", "poke") for e in w.events)
    assert w.tip.snapshot()["apex_sigma_nm"] == pytest.approx(SHARP["apex_sigma_nm"])
    # the benchmark's own measurement agrees: it counts frames on disk, not MAST's word
    from stmbench.harness.episode import count_atomic_frames
    from stmbench.trackB.truth_criteria import judge
    frames = count_atomic_frames(w.session_dir, w.surface.material.name)
    assert frames["error"] is None and frames["n_passed"] >= 1, frames
    verdict = judge("atomic_resolution", w.truth(), frames_passed_atomic=frames["n_passed"])
    assert verdict.success, verdict.as_dict()
    assert not _wire_errors(host), _wire_errors(host)[:5]
    assert elapsed < 60.0, elapsed


def test_achieve_atomic_resolution_on_a_blunt_tip_stops_with_a_clean_verdict(host_factory):
    """No lattice on a σ = 0.5 nm apex. With tip conditioning and relocation disabled the
    ladder has nowhere to go: the verdict must be "absent" (not "undecidable" — that would
    mean the evidence, not the tip), the tip must be left alone, and the report must say why."""
    host = host_factory(BLUNT, "atomic_blunt")
    w = host.world
    snap_before = w.tip.snapshot()
    _set_atomic_working_point(host)
    r = host.run_skill("AchieveAtomicResolution",
                       {"total_budget_min": 5.0, "allow_relocate": False, "allow_tip_conditioning": False})
    d = r.data
    assert d, r.error
    assert d["achieved"] is False and d["verified"] is None and d["final_path"] is None, d
    assert not d["aborted"] and not d.get("abort_reason"), d
    outcomes = [(x["rung"], x["outcome"]) for x in d["rungs"]]
    assert outcomes == [("r1", "no_lattice"), ("triage", "tip_conditioning_disabled")], outcomes
    r1 = d["rungs"][0]
    assert r1["attempts"] == 3 and r1["scans_failed"] == 0 and r1["assess_failed"] == 0, r1
    # Every frame is judged "absent" (concentration ≈ 0). When two consecutive frames read
    # bit-identically (0.0 / coverage 1.0 — what a lattice-free frame always reads),
    # ScanUntilAtomicResolution's repeat-frame detector books the later one as a repeat
    # rather than a judged frame; either way all three attempts are accounted for and
    # none is "undetermined".
    assert r1["frames_judged"] >= 1 and r1["frames_judged"] + r1["repeat_frames"] == 3, r1
    assert r1["n_undetermined"] == 0, "blunt tip must read as 'absent', not 'undecidable'"
    # One reading per attempt. The concentration itself is NOT asserted: it is an
    # intermediate number whose noise realisation differs run to run (the sim's noise RNG
    # is consumed by MAST's monitor threads at wall-clock pace), and a lattice-free frame
    # has read 71.6 on one run while AssessAtomicResolution still (rightly) said "absent"
    # on its other gates. The verdict is the contract; the number is a report field.
    assert len(r1["concentrations"]) == 3, r1
    assert d["rungs_used"] == ["r1"]
    assert d["advice"].startswith("走完了允许的档位仍没拿到原子分辨"), d["advice"]
    # clean = the instrument was not asked to do anything to the tip
    assert not any(e["kind"] in ("pulse", "poke") for e in w.events)
    assert w.tip.snapshot() == snap_before
    from stmbench.trackB.truth_criteria import judge
    assert not judge("atomic_resolution", w.truth(), frames_passed_atomic=0).success
    assert not _wire_errors(host), _wire_errors(host)[:5]


# ────────────────────────────── ForgeAuTip ──────────────────────────────

FORGE_PARAMS = {"forge_pixels": 64, "time_budget_h": 0.1}   # 0.1 h is MAST's floor for the budget


def test_forge_au_tip_smoke_one_site_one_round(host_factory):
    """The whole outer loop once: forced first pulse → verify scan → out of sites. Bounded by
    the skill's own caps (max_sites / max_rounds_per_site), so it does not wait for the
    time budget; measured 6 s wall on the sim (host trial 2026-08-28)."""
    host = host_factory(SHARP, "forge1")
    w = host.world
    snap_before = w.tip.snapshot()
    t0 = time.perf_counter()
    r = host.run_skill("ForgeAuTip", {**FORGE_PARAMS, "max_sites": 1, "max_rounds_per_site": 1})
    elapsed = time.perf_counter() - t0
    assert elapsed < 60.0, elapsed
    d = r.data
    assert d, r.error
    # not "ready" ⇒ not a success: the skill makes no "good enough" concession
    assert r.success is False and "针尖未达标" in str(r.error), r.error
    assert d["outcome"] == "sites_exhausted" and d["sites_worked"] == 1, d["outcome"]
    site = d["sites"][0]
    assert site["site"] == 1 and site["rounds"] == 1 and site["outcome"] == "verify_exhausted", site
    phases = [p["phase"] for p in site["phases"]]
    assert phases == ["pulse", "verify"], phases
    pulse, verify = site["phases"]
    assert pulse["fired"] == 1 and pulse["log"][0]["bias_v"] == pytest.approx(10.0)
    assert pulse["log"][0]["outcome"] == "satisfied"
    assert Path(verify["scan_path"]).exists() and verify["passed"] is False
    # the pulse reached the physics: one pulse event, the tip reshaped, a frame saved after
    kinds = [e["kind"] for e in w.events]
    assert kinds.count("pulse") == 1 and "scan_saved" in kinds, kinds
    assert [e.kind for e in w.tip.events] == ["pulse"]
    assert w.tip.snapshot() != snap_before and w.tip.length_m != 0.0
    # the report ends with a measured readback of the instrument, not a plan
    left = d["left_at_measured"]
    assert left["any_read"] and left["feedback_on"] is True and left["z_controller_status"] == "On"
    assert not d["aborted"]
    assert not _wire_errors(host), _wire_errors(host)[:5]


@slow
def test_forge_au_tip_runs_out_its_minimum_budget(host_factory):
    """The task-level cap: 0.1 h (= 360 s of ``time.time()``) of forging on a sharp tip that
    the verify keeps vetoing. Budget exhaustion is a *failure* with the per-site record;
    measured 367 s wall (2026-08-28). Deselect with ``-m "not slow"``."""
    host = host_factory(SHARP, "forge_full")
    t0 = time.perf_counter()
    r = host.run_skill("ForgeAuTip", FORGE_PARAMS)
    elapsed = time.perf_counter() - t0
    d = r.data
    assert d, r.error
    assert r.success is False and d["outcome"] == "time_budget_exhausted", (r.error, d.get("outcome"))
    assert d["time_budget_s"] == pytest.approx(360.0)
    assert 360.0 <= elapsed < 600.0, elapsed          # ran the budget out, overran < one site
    assert d["sites_worked"] >= 1 and all(s.get("rounds", 0) >= 1 for s in d["sites"] if s.get("site"))
    assert d["left_at_measured"]["any_read"]
    assert not d["aborted"]
    assert not _wire_errors(host), _wire_errors(host)[:5]
