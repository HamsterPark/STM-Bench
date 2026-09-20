"""Fidelity validation (docs/DESIGN.md §4.6 item 1) + the §4.5 calibration leftovers.

* the sim's good tip at MAST's verify point must land where the real corpus puts a good
  tip — judged by MAST's own detectors through ``read_sxm`` (needs MAST);
* the sim's crashed tip must NOT look like a good one to MAST's trace/retrace detector
  (the plan's ~0.43 band; a static blunt multi-apex tip read 0.96 on 2026-08-28);
* the retrace offset must read back at the real rig's median (1.17 % of the axis) with the
  measured along-line shape (a bulge mid-line, not a closed loop, not a constant);
* the indexer's recipe re-implemented in ``index_stats`` must agree with a hand-built pair;
* the real-side loader / KS / bands must work on a tiny synthetic index (no MAST, no E:);
* ``lambda_from_rowjump`` must reproduce the hand-computed bound and the per-session split;
* ``hysteresis`` must state the storage convention and the nx-trend bound, and refuse to
  invent a per-frame shift.

Bands the simulator does not reach are pinned as STRICT xfails with the finding, so a fix
on the owning side (the surface) flips them to failures that remove the marker. No test
here skips silently: everything that needs MAST carries ``requires_mast``.

Rendering: each verify frame ≈ 0.5–1 s including MAST's detectors.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import requires_mast

VERIFY_BAND = (0.96, 0.99)
CRASHED_TARGET = 0.43              # one real failed verify at the correct speed (forge_tip_reference.txt)
MAST_TIP_READY = 0.80              # MAST's verify gate on trace_retrace_correlation

# MAST's own detectors on 40 real complete 80–120 nm Au frames (the ones the hysteresis
# calibration measured, fb_corr_flip ≥ 0.9), re-run 2026-08-28: the real reference the
# sim is held against below.
REAL_TRC_QUANTILES = (0.856, 0.940, 0.972, 0.981, 0.991)      # p05 p25 p50 p75 p95
REAL_STEP_SPLIT_DECIDED_MAX = 0.125                           # 9 of 10 decided single-tip frames (one 128 px frame: 0.199)


# ── synthetic index used by the no-MAST tests ───────────────────────────────────────
def _write_index(path: Path, *, n_au: int = 40, seed: int = 0) -> Path:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_au):
        big = i % 2 == 0
        rows.append({"path": f"X:/mirror/sessionA/2025-01-0{1 + i % 3}/f{i:03d}.sxm", "comment": "Au(111) clean",
                     "err": "", "acq_frac": 1.0, "range_x_m": (100e-9 if big else 5e-9), "nx": 256.0 if i % 4 else 512.0,
                     "ny": 256.0, "t_fwd_s": 0.586, "t_bwd_s": 0.586, "rec_date": f"2025-01-0{1 + i % 3}",
                     "fb_corr_raw": float(rng.uniform(-0.3, 0.1)), "fb_corr_flip": float(rng.uniform(0.93, 0.995)),
                     "rowjump_frac": float(rng.uniform(0.0, 0.2)), "rowjump_sigma_m": float(rng.uniform(1e-12, 4e-12)),
                     "z_p2p98_m": 5e-10})
    # decoys that every filter must drop: incomplete, errored, other material
    rows.append({**rows[0], "path": "X:/m/incomplete.sxm", "acq_frac": 0.5})
    rows.append({**rows[0], "path": "X:/m/broken.sxm", "err": "short_data"})
    rows.append({**rows[0], "path": "X:/m/cu.sxm", "comment": "Cu(111)"})
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return path


def _overlap_shift(a: np.ndarray, b: np.ndarray, window: int = 12) -> int:
    """Lateral shift maximising the overlap NCC of row-median-removed frames (b's features
    sit ``s`` px to the left of a's when the result is +s) — the calibration's measurement."""
    a = a - np.median(a, axis=1, keepdims=True)
    b = b - np.median(b, axis=1, keepdims=True)
    best = (0, -2.0)
    for s in range(-window, window + 1):
        aa, bb = (a[:, s:], b[:, :a.shape[1] - s]) if s >= 0 else (a[:, :s], b[:, -s:])
        aa = aa - aa.mean()
        bb = bb - bb.mean()
        c = float((aa * bb).sum() / (np.sqrt((aa * aa).sum() * (bb * bb).sum()) + 1e-30))
        if c > best[1]:
            best = (s, c)
    return best[0]


# ── MAST-judged rendering ───────────────────────────────────────────────────────────
@requires_mast
def test_good_tip_verify_frames_match_the_real_corpus_on_the_index_recipe(tmp_path: Path):
    """The apples-to-apples statistic: the indexer's fb_corr (row-median removed, ±6 px)
    on the sim's good tip must sit where the real verify genre sits (real complete Au
    80–120 nm frames, 1352 of them, tips unlabelled: p05 0.875 / p25 0.961 / p50 0.979 /
    p75 0.989) — every frame above the real p05 and the median inside the real IQR — and
    must not be an identical pair (hysteresis exists). The crashed tip must be what the
    truth says it is."""
    from stmsim.validate.fidelity import FidelitySet, compare

    fs = FidelitySet(n=2, tips=("good", "crashed"), points=("verify",), session_dir=tmp_path / "s")
    recs = fs.run()
    assert len(recs) == 4 and all(Path(r["path"]).is_file() for r in recs)
    good = [r for r in recs if r["tip"] == "good"]
    idx = [r["mast"]["index"]["fb_corr_flip"] for r in good]
    assert all(0.875 <= v < 0.9999 for v in idx), idx
    assert 0.961 <= float(np.median(idx)) <= 0.989, idx
    raw = [r["mast"]["index"]["fb_corr_raw"] for r in good]
    assert all(raw[i] < idx[i] - 0.3 for i in range(len(good))), (raw, idx)   # mirrored storage seen
    for r in good:
        assert not r["truth"]["multi"] and r["truth"]["lambda_per_s"] == 0.0 and r["truth"]["tip_change_rows"] == []
        assert r["truth"]["flicker_dz_pm"] == 0.0 and r["truth"]["hysteresis_shape"] == "loop"
        assert r["mast"]["trace_retrace_corr"] > MAST_TIP_READY          # MAST's verify gate (tip_ready at 0.80)
    for r in [r for r in recs if r["tip"] == "crashed"]:
        assert r["truth"]["radius_nm"] > 1.0 and r["truth"]["lambda_per_s"] >= 0.05
        assert r["truth"]["metastable"] and r["truth"]["flicker_dz_pm"] > 0 and r["truth"]["flicker_rate_hz"] > 0
        assert r["truth"]["n_apex"] >= 2 and r["truth"]["tip_change_rows"]
    # the comparison machinery runs with no real side and still reports the bands
    cmp_ = compare(recs, {"Au(111)": {"available": False}})
    b = cmp_["by_material"]["Au(111)"]["bands"]
    assert b["verify_corr"]["good"]["frac_in_band"] is not None
    assert cmp_["by_material"]["Au(111)"]["ks"]["verify"]["fb_corr_flip"]["good"]["ks"] is None


@requires_mast
def test_crashed_tip_verify_frames_fail_masts_gate_and_sit_near_the_real_failed_verify(tmp_path: Path):
    """Band (b): a crashed tip's verify frame must read like the real failed verify
    (~0.43), not like a good tip. Eight seeds: every frame below MAST's tip_ready gate
    (0.80), the median inside 0.35–0.65 (measured 0.48 over 12 seeds with
    ``Tip.CRASH_FLICKER_DZ_M`` = 0.12–0.20 nm), and the mechanism visible in the truth —
    a flickering, metastable apex with a high change hazard that fired during the frame."""
    from stmsim.validate.fidelity import FidelitySet

    fs = FidelitySet(n=8, tips=("crashed",), points=("verify",), session_dir=tmp_path / "s")
    recs = fs.run()
    corr = [r["mast"]["trace_retrace_corr"] for r in recs]
    assert len(corr) == 8
    assert all(c < MAST_TIP_READY for c in corr), corr
    assert 0.35 <= float(np.median(corr)) <= 0.65, corr
    assert sum(0.3 <= c <= 0.7 for c in corr) >= 4, corr
    for r in recs:
        t = r["truth"]
        assert t["metastable"] and t["flicker_dz_pm"] >= 120 and t["flicker_rate_hz"] >= 1.0
        assert t["lambda_per_s"] >= 0.05 and t["n_apex"] >= 2 and len(t["tip_change_rows"]) >= 5
    # and the index recipe (what the corpus carries) also separates them from the good band
    idx = [r["mast"]["index"]["fb_corr_flip"] for r in recs]
    assert float(np.median(idx)) < 0.961, idx                       # below the real good-tip IQR


@requires_mast
@pytest.mark.xfail(strict=True, reason=(
    "Band (a): MAST's trace_retrace_correlation on the sim's good tip reads median 0.93–0.95 "
    "(8 seeds, hyst_frac 0.012 with the measured loop shape / constant shift), band 0.96–0.99; "
    "real reference on 40 frames: median 0.972. NOT the hysteresis: the loss is MAST's circular "
    "2-D NCC wrapping the frame's edge columns onto each other, and on the sim's monotonic "
    "staircase with perfectly flat terraces (detrended rms ~120 pm vs real 200 pm) those edges "
    "carry ~5 % of the variance. Adding the same 50–100 pm smooth roughness to both passes "
    "lifts the sim to 0.97–0.985; the smoothest real frames (rms ≤ 120 pm) read 0.85–0.98 too. "
    "Surface owner's lever (terrace texture / adsorbates); strict so a surface fix flips this."))
def test_good_tip_verify_frames_land_in_masts_band(tmp_path: Path):
    from stmsim.validate.fidelity import FidelitySet

    fs = FidelitySet(n=6, tips=("good",), points=("verify",), session_dir=tmp_path / "s")
    corr = [r["mast"]["trace_retrace_corr"] for r in fs.run()]
    assert VERIFY_BAND[0] <= float(np.median(corr)) <= VERIFY_BAND[1], corr
    assert sum(VERIFY_BAND[0] <= c <= VERIFY_BAND[1] for c in corr) >= 4, corr


@requires_mast
@pytest.mark.xfail(strict=True, reason=(
    "Band (c), single half: step_splitting on the sim's single tip reads 0.13–0.30 where decided "
    "(real single tips ≤ 0.147; the ten decided real frames 0.07–0.125). The score is reproduced by "
    "the SURFACE ALONE — Surface.terrace_height rendered with no tip and no scanner gives the same "
    "numbers on the same seeds — it is the autocorrelation ridge of the sim's straight, parallel "
    "step edges (the separation vectors line up with Site.step_angle); real Au/mica edges meander. "
    "Surface owner's finding; strict so a surface fix flips this."))
def test_single_tip_step_splitting_stays_at_or_below_the_real_single_band(tmp_path: Path):
    from stmsim.validate.fidelity import BANDS, FidelitySet

    fs = FidelitySet(n=6, tips=("good",), points=("verify",), session_dir=tmp_path / "s")
    scores = [r["mast"]["step_splitting"]["score"] for r in fs.run()]
    assert all(s is not None and s <= BANDS["step_splitting_single_max"] for s in scores), scores


@requires_mast
@pytest.mark.xfail(strict=True, reason=(
    "Band (c), multi half: MAST's autocorrelation replica score cannot see the two-apex soft-max "
    "ghost on the sim's frames — equal-height apexes 3/2 nm off at 0.7 weight, one-step-lower "
    "apexes, curved edges and adsorbates up to 1000/µm² all leave the score at the single-tip "
    "value (double_tip.py header point 3: the echo is a soft-max, not a linear echo; the real "
    "0.171–0.185 band came from six labelled frames). Whatever separates multi from single on "
    "real frames is not in the sim's surface; the sim's single-tip ridge (0.13–0.30) already "
    "overlaps the multi band. Surface owner's finding; strict so a fix flips this."))
def test_multi_tip_probe_reads_in_the_real_multi_band(tmp_path: Path):
    from stmsim.validate.fidelity import FidelitySet

    fs = FidelitySet(n=6, tips=("good", "multi"), points=("verify",), session_dir=tmp_path / "s")
    recs = fs.run()
    multi = [r["mast"]["step_splitting"]["score"] for r in recs if r["tip"] == "multi"]
    single = [r["mast"]["step_splitting"]["score"] for r in recs if r["tip"] == "good"]
    assert all(m is not None and m >= 0.17 for m in multi), multi
    assert all(s is not None and s < m for s, m in zip(single, multi)), (single, multi)


@requires_mast
def test_atomic_point_is_judged_by_masts_gate(tmp_path: Path):
    """Band (d): one sharp tip passes ``assess_atomic_phase`` fwd+bwd at 5 nm / 256 px; the
    crashed tip (R ×2–6, flickering) does not — the gate is MAST's, the tip is the
    simulator's. σ* = 0.09 nm is derived by ``stmbench.trackB.derive_thresholds``."""
    from stmsim.validate.fidelity import FidelitySet

    fs = FidelitySet(n=1, tips=("good", "crashed"), points=("atomic",), session_dir=tmp_path / "s")
    recs = {r["tip"]: r for r in fs.run()}
    assert recs["good"]["mast"]["atomic"]["both_passed"], recs["good"]["mast"]["atomic"]
    assert not recs["crashed"]["mast"]["atomic"]["both_passed"]
    assert abs(recs["good"]["mast"]["nm_per_px"] - 5.0 / 256) < 1e-6


# ── hysteresis: the sim's retrace offset reads back like the rig's ──────────────────
def test_retrace_offset_reads_back_at_the_real_median_with_the_measured_shape(tmp_path: Path):
    """The calibration's overlap-NCC peak shift (``calibrate.hysteresis.peak_shift``) on the
    sim's verify frames must read the real median (1.17 % of the axis = 3 px at 256 px;
    real frames read 3–4 px in the same recipe's ±1 px quantisation), and the offset must
    bulge mid-line the way 39 real frames do (thirds 0.98 / 1.37 / 1.17 % of the axis:
    centre ≥ edges, edges ≥ half the centre) — neither a constant shift nor a closed loop
    with zero offset at the turnarounds. Sim frames only; no MAST needed."""
    from stmsim.calibrate.hysteresis import peak_shift
    from stmsim.physics.scanner import SIG_Z
    from stmsim.validate.fidelity import VERIFY, make_world, scan_frame

    for seed in range(3):
        w = make_world(seed, "Au(111)", "good", tmp_path / f"s{seed}", VERIFY)
        assert w.hyst_frac == pytest.approx(0.012) and w.hyst_shape == "loop"
        scan_frame(w, VERIFY)
        fwd, bwd_acq = w.frame.data[SIG_Z]
        bwd = bwd_acq[:, ::-1]                       # acquisition order is right→left
        res = peak_shift(fwd, bwd)
        assert res["shift_px"] in (3, 4) and res["ncc_peak"] > 0.995, res
        t = fwd.shape[1] // 3
        left, centre, right = (_overlap_shift(fwd[:, lo:hi], bwd[:, lo:hi])
                               for lo, hi in ((0, t), (t, 2 * t), (2 * t, fwd.shape[1])))
        assert centre >= max(left, right) >= 2 and min(left, right) >= 0.5 * centre, (left, centre, right)
        assert centre > min(left, right), (left, centre, right)   # not a constant shift


def test_hysteresis_profile_keeps_the_mean_and_rejects_the_closed_loop():
    """``Renderer.hysteresis_profile_px``: the mean over the line is ``hyst_px`` for every
    edge fraction, the constant model is the ``e = 1`` limit, and the thirds pattern of the
    measured ``e = 0.62`` is 1.07 / 1.37 / 1.07 (% of axis) — the real 0.98 / 1.37 / 1.17
    with the two edge thirds pooled; a fully closed loop (``e = 0``) puts the edge thirds
    at 54 % of the centre, which the real frames (≥ 71 %) rule out."""
    from stmsim.physics.scanner import Renderer

    cols = np.arange(256)
    h = 3.07
    for e in (0.0, 0.3, 0.62, 1.0):
        prof = Renderer.hysteresis_profile_px(cols, 256, h, "loop", e)
        assert prof.mean() == pytest.approx(h, rel=1e-3), e
        assert h * e * 0.99 <= prof.min() <= h * e + 0.05 * h, e            # the edges sit at e × mean
        assert prof.max() == pytest.approx(h * (e + 1.5 * (1 - e)), rel=1e-3), e   # the centre at e + 1.5(1−e)
        assert prof.argmax() in (127, 128) or e == 1.0
    const = Renderer.hysteresis_profile_px(cols, 256, h, "shift", 0.62)
    assert np.allclose(const, h) and np.allclose(Renderer.hysteresis_profile_px(cols, 256, h, "loop", 1.0), h)
    thirds = lambda p: [float(p[lo:hi].mean()) for lo, hi in ((0, 85), (85, 170), (170, 256))]
    l, c, r = thirds(Renderer.hysteresis_profile_px(cols, 256, 1.17, "loop", 0.62))
    assert 1.0 < l < 1.12 and 1.3 < c < 1.45 and abs(l - r) < 0.01, (l, c, r)
    l0, c0, _ = thirds(Renderer.hysteresis_profile_px(cols, 256, 1.17, "loop", 0.0))
    assert l0 / c0 < 0.6 < 0.71                                     # closed loop vs real 0.98/1.37


# ── the indexer's recipe ────────────────────────────────────────────────────────────
def test_index_stats_reproduces_the_indexer_recipe():
    from stmsim.validate.fidelity import index_stats

    rng = np.random.default_rng(1)
    line = np.cumsum(rng.normal(0, 1e-11, 128))                # one corrugated line, same on every row
    base = np.tile(line, (64, 1)) + rng.uniform(-1e-12, 1e-12, 64)[:, None]   # ±1 pm row wobble (bounded):
    fwd = base.copy()                                          # |Δmedian| ≤ 2 pm while 4×MAD ≈ 2.8 pm → no false jump
    fwd[40:] += 3e-10                                          # one 300 pm row jump (a step / tip change)
    bwd = np.roll(fwd, -3, axis=1)                             # hysteresis: bwd features 3 px to the left
    s = index_stats(fwd, bwd)
    assert s["rows_used"] == 64
    assert s["fb_corr_flip"] > 0.999                           # inside the ±6 px window → recovered
    assert s["fb_corr_raw"] < 0.6                              # mirrored block does not match
    assert s["rowjump_frac"] == pytest.approx(1 / 63)          # exactly the one jump above 4×MAD
    assert 0 < s["rowjump_sigma_m"] < 1e-10
    assert s["z_p2p98_m"] > 0
    # NaN (unscanned) rows are dropped, not counted
    fwd2 = fwd.copy(); fwd2[50:] = np.nan
    assert index_stats(fwd2, bwd)["rows_used"] == 50
    assert index_stats(fwd[:4], bwd[:4])["fb_corr_flip"] is None


# ── the real side without MAST or the lab drive ─────────────────────────────────────
def test_real_distributions_filters_and_ks(tmp_path: Path):
    from stmsim.validate.fidelity import compare, ks_distance, real_distributions, resolve_index

    idx = _write_index(tmp_path / "sxm_index_full.parquet")
    assert resolve_index(tmp_path) == idx                      # a directory resolves to the parquet
    real = real_distributions(idx, "Au(111)")
    assert real["available"] and real["n_complete"] == 40      # three decoys dropped
    assert real["genres"]["verify"]["n"] == 20 and real["genres"]["atomic"]["n"] == 20
    q = real["genres"]["verify"]["quantiles"]["fb_corr_flip"]
    assert len(q) == 5 and 0.93 <= q[0] <= q[4] <= 0.995
    missing = real_distributions(tmp_path / "nope.parquet")
    assert not missing["available"]
    # KS: identical samples → 0; shifted → large
    a = real["genres"]["verify"]["arrays"]["fb_corr_flip"]
    assert ks_distance(a, a)["ks"] == 0.0
    assert ks_distance(a - 0.5, a)["ks"] == 1.0
    assert ks_distance([None, 0.9], a)["ks"] is None            # too few sim frames → no number
    # compare() consumes records shaped like FidelitySet.run() output
    recs = [{"material": "Au(111)", "tip": t, "point": "verify", "seed": i,
             "truth": {"tip_change_rows": [3] if t == "crashed" else []},
             "mast": {"trace_retrace_corr": 0.97 if t == "good" else 0.5,
                      "tip_change": {"changed": t == "crashed", "score": 5.0, "threshold": 10.0},
                      "step_splitting": {"verdict": "single", "score": 0.1},
                      "index": {"fb_corr_flip": 0.97, "rowjump_frac": 0.05, "rowjump_sigma_m": 2e-12}}}
            for t in ("good", "crashed") for i in range(3)]
    c = compare(recs, {"Au(111)": real})["by_material"]["Au(111)"]
    assert c["ks"]["verify"]["fb_corr_flip"]["good"]["n_real"] == 20
    assert c["ks"]["verify"]["fb_corr_flip"]["good"]["ks"] is not None
    assert c["ks"]["atomic"]["rowjump_frac"]["good"]["ks"] is None      # no atomic records rendered
    assert c["bands"]["verify_corr"]["good"]["frac_in_band"] == 1.0
    assert c["bands"]["verify_corr"]["crashed"]["median"] == 0.5
    assert c["bands"]["tip_change"]["detected_frac"] == 1.0 and c["bands"]["tip_change"]["false_alarm_frac"] == 0.0
    assert c["bands"]["step_splitting"]["single"]["frac_le_single_max"] == 1.0


# ── λ* upper bound (calibrate-side owner) ───────────────────────────────────────────
def test_lambda_from_rowjump_matches_hand_computation_and_splits_sessions(tmp_path: Path):
    from stmsim.calibrate.lambda_from_rowjump import (
        lambda_star_from_index, lambda_star_from_task, lambda_star_per_session, main,
    )

    idx = _write_index(tmp_path / "sxm_index_full.parquet")
    pd = pytest.importorskip("pandas")
    df = pd.read_parquet(idx)
    d = df[(df.err == "") & (df.acq_frac >= 0.99) & (df.fb_corr_flip >= 0.9)
           & df.comment.str.contains("Au", case=False)]
    rate = (d.rowjump_frac * d.ny / (d.ny * (d.t_fwd_s + d.t_bwd_s))).to_numpy()
    out = lambda_star_from_index(idx, quantile=0.85)
    assert out["n_frames"] == len(d) and out["lambda_star_per_s"] == pytest.approx(np.quantile(rate, 0.85))
    assert "upper bound" in out["note"]
    ps = lambda_star_per_session(idx, min_frames=5)
    assert ps["n_sessions"] == 3 and sum(s["n_frames"] for s in ps["sessions"]) == len(d)
    assert ps["sessions"] == sorted(ps["sessions"], key=lambda s: s["lambda_q_per_s"])
    task = lambda_star_from_task()
    assert task["lambda_star_per_s"] == pytest.approx(-np.log(0.9) / (256 * 2 * 0.586))
    # the CLI writes both numbers; a missing index still writes the task-derived one
    out_json = tmp_path / "calib" / "lambda.json"
    assert main(["--index", str(idx), "--out", str(out_json)]) == 0
    j = json.loads(out_json.read_text(encoding="utf-8"))
    assert j["lambda_star_index"]["lambda_star_per_s"] == pytest.approx(out["lambda_star_per_s"])
    assert j["lambda_star_task"]["lambda_star_per_s"] == pytest.approx(task["lambda_star_per_s"])
    assert main(["--index", str(tmp_path / "missing.parquet"), "--out", str(out_json)]) == 0
    j = json.loads(out_json.read_text(encoding="utf-8"))
    assert j["lambda_star_index"]["lambda_star_per_s"] is None and j["lambda_star_task"]["lambda_star_per_s"] > 0


# ── hysteresis: what the index can and cannot say ───────────────────────────────────
def test_hysteresis_reports_convention_and_bound_but_never_a_per_frame_shift(tmp_path: Path):
    from stmsim.calibrate.hysteresis import fit_from_index, main, peak_shift

    idx = _write_index(tmp_path / "sxm_index_full.parquet", n_au=200)
    out = fit_from_index(idx, "Au(111)", min_frames_per_nx=10)
    assert out["available"] and out["storage_convention"]["frac_flip_gt_raw_among_differing"] == 1.0
    assert "mirrored" in out["storage_convention"]["verdict"]
    assert set(out["by_nx"]) == {"256", "512"}
    b = out["bound_from_nx_trend"]
    assert b is not None and b["hyst_frac_upper_bound"] is None and b["inference"] == "none"
    assert "circular" in b["note"]                              # the nx trend is description, not a bound
    assert "per-frame hysteresis in px" in out["cannot_infer"][0]
    assert out["index_recipe"]["argmax_stored"] is False
    # the direct measurement finds a planted shift, with its sign, on a stepped + tilted
    # frame where the circular recipe would sit at zero
    rng = np.random.default_rng(3)
    fwd = np.cumsum(rng.normal(0, 1e-11, (64, 256)), axis=1)
    fwd += np.linspace(0, 1e-9, 256)[None, :]                  # 1 nm of fast-axis tilt
    fwd[:, 120:] += 2.35e-10                                    # a step
    bwd = np.empty_like(fwd); bwd[:, :-5] = fwd[:, 5:]; bwd[:, -5:] = fwd[:, -5:]   # bwd features 5 px left
    res = peak_shift(fwd, bwd)
    assert res["shift_px"] == 5 and res["ncc_peak"] > 0.999 and res["ncc_zero"] < res["ncc_peak"]
    # CLI without --sample writes the JSON and says the direct measurement did not run
    out_json = tmp_path / "calib" / "hysteresis.json"
    assert main(["--index", str(idx), "--out", str(out_json)]) == 0
    j = json.loads(out_json.read_text(encoding="utf-8"))
    assert j["direct_measurement"]["ran"] is False and j["storage_convention"]["n"] == 200
