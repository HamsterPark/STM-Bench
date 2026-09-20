"""Detector-distribution fidelity (docs/DESIGN.md §4.6 item 1).

Render simulator frames across seeds / materials / tip states at the two imaging points
MAST itself uses, run **MAST's own detectors** on the files (through ``read_sxm`` — the
same path a real frame takes), and compare against the real corpus:

* ``verify``  — the tip-forging verify frame: 100 nm / 256 px / 0.586 s per line at
  1.0 V / 100 pA (``config/forge_tip_reference.txt`` §4: ``verify_scan_nm=100``,
  ``verify_pixels=256``, ``verify_line_time_s=0.586``, ``verify_bias_v=1.0``,
  ``verify_setpoint_a=100 pA``);
* ``atomic``  — the atomic-resolution point: 5 nm / 256 px at 20 mV / 500 pA on Au(111)
  (the point ``derive_thresholds`` sweeps r* at; 0.0195 nm/px is inside MAST's scale gate).

Detectors (all MAST, none re-implemented here):

* ``mast.vision.tip_metrics.trace_retrace_correlation`` / ``assess_tip_classical``
* ``mast.vision.double_tip.step_splitting`` (the large-frame multi-tip verdict)
* ``mast.vision.tip_change.detect_tip_change`` (v2, null-calibrated row-jump)
* ``mast.vision.atomic_phase.assess_atomic_phase`` (atomic point only)

Real side — **without re-reading a single .sxm** — comes from the corpus index
(``sxm_index_full.parquet``): ``fb_corr_flip`` / ``fb_corr_raw`` / ``rowjump_frac`` /
``rowjump_sigma_m`` of *complete* frames (``acq_frac ≥ 0.99``) of the requested material,
split by genre (verify-like 80–120 nm; atomic-like ≤ 10 nm). Those columns were computed by
the indexer with a fixed recipe (row-median removal, ≤128 rows, best NCC over ±6 px); the
**same recipe** is applied to the simulator frames (:func:`index_stats`) so the KS distance
compares like with like. MAST's ``trace_retrace_correlation`` is a different statistic
(plane-detrended 2-D NCC, ±12 % shift window) and is compared against the *bands* the plan
names, not against the index.

Target bands (docs/DESIGN.md §4.6; the real numbers they come from are cited inline):

* good tip, verify frame, ``trace_retrace_correlation`` ∈ [0.96, 0.99]
  (0.994 / 0.990 / 0.959 on the real rig, ``forge_tip_reference.txt:247-249``);
  bad tip ≈ 0.43 (one real failed verify at the correct speed);
* ``step_splitting`` score: multi-tip 0.171–0.185 vs single ≤ 0.147, 40 labelled real
  frames (``double_tip.py`` ``DOUBLE_TIP_SCORE_ON_STEPS``);
* ``edge_resolution_px`` median 1.21 on 72 real rig frames — **reported, not a ruler**
  (seven tenths of that width is the pixel grid).

Nothing here changes the simulator; a miss is a finding for the tip / renderer / surface
owner.

Findings 2026-08-28 (G4, MAST's detectors re-run on 40 real complete 80–120 nm Au frames
from ``calib/hysteresis.json`` as the reference — ``trace_retrace_correlation`` p05/p25/
p50/p75/p95 = 0.856/0.940/0.972/0.981/0.991; ``step_splitting`` 30/40 undecidable, the
decided ten 0.07–0.125 plus one 0.199 on a 128 px frame):

* good-tip ``trace_retrace_correlation``: the loss is NOT the hysteresis. MAST's statistic
  is a circular 2-D NCC after row-median + plane detrending; the retrace's fast-axis offset
  makes the frame's two edge columns wrap onto each other, and on the sim's frames those
  edges carry ~5 % of the detrended variance (a monotonic staircase, 10–17 mrad, with
  perfectly flat terraces: detrended rms ~120 pm vs 200 pm real median). Adding the
  *same* 50–100 pm smooth roughness to trace and retrace lifts the sim to 0.97–0.985; the
  smoothest real frames (rms ≤ 120 pm) sit at 0.85–0.98, median ≈ 0.955, where the sim is.
  ⇒ terrace texture / adsorbate density is the surface owner's lever; ``hyst_frac`` 0.012
  (real median 1.17 %) and the measured loop shape are kept as the physics says.
* crashed tip: a *static* blunt multi-apex tip reads 0.96 — indistinguishable from good.
  ``Tip.crash`` now leaves a flickering apex (``flicker_dz_m`` / ``flicker_rate_hz``, per
  pass and mid-line), which puts the verify frame at ~0.5 (real failed verify: 0.43).
* ``step_splitting``: the single-tip scores of 0.13–0.30 are reproduced by the *surface
  alone* (``terrace_height`` rendered without tip or scanner, same seeds, same numbers):
  the autocorrelation replica the detector scores is the ridge of the sim's straight,
  parallel step edges (separation vectors line up with ``Site.step_angle``), which real
  Au/mica edges (meandering, islands) do not produce. The two-apex soft-max ghost adds
  nothing the detector can see on these frames (also tried: equal-height apexes, curved
  edges, adsorbates up to 1000/µm²) — ``double_tip.py`` header point 3 says why. Both
  halves of that band are therefore a surface finding, pinned as strict xfails.
* atomic-phase gate: σ* = 0.09 nm before and after (``derive_thresholds --seeds 4``).

Run::

    python -m stmsim.validate.fidelity --out $STM_BENCH_DATA/fidelity/report.json --n 12

``--real`` names the corpus index directory or parquet (default
``stmsim.paths.corpus_index_dir()``); when it is absent the report still carries the
simulator side and says the real side was unavailable.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from stmsim.paths import corpus_index_dir, fidelity_dir

# ── imaging points (provenance in the module docstring) ─────────────────────────────


@dataclass(frozen=True)
class ImagingPoint:
    name: str
    w_nm: float
    px: int
    line_s: float
    bias_v: float
    setpoint_a: float
    flat_window: bool = False     # centre the frame on a step-free window (atomic point)

    @property
    def nm_per_px(self) -> float:
        return self.w_nm / self.px

    @property
    def frame_time_s(self) -> float:
        return 2.0 * self.px * self.line_s


VERIFY = ImagingPoint("verify", 100.0, 256, 0.586, 1.0, 100e-12)
ATOMIC = ImagingPoint("atomic", 5.0, 256, 0.1, 0.02, 500e-12, flat_window=True)
POINTS = {p.name: p for p in (VERIFY, ATOMIC)}

#: genre filters on the real index, in nm of fast-axis range — the sim point sits inside
GENRE_RANGE_NM = {"verify": (80.0, 120.0), "atomic": (0.0, 10.0)}

#: material → regex on ``path + comment`` (same table as ``stmsim.calibrate.working_points``)
MATERIAL_PATTERNS = {
    "Au(111)": r"au\s*\(?111|gold|金",
    "Cu(111)": r"cu\s*\(?111|copper|铜",
    "Ag(111)": r"ag\s*\(?111|silver|银",
    "HOPG": r"hopg|graphite|石墨",
}

TIP_KINDS = ("good", "crashed", "multi")

BANDS = {
    "verify_corr_good": (0.96, 0.99),
    "verify_corr_bad_target": 0.43,
    "step_splitting_multi": (0.171, 0.185),
    "step_splitting_single_max": 0.147,
    "edge_px_real_median": 1.21,
}


# ── simulator side ──────────────────────────────────────────────────────────────────

def make_world(seed: int, material: str, tip_kind: str, session_dir: Path, point: ImagingPoint):
    """A World in tunnelling at ``point`` with a tip of ``tip_kind``.

    ``good``: R = 1 nm, σ_a = 0.06 nm (inside σ* from ``derive_thresholds``), no spontaneous
    changes. ``crashed``: the good tip after :meth:`Tip.crash` (R ×2–6, extra apexes 2–6 nm
    off, λ ≥ 0.05/s, φ lowered, and a flickering apex — ``flicker_dz_m`` / ``flicker_rate_hz``
    — so trace and retrace disagree) — the simulator's own notion of a damaged tip.
    ``multi``: the good tip plus a second apex of equal height 3 nm / 2 nm away at 0.7
    weight — :meth:`Tip.is_multi` is True and it is a clean probe for ``step_splitting``
    with everything else held at the good values. Equal height (dz = 0) is the operator's
    labelled multi tip (``double_tip.py``: levels at whole steps, every edge drawn twice
    laterally); the soft-max then splits each 235 pm step into a 25 + 210 pm or
    42 + 194 pm pair ``|d|`` apart depending on which apex leads (``Tip.height_parts``;
    tests/test_physics.py), whereas an apex one step lower only leaves a 25 pm ledge.
    The separation is off both scan axes because ``detect_double_tip`` notches |dy| ≤ 3 px
    and |dx| ≤ 3 px.
    """
    from stmsim.physics.rig import RigProfile
    from stmsim.physics.tip import Apex
    from stmsim.physics.world import World

    w = World(rig=RigProfile.load("reference-stm"), seed=seed, material=material,
              session_dir=session_dir, time_scale=1000.0)
    # These targets characterise the SCANNER — the retrace loop, the corrugation, the step
    # width a tip radius produces — so the sample is held still while they are measured. The
    # 1000× clock is here to render frames quickly, not to model a long experiment: drift over
    # a 512-second frame would be twenty pixels across a 100 nm window and every one of these
    # numbers would become a measurement of the drift instead. (Before drift was tied to sim
    # time this held by accident, because 1000× made the frame half a wall-second long.)
    w.drift_v_m_per_s = np.zeros(3)
    w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
    w.withdrawn = False
    w.zctrl_set(True)
    w.transients.clear()
    w.set_bias(point.bias_v)
    w.set_setpoint(point.setpoint_a)
    w.tip.radius_m = 1.0e-9
    w.tip.apex_sigma_m = 0.06e-9
    w.tip.apex_radius_ref_m = w.tip.radius_m      # pin: no resampling
    w.tip.lambda_per_s = 0.0
    w.tip.metastable = False
    if tip_kind == "crashed":
        w.tip.crash(0.0, 1.0)
    elif tip_kind == "multi":
        w.tip.apexes = [Apex(0.0, 0.0, 0.0, 1.0), Apex(3e-9, 2e-9, 0.0, 0.7)]
    elif tip_kind != "good":
        raise ValueError(f"unknown tip kind {tip_kind!r}; choose from {TIP_KINDS}")
    w.achievable_z_tip()
    return w


def flat_centre(world, w_nm: float, *, max_step_pm: float = 20.0, reach_nm: float = 400.0):
    """A scan centre whose window holds no step (what ``FindFlatRegion`` gives an operator
    before an atomic frame), from the simulator's terrace truth. Mirrors
    ``stmbench.trackB.derive_thresholds._flat_centre`` — duplicated because ``stmsim`` must
    not import ``stmbench``."""
    half = 0.5 * w_nm * 1e-9
    g = np.linspace(-half, half, 9)
    gx, gy = np.meshgrid(g, g)
    cands = [(0.0, 0.0)]
    for r in range(1, int(reach_nm // 10) + 1):
        for dx, dy in itertools.product((-r, 0, r), repeat=2):
            if (dx, dy) != (0, 0):
                cands.append((dx * 10e-9, dy * 10e-9))
    A = np.c_[gx.ravel(), gy.ravel(), np.ones(gx.size)]
    for cx, cy in cands:
        h = world.surface.terrace_height(gx + cx, gy + cy)
        coef, *_ = np.linalg.lstsq(A, h.ravel(), rcond=None)
        resid = h.ravel() - A @ coef
        if (resid.max() - resid.min()) * 1e12 < max_step_pm:
            return cx, cy
    return 0.0, 0.0


def scan_frame(world, point: ImagingPoint) -> str:
    """Scan one frame at ``point`` and save it; returns the .sxm path.

    The frame is rendered at ``scan_start``; the sim clock is jumped past the frame time so
    no wall-clock wait is needed (``time_scale`` only sets how fast it *would* have run)."""
    st = world.scan
    st.nx = st.ny = point.px
    st.w = st.h = point.w_nm * 1e-9
    if point.flat_window:
        st.cx, st.cy = flat_centre(world, point.w_nm)
    else:
        st.cx, st.cy = 0.0, 0.0
    st.line_time_fwd_s = st.line_time_bwd_s = point.line_s
    world.scan_start()
    world.clock.advance_sim(point.frame_time_s + 1.0)
    if world.scan_running():   # progresses the frame; must be finished after the jump
        raise RuntimeError("frame did not finish after the clock jump")
    path = world.save_frame()
    if not path:
        raise RuntimeError("save_frame returned no path")
    return path


# ── the indexer's recipe, applied to a sim frame ────────────────────────────────────

def index_stats(fwd: np.ndarray, bwd: np.ndarray | None, *, shift_px: int = 6,
                max_rows: int = 128) -> dict:
    """``rowjump_frac`` / ``rowjump_sigma_m`` / ``z_p2p98_m`` / ``fb_corr_flip`` /
    ``fb_corr_raw`` exactly as the corpus indexer computed them (``probe/sxm_fast.py``,
    2026-08-27): rows = complete rows; row-median removed; ≤ ``max_rows`` rows subsampled;
    NCC maximised over ±``shift_px`` circular shifts. ``fwd``/``bwd`` are in the *oriented*
    frame (``sxm_oriented_frames``), so ``raw`` — the indexer's correlation against the
    block as stored, i.e. mirrored — is the correlation against ``bwd[:, ::-1]``.

    Reproduced bias included: the circular ``np.roll`` wraps ``s`` columns of a frame whose
    range is dominated by tilt + steps, so on stepped frames the maximum sits at ``s = 0``
    even for a clean lateral shift (see ``stmsim.calibrate.hysteresis.peak_shift``). That is
    what the real index contains, so it is what the sim frames get too."""
    a = np.asarray(fwd, dtype=np.float64)
    full = np.isfinite(a).all(axis=1)
    a = a[full]
    out: dict = {"rowjump_frac": None, "rowjump_sigma_m": None, "z_p2p98_m": None,
                 "fb_corr_flip": None, "fb_corr_raw": None, "rows_used": int(a.shape[0])}
    if a.shape[0] < 8:
        return out
    rm = np.median(a, axis=1)
    dr = np.diff(rm)
    dmad = float(np.median(np.abs(dr - np.median(dr)))) + 1e-15
    out["rowjump_frac"] = float((np.abs(dr) > 4 * dmad).mean())
    out["rowjump_sigma_m"] = float(dmad * 1.4826)
    al = a - rm[:, None]
    out["z_p2p98_m"] = float(np.percentile(al, 98) - np.percentile(al, 2))
    if bwd is None:
        return out
    b = np.asarray(bwd, dtype=np.float64)[full]
    bok = np.isfinite(b).all(axis=1)
    if bok.sum() < 8 or b.shape != a.shape:
        return out
    a2 = al[bok]
    b2 = b[bok] - np.median(b[bok], axis=1)[:, None]
    if a2.shape[0] > max_rows:
        ii = np.linspace(0, a2.shape[0] - 1, max_rows).astype(int)
        a2, b2 = a2[ii], b2[ii]
    a2 = a2 - a2.mean()

    def best(cand: np.ndarray) -> float:
        c0 = cand - cand.mean()
        bst = -1.0
        for s in range(-shift_px, shift_px + 1):
            cs = np.roll(c0, s, axis=1)
            den = math.sqrt(float((a2 * a2).sum()) * float((cs * cs).sum())) + 1e-30
            bst = max(bst, float((a2 * cs).sum() / den))
        return bst

    out["fb_corr_flip"] = best(b2)            # geometric orientation (what "flip" undid)
    out["fb_corr_raw"] = best(b2[:, ::-1])    # against the mirrored block as stored
    return out


# ── MAST detectors on one saved frame ───────────────────────────────────────────────

def evaluate_frame(path: str, point: ImagingPoint, material: str) -> dict:
    """Run MAST's detectors on a saved .sxm and return plain JSON-able numbers."""
    from mast.io.nanonis_files import read_sxm, sxm_oriented_frames
    from mast.vision.atomic_phase import assess_atomic_phase
    from mast.vision.double_tip import step_splitting
    from mast.vision.lattice_calibration import first_order_period_nm
    from mast.vision.tip_change import detect_tip_change
    from mast.vision.tip_metrics import assess_tip_classical, trace_retrace_correlation

    fr = sxm_oriented_frames(read_sxm(path), "Z")
    fwd, bwd = fr["forward"], fr["backward"]
    nmpp = float(fr.get("nm_per_px") or point.nm_per_px)
    out: dict = {"nm_per_px": nmpp}
    out["trace_retrace_corr"] = float(trace_retrace_correlation(fwd, bwd))
    # assess_tip_classical opens with ``if std < 1e-9: return <flat>`` on the detrended
    # frame — an absolute gate in the frame's own unit, so a metre-unit frame with < 1 nm
    # of rms relief (every 100 nm Au frame) comes back as "flat" (sharpness 0, no edge, no
    # instability). MAST's callers hand it whatever unit the frame arrived in; here it gets
    # nanometres so the metrics are computed at all. Recorded so the reader knows.
    tm = assess_tip_classical(fwd * 1e9, bwd * 1e9, nm_per_px=nmpp)
    out["tip_metrics_unit"] = "nm (metre-unit frames trip the std<1e-9 flat gate)"
    out["fwd_bwd_instability"] = _f(getattr(tm, "fwd_bwd_instability", None))
    out["edge_resolution_px"] = _f(getattr(tm, "edge_resolution_px", None))
    out["fft_sharpness"] = _f(getattr(tm, "fft_sharpness", None))
    out["n_terrace_levels"] = getattr(tm, "n_terrace_levels", None)
    tc = detect_tip_change(np.stack([fwd, bwd]), nm_per_px=nmpp)
    out["tip_change"] = {"changed": bool(tc.changed), "score": _f(tc.score), "change_row": tc.change_row,
                         "threshold": _f(tc.threshold), "lod_m": _f(tc.lod), "calib": tc.calib}
    if point.name == "verify":
        ss = step_splitting(fwd, frame_m=point.w_nm * 1e-9)
        out["step_splitting"] = {"verdict": ss.get("verdict"), "score": _f(ss.get("score")),
                                 "reason": str(ss.get("reason", ""))[:120],
                                 "replica_reason": str(ss.get("replica_reason", ""))[:80],
                                 "separation_px": ss.get("separation_px")}
    if point.name == "atomic":
        expect = first_order_period_nm(material)
        res = {}
        for name, img in (("fwd", fwd), ("bwd", bwd)):
            r = assess_atomic_phase(img, nm_per_px=nmpp, expected_a_nm=expect)
            res[name] = {"passed": bool(r.passed), "reasons": list(getattr(r, "reasons", ()) or ()),
                         "concentration": _f(getattr(r, "angular_concentration", None)),
                         "sharpness": _f(getattr(r, "fft_sharpness", None)),
                         "period_nm": _f(getattr(r, "period_nm", None))}
        res["both_passed"] = bool(res["fwd"]["passed"] and res["bwd"]["passed"])
        res["expected_a_nm"] = expect
        out["atomic"] = res
    out["index"] = index_stats(fwd, bwd)
    return out


def _f(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ── render + evaluate a whole set ───────────────────────────────────────────────────

@dataclass
class FidelitySet:
    n: int = 12
    materials: tuple[str, ...] = ("Au(111)",)
    tips: tuple[str, ...] = TIP_KINDS
    points: tuple[str, ...] = ("verify", "atomic")
    seed_base: int = 0
    session_dir: Path | None = None
    verbose: bool = False
    records: list[dict] = field(default_factory=list)

    def run(self) -> list[dict]:
        root = Path(self.session_dir) if self.session_dir else fidelity_dir() / "sessions"
        for material, tip_kind, i in itertools.product(self.materials, self.tips, range(self.n)):
            seed = self.seed_base + i
            for pname in self.points:
                point = POINTS[pname]
                sd = root / f"{_slug(material)}_{tip_kind}_{pname}_s{seed}"
                t0 = time.time()
                w = make_world(seed, material, tip_kind, sd, point)
                path = scan_frame(w, point)
                rec = {"material": material, "tip": tip_kind, "point": pname, "seed": seed,
                       "path": path, "truth": _truth(w), "mast": evaluate_frame(path, point, material),
                       "wall_s": round(time.time() - t0, 2)}
                self.records.append(rec)
                if self.verbose:
                    m = rec["mast"]
                    print(f"{material} {tip_kind:8s} {pname:6s} seed={seed:3d} corr={m['trace_retrace_corr']:.3f} "
                          f"idx_corr={_fmt(m['index']['fb_corr_flip'])} rowjump={_fmt(m['index']['rowjump_frac'])} "
                          f"tc={m['tip_change']['score']:.2f}/{len(rec['truth']['tip_change_rows'])} "
                          f"{'ss=' + _fmt(m['step_splitting']['score']) if 'step_splitting' in m else ''}"
                          f"{'atomic=' + str(m['atomic']['both_passed']) if 'atomic' in m else ''}  {rec['wall_s']}s",
                          flush=True)
        return self.records


def _truth(w) -> dict:
    snap = w.tip.snapshot()
    return {"multi": bool(snap["multi"]), "n_apex": int(snap["n_apex"]), "radius_nm": snap["radius_nm"],
            "apex_sigma_nm": snap["apex_sigma_nm"], "lambda_per_s": snap["lambda_per_s"],
            "metastable": bool(snap["metastable"]), "flicker_dz_pm": snap["flicker_dz_pm"],
            "flicker_rate_hz": snap["flicker_rate_hz"],
            "phi_ev": snap["phi_ev"], "tip_change_rows": list(w.frame.tip_change_rows) if w.frame else [],
            "hysteresis_px": float(w.hysteresis_px(w.frame.settings)) if w.frame else None,
            "hysteresis_shape": getattr(w, "hyst_shape", "shift")}


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", s)


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


# ── real side: the corpus index, no .sxm re-read ────────────────────────────────────

REAL_COLUMNS = ("fb_corr_flip", "fb_corr_raw", "rowjump_frac", "rowjump_sigma_m")


def resolve_index(real: str | Path | None) -> Path:
    p = Path(real) if real else corpus_index_dir()
    return p / "sxm_index_full.parquet" if p.is_dir() else p


def real_distributions(index_path: str | Path, material: str = "Au(111)") -> dict:
    """Per genre: arrays of the index columns over complete frames of ``material``.

    Returns ``{"available": bool, "index": str, "material": ..., "n_complete": int,
    "genres": {genre: {"n": int, "range_nm": [lo, hi], "arrays": {col: ndarray},
    "quantiles": {col: [p05, p25, p50, p75, p95]}}}}``. The real frames are whatever the
    operators saved — mostly good tips, unlabelled — so the comparison partner for a KS
    distance is the simulator's *good* tip first, the pooled tips second."""
    p = Path(index_path)
    if not p.is_file():
        return {"available": False, "index": str(p), "error": "index file not found"}
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        return {"available": False, "index": str(p), "error": f"pandas missing: {exc}"}
    cols = ["path", "comment", "err", "acq_frac", "range_x_m", "nx", *REAL_COLUMNS]
    df = pd.read_parquet(p, columns=cols)
    d = df[(df["err"].astype(str) == "") & (df["acq_frac"] >= 0.99)]
    pat = MATERIAL_PATTERNS.get(material, re.escape(material))
    hay = d["path"].astype(str) + " " + d["comment"].astype(str)
    d = d[hay.str.contains(pat, case=False, regex=True, na=False)]
    rng_nm = d["range_x_m"].astype(float) * 1e9
    out = {"available": True, "index": str(p), "material": material, "n_complete": int(len(d)),
           "filters": {"err": "empty", "acq_frac": ">=0.99", "material_regex": pat}, "genres": {}}
    for genre, (lo, hi) in GENRE_RANGE_NM.items():
        s = d[(rng_nm >= lo) & (rng_nm <= hi)]
        arrays = {c: s[c].dropna().to_numpy(dtype=float) for c in REAL_COLUMNS}
        out["genres"][genre] = {"n": int(len(s)), "range_nm": [lo, hi], "arrays": arrays,
                                "quantiles": {c: _quant(v) for c, v in arrays.items()}}
    return out


def _quant(v) -> list | None:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    return [float(x) for x in np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])]


# ── comparison ──────────────────────────────────────────────────────────────────────

def ks_distance(a, b) -> dict:
    a = np.asarray([x for x in a if x is not None], dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return {"ks": None, "p": None, "n_sim": int(a.size), "n_real": int(b.size)}
    from scipy.stats import ks_2samp
    r = ks_2samp(a, b)
    return {"ks": float(r.statistic), "p": float(r.pvalue), "n_sim": int(a.size), "n_real": int(b.size),
            "sim_quantiles": _quant(a), "real_quantiles": _quant(b)}


def _pick(records, **where) -> list[dict]:
    return [r for r in records if all(r.get(k) == v for k, v in where.items())]


def _vals(recs, *keys):
    out = []
    for r in recs:
        v = r["mast"]
        for k in keys:
            v = v.get(k) if isinstance(v, dict) else None
            if v is None:
                break
        out.append(v)
    return out


def _frac_in(vals, lo, hi) -> float | None:
    v = [x for x in vals if x is not None]
    return (sum(lo <= x <= hi for x in v) / len(v)) if v else None


def _median(vals) -> float | None:
    v = [x for x in vals if x is not None]
    return float(np.median(v)) if v else None


def compare(records: list[dict], real_by_material: dict[str, dict]) -> dict:
    """KS distances (index recipe, per genre) + the plan's bands (MAST recipe).

    ``real_by_material`` maps material → :func:`real_distributions` output (may be
    ``{"available": False}``: the bands are still computed, the KS entries carry ``None``)."""
    materials = sorted({r["material"] for r in records})
    out: dict = {"bands": BANDS, "by_material": {}}
    for material in materials:
        m: dict = {"ks": {}, "bands": {}}
        real = real_by_material.get(material) or {}
        # ── KS on the indexer's statistics, sim vs real, per genre ──
        for genre in ("verify", "atomic"):
            g_real = (real.get("genres") or {}).get(genre) if real.get("available") else None
            m["ks"][genre] = {}
            for col in ("fb_corr_flip", "rowjump_frac", "rowjump_sigma_m"):
                entry = {}
                for tip_set, recs in (("good", _pick(records, material=material, point=genre, tip="good")),
                                      ("all_tips", _pick(records, material=material, point=genre))):
                    sim_vals = _vals(recs, "index", col)
                    if g_real is None:
                        entry[tip_set] = {"ks": None, "n_sim": len([v for v in sim_vals if v is not None]),
                                          "n_real": 0, "sim_quantiles": _quant([v for v in sim_vals if v is not None])}
                    else:
                        entry[tip_set] = ks_distance(sim_vals, g_real["arrays"][col])
                m["ks"][genre][col] = entry
        # ── bands on MAST's own detectors ──
        vg = _pick(records, material=material, point="verify", tip="good")
        vc = _pick(records, material=material, point="verify", tip="crashed")
        vm = _pick(records, material=material, point="verify", tip="multi")
        corr_g = _vals(vg, "trace_retrace_corr")
        corr_c = _vals(vc, "trace_retrace_corr")
        lo, hi = BANDS["verify_corr_good"]
        m["bands"]["verify_corr"] = {
            "good": {"values": corr_g, "median": _median(corr_g), "frac_in_band": _frac_in(corr_g, lo, hi),
                     "band": [lo, hi]},
            "crashed": {"values": corr_c, "median": _median(corr_c), "target": BANDS["verify_corr_bad_target"],
                        "frac_below_0p8": _frac_in(corr_c, -1.0, 0.8)},
            "separation": ((_median(corr_g) or 0) - (_median(corr_c) or 0)) if corr_g and corr_c else None,
        }
        ss_single = _vals(vg, "step_splitting", "score")
        ss_multi = _vals(vm, "step_splitting", "score")
        slo, shi = BANDS["step_splitting_multi"]
        m["bands"]["step_splitting"] = {
            "single": {"values": ss_single, "median": _median(ss_single),
                       "frac_le_single_max": _frac_in(ss_single, -1.0, BANDS["step_splitting_single_max"]),
                       "verdicts": _count(_vals(vg, "step_splitting", "verdict"))},
            "multi": {"values": ss_multi, "median": _median(ss_multi), "frac_in_band": _frac_in(ss_multi, slo, shi),
                      "frac_split": _frac_in(ss_multi, 0.16, 10.0), "verdicts": _count(_vals(vm, "step_splitting", "verdict"))},
        }
        # tip_change: detection on frames with a true change vs false alarms on frames without
        all_v = _pick(records, material=material, point="verify")
        with_ev = [r for r in all_v if r["truth"]["tip_change_rows"]]
        without = [r for r in all_v if not r["truth"]["tip_change_rows"]]
        m["bands"]["tip_change"] = {
            "n_frames_with_true_change": len(with_ev),
            "detected_frac": _frac_in([1.0 if r["mast"]["tip_change"]["changed"] else 0.0 for r in with_ev], 0.5, 1.5),
            "false_alarm_frac": _frac_in([1.0 if r["mast"]["tip_change"]["changed"] else 0.0 for r in without], 0.5, 1.5),
            "score_median_with": _median([r["mast"]["tip_change"]["score"] for r in with_ev]),
            "score_median_without": _median([r["mast"]["tip_change"]["score"] for r in without]),
            "threshold": _median([r["mast"]["tip_change"]["threshold"] for r in all_v]),
            "true_changes_per_frame_median": _median([len(r["truth"]["tip_change_rows"]) for r in with_ev]),
        }
        ag = _pick(records, material=material, point="atomic", tip="good")
        ac = _pick(records, material=material, point="atomic", tip="crashed")
        m["bands"]["atomic_phase"] = {
            "good_pass_frac": _frac_in([1.0 if r["mast"]["atomic"]["both_passed"] else 0.0 for r in ag], 0.5, 1.5),
            "crashed_pass_frac": _frac_in([1.0 if r["mast"]["atomic"]["both_passed"] else 0.0 for r in ac], 0.5, 1.5),
            "good_fail_reasons": _count(sum((r["mast"]["atomic"]["fwd"]["reasons"] for r in ag), [])),
            "crashed_fail_reasons": _count(sum((r["mast"]["atomic"]["fwd"]["reasons"] for r in ac), [])),
        }
        edge = _vals(vg, "edge_resolution_px")
        m["bands"]["edge_resolution_px"] = {"good_median": _median(edge), "real_median_reported": BANDS["edge_px_real_median"],
                                            "note": "reported only — not a ruler (DESIGN §4.6)"}
        out["by_material"][material] = m
    return out


def _count(vals) -> dict:
    c: dict = {}
    for v in vals:
        k = str(v)
        c[k] = c.get(k, 0) + 1
    return c


# ── report ──────────────────────────────────────────────────────────────────────────

def build_report(fs: FidelitySet, real_by_material: dict[str, dict]) -> dict:
    real_json = {}
    for material, real in real_by_material.items():
        rj = {k: v for k, v in real.items() if k != "genres"}
        if real.get("available"):
            rj["genres"] = {g: {k: v for k, v in gv.items() if k != "arrays"} for g, gv in real["genres"].items()}
        real_json[material] = rj
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "points": {p.name: {"w_nm": p.w_nm, "px": p.px, "line_s": p.line_s, "bias_v": p.bias_v,
                            "setpoint_a": p.setpoint_a, "nm_per_px": p.nm_per_px} for p in POINTS.values()},
        "set": {"n": fs.n, "materials": list(fs.materials), "tips": list(fs.tips), "points": list(fs.points),
                "seed_base": fs.seed_base},
        "real": real_json,
        "comparison": compare(fs.records, real_by_material),
        "frames": fs.records,
    }


def summary_lines(report: dict) -> list[str]:
    lines = []
    for material, m in report["comparison"]["by_material"].items():
        b = m["bands"]
        vc = b["verify_corr"]
        lines.append(f"[{material}] verify trace_retrace_corr good: median {_fmt(vc['good']['median'])}, "
                     f"in band {vc['good']['band']}: {_fmt(vc['good']['frac_in_band'])}; "
                     f"crashed median {_fmt(vc['crashed']['median'])} (target ~{vc['crashed']['target']})")
        ss = b["step_splitting"]
        lines.append(f"[{material}] step_splitting single median {_fmt(ss['single']['median'])} "
                     f"(≤0.147: {_fmt(ss['single']['frac_le_single_max'])}) {ss['single']['verdicts']}; "
                     f"multi median {_fmt(ss['multi']['median'])} (in 0.171–0.185: {_fmt(ss['multi']['frac_in_band'])}) "
                     f"{ss['multi']['verdicts']}")
        tc = b["tip_change"]
        lines.append(f"[{material}] tip_change detected {_fmt(tc['detected_frac'])} of {tc['n_frames_with_true_change']} "
                     f"frames with a true change, false alarms {_fmt(tc['false_alarm_frac'])}; "
                     f"score {_fmt(tc['score_median_with'])} vs {_fmt(tc['score_median_without'])} (tau {_fmt(tc['threshold'])})")
        ap = b["atomic_phase"]
        lines.append(f"[{material}] atomic_phase pass: good {_fmt(ap['good_pass_frac'])}, crashed {_fmt(ap['crashed_pass_frac'])}")
        for genre, cols in m["ks"].items():
            parts = []
            for col, entry in cols.items():
                e = entry["good"]
                parts.append(f"{col} KS={_fmt(e.get('ks'))} (n {e.get('n_sim')} vs {e.get('n_real')})")
            lines.append(f"[{material}] {genre} good-tip vs real: " + "; ".join(parts))
    for material, real in report["real"].items():
        if not real.get("available"):
            lines.append(f"[{material}] real side unavailable: {real.get('error')} ({real.get('index')})")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=None, help="report JSON (default $STM_BENCH_DATA/fidelity/report.json)")
    ap.add_argument("--n", type=int, default=12, help="seeds per material × tip")
    ap.add_argument("--materials", default="Au(111)", help="comma-separated; the real index is filtered per material")
    ap.add_argument("--tips", default=",".join(TIP_KINDS))
    ap.add_argument("--points", default="verify,atomic")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--real", default=None, help="corpus index dir or parquet (default stmsim.paths.corpus_index_dir())")
    ap.add_argument("--session-dir", default=None, help="where the sim .sxm go (default $STM_BENCH_DATA/fidelity/sessions)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    fs = FidelitySet(n=args.n, materials=tuple(s.strip() for s in args.materials.split(",") if s.strip()),
                     tips=tuple(s.strip() for s in args.tips.split(",") if s.strip()),
                     points=tuple(s.strip() for s in args.points.split(",") if s.strip()),
                     seed_base=args.seed_base,
                     session_dir=Path(args.session_dir) if args.session_dir else None, verbose=not args.quiet)
    fs.run()
    index_path = resolve_index(args.real)
    real_by_material = {material: real_distributions(index_path, material) for material in fs.materials}
    report = build_report(fs, real_by_material)
    out = Path(args.out) if args.out else fidelity_dir() / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=_json_default), encoding="utf-8")
    for line in summary_lines(report):
        print(line)
    print("wrote", out)
    return 0


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


if __name__ == "__main__":
    raise SystemExit(main())
