"""Fast-axis hysteresis from the corpus index (docs/DESIGN.md §4.5, row "快轴迟滞").

The index (``sxm_index_full.parquet``) carries, per frame, two trace/retrace correlations
computed by the indexer (``probe/sxm_fast.py``, 2026-08-27):

* ``fb_corr_raw``  — NCC of the forward block against the backward block **as stored**;
* ``fb_corr_flip`` — the same against the backward block mirrored along x;

both after row-median removal, ≤128 subsampled rows, and **maximised over circular shifts
of ±6 px** — only the maximum was kept, not the shift at which it occurred.

What the pair CAN tell us
-------------------------
1. The storage convention. ``flip ≫ raw`` on essentially every frame ⇒ controller writes the
   backward block mirrored, and the simulator's writer must do the same (it does:
   ``stmsim/io/sxm_writer.py``). The few frames with ``raw > flip`` are the left–right
   symmetric / featureless ones.
2. The population-level distribution of trace/retrace agreement per genre (used by
   :mod:`stmsim.validate.fidelity` as the KS partner) — with the recipe's bias built in.

What it CANNOT tell us
----------------------
* The hysteresis in px for any frame — the argmax shift was never stored, and a scalar
  correlation cannot be inverted for a shift without the frame's lateral autocorrelation.
* Even a *bound* from the ``nx`` trend. The first draft of this module argued: the window
  is ±6 px whatever ``nx``, so a hysteresis that is a fixed fraction of the range would
  leave the window at 512 / 1024 px and lower ``fb_corr_flip`` there; a flat trend would
  bound it. That is wrong because the indexer's search is *circular* (``np.roll``): on a
  100 nm frame whose range is ~1 nm of tilt + steps, the wrapped columns mismatch by the
  whole range and the NCC peaks at ``s = 0`` whatever the true shift (sim frame with a
  known 4 px shift: circular NCC 0.989 at 0 vs 0.943 at 4; overlap NCC 0.9999 at 4). The
  ±6 px window is a zero-shift NCC in practice, and the trend with ``nx`` is flat for that
  reason. The per-``nx`` table is kept as description only.
* Whether 6–7 px (MAST's ``_fwd_bwd_instability`` docstring, "real Createc hardware")
  applies to *this* rig; that number is from a different machine.

The measurement the index cannot give is ``--sample N``: it re-reads N real complete frames
of the verify genre (needs the raw mirror the index paths point at) and locates the NCC peak
over ±24 px on the **overlap** (no wrap), row medians removed. Off by default; the JSON says
whether it ran. First run (2026-08-28, 40 complete Au 80–120 nm frames, ``fb_corr_flip ≥
0.9``): |shift| median 6 px = 1.2 % of the fast axis, a third of the frames beyond the
indexer's ±6 px — the simulator's ``hyst_frac = 0.015`` is the right order.
Output: ``$STM_BENCH_DATA/calib/hysteresis.json``. Run::

    python -m stmsim.calibrate.hysteresis --out $STM_BENCH_DATA/calib/hysteresis.json
    python -m stmsim.calibrate.hysteresis --sample 40      # + direct peak-shift measurement
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from stmsim.paths import calib_dir, corpus_index_dir

MATERIAL_PATTERNS = {
    "Au(111)": r"au\s*\(?111|gold|金",
    "Cu(111)": r"cu\s*\(?111|copper|铜",
    "Ag(111)": r"ag\s*\(?111|silver|银",
    "HOPG": r"hopg|graphite|石墨",
}
INDEX_SHIFT_WINDOW_PX = 6          # the indexer's window — fixed, in px, independent of nx
SIM_HYST_FRAC = 0.015              # stmsim World.hyst_frac (documented there: 3.5–4 px at 256 px)


def _quant(v) -> list | None:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    return [float(x) for x in np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])]


def load_complete(index_path: str | Path, material: str = "Au(111)"):
    """Complete frames (``err`` empty, ``acq_frac ≥ 0.99``) of ``material`` with both
    correlations present. Returns a DataFrame."""
    import pandas as pd

    cols = ["path", "comment", "err", "acq_frac", "range_x_m", "nx", "ny", "fb_corr_raw", "fb_corr_flip",
            "rowjump_frac", "z_p2p98_m"]
    df = pd.read_parquet(index_path, columns=cols)
    d = df[(df["err"].astype(str) == "") & (df["acq_frac"] >= 0.99)]
    pat = MATERIAL_PATTERNS.get(material, re.escape(material))
    hay = d["path"].astype(str) + " " + d["comment"].astype(str)
    d = d[hay.str.contains(pat, case=False, regex=True, na=False)]
    return d.dropna(subset=["fb_corr_raw", "fb_corr_flip"])


def fit_from_index(index_path: str | Path, material: str = "Au(111)", *,
                   size_window_nm: tuple[float, float] = (50.0, 200.0), min_frames_per_nx: int = 50) -> dict:
    """Everything the index can say about the fast-axis hysteresis (see module docstring)."""
    p = Path(index_path)
    if not p.is_file():
        return {"available": False, "index": str(p), "error": "index file not found"}
    d = load_complete(p, material)
    if d.empty:
        return {"available": False, "index": str(p), "error": f"no complete {material} frames with fb_corr"}
    raw = d["fb_corr_raw"].to_numpy(float)
    flip = d["fb_corr_flip"].to_numpy(float)
    # ties (raw == flip, typically both 0.0 on tiny / flat frames whose backward block has
    # no structure) say nothing about the convention — judge on the frames that differ
    differ = flip != raw
    frac_gt = float(np.mean(flip[differ] > raw[differ])) if differ.any() else float("nan")
    convention = {
        "n": int(len(d)), "n_tied": int((~differ).sum()),
        "frac_flip_gt_raw_among_differing": frac_gt,
        "raw_quantiles": _quant(raw), "flip_quantiles": _quant(flip),
        "verdict": ("backward block stored mirrored (flip ≫ raw)" if frac_gt > 0.95
                    else "storage convention unclear from the index"),
    }
    rng_nm = d["range_x_m"].to_numpy(float) * 1e9
    lo, hi = size_window_nm
    s = d[(rng_nm >= lo) & (rng_nm <= hi)]
    by_nx = {}
    for nx, g in s.groupby("nx"):
        if len(g) < min_frames_per_nx:
            continue
        v = g["fb_corr_flip"].to_numpy(float)
        by_nx[int(nx)] = {"n": int(len(g)), "median": float(np.median(v)), "p25": float(np.quantile(v, 0.25)),
                          "window_frac_of_axis": INDEX_SHIFT_WINDOW_PX / float(nx),
                          "sim_hyst_px_at_this_nx": SIM_HYST_FRAC * float(nx)}
    nxs = sorted(by_nx)
    bound = None
    if len(nxs) >= 2:
        med = [by_nx[n]["median"] for n in nxs]
        bound = {
            "median_by_nx": dict(zip(map(str, nxs), med)),
            "median_falls_with_nx": bool(med[-1] < med[0] - 0.02),
            "hyst_frac_upper_bound": None,
            "inference": "none",
            "note": ("NOT a bound. The indexer's shift search is circular (np.roll): on a frame whose range "
                     "is tilt + steps the wrapped columns mismatch by the whole range, so the maximum sits "
                     "at s = 0 whatever the true shift (verified on a sim frame with a known 4 px shift: "
                     "circular peak at 0, overlap peak at 4). The ±6 px window is therefore effectively a "
                     "zero-shift NCC on stepped frames and a flat trend with nx says nothing about the "
                     "hysteresis. Use --sample (overlap NCC) for px."),
            "sim_hyst_frac": SIM_HYST_FRAC,
        }
    genres = {}
    for name, (glo, ghi) in {"verify": (80.0, 120.0), "atomic": (0.0, 10.0), "all": (0.0, float("inf"))}.items():
        g = d[(rng_nm >= glo) & (rng_nm <= ghi)]
        genres[name] = {"n": int(len(g)), "fb_corr_flip_quantiles": _quant(g["fb_corr_flip"].to_numpy(float))}
    return {
        "available": True, "index": str(p), "material": material,
        "index_recipe": {"row_median_removed": True, "rows_subsampled_to": 128,
                         "shift_window_px": INDEX_SHIFT_WINDOW_PX, "circular": True, "argmax_stored": False},
        "storage_convention": convention,
        "by_nx": {str(k): v for k, v in by_nx.items()},
        "size_window_nm": list(size_window_nm),
        "bound_from_nx_trend": bound,
        "genres": genres,
        "cannot_infer": ["per-frame hysteresis in px (argmax shift not stored)",
                         "whether MAST's Createc '6-7 px' applies to this rig"],
        "reference_other_rig_px": [6, 7],
    }


# ── the direct measurement (optional; reads .sxm) ───────────────────────────────────

def peak_shift(fwd: np.ndarray, bwd: np.ndarray, *, max_shift_px: int = 24, max_rows: int = 128) -> dict:
    """NCC of the oriented forward / backward frames over lateral shifts, **on the overlap
    only** (no circular wrap). Returns the argmax shift (px; positive = backward features
    sit to the LEFT of the forward ones by that many px, i.e. the backward line samples
    ahead) and the NCC at zero / at the peak.

    Why not ``np.roll`` like the indexer (and like MAST's FFT cross-correlation): a
    circular shift wraps ``s`` columns from the far edge, and on a 100 nm frame whose
    range is ~1 nm of tilt + steps those wrapped columns mismatch by the whole range —
    enough to pull the peak back to ``s = 0`` even when every row is a clean 4 px shift
    (measured on a sim frame: raw circular NCC 0.989 at 0 vs 0.943 at 4; overlap NCC
    0.9999 at 4). The circular recipe is therefore biased toward zero on stepped frames;
    the index's ``fb_corr_flip`` inherits that bias and so does its ±6 px "window"."""
    a = np.asarray(fwd, float)
    b = np.asarray(bwd, float)
    ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    a, b = a[ok], b[ok]
    if a.shape[0] < 8 or a.shape[1] <= 2 * max_shift_px + 8:
        return {"shift_px": None, "ncc_zero": None, "ncc_peak": None}
    if a.shape[0] > max_rows:
        ii = np.linspace(0, a.shape[0] - 1, max_rows).astype(int)
        a, b = a[ii], b[ii]
    W = a.shape[1]
    best_s, best_v, zero_v = 0, -1.0, None
    for s in range(-max_shift_px, max_shift_px + 1):
        fa = a[:, max(s, 0):W + min(s, 0)]
        ba = b[:, max(-s, 0):W - max(s, 0)]
        # row medians on the overlap, then a global mean — the indexer's normalisation
        fa = fa - np.median(fa, axis=1, keepdims=True)
        ba = ba - np.median(ba, axis=1, keepdims=True)
        fa = fa - fa.mean()
        ba = ba - ba.mean()
        v = float((fa * ba).sum() / (math.sqrt(float((fa * fa).sum()) * float((ba * ba).sum())) + 1e-30))
        if s == 0:
            zero_v = v
        if v > best_v:
            best_s, best_v = s, v
    return {"shift_px": int(best_s), "ncc_zero": zero_v, "ncc_peak": best_v}


def measure_sample(d, n: int, *, seed: int = 0, genre_nm: tuple[float, float] = (80.0, 120.0),
                   min_corr: float = 0.9, max_shift_px: int = 24) -> dict:
    """Re-read ``n`` real complete frames of the verify genre and measure the peak shift."""
    from mast.io.nanonis_files import read_sxm, sxm_oriented_frames

    rng_nm = d["range_x_m"].to_numpy(float) * 1e9
    s = d[(rng_nm >= genre_nm[0]) & (rng_nm <= genre_nm[1]) & (d["fb_corr_flip"] >= min_corr)]
    if s.empty:
        return {"ran": False, "error": "no candidate frames"}
    s = s.sample(n=min(n, len(s)), random_state=seed)
    rows = []
    for _, r in s.iterrows():
        path = str(r["path"])
        try:
            fr = sxm_oriented_frames(read_sxm(path), "Z")
            if fr["forward"] is None or fr["backward"] is None:
                rows.append({"path": path, "error": "no Z fwd/bwd"})
                continue
            res = peak_shift(fr["forward"], fr["backward"], max_shift_px=max_shift_px)
            nx = int(fr["forward"].shape[1])
            res.update(path=path, nx=nx, range_nm=float(r["range_x_m"]) * 1e9,
                       shift_frac_of_axis=(res["shift_px"] / nx) if res["shift_px"] is not None else None)
            rows.append(res)
        except Exception as exc:  # noqa: BLE001 — a single unreadable file must not kill the sample
            rows.append({"path": path, "error": repr(exc)[:120]})
    ok = [r for r in rows if r.get("shift_px") is not None]
    shifts = np.array([r["shift_px"] for r in ok], float)
    fracs = np.array([r["shift_frac_of_axis"] for r in ok], float)
    return {
        "ran": True, "n_requested": n, "n_measured": len(ok), "n_failed": len(rows) - len(ok),
        "window_px": max_shift_px, "genre_nm": list(genre_nm), "min_fb_corr_flip": min_corr,
        "shift_px_quantiles": _quant(shifts), "abs_shift_px_median": float(np.median(np.abs(shifts))) if ok else None,
        "shift_frac_quantiles": _quant(fracs), "abs_shift_frac_median": float(np.median(np.abs(fracs))) if ok else None,
        "frac_beyond_index_window": float(np.mean(np.abs(shifts) > INDEX_SHIFT_WINDOW_PX)) if ok else None,
        "ncc_gain_median": float(np.median([r["ncc_peak"] - r["ncc_zero"] for r in ok])) if ok else None,
        "frames": rows,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="fast-axis hysteresis: what the corpus index can and cannot say")
    ap.add_argument("--index", default=None, help="index dir or parquet (default stmsim.paths.corpus_index_dir())")
    ap.add_argument("--material", default="Au(111)")
    ap.add_argument("--out", default=None, help="default $STM_BENCH_DATA/calib/hysteresis.json")
    ap.add_argument("--sample", type=int, default=0, help="re-read N real frames and measure the peak shift")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    p = Path(args.index) if args.index else corpus_index_dir()
    if p.is_dir():
        p = p / "sxm_index_full.parquet"
    out = fit_from_index(p, args.material)
    if args.sample > 0 and out.get("available"):
        out["direct_measurement"] = measure_sample(load_complete(p, args.material), args.sample, seed=args.seed)
    else:
        out["direct_measurement"] = {"ran": False, "note": "pass --sample N to re-read real frames"}
    op = Path(args.out) if args.out else calib_dir() / "hysteresis.json"
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    if out.get("available"):
        print("storage:", out["storage_convention"]["verdict"], "n =", out["storage_convention"]["n"])
        for nx, v in out["by_nx"].items():
            print(f"  nx={nx}: n={v['n']} median flip corr {v['median']:.3f} (p25 {v['p25']:.3f}); sim hyst would be {v['sim_hyst_px_at_this_nx']:.1f} px")
        if out.get("bound_from_nx_trend"):
            print("nx trend:", out["bound_from_nx_trend"]["note"])
        dm = out["direct_measurement"]
        if dm.get("ran"):
            print(f"direct: {dm['n_measured']} frames, |shift| median {dm['abs_shift_px_median']} px "
                  f"({dm['abs_shift_frac_median']:.4f} of axis; sim hyst_frac {SIM_HYST_FRAC}), "
                  f"beyond the indexer's ±6 px: {dm['frac_beyond_index_window']:.2f}")
    else:
        print("unavailable:", out.get("error"), out.get("index"))
    print("wrote", op)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
