"""λ* — the tip-change-rate UPPER BOUND read off real complete frames (docs/DESIGN.md §4.5,
row "换针率 λ 上界"; §5.2 "λ* 由完整 Au 帧 rowjump_frac p85 反推").

This is the calibrate-side owner of the computation that ``stmbench.trackB.derive_thresholds
.lambda_star_from_index`` also carries; the two are kept identical on purpose (same filters,
same quantile) — change both or neither. What is added here:

* a per-session view ("逐会话"): session = the frame's directory + recording date; the
  per-session p85 rates say how much the bound moves from one night to the next;
* the task-derived λ* (MAST's verify frame must survive in ≥ 90 % of attempts) next to it,
  so ``lambda.json`` carries both numbers with their provenance — ``truth_criteria`` pins
  the task-derived one (3.5e-4 /s).

Why an upper bound: the index's ``rowjump_frac`` = fraction of adjacent-row median jumps
above 4×MAD; feedback settling, noise bursts and adsorbate hops fire it too, and clean
frames still show a median of a few percent. So the per-frame "rate" ``rowjump_frac × rows /
t_frame`` counts every row jump as if it were a tip change: the true spontaneous rate is at
most that.

Output ``$STM_BENCH_DATA/calib/lambda.json``. Run::

    python -m stmsim.calibrate.lambda_from_rowjump --out $STM_BENCH_DATA/calib/lambda.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from stmsim.paths import calib_dir, corpus_index_dir

FILTERS = {"acq_frac": 0.99, "fb_corr_flip": 0.9}


def _frames(index_path: str | Path, material_filter: str):
    """The filtered frame table + the per-frame rate series (shared by both estimators)."""
    import pandas as pd

    df = pd.read_parquet(index_path)
    cols = set(df.columns)
    need = {"rowjump_frac", "ny", "t_fwd_s", "t_bwd_s", "acq_frac", "fb_corr_flip", "path"}
    if not need <= cols:
        raise KeyError(f"index lacks {sorted(need - cols)}")
    d = df[df["err"].isna() | (df["err"].astype(str) == "")] if "err" in cols else df
    if material_filter:
        hay = d["path"].astype(str)
        if "comment" in cols:
            hay = hay + " " + d["comment"].astype(str)
        d = d[hay.str.contains(material_filter, case=False, na=False)]
    d = d[d["acq_frac"] >= FILTERS["acq_frac"]]              # complete frames only
    d = d[d["fb_corr_flip"] >= FILTERS["fb_corr_flip"]]      # MAST's good-tip trace/retrace band
    rows_n = d["ny"].astype(float)
    t_frame = rows_n * (d["t_fwd_s"].astype(float) + d["t_bwd_s"].astype(float))
    keep = (t_frame > 0) & np.isfinite(t_frame)
    d, t_frame, rows_n = d[keep], t_frame[keep], rows_n[keep]
    n_jumps = d["rowjump_frac"].astype(float) * rows_n
    rate = (n_jumps / t_frame).replace([np.inf, -np.inf], np.nan)
    d = d.assign(_rate=rate, _t_frame=t_frame, _n_jumps=n_jumps).dropna(subset=["_rate"])
    return d


def lambda_star_from_index(index_path: str | Path, *, quantile: float = 0.85,
                           material_filter: str = "Au") -> dict:
    """Upper bound on the acceptable tip-change rate from real complete frames — identical
    to ``derive_thresholds.lambda_star_from_index`` (kept in step by hand)."""
    try:
        d = _frames(index_path, material_filter)
    except KeyError as exc:
        return {"lambda_star_per_s": None, "error": str(exc)}
    if d.empty:
        return {"lambda_star_per_s": None, "error": "no frames survived the filters"}
    rate = d["_rate"].to_numpy(float)
    n_jumps = d["_n_jumps"].to_numpy(float)
    t_frame = d["_t_frame"].to_numpy(float)
    q = float(np.quantile(rate, quantile))
    return {"lambda_star_per_s": q, "quantile": quantile, "n_frames": int(rate.size),
            "rate_median_per_s": float(np.median(rate)),
            "rate_p50_p85_p95": [float(np.quantile(rate, p)) for p in (0.5, 0.85, 0.95)],
            "frac_frames_with_any_jump": float((n_jumps > 0).mean()),
            "t_frame_median_s": float(np.median(t_frame)),
            "filters": {"material": material_filter, "acq_frac": f">={FILTERS['acq_frac']}",
                        "fb_corr_flip": f">={FILTERS['fb_corr_flip']}"},
            "note": ("upper bound: the row-jump detector fires on feedback/noise too "
                     "(clean frames still show a median rowjump_frac of a few percent)")}


def lambda_star_per_session(index_path: str | Path, *, quantile: float = 0.85,
                            material_filter: str = "Au", min_frames: int = 20) -> dict:
    """The same bound per session (directory + recording date) — the spread across sessions
    is the honest error bar on the pooled number."""
    try:
        d = _frames(index_path, material_filter)
    except KeyError as exc:
        return {"error": str(exc), "sessions": []}
    if d.empty:
        return {"error": "no frames survived the filters", "sessions": []}
    key = d["path"].astype(str).str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[0]
    if "rec_date" in d.columns:
        key = key + " @ " + d["rec_date"].astype(str)
    out = []
    for name, g in d.groupby(key):
        if len(g) < min_frames:
            continue
        r = g["_rate"].to_numpy(float)
        out.append({"session": name, "n_frames": int(len(g)), "lambda_q_per_s": float(np.quantile(r, quantile)),
                    "rate_median_per_s": float(np.median(r)),
                    "t_frame_median_s": float(np.median(g["_t_frame"].to_numpy(float)))})
    out.sort(key=lambda s: s["lambda_q_per_s"])
    lam = np.array([s["lambda_q_per_s"] for s in out], float)
    return {"quantile": quantile, "min_frames_per_session": min_frames, "n_sessions": len(out),
            "session_lambda_quantiles": ([float(x) for x in np.quantile(lam, [0.05, 0.25, 0.5, 0.75, 0.95])]
                                         if lam.size else None),
            "sessions": out}


def lambda_star_from_task(*, verify_rows: int = 256, verify_line_s: float = 0.586,
                          max_interrupt_p: float = 0.10) -> dict:
    """λ* defined by what the task needs (same as ``derive_thresholds.lambda_star_from_task``):
    MAST's verify frame (256 rows × 2 × 0.586 s ≈ 300 s) survives without a tip change in
    ≥ 90 % of attempts ⇒ λ* = −ln(1 − p) / T."""
    t_frame = verify_rows * 2.0 * verify_line_s
    lam = -math.log(1.0 - max_interrupt_p) / t_frame
    return {"lambda_star_per_s": lam, "t_verify_frame_s": t_frame, "max_interrupt_p": max_interrupt_p,
            "note": "task-derived: a Poisson tip-change rate at which MAST's own verify frame "
                    "is interrupted in at most max_interrupt_p of attempts"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="λ* upper bound from real complete frames' row jumps")
    ap.add_argument("--index", default=None, help="index dir or parquet (default stmsim.paths.corpus_index_dir())")
    ap.add_argument("--out", default=None, help="default $STM_BENCH_DATA/calib/lambda.json")
    ap.add_argument("--quantile", type=float, default=0.85)
    ap.add_argument("--material", default="Au")
    args = ap.parse_args(argv)
    p = Path(args.index) if args.index else corpus_index_dir()
    if p.is_dir():
        p = p / "sxm_index_full.parquet"
    out: dict = {"index": str(p), "lambda_star_task": lambda_star_from_task()}
    if p.is_file():
        try:
            out["lambda_star_index"] = lambda_star_from_index(p, quantile=args.quantile, material_filter=args.material)
            out["per_session"] = lambda_star_per_session(p, quantile=args.quantile, material_filter=args.material)
        except Exception as exc:  # noqa: BLE001 — the task-derived number must still be written
            out["lambda_star_index"] = {"lambda_star_per_s": None, "error": repr(exc)}
    else:
        out["lambda_star_index"] = {"lambda_star_per_s": None, "error": "index file not found"}
    op = Path(args.out) if args.out else calib_dir() / "lambda.json"
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    li = out["lambda_star_index"]
    print("lambda* (task)        =", f"{out['lambda_star_task']['lambda_star_per_s']:.3g} /s")
    print("lambda* (index bound) =", (f"{li['lambda_star_per_s']:.3g} /s over {li['n_frames']} frames"
                                     if li.get("lambda_star_per_s") is not None else li.get("error")))
    ps = out.get("per_session")
    if ps and ps.get("n_sessions"):
        q = ps["session_lambda_quantiles"]
        print(f"per-session p{int(args.quantile * 100)}: {ps['n_sessions']} sessions, "
              f"median {q[2]:.3g} /s, p05–p95 {q[0]:.3g}–{q[4]:.3g} /s")
    print("wrote", op)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
