"""Fit the post-move piezo creep constants from calibration "glance" episodes.

Data: ``<corpus index>/glance_drift.csv`` (983 rows; ``stmsim.paths.corpus_index_dir()``,
``STM_BENCH_CORPUS_INDEX``) — one row per step in
an episode of repeated short scans at one position after a move; ``dx_nm`` / ``dz_pm`` are
the lateral / vertical change between consecutive glances, ``dt_s`` the time between them,
``rel`` the step's position in the episode (0 = right after the move, 1 = the frame the
acquisition was allowed to run to completion).

Model (docs/DESIGN.md §4.2): after a move of size D the position keeps creeping
``x(t) = D · γ · ln(1 + t/τ)`` so the velocity between glances is ``v(t) = D γ /(τ + t)``.
We fit ``A = D γ`` and ``τ`` on the median |v| vs elapsed time, then report γ for the median
move size of the episodes (from ``sxm_sequence.parquet`` when available, else the
data guide's ~half a field of view = 25 nm at 50 nm frames).

Output: ``$STM_BENCH_DATA/calib/creep.json`` (``stmsim.paths.calib_dir()``) with the
fitted numbers and the curves.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from stmsim.paths import calib_dir, corpus_index_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(corpus_index_dir() / "glance_drift.csv"),
                    help="glance step table (default <corpus index>/glance_drift.csv)")
    ap.add_argument("--out", default=str(calib_dir() / "creep.json"),
                    help="default $STM_BENCH_DATA/calib/creep.json")
    ap.add_argument("--move-nm", type=float, default=25.0, help="typical move size before an episode")
    a = ap.parse_args(argv)
    import pandas as pd

    d = pd.read_csv(a.csv)
    d = d[(d.dt_s > 0) & np.isfinite(d.dx_nm) & np.isfinite(d.dz_pm)].copy()
    d = d.sort_values(["ep", "step"])
    d["t_elapsed"] = d.groupby("ep")["dt_s"].cumsum() - d["dt_s"] / 2.0     # mid-interval time since first glance
    d["vx"] = d.dx_nm / d.dt_s              # nm/s
    d["vz"] = d.dz_pm / d.dt_s              # pm/s
    bins = [0, 10, 20, 40, 80, 160, 320, 10000]
    d["tbin"] = pd.cut(d.t_elapsed, bins)
    g = d.groupby("tbin", observed=True)
    tab = g.agg(n=("vx", "size"), t=("t_elapsed", "median"), vx=("vx", "median"), vz=("vz", "median"),
                dx=("dx_nm", "median"), dz=("dz_pm", "median"), corr=("corr", "median")).reset_index()
    print(tab.to_string(index=False))
    # fit v = A / (tau + t) on the binned medians (weights = sqrt(n))
    t = tab.t.to_numpy(float)
    vx = tab.vx.to_numpy(float)
    wgt = np.sqrt(tab.n.to_numpy(float))
    best = None
    for tau in np.logspace(0, 3, 300):
        A = np.sum(wgt * vx * (tau + t)) / np.sum(wgt * (tau + t) ** 0 + 0.0) if False else None
        # least squares for A given tau: minimise Σ w (vx − A/(tau+t))²
        f = 1.0 / (tau + t)
        A = np.sum(wgt * vx * f) / np.sum(wgt * f * f)
        res = float(np.sum(wgt * (vx - A * f) ** 2))
        if best is None or res < best[0]:
            best = (res, tau, A)
    res, tau, A = best
    gamma = A / a.move_nm
    vz_t = tab.vz.to_numpy(float)
    f = 1.0 / (tau + t)
    Az = float(np.sum(wgt * vz_t * f) / np.sum(wgt * f * f))
    out = {
        "n_rows": int(len(d)), "n_episodes": int(d.ep.nunique()),
        "tau_s": float(tau), "A_nm": float(A), "assumed_move_nm": a.move_nm, "gamma": float(gamma),
        "Az_pm": Az, "median_dt_s": float(d.dt_s.median()),
        "final_step_dx_nm": float(d[d.is_final].dx_nm.median()), "final_step_dz_pm": float(d[d.is_final].dz_pm.median()),
        "nonfinal_dx_nm": float(d[~d.is_final].dx_nm.median()), "nonfinal_dz_pm": float(d[~d.is_final].dz_pm.median()),
        "bins": tab.assign(tbin=tab.tbin.astype(str)).to_dict(orient="records"),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nfit: v_x(t) = {A:.3f} nm / ({tau:.1f} s + t)  → gamma = {gamma:.4f} for a {a.move_nm:.0f} nm move; "
          f"v_z(t) = {Az:.1f} pm / (tau + t)")
    print(f"release step (is_final) dx {out['final_step_dx_nm']:.2f} nm / dz {out['final_step_dz_pm']:.0f} pm vs "
          f"earlier steps {out['nonfinal_dx_nm']:.2f} nm / {out['nonfinal_dz_pm']:.0f} pm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
