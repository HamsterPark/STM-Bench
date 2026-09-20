"""Inventory every controller ``.dat`` spectroscopy file under the raw mirror.

The calibration corpus contains approximately 17k ``.dat`` files. This builds an index so that
I(V) / I(z) templates and barrier-height (κ, φ) distributions can be fitted for the
simulator's junction model.

The group is the first directory below the configured raw root. Root-level files are
assigned to ``other``. Indexes retain source metadata and are private calibration artifacts.

Per file we keep: path, group, size, header fields of interest, column names, row count,
sweep column range, and — when the columns allow it — a first-pass κ fit for I(z) sweeps
(same recipe as ``mast.vision.spectroscopy.assess_iz``: log|I| vs z, points above the noise
floor only).

Run (from STM-Bench root, MAST venv, PYTHONPATH must include MASTv2)::

    python -m stmsim.calibrate.index_dat --root "$STM_BENCH_RAW" --out "$STM_BENCH_DATA/index/dat_index.parquet"

Defaults come from ``stmsim.paths`` (``raw_mirror_root()``, ``index_dir()``).
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from stmsim.paths import index_dir, raw_mirror_root

_HEADER_KEYS = [
    "Experiment", "Date", "Saved Date", "User", "Comment01", "Comment",
    "Bias>Bias (V)", "Bias>Calibration (V/V)",
    "Z-Controller>Setpoint", "Z-Controller>Controller status", "Z-Controller>Z (m)",
    "Bias Spectroscopy>Sweep Start (V)", "Bias Spectroscopy>Sweep End (V)",
    "Bias Spectroscopy>Num Pixel", "Bias Spectroscopy>Z offset (m)",
    "Bias Spectroscopy>Z (m)", "Bias Spectroscopy>Z-controller hold",
    "Bias Spectroscopy>Integration time (s)", "Bias Spectroscopy>Settling time (s)",
    "Z Spectroscopy>Sweep Start (m)", "Z Spectroscopy>Sweep End (m)",
    "Lock-in>Modulated signal", "Lock-in>Amplitude", "Lock-in>Frequency (Hz)",
    "Lock-in>Lock-in status", "X (m)", "Y (m)", "Z (m)",
    "Current>Current (A)", "Current>Calibration (A/V)", "Current>Gain",
    "Temperature 1>Temperature 1 (K)",
]


def _group(path: str, root: str | os.PathLike) -> str:
    """Return the first directory below ``root``; root-level and outside files are ``other``."""
    try:
        parts = Path(path).resolve().relative_to(Path(root).resolve()).parts
    except ValueError:
        return "other"
    return parts[0] if len(parts) > 1 else "other"


def _fit_kappa(z_m: np.ndarray, i_a: np.ndarray) -> dict:
    """log|I| vs z linear fit on points above a noise floor (assess_iz recipe)."""
    out = {"iz_kappa_per_nm": np.nan, "iz_phi_ev": np.nan, "iz_r2": np.nan, "iz_n_ok": 0}
    i = np.abs(np.asarray(i_a, float))
    z = np.asarray(z_m, float) * 1e9
    ok = np.isfinite(i) & np.isfinite(z) & (i > max(1e-30, 1e-4 * np.nanmax(i)))
    if ok.sum() < 5:
        return out
    y = np.log(i[ok])
    x = z[ok]
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope = coef[0]
    yhat = A @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-30
    kappa = abs(slope) / 2.0  # 1/nm
    out.update(iz_kappa_per_nm=kappa, iz_phi_ev=(kappa / 5.123) ** 2,
               iz_r2=1 - ss_res / ss_tot, iz_n_ok=int(ok.sum()))
    return out


def _one(path: str, root: str | os.PathLike) -> dict:
    from mast.io.nanonis_files import read_dat  # PYTHONPATH must include MASTv2

    rec: dict = {"path": path, "group": _group(path, root), "size": os.path.getsize(path),
                 "mtime": os.path.getmtime(path), "err": ""}
    try:
        d = read_dat(path)
    except Exception as exc:  # noqa: BLE001
        rec["err"] = f"{type(exc).__name__}: {exc}"[:200]
        return rec
    hdr = d.get("header", {}) or {}
    cols = d.get("columns", {}) or {}
    for k in _HEADER_KEYS:
        rec["h:" + k] = hdr.get(k, "")
    names = list(cols.keys())
    rec["columns"] = "|".join(names)
    rec["n_cols"] = len(names)
    rec["n_rows"] = int(len(next(iter(cols.values())))) if cols else 0
    if names:
        first = np.asarray(cols[names[0]], float)
        rec["sweep_col"] = names[0]
        rec["sweep_min"] = float(np.nanmin(first)) if first.size else np.nan
        rec["sweep_max"] = float(np.nanmax(first)) if first.size else np.nan
    cur_name = next((n for n in names if n.lower().startswith("current")), None)
    z_name = next((n for n in names if n.lower().startswith("z (m)") or n.lower() == "z"), None)
    if cur_name is not None:
        cur = np.asarray(cols[cur_name], float)
        rec["cur_absmax"] = float(np.nanmax(np.abs(cur))) if cur.size else np.nan
        rec["cur_absmin"] = float(np.nanmin(np.abs(cur))) if cur.size else np.nan
        if z_name is not None and names and names[0] == z_name:
            rec.update(_fit_kappa(cols[z_name], cur))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(raw_mirror_root()),
                    help="raw .dat tree (default $STM_BENCH_RAW)")
    ap.add_argument("--out", default=str(index_dir() / "dat_index.parquet"),
                    help="default $STM_BENCH_DATA/index/dat_index.parquet")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    t0 = time.time()
    files = [str(p) for p in Path(a.root).rglob("*.dat")]
    if a.limit:
        files = files[: a.limit]
    print(f"{len(files)} .dat files found in {time.time()-t0:.0f}s", flush=True)
    recs = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for k, rec in enumerate(ex.map(_one, files, itertools.repeat(a.root), chunksize=32)):
            recs.append(rec)
            if k % 1000 == 0:
                print(f"  {k}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
    import pandas as pd

    df = pd.DataFrame(recs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    df.drop(columns=[c for c in df.columns if c.startswith("h:")]).head(2000).to_csv(
        str(Path(a.out).with_suffix(".head.csv")), index=False)
    print(f"wrote {a.out}: {len(df)} rows, {df['err'].astype(bool).sum()} errors, "
          f"{time.time()-t0:.0f}s", flush=True)
    print(df["group"].value_counts().to_string())
    exp = df["h:Experiment"].value_counts().head(20)
    print(exp.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
