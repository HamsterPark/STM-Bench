"""Working-point priors (bias / setpoint / frame size / pixels) per material from the
105k-frame index — for Track A T4 baselines and for scenario defaults.

    python -m stmsim.calibrate.working_points --out "$STM_BENCH_DATA/calib/working_points.json"

Defaults: ``--index`` = ``stmsim.paths.corpus_index_dir()/sxm_index_full.parquet``,
``--out`` = ``stmsim.paths.calib_dir()/working_points.json``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from stmsim.paths import calib_dir, corpus_index_dir

_MATERIALS = [("Au", r"au\s*\(?111|gold|金"), ("Cu", r"cu\s*\(?111|copper|铜"), ("Ag", r"ag\s*\(?111|silver|银"),
              ("HOPG", r"hopg|graphite|石墨"), ("Si", r"si\s*\(?111|si\s*\(?100|硅"),
              ("WO2I2", r"wo2i2|woi"), ("MoS2", r"mos2"), ("Bi2Se3", r"bi2se3|bi2te3")]


def _material(comment: str) -> str:
    c = str(comment or "").lower()
    for name, pat in _MATERIALS:
        if re.search(pat, c):
            return name
    return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(corpus_index_dir() / "sxm_index_full.parquet"),
                    help="frame index parquet (default <corpus index>/sxm_index_full.parquet)")
    ap.add_argument("--out", default=str(calib_dir() / "working_points.json"),
                    help="default $STM_BENCH_DATA/calib/working_points.json")
    a = ap.parse_args(argv)
    import pandas as pd

    d = pd.read_parquet(a.index)
    d = d[d.err.isna() | (d.err == "")] if "err" in d else d
    d["material"] = d["comment"].map(_material) if "comment" in d else "unknown"
    d["size_nm"] = d["range_x_m"].astype(float) * 1e9
    d["bias_mv"] = d["bias_v"].astype(float) * 1e3
    d["setpoint_pa"] = d["setpoint"].astype(str).str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")[0].astype(float) * 1e12
    complete = d[d.get("acq_frac", 1.0) >= 0.99] if "acq_frac" in d else d
    out = {"n_frames": int(len(d)), "n_complete": int(len(complete)), "materials": {}}

    def q(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return None
        return {"n": int(s.size), "p10": float(s.quantile(.1)), "p50": float(s.median()),
                "p90": float(s.quantile(.9))}

    for mat, g in complete.groupby("material"):
        if len(g) < 50:
            continue
        out["materials"][mat] = {
            "n": int(len(g)),
            "bias_mv": q(g.bias_mv.abs()), "setpoint_pa": q(g.setpoint_pa.abs()), "size_nm": q(g.size_nm),
            "pixels": q(g["nx"]) if "nx" in g else None,
            "bias_sign_negative_frac": float((g.bias_mv < 0).mean()),
            "by_size_bin": {
                str(b): q(gg.bias_mv.abs()) for b, gg in g.groupby(pd.cut(g.size_nm, [0, 10, 30, 100, 300, 1e5]), observed=True)
            },
        }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    for mat, v in out["materials"].items():
        print(f"{mat:8s} n={v['n']:6d} |bias| p50={v['bias_mv']['p50']:.0f} mV  setpoint p50={v['setpoint_pa']['p50']:.0f} pA  "
              f"size p50={v['size_nm']['p50']:.0f} nm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
