"""Calibration of the P7 reference solution: many seeds in one process pool (one cluster job).

    python -m stmbench.reference.p7_calibrate --seeds 30 --jobs 8 --n-drags 32 --out p7.json

Writes one JSON row per seed (and, inside it, one per atom: R̂, the truth the judge will use, the
relative error, the drags, the tip's state on arrival and how it changed) and prints the summary:
the relative error |R̂ / R½ - 1| (p50 / p90 / max, per atom), the precision-cost curve (the same
series cut after n drags) and the instrument time against the scenario's budget.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import p7


def _one(args: tuple) -> dict:
    seed, kw = args
    try:
        return p7.run(seed, finish=False, **kw)
    except Exception as exc:  # noqa: BLE001 - a failed seed is a result, not a crash
        return {"paper": "P7", "seed": int(seed), "success": False, "error": f"{type(exc).__name__}: {exc}"}


def _pct(v, q):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    return float(np.percentile(v, q)) if v.size else float("nan")


def summarize(rows: list[dict], budget_s: float = 5400.0) -> dict:
    done = [r for r in rows if "error" not in r]
    atoms = [a for r in done for a in r["atoms"] if a.get("rel_err") is not None]
    err = [abs(a["rel_err"]) for a in atoms]
    out = {"n_seeds": len(rows), "failed": [{"seed": r["seed"], "error": r["error"]} for r in rows if "error" in r],
           "n_atoms": len(atoms),
           "abs_rel_err": {"p50": _pct(err, 50), "p90": _pct(err, 90), "max": max(err, default=float("nan"))},
           "signed_rel_err_median": _pct([a["rel_err"] for a in atoms], 50)}
    curve = {}
    for n in p7.CURVE_N:
        e = [abs(a["curve"][str(n)]) for a in atoms if a.get("curve", {}).get(str(n)) is not None]
        curve[str(n)] = {"n": len(e), "p50": _pct(e, 50), "p90": _pct(e, 90), "max": max(e, default=float("nan"))}
    out["curve"] = curve
    t = [r["instrument_s"] for r in done]
    out["instrument_s"] = {"p50": _pct(t, 50), "p90": _pct(t, 90), "max": max(t, default=float("nan")),
                           "over_budget": int(sum(x > budget_s for x in t)), "budget_s": budget_s}
    out["evidence_ok"] = {"atoms_ok": int(sum(bool(a.get("evidence", {}).get("all")) for a in atoms)),
                          "atoms": len(atoms)}
    out["both_atoms"] = int(sum(len(r["atoms"]) == 2 and all(a.get("rel_err") is not None for a in r["atoms"])
                                for r in done))
    out["counts"] = {k: int(sum(r.get(k, 0) for r in done))
                     for k in ("conditionings", "pickups", "restarts", "discarded", "tip_changes", "atoms_tried")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--n-drags", type=int, default=32)
    ap.add_argument("--manip-v", type=float, default=p7.MANIP_V)
    ap.add_argument("--step", type=float, default=p7.STAIR_STEP)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kw = {"n_drags": a.n_drags, "manip_v": a.manip_v, "step": a.step}
    tasks = [(s, kw) for s in range(a.first_seed, a.first_seed + a.seeds)]
    if a.jobs > 1:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            rows = list(ex.map(_one, tasks))
    else:
        rows = [_one(t) for t in tasks]
    s = summarize(rows)
    Path(a.out).write_text(json.dumps({"config": kw, "summary": s, "runs": rows}, indent=1, default=str),
                           encoding="utf-8")
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
