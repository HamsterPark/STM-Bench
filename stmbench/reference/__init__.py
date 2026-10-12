"""Privileged reference solutions: how well each paper can be reproduced on this instrument.

A reference solution may use the simulator's truth to navigate (where the atom is, how far the
sample has drifted, whether the tip needs conditioning) but every reported number comes from
data the instrument measured, through :mod:`stmbench.analysis`. Run over many seeds, the
reference solutions give what the plan's M4 needs: whether each paper is solvable, the error
distribution of a careful measurement (tolerances are set at about twice its p90) and the
instrument time it takes (budgets at its p90 times a margin).

    python -m stmbench.cli reference --paper P5 --seeds 30 --jobs 4 --json p5.json
    python -m stmbench.cli reference --paper P3 --seeds 30 --jobs 4
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import p3, p5

PAPERS = {"P3": p3.run, "P5": p5.run}


def _one(args: tuple[str, int]) -> dict:
    paper, seed = args
    try:
        return PAPERS[paper](seed)
    except Exception as exc:  # noqa: BLE001 — a failed seed is a result, not a crash
        return {"paper": paper, "seed": int(seed), "success": False, "error": f"{type(exc).__name__}: {exc}"}


def run_many(paper: str, seeds: list[int], *, jobs: int = 1) -> list[dict]:
    if paper not in PAPERS:
        raise KeyError(f"no reference solution for {paper!r} (have: {', '.join(PAPERS)})")
    tasks = [(paper, int(s)) for s in seeds]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            return list(ex.map(_one, tasks))
    return [_one(t) for t in tasks]


def summarize(rows: list[dict]) -> dict:
    """Success rate, per-claim pass rate and |error| (median, p90), instrument time. Errors are
    relative (``rel_err``) or in the claim's unit (``abs_err``), as the paper's tolerances are."""
    done = [r for r in rows if "error" not in r]
    out = {"n": len(rows), "errors": [r for r in rows if "error" in r],
           "success_rate": float(np.mean([bool(r["success"]) for r in rows])) if rows else 0.0}
    per = {}
    for key, rel in (("rel_err", True), ("abs_err", False)):
        for c in sorted({c for r in done for c in r.get(key, {})}):
            vals = [r[key][c] for r in done if c in r.get(key, {}) and np.isfinite(r[key][c])]
            e = np.abs(np.array(vals))
            ok = [bool(r["claims_ok"].get(c)) for r in done if c in r.get("claims_ok", {})]
            per[c] = {"relative": rel, "pass_rate": float(np.mean(ok)) if ok else 0.0,
                      "err_median": float(np.median(e)) if e.size else float("nan"),
                      "err_p90": float(np.percentile(e, 90)) if e.size else float("nan"),
                      "err_max": float(e.max()) if e.size else float("nan"),
                      "signed_median": float(np.median(vals)) if vals else float("nan")}
    out["claims"] = per
    t = np.array([r["instrument_s"] for r in done]) if done else np.array([])
    out["instrument_s"] = {"median": float(np.median(t)) if t.size else float("nan"),
                           "p90": float(np.percentile(t, 90)) if t.size else float("nan")}
    return out


def format_summary(paper: str, s: dict) -> str:
    lines = [f"{paper} reference: {s['n']} seeds, success {s['success_rate']:.0%}"
             + (f", {len(s['errors'])} runs failed" if s["errors"] else "")]
    lines.append(f"  {'claim':<14}{'pass':>7}{'|err| med':>11}{'|err| p90':>11}{'|err| max':>11}{'signed med':>12}")
    for c, d in s["claims"].items():
        f = (lambda x: f"{x:.1%}") if d["relative"] else (lambda x: f"{x:.3g}")
        g = (lambda x: f"{x:+.1%}") if d["relative"] else (lambda x: f"{x:+.3g}")
        lines.append(f"  {c:<14}{d['pass_rate']:>7.0%}{f(d['err_median']):>11}{f(d['err_p90']):>11}"
                     f"{f(d['err_max']):>11}{g(d['signed_median']):>12}")
    lines.append(f"  instrument time: median {s['instrument_s']['median']:.0f} s, p90 {s['instrument_s']['p90']:.0f} s")
    return "\n".join(lines)


__all__ = ["PAPERS", "format_summary", "run_many", "summarize"]
