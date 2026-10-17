"""Scripted baselines on the benchmark instrument: no truth.

A baseline drives the instrument an agent gets — the same actions, frames, spectra, status and
task (:class:`stmbench.baselines._common.Blind`) — and decides everything from what it observes.
The privileged reference solutions (:mod:`stmbench.reference`) navigate by the truth and bound
what a careful measurement can reach; a baseline is what a script reaches from the observations
alone, which is the bar an agent should clear.

    python -m stmbench.cli baseline --paper P4 --seeds 30 --jobs 4
"""
from __future__ import annotations

import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from . import p4

PAPERS = {"P4": p4.run}


def _one(args: tuple[str, int]) -> dict:
    paper, seed = args
    try:
        return PAPERS[paper](seed)
    except Exception as exc:  # noqa: BLE001 — a failed seed is a result, not a crash
        return {"paper": paper, "seed": int(seed), "success": False, "error": f"{type(exc).__name__}: {exc}"}


def run_many(paper: str, seeds: list[int], *, jobs: int = 1) -> list[dict]:
    if paper not in PAPERS:
        raise KeyError(f"no baseline for {paper!r} (have: {', '.join(PAPERS)})")
    tasks = [(paper, int(s)) for s in seeds]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            return list(ex.map(_one, tasks))
    return [_one(t) for t in tasks]


def _p90(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.9 * (len(xs) - 1))))]


def summarize(rows: list[dict]) -> dict:
    """Success rate, overall and by the state the tip arrived in; instrument time of the runs
    that succeeded; why the others failed (the session's error, else the claims that failed)."""
    n = len(rows)
    ok = [r for r in rows if r.get("success")]
    by_tip: dict[str, list[int]] = {}
    for r in rows:
        c = by_tip.setdefault(str(r.get("tip_condition")), [0, 0])
        c[0] += int(bool(r.get("success")))
        c[1] += 1
    why = Counter()
    for r in rows:
        if r.get("success"):
            continue
        if r.get("error"):
            why[str(r["error"]).split(" (")[0][:80]] += 1
        else:
            bad = sorted(k for k, v in (r.get("claims_ok") or {}).items() if not v)
            why["claims failed: " + ", ".join(bad) if bad else "no report"] += 1
    t = [float(r["instrument_s"]) for r in ok if "instrument_s" in r]
    return {"n": n, "success_rate": len(ok) / n if n else 0.0,
            "by_tip": {k: {"success": v[0], "n": v[1]} for k, v in sorted(by_tip.items())},
            "instrument_s": {"median": statistics.median(t), "p90": _p90(t), "max": max(t)} if t else {},
            "failures": dict(why.most_common())}


def format_summary(paper: str, s: dict) -> str:
    lines = [f"{paper} baseline: {s['n']} seeds, success {s['success_rate']:.0%}"]
    lines.append("  by the tip it arrived with: " + ", ".join(
        f"{k} {v['success']}/{v['n']}" for k, v in s["by_tip"].items()))
    if s["instrument_s"]:
        t = s["instrument_s"]
        lines.append(f"  instrument time (successes): median {t['median']:.0f} s, p90 {t['p90']:.0f} s, max {t['max']:.0f} s")
    for k, v in s["failures"].items():
        lines.append(f"  failed {v}×: {k}")
    return "\n".join(lines)


__all__ = ["PAPERS", "format_summary", "run_many", "summarize"]
