"""Statistics for the family × mode × model table (docs/DESIGN.md §5.2 metrics).

Three estimators, all on plain rows (``dict`` with at least ``scenario``, ``seed``,
``mode``, ``model``, ``success``; ``family`` and ``partial`` optional):

* :func:`wilson_ci` — Wilson score interval for a success rate (k of n). ``n = 0``
  gives ``(None, None)``: no episodes is not a rate of zero.
* :func:`paired_delta` — Δ success rate of ``model`` against ``ref_model`` on the
  episodes both ran (same scenario, same seed, same mode: the paired design of the
  shared seed set), with a **bootstrap over seeds** — the seed is the unit that is
  resampled, because the world behind an episode is drawn from the seed and every
  model sees the same worlds. Repeated runs of one cell are averaged before pairing.
* :func:`variance_split` — with some seeds run more than once for the same
  (scenario, mode, model), the success indicator's variance splits into an
  *LLM-sampling* part (same world, different answer) and a *sim* part (different
  world): a one-way random-effects ANOVA per stratum, pooled over strata. Every
  field is ``None`` when no seed was repeated — unknown is not zero.

Nothing here imports numpy beyond the bootstrap; the rows come from
``summarize.load_rows`` or from tests.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np

Z_95 = 1.959963984540054


# ── success rate ────────────────────────────────────────────────────────────
def wilson_ci(k: int, n: int, z: float = Z_95) -> tuple[float | None, float | None]:
    """Wilson score interval for ``k`` successes in ``n`` trials; ``(None, None)`` when n == 0."""
    if n <= 0:
        return None, None
    if k < 0 or k > n:
        raise ValueError(f"k={k} outside 0..n={n}")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    # analytically the interval touches 0 for k == 0 and 1 for k == n; do not let rounding
    # report 0.9999999999999999 for "every episode succeeded"
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return lo, hi


def _succ(row: dict) -> float | None:
    s = row.get("success")
    if s is None:
        return None
    return 1.0 if s else 0.0


def success_rate(rows: Iterable[dict]) -> dict:
    """``{"n", "k", "rate", "ci_lo", "ci_hi"}`` over the rows whose ``success`` is known."""
    vals = [v for v in (_succ(r) for r in rows) if v is not None]
    n = len(vals)
    k = int(sum(vals))
    lo, hi = wilson_ci(k, n)
    return {"n": n, "k": k, "rate": (k / n) if n else None, "ci_lo": lo, "ci_hi": hi}


# ── paired difference vs a reference model ──────────────────────────────────
def _cell_means(rows: Iterable[dict], value: str) -> dict[tuple, float]:
    """(scenario, mode, seed) → mean of ``value`` over the runs of that cell."""
    acc: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        v = _succ(r) if value == "success" else r.get(value)
        if v is None:
            continue
        acc[(r.get("scenario"), r.get("mode"), r.get("seed"))].append(float(v))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def bootstrap_mean_ci(values_by_seed: dict[Any, Sequence[float]], *, n_boot: int = 2000,
                      seed: int = 0, alpha: float = 0.05) -> tuple[float | None, float | None]:
    """Percentile CI of the grand mean, resampling **seeds** (clusters) with replacement.

    Every value of a drawn seed is kept, so a seed with more paired scenarios weighs
    more — the same as in the point estimate. Fewer than two seeds → ``(None, None)``.
    """
    seeds = [s for s, v in values_by_seed.items() if len(v)]
    if len(seeds) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    sums = np.array([float(np.sum(values_by_seed[s])) for s in seeds])
    cnts = np.array([float(len(values_by_seed[s])) for s in seeds])
    idx = rng.integers(0, len(seeds), size=(int(n_boot), len(seeds)))
    means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_delta(rows: Iterable[dict], model: str, ref_model: str, *, value: str = "success",
                 n_boot: int = 2000, seed: int = 0) -> dict:
    """Δ = mean(model − ref) over the (scenario, mode, seed) cells both models ran.

    Returns ``{"model", "ref_model", "n_pairs", "n_seeds", "delta", "ci_lo", "ci_hi",
    "rate", "ref_rate"}``; ``delta`` is ``None`` with no pairs, the CI is ``None`` with
    fewer than two seeds.
    """
    rows = list(rows)
    a = _cell_means((r for r in rows if r.get("model") == model), value)
    b = _cell_means((r for r in rows if r.get("model") == ref_model), value)
    common = sorted(set(a) & set(b), key=str)
    by_seed: dict[Any, list[float]] = defaultdict(list)
    for key in common:
        by_seed[key[2]].append(a[key] - b[key])
    diffs = [d for v in by_seed.values() for d in v]
    out = {"model": model, "ref_model": ref_model, "n_pairs": len(diffs), "n_seeds": len(by_seed),
           "delta": None, "ci_lo": None, "ci_hi": None, "rate": None, "ref_rate": None}
    if not diffs:
        return out
    out["delta"] = float(sum(diffs) / len(diffs))
    out["rate"] = float(sum(a[k] for k in common) / len(common))
    out["ref_rate"] = float(sum(b[k] for k in common) / len(common))
    out["ci_lo"], out["ci_hi"] = bootstrap_mean_ci(by_seed, n_boot=n_boot, seed=seed)
    return out


# ── LLM sampling variance vs sim (seed) variance ────────────────────────────
def variance_split(rows: Iterable[dict], *, value: str = "success") -> dict:
    """Split the variance of ``value`` into LLM-sampling (within a repeated seed) and
    sim (between seeds of one scenario) components.

    Per stratum (scenario, mode, model) the seeds are the groups of a one-way
    random-effects ANOVA; the strata are pooled by degrees of freedom::

        var_llm = Σ SS_within / Σ df_within
        var_sim = max(0, (Σ SS_between / Σ df_between − var_llm) / n0)

    with ``n0`` the usual unbalanced-design group size. Strata without a repeated seed
    contribute nothing to ``var_llm``; with no repeated seed anywhere every estimate is
    ``None``. ``n_episodes`` counts every row that reached the estimator.
    """
    strata: dict[tuple, dict[Any, list[float]]] = defaultdict(lambda: defaultdict(list))
    n_rows = 0
    for r in rows:
        v = _succ(r) if value == "success" else r.get(value)
        if v is None:
            continue
        n_rows += 1
        strata[(r.get("scenario"), r.get("mode"), r.get("model"))][r.get("seed")].append(float(v))
    ssw = dfw = ssb = dfb = 0.0
    n0_num = n0_den = 0.0
    n_groups = n_rep = 0
    for groups in strata.values():
        n_groups += len(groups)
        n_rep += sum(1 for g in groups.values() if len(g) >= 2)
        for g in groups.values():
            if len(g) >= 2:
                m = sum(g) / len(g)
                ssw += sum((x - m) ** 2 for x in g)
                dfw += len(g) - 1
        if len(groups) >= 2:
            allv = [x for g in groups.values() for x in g]
            gm = sum(allv) / len(allv)
            N = len(allv)
            ssb += sum(len(g) * ((sum(g) / len(g)) - gm) ** 2 for g in groups.values())
            dfb += len(groups) - 1
            n0 = (N - sum(len(g) ** 2 for g in groups.values()) / N) / (len(groups) - 1)
            n0_num += n0 * (len(groups) - 1)
            n0_den += len(groups) - 1
    out = {"n_strata": len(strata), "n_groups": n_groups, "n_repeated_groups": n_rep,
           "n_episodes": n_rows, "var_llm": None, "var_sim": None, "frac_llm": None, "frac_sim": None}
    if dfw <= 0:
        return out                                   # no repeated seed: unknown, not zero
    var_llm = ssw / dfw
    out["var_llm"] = float(var_llm)
    if dfb > 0 and n0_den > 0:
        msb = ssb / dfb
        n0 = n0_num / n0_den
        var_sim = max(0.0, (msb - var_llm) / n0)
        out["var_sim"] = float(var_sim)
        tot = var_llm + var_sim
        if tot > 0:
            out["frac_llm"] = float(var_llm / tot)
            out["frac_sim"] = float(var_sim / tot)
    return out


# ── diagnosis accuracy (ledger diag_correct: 1 | 0 | 'abstain') ─────────────
def diag_accuracy(rows: Iterable[dict]) -> dict:
    """``{"diag_n", "diag_correct_n", "diag_abstain", "diag_acc"}`` over the rows' ``diag_correct``.

    ``diag_n`` counts the episodes that made a statement (``1`` or ``0``), ``diag_abstain``
    those that made none; ``diag_acc`` is correct / stated, ``None`` when nothing was stated
    — an agent that never reports has no accuracy, not a bad one. Rows without the key (or
    with ``-`` / None) count for neither.
    """
    n = k = abstain = 0
    for r in rows:
        v = r.get("diag_correct")
        if v == "abstain":
            abstain += 1
        elif isinstance(v, bool) or v in (0, 1):
            n += 1
            k += int(v)
    return {"diag_n": n, "diag_correct_n": k, "diag_abstain": abstain, "diag_acc": (k / n) if n else None}


# ── the table ───────────────────────────────────────────────────────────────
def aggregate(rows: Iterable[dict], *, ref_model: str | None = None, n_boot: int = 2000,
              seed: int = 0) -> list[dict]:
    """One record per (family, mode, model) plus an ``ALL``-family record per (mode, model):
    n, k, rate, Wilson CI, median sim_s / cmds, and — with ``ref_model`` — the paired Δ
    against it (same family, same mode) with its seed-bootstrap CI.
    """
    rows = list(rows)
    for r in rows:
        r.setdefault("family", str(r.get("scenario", "")).split("_", 1)[0])
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["family"], r.get("mode"), r.get("model"))].append(r)
        cells[("ALL", r.get("mode"), r.get("model"))].append(r)
    out = []
    for key in sorted(cells, key=lambda k: (k[0] == "ALL", str(k))):
        fam, mode, model = key
        rs = cells[key]
        rec = {"family": fam, "mode": mode, "model": model, **success_rate(rs),
               "median_sim_s": _median([r.get("sim_s") for r in rs]),
               "median_cmds": _median([r.get("cmds") for r in rs]),
               "mean_partial": _mean([r.get("partial") for r in rs]),
               **diag_accuracy(rs)}
        if ref_model is not None and model != ref_model:
            pool = [r for r in rows if r.get("mode") == mode and (fam == "ALL" or r["family"] == fam)]
            d = paired_delta(pool, model, ref_model, n_boot=n_boot, seed=seed)
            rec.update({"delta": d["delta"], "delta_lo": d["ci_lo"], "delta_hi": d["ci_hi"],
                        "n_pairs": d["n_pairs"], "n_seeds": d["n_seeds"]})
        out.append(rec)
    return out


def _median(vals) -> float | None:
    xs = sorted(float(v) for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool))
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2.0


def _mean(vals) -> float | None:
    xs = [float(v) for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return (sum(xs) / len(xs)) if xs else None


def fmt_ci(lo: float | None, hi: float | None, digits: int = 2) -> str:
    if lo is None or hi is None:
        return "-"
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


__all__ = ["wilson_ci", "success_rate", "bootstrap_mean_ci", "paired_delta", "variance_split",
           "diag_accuracy", "aggregate", "fmt_ci"]
