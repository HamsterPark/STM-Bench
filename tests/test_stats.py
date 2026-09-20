"""report.stats on synthetic rows: Wilson CI, paired Δ with a seed bootstrap, variance split."""
from __future__ import annotations

import pytest

from stmbench.report import stats
from stmbench.report.stats import (aggregate, bootstrap_mean_ci, paired_delta, success_rate,
                                   variance_split, wilson_ci)


def _row(scenario, seed, model, success, mode="A", partial=None, sim_s=100, cmds=10):
    return {"scenario": scenario, "family": scenario[:2], "seed": seed, "mode": mode, "model": model,
            "success": success, "partial": (1.0 if success else 0.0) if partial is None else partial,
            "sim_s": sim_s, "cmds": cmds}


# ── Wilson ──────────────────────────────────────────────────────────────────
def test_wilson_known_values_and_bounds():
    lo, hi = wilson_ci(5, 10)
    assert lo == pytest.approx(0.2366, abs=2e-3) and hi == pytest.approx(0.7634, abs=2e-3)
    assert wilson_ci(10, 10)[1] == 1.0 and wilson_ci(10, 10)[0] == pytest.approx(0.722, abs=2e-3)
    assert wilson_ci(0, 10)[0] == 0.0 and wilson_ci(0, 10)[1] == pytest.approx(0.278, abs=2e-3)
    lo1, hi1 = wilson_ci(50, 100)
    assert lo < lo1 < 0.5 < hi1 < hi                       # more trials → narrower interval


def test_wilson_no_trials_is_unknown_not_zero():
    assert wilson_ci(0, 0) == (None, None)
    r = success_rate([])
    assert r["n"] == 0 and r["rate"] is None and r["ci_lo"] is None
    with pytest.raises(ValueError):
        wilson_ci(11, 10)


def test_success_rate_ignores_unknown_success():
    rows = [_row("B1_a", 0, "m", True), _row("B1_a", 1, "m", False), _row("B1_a", 2, "m", None)]
    r = success_rate(rows)
    assert (r["n"], r["k"], r["rate"]) == (2, 1, 0.5)


# ── paired Δ ────────────────────────────────────────────────────────────────
def _paired_rows():
    rows = []
    for sc in ("B1_a", "B5_b"):
        for seed in range(5):
            rows.append(_row(sc, seed, "X", True))
            rows.append(_row(sc, seed, "R", seed < 3))     # the reference fails on seeds 3, 4
    return rows


def test_paired_delta_point_estimate_and_sign():
    d = paired_delta(_paired_rows(), "X", "R", n_boot=500)
    assert d["n_pairs"] == 10 and d["n_seeds"] == 5
    assert d["delta"] == pytest.approx(0.4) and d["rate"] == 1.0 and d["ref_rate"] == pytest.approx(0.6)
    assert d["ci_lo"] is not None and d["ci_lo"] <= 0.4 <= d["ci_hi"]
    assert 0.0 <= d["ci_lo"] and d["ci_hi"] <= 1.0
    back = paired_delta(_paired_rows(), "R", "X", n_boot=500)
    assert back["delta"] == pytest.approx(-0.4)


def test_paired_delta_pairs_only_shared_cells_and_averages_repeats():
    rows = _paired_rows()
    rows = [r for r in rows if not (r["model"] == "X" and r["scenario"] == "B5_b" and r["seed"] == 4)]
    rows.append(_row("B1_a", 0, "X", False))               # a repeat of one cell: X is 0.5 there
    d = paired_delta(rows, "X", "R", n_boot=200)
    assert d["n_pairs"] == 9
    assert d["delta"] == pytest.approx((0.5 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 - 6) / 9)


def test_paired_delta_without_pairs_or_seeds_stays_unknown():
    d = paired_delta(_paired_rows(), "X", "nobody")
    assert d["n_pairs"] == 0 and d["delta"] is None and d["ci_lo"] is None
    one_seed = [r for r in _paired_rows() if r["seed"] == 0]
    d1 = paired_delta(one_seed, "X", "R")
    assert d1["n_pairs"] == 2 and d1["delta"] == 0.0 and d1["ci_lo"] is None    # < 2 seeds: no CI


def test_bootstrap_resamples_seeds_deterministically():
    by_seed = {0: [1.0, 1.0], 1: [0.0, 0.0], 2: [1.0, 0.0]}
    a = bootstrap_mean_ci(by_seed, n_boot=300, seed=7)
    b = bootstrap_mean_ci(by_seed, n_boot=300, seed=7)
    assert a == b and a[0] is not None and a[0] <= 0.5 <= a[1]
    assert bootstrap_mean_ci({0: [1.0, 0.0]}) == (None, None)


# ── variance split ──────────────────────────────────────────────────────────
def test_variance_split_without_repeats_is_unknown():
    vs = variance_split(_paired_rows())
    assert vs["n_repeated_groups"] == 0 and vs["var_llm"] is None and vs["frac_llm"] is None
    assert vs["n_episodes"] == 20


def test_variance_split_pure_sim_variance():
    # every repeat of a seed agrees; seeds disagree → nothing from LLM sampling
    rows = []
    for seed in range(4):
        for rep in range(2):
            rows.append(_row("B5_b", seed, "X", seed % 2 == 0))
    vs = variance_split(rows)
    assert vs["n_repeated_groups"] == 4 and vs["var_llm"] == 0.0
    assert vs["var_sim"] > 0 and vs["frac_llm"] == 0.0 and vs["frac_sim"] == 1.0


def test_variance_split_pure_llm_variance():
    # each seed splits 1/0 across its repeats; every seed mean is 0.5 → nothing from the sim
    rows = []
    for seed in range(4):
        rows.append(_row("B5_b", seed, "X", True))
        rows.append(_row("B5_b", seed, "X", False))
    vs = variance_split(rows)
    assert vs["var_llm"] == pytest.approx(0.5) and vs["var_sim"] == 0.0 and vs["frac_llm"] == 1.0


def test_variance_split_pools_strata_and_keeps_single_runs_out_of_within():
    rows = []
    for sc in ("B1_a", "B5_b"):
        for seed in range(3):
            rows.append(_row(sc, seed, "X", True))
        rows.append(_row(sc, 0, "X", False))            # seed 0 repeated once per scenario
    vs = variance_split(rows)
    assert vs["n_strata"] == 2 and vs["n_groups"] == 6 and vs["n_repeated_groups"] == 2
    assert vs["var_llm"] == pytest.approx(0.5)          # two groups [1,0], df 1 each
    assert vs["var_sim"] is not None and 0.0 <= vs["frac_llm"] <= 1.0


# ── the table ───────────────────────────────────────────────────────────────
def test_aggregate_family_and_all_rows_with_reference_delta():
    recs = aggregate(_paired_rows(), ref_model="R", n_boot=200)
    keyed = {(r["family"], r["mode"], r["model"]): r for r in recs}
    assert set(keyed) == {("B1", "A", "X"), ("B1", "A", "R"), ("B5", "A", "X"), ("B5", "A", "R"),
                          ("ALL", "A", "X"), ("ALL", "A", "R")}
    x_all = keyed[("ALL", "A", "X")]
    assert (x_all["n"], x_all["k"]) == (10, 10) and x_all["ci_hi"] == 1.0
    assert x_all["delta"] == pytest.approx(0.4) and x_all["n_pairs"] == 10 and x_all["n_seeds"] == 5
    assert keyed[("B1", "A", "X")]["delta"] == pytest.approx(0.4) and keyed[("B1", "A", "X")]["n_pairs"] == 5
    assert "delta" not in keyed[("ALL", "A", "R")]            # the reference has no Δ against itself
    assert x_all["median_sim_s"] == 100 and x_all["median_cmds"] == 10 and x_all["mean_partial"] == 1.0
    assert recs[-1]["family"] == "ALL"                        # ALL rows come last


def test_fmt_ci_marks_unknown():
    assert stats.fmt_ci(None, None) == "-" and stats.fmt_ci(0.1, 0.25) == "[0.10, 0.25]"
