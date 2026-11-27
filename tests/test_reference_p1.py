"""The P1 privileged reference solution: measured period and stripe directions, through the judge."""
from __future__ import annotations

import pytest

from stmbench.reference import format_summary, p1, run_many, summarize


@pytest.mark.parametrize("seed", [13, 0])
def test_the_reference_measures_period_and_two_domains_and_the_judge_accepts(seed):
    """Seed 13 arrives with a good tip, seed 0 with a blunt one (conditioned first, far from both
    areas). The reported numbers come from windows of the frames, not from the truth."""
    r = p1.run(seed)
    assert r["success"], r
    assert all(r["claims_ok"].values())
    assert abs(r["abs_err"]["stripe_period_nm"]) < 0.1                  # the claim allows 0.15
    assert all(r["abs_err"][c] < 4.0 for c in r["abs_err"] if c.endswith("_deg"))     # it allows 6
    assert r["values"]["domain_a_orientation_deg"] != r["values"]["domain_b_orientation_deg"]
    assert r["instrument_s"] < 0.5 * r["budget_s"]
    assert r["frames"] >= 2 and r["measured_windows"] >= 4
    if seed == 0:
        assert r["tip_rounds"] >= 1


def test_the_runner_knows_p1_and_summarizes_it():
    rows = run_many("P1", [2])
    s = summarize(rows)
    assert s["n"] == 1 and s["success_rate"] == 1.0
    assert "stripe_period_nm" in s["claims"] and "domain_b_orientation_deg" in s["claims"]
    assert "P1 reference: 1 seeds, success 100%" in format_summary("P1", s)
