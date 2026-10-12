"""The P5 privileged reference solution: measured numbers, through the instrument and the judge."""
from __future__ import annotations

import pytest

from stmbench.reference import format_summary, run_many, summarize
from stmbench.reference import p5


@pytest.mark.parametrize("seed", [0, 3])
def test_the_reference_reproduces_p5_from_measured_curves(seed):
    """Seed 0 has a blunt tip and a large short-range background, seed 3 an unstable tip:
    the reference pokes it stable, compensates the drift, and its numbers - not the truth's -
    pass the judge."""
    r = p5.run(seed)
    assert r["success"], r
    assert all(r["claims_ok"].values())
    assert abs(r["rel_err"]["f_min_pn"]) < 0.08 and abs(r["rel_err"]["e_bind_mev"]) < 0.10
    assert abs(r["rel_err"]["f_decay_pm"]) < 0.03
    assert r["values"] != r["truth"]                       # measured, not copied
    assert r["curves"] == 7 and r["instrument_s"] < 900.0
    if seed == 3:
        assert r["tip_pokes"] >= 1


def test_the_runner_summarizes_success_errors_and_time():
    rows = run_many("P5", [1, 2])
    s = summarize(rows)
    assert s["n"] == 2 and s["success_rate"] == 1.0
    assert set(s["claims"]) == {"f_min_pn", "f_decay_pm", "e_bind_mev"}
    assert s["instrument_s"]["p90"] > 0
    assert "P5 reference: 2 seeds, success 100%" in format_summary("P5", s)
    with pytest.raises(KeyError):
        run_many("P9", [0])
