"""Instrument time beyond the scenario budget is not success, whatever the truth says."""
from __future__ import annotations

from stmbench.harness.episode import OVERRUN_PARTIAL_CAP, apply_budget_overrun
from stmbench.trackB.truth_criteria import Verdict


def test_overrun_fails_and_caps_partial():
    v = Verdict(True, 1.0, {"checks": {"x": True}})
    out = apply_budget_overrun(v, consumed_sim_s=16547.0, max_sim_s=7200.0)
    assert out.success is False and out.partial == OVERRUN_PARTIAL_CAP
    assert out.details["budget_overrun"]["factor"] > 2.0
    assert out.details["checks"] == {"x": True}          # the truth checks are kept


def test_within_tolerance_or_uncapped_is_untouched():
    v = Verdict(True, 0.9, {})
    assert apply_budget_overrun(v, 7300.0, 7200.0) is v      # < 5 % over
    assert apply_budget_overrun(v, 99999.0, None) is v       # uncapped scenario
    assert apply_budget_overrun(v, None, 7200.0) is v        # unknown consumption stays unknown
