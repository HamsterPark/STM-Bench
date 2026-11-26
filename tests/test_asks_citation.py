"""``trackB.rules.asks_citation``: whether a claim's evidence line asks the reader to cite data in the evidence field (screen trial 24:
a report went out without the files, the label had no mark and the reply read like any other). The helper is pure: it must agree with
the card's own words for every claim of every paper scenario in both languages, and it changes no card text."""
from __future__ import annotations

from pathlib import Path

import pytest

from stmbench.trackB import rules as R
from stmsim.scenario import Scenario

SCEN = Path(__file__).resolve().parent.parent / "stmbench" / "trackB" / "scenarios"
PAPERS = sorted(SCEN.glob("P*.yaml"))


def cites_in_words(text: str, lang: str) -> bool:
    """Whether one evidence sentence of the card asks to cite something (read from the wording alone)."""
    if lang == "en":
        return "cite " in text and "no citation needed" not in text
    return "evidence 中" in text and "不需要引用" not in text



@pytest.mark.parametrize("path", PAPERS, ids=lambda p: p.name)
@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_helper_agrees_with_the_cards_wording_for_every_claim(path, lang):
    sc = Scenario.load(path)
    for c in sc.claims:
        if c.get("kind") == "constraint":
            continue
        rule = c.get("evidence") or {}
        if rule.get("kind") == "any_of":
            alts = [R._evidence({"evidence": r}, lang) for r in rule.get("rules") or []]
            words = bool(alts) and all(cites_in_words(a, lang) for a in alts)
        else:
            words = cites_in_words(R._evidence(c, lang, sc), lang)
        assert R.asks_citation(c) == words, (path.name, c["id"], lang)


def test_the_cases_by_kind():
    ask = R.asks_citation
    for kind in ("frame", "spectra_near_scatterer", "sts_at_corral_centre", "zspec_pair", "didv_maps", "drag_series"):
        assert ask({"evidence": {"kind": kind}}), kind
    for kind in ("events", "frame_covers", "frame_covers_after", "nothing_like_it"):
        assert not ask({"evidence": {"kind": kind}}), kind
    assert not ask({"evidence": {"kind": "frame_covers_after", "point": "corral_centre"}})
    assert ask({"evidence": {"kind": "any_of", "rules": [{"kind": "frame"}, {"kind": "drag_series"}]}})
    assert not ask({"evidence": {"kind": "any_of", "rules": [{"kind": "frame"}, {"kind": "events"}]}})
    assert not ask({"evidence": {"kind": "any_of", "rules": []}})
    assert ask({})                                   # no evidence field: the card says "frame", as _evidence does


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_asking_changes_no_claim_and_no_card_text(lang):
    import copy
    for path in PAPERS:
        sc = Scenario.load(path)
        before_claims, before_card, before_task = copy.deepcopy(sc.claims), R.rule_card(sc, lang), R.task_text(sc, lang)
        for c in sc.claims:
            R.asks_citation(c)
        assert sc.claims == before_claims and R.rule_card(sc, lang) == before_card and R.task_text(sc, lang) == before_task
