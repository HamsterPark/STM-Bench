"""The rule card: every rule the judge applies to a paper is stated in the task the agent gets,
from the same scenario fields the judge reads."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from stmbench.trackB import rules as R
from stmsim.scenario import Scenario

SCEN = Path(__file__).resolve().parent.parent / "stmbench" / "trackB" / "scenarios"
PAPERS = sorted(SCEN.glob("P*.yaml"))
NAMES = [p.name for p in PAPERS]


def _num(x) -> str:
    return f"{float(x):g}"


@pytest.mark.parametrize("name", NAMES)
def test_no_paper_task_states_its_budget_by_hand(name):
    """P1's task once promised 2 h while the overrun rule used the file's 3 h."""
    assert not R.has_budget_sentence(Scenario.load(SCEN / name).task)


@pytest.mark.parametrize("name", NAMES)
def test_the_card_states_the_budget_the_overrun_rule_uses(name):
    sc = Scenario.load(SCEN / name)
    card = R.rule_card(sc)
    assert f"预算 {_num(sc.budget['sim_hours'])} 小时仪器时间" in card
    assert f"{_num(round(100 * (R.OVERRUN_TOLERANCE - 1), 6))}%" in card


@pytest.mark.parametrize("name", NAMES)
def test_every_claim_is_explained_with_its_tolerance_and_evidence(name):
    sc = Scenario.load(SCEN / name)
    card = R.rule_card(sc)
    for c in sc.claims:
        assert c.get("say"), f"{sc.id}: claim {c['id']} has no `say` for the rule card"
        block = card.split(f"{c['id']}", 1)[1]
        assert "判对条件" in block and "证据" in block
        tol = c.get("tol") or {}
        for key in ("abs", "report_nm"):
            if key in tol:
                assert _num(tol[key]) in block, (c["id"], key)
        if "rel" in tol:
            assert f"{_num(100 * float(tol['rel']))}%" in block, c["id"]
        ev = c["evidence"]
        for key in ("min_fov_nm", "max_nm_per_px", "min_points", "near_nm", "max_centre_dist_nm"):
            if key in ev:
                assert _num(ev[key]) in block, (c["id"], key)


def test_the_p4_verification_frame_is_on_the_card():
    """The rule that was missing from P4's task: a close frame, after the last hop, in the
    frame the report names."""
    sc = Scenario.load(SCEN / "P4_atom_positioning_cu111.yaml")
    rule = next(c for c in sc.claims if c["id"] == "target_site")["evidence"]
    card = R.rule_card(sc)
    assert f"像素尺寸（宽度 ÷ 列数）不大于 {_num(rule['max_nm_per_px'])} nm" in card
    assert "最后一次原子跳动" in card and "frame" in card
    # and the near-tied sites the judge also accepts
    tie = next(c for c in sc.claims if c["id"] == "target_site")["tol"]["site_tie_nm"]
    assert f"只比最近的远不到 {_num(tie)} nm" in card


def test_the_card_explains_citations_and_coordinates():
    card = R.rule_card(Scenario.load(PAPERS[0]))
    for needle in ("evidence", "frame", ".sxm", ".dat", "逆时针", "漂移"):
        assert needle in card


def test_task_text_appends_the_card_to_papers_only():
    paper = Scenario.load(SCEN / "P5_sader_jarvis_qplus.yaml")
    assert R.task_text(paper).startswith(paper.task.strip())
    assert R.task_text(paper).endswith(R.rule_card(paper))
    regression = Scenario.load(SCEN / "B5_repair_blunt.yaml")
    assert R.task_text(regression) == regression.task.strip()


def test_the_card_never_states_a_hidden_value():
    """Tolerances are public; the values drawn for a seed are not."""
    for p in PAPERS:
        sc = Scenario.load(p)
        card = R.rule_card(sc)
        for seed in (0, 1):
            for block, body in sc.draw_hidden(seed).items():
                for key, val in body.items():
                    if isinstance(val, float):
                        assert f"{val:.4f}" not in card, (sc.id, block, key, val)


def test_the_judge_reads_the_version_the_card_announces():
    sc = Scenario.load(PAPERS[0])
    assert re.search(rf"评分规则 v{R.RULES_VERSION}", R.rule_card(sc))
    assert R.RULES_VERSION >= 1
