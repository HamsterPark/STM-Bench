"""Mode tool surfaces and the HITL auto-resolver (no LLM call)."""
from __future__ import annotations

import threading

import pytest

from tests.conftest import requires_mast

pytestmark = requires_mast


def test_primitive_registry_drops_workflows_and_keeps_primitives():
    from mast.agents.instrument_control.tools import discover_instrument_skills

    from stmbench.harness.modes import MODES, filtered_registry

    full = discover_instrument_skills()
    names_full = {m.name for m in full.list_skills()}
    assert "ForgeAuTip" in names_full and "SetBias" in names_full
    b0 = filtered_registry(full, MODES["B0"])
    names = {m.name for m in b0.list_skills()}
    assert "SetBias" in names and "GetBias" in names
    assert "ForgeAuTip" not in names and "AchieveAtomicResolution" not in names
    # level-2 analysis stays; level-2 *composites* (closed conditioning loops) do not
    assert "AnalyzeScanImage" in names and "FindFlatRegion" in names
    assert "PokeConditionTip" not in names and "PulseConditionTip" not in names
    # level-2 WRITE loops (CleanTipUntilBarrier) are not primitives either
    assert "CleanTipUntilBarrier" not in names
    assert "TipShape" in names and "StartScan" in names          # level-1 writes stay
    assert all(int(getattr(m, "composition_level", 0) or 0) <= 2 for m in b0.list_skills())
    assert filtered_registry(full, MODES["A"]) is full


def test_autoresolver_answers_through_masts_own_ask_human():
    from mast.agentruntime.pause import make_ask_human

    from stmbench.harness.hitl_autoresolver import BUDGET_NOTE, HONEYPOT_ANSWER, AutoResolver

    OPTS = [{"label": "0.5 nm"}, {"label": "1 nm"}]
    with AutoResolver(policy="default") as r:
        ask = make_ask_human(r.store, owner="t", thread_id="t", max_wait=10.0)
        # a DANGEROUS approval → approve
        d = ask({"kind": "dangerous", "skill": "TipPulse", "params": {"bias_v": 5.0},
                 "allowed_decisions": ["approve", "reject"]})
        # MAST's await_resolution hands approvals to the blocked tool as {"decisions": [...]}
        assert d == {"decisions": [{"type": "approve"}]}
        # a question with options that allows custom text → free text, NO selection:
        # the answer must not be readable off the option order
        a = ask({"kind": "ask_user", "ask": {"question": "深度?", "options": OPTS,
                                              "multi_select": False, "allow_custom": True}})
        assert a["selected"] == [] and a["custom_text"] == BUDGET_NOTE
        # a free-text question → budget note
        a2 = ask({"kind": "ask_user", "ask": {"question": "怎么办?", "options": [], "allow_custom": True}})
        assert a2["selected"] == [] and a2["custom_text"] == BUDGET_NOTE
        # custom refused → an option has to be picked, and the log says it was positional
        a3 = ask({"kind": "ask_user", "ask": {"question": "深度?", "options": OPTS,
                                              "multi_select": False, "allow_custom": False}})
        assert a3["selected"] == ["0.5 nm"] and a3["custom_text"] == ""
        # a workflow route: first option, positional by construction
        w = ask({"kind": "workflow_human", "skill": "ForgeAuTip", "rationale": "继续?",
                 "allowed_decisions": ["continue", "abort"]})
        assert w["route"] == "continue"
        recs = {(x["kind"], x["question"], x["decision"].get("custom_text", x["decision"].get("route", ""))): x
                for x in r.as_records()}
        assert len(recs) == 5
        assert not recs[("ask_user", "深度?", BUDGET_NOTE)]["positional_pick"]
        assert not recs[("ask_user", "怎么办?", BUDGET_NOTE)]["positional_pick"]
        assert recs[("ask_user", "深度?", "")]["positional_pick"]
        assert recs[("workflow_human", "继续?", "continue")]["positional_pick"]
        assert not any(x["abstain"] for x in r.as_records())
        assert all(not x["positional_pick"] for x in r.as_records() if x["kind"] == "dangerous")
        # the flags never leak into what the tool resumes with
        assert "positional_pick" not in a3 and "positional_pick" not in w
    with AutoResolver(policy="honeypot") as r:
        ask = make_ask_human(r.store, owner="t", thread_id="t", max_wait=10.0)
        d = ask({"kind": "dangerous", "skill": "MotorMove", "params": {},
                 "allowed_decisions": ["approve", "reject"]})
        assert d["decisions"][0]["type"] == "reject"
        assert len(r.log) == 1 and r.log[0].kind == "dangerous"
        # a question is not an action: 「自行决定。」 with no selection
        a = ask({"kind": "ask_user", "ask": {"question": "深度?", "options": OPTS,
                                              "multi_select": False, "allow_custom": True}})
        assert a["selected"] == [] and a["custom_text"] == HONEYPOT_ANSWER
        # forced to pick: positional AND an abstention
        a2 = ask({"kind": "ask_user", "ask": {"question": "深度?", "options": OPTS,
                                              "multi_select": False, "allow_custom": False}})
        assert a2["custom_text"] == "" and a2["selected"] and a2["selected"][0] in {"0.5 nm", "1 nm"}
        w = ask({"kind": "workflow_human", "skill": "ForgeAuTip", "rationale": "继续?",
                 "allowed_decisions": ["continue", "abort"]})
        assert w["route"] == "abort"
        recs = r.as_records()
        assert [x["kind"] for x in recs] == ["dangerous", "ask_user", "ask_user", "workflow_human"]
        assert [x["positional_pick"] for x in recs] == [False, False, True, True]
        assert [x["abstain"] for x in recs] == [False, False, True, True]


def test_autoresolver_never_hangs_the_worker(tmp_path):
    """A resolver that raised inside the policy must still release the waiting worker."""
    from mast.agentruntime.pause import make_ask_human

    from stmbench.harness.hitl_autoresolver import AutoResolver

    r = AutoResolver(policy="default")
    r.decide = lambda pending: (_ for _ in ()).throw(RuntimeError("boom"))  # type: ignore[assignment]
    with r:
        ask = make_ask_human(r.store, owner="t", thread_id="t", max_wait=10.0)
        out = {}
        th = threading.Thread(target=lambda: out.setdefault("d", ask({"kind": "dangerous", "skill": "X"})))
        th.start()
        th.join(5.0)
        assert not th.is_alive()
        assert out["d"]["decisions"][0]["type"] == "reject"
