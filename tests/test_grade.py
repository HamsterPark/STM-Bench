"""Replaying an episode from its log records (stmbench.instrument.core.replay_records): the same calls
in order on a fresh instrument, a divergence reported (never raised, never silently scored), and what
a ledger replayed by ``replay`` still does."""
from __future__ import annotations

import json

import pytest

from stmbench.instrument import ActionError, replay
from stmbench.instrument.core import replay_records

P4 = "P4_atom_positioning_cu111"


# ── replay_records ──────────────────────────────────────────────────────────
def _wait(i: int, seconds: float, t0: float) -> dict:
    return {"i": i, "kind": "act", "name": "wait", "params": {"seconds": seconds}, "t0_s": t0,
            "t1_s": t0 + seconds, "result": {"waited_s": seconds}}


def test_replay_records_runs_the_calls_and_skips_what_changed_nothing():
    recs = [_wait(0, 30.0, 0.0),
            {"i": 1, "kind": "observe", "what": "status", "args": {}, "t_s": 30.0},
            {"i": 2, "kind": "refused", "name": "scan", "args": {}, "error": "x", "t_s": 30.0},
            {"i": 3, "kind": "diagnosis", "args": {"tip_state": "sharp", "junction": None, "note": ""}, "t_s": 30.0},
            {"i": 4, "kind": "finish", "t_s": 30.0},
            {"i": 5, "kind": "observe", "what": "status", "args": {}, "t_s": 30.0},
            {"i": 6, "kind": "diagnosis", "args": {"tip_state": "blunt", "junction": None, "note": ""}, "t_s": 30.0},
            _wait(7, 99.0, 30.0),                           # after finish the instrument takes no act:
            {"i": 8, "kind": "report", "args": {"claim_id": "target_site", "value": 1.0}}]  # nor a report
    inst, mismatches = replay_records(P4, 0, recs)
    assert mismatches == [{"i": 7, "name": "wait", "error": "a call after finish"},
                          {"i": 8, "name": "report", "error": "a call after finish"}]
    assert inst.finished and inst.n_actions == 1 and inst.used_s == pytest.approx(30.0)
    assert [d["tip_state"] for d in inst.diagnosis] == ["sharp"]       # nothing after finish is run
    assert inst.verdict.success is False                    # judged as it stands: nothing reported
    assert replay_records(P4, 0, recs[:5])[1] == []


def test_replay_records_names_the_reply_or_time_that_differs():
    bad = _wait(0, 30.0, 0.0)
    bad["result"] = {"waited_s": 31.0, "extra": 1}
    bad["t1_s"] = 31.0
    _, mismatches = replay_records(P4, 0, [bad])
    assert mismatches == [{"i": 0, "name": "wait", "field": "extra"}, {"i": 0, "name": "wait", "field": "waited_s"},
                          {"i": 0, "name": "wait", "field": "t1_s"}]
    no_result = {k: v for k, v in _wait(0, 30.0, 0.0).items() if k != "result"}
    assert replay_records(P4, 0, [no_result])[1] == [{"i": 0, "name": "wait", "field": "result"}]


def test_a_call_the_replay_refuses_is_a_mismatch_not_a_crash():
    recs = [{"i": 0, "kind": "act", "name": "scan",
             "params": {"center_x_nm": 0.0, "center_y_nm": 0.0, "size_nm": 10.0, "pixels": 7}, "result": {}},
            {"i": 1, "kind": "report", "args": {"claim_id": "nope", "value": 1.0}},
            {"i": 2, "kind": "act", "name": "wait", "params": {"seconds": 1.0, "name": 5}},
            {"i": 3, "kind": "act"},
            {"i": 4, "kind": "report", "args": {"claim_id": "target_site"}},
            {"i": 5, "kind": "report", "args": {"claim_id": "target_site", "value": "abc"}},
            {"i": 6, "kind": "diagnosis", "args": {"nonsense": 1}},
            {"i": 7, "kind": "mystery"},
            "junk",
            {"i": 9, "kind": "act", "name": "wait", "params": {"seconds": 1.0}, "result": {"waited_s": 1.0},
             "t0_s": 0.0, "t1_s": 1.0}]                     # the replay goes on after them
    inst, mismatches = replay_records(P4, 0, recs)
    assert [(m["i"], m.get("name")) for m in mismatches] == [
        (0, "scan"), (1, "report"), (2, "wait"), (3, None), (4, "report"), (5, "report"), (6, "diagnosis"),
        (7, "mystery"), (None, None)]
    assert all("error" in m and "field" not in m for m in mismatches)
    assert "outside" in mismatches[0]["error"] and "unknown claim" in mismatches[1]["error"]
    assert inst.n_actions == 1 and inst.finished


def test_replay_of_a_ledger_still_raises_a_call_the_replay_refuses(tmp_path):
    """``replay`` is ``replay_records`` that raises: a ledger that cannot be replayed is not
    silently replayed to the end."""
    rec = {"i": 0, "kind": "act", "name": "scan",
           "params": {"center_x_nm": 0.0, "center_y_nm": 0.0, "size_nm": 10.0, "pixels": 7}, "result": {}}
    (tmp_path / "episode.json").write_text(json.dumps({"scenario_id": P4, "seed": 0}), encoding="utf-8")
    (tmp_path / "actions.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    with pytest.raises(ActionError, match="outside"):
        replay(tmp_path)
