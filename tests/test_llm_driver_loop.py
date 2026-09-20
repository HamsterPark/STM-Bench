"""The multi-turn driver on a scripted model: a stop without [DONE] gets a 'continue',
[DONE] ends the episode (only at the END of the reply), tools reach the simulator through
MAST's real IC loop, the harness's ReportTipState tool is offered in every mode, the sim
clock stands still while the model is called, and call budgets are exact."""
from __future__ import annotations

import time

import pytest

from stmsim.physics.rig import RigProfile
from stmsim.physics.world import World

from tests.conftest import requires_mast

pytestmark = requires_mast


def _tunnelling_world(tmp_path, seed: int, time_scale: float) -> World:
    w = World(rig=RigProfile.load("reference-stm"), seed=seed, session_dir=tmp_path / "s", time_scale=time_scale)
    w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
    w.withdrawn = False
    w.zctrl_set(True)
    w.transients.clear()
    w.achievable_z_tip()
    return w


def _clock_can_pause(clock) -> bool:
    return callable(getattr(clock, "paused", None)) or (
        callable(getattr(clock, "pause", None)) and callable(getattr(clock, "resume", None)))


def test_driver_continues_until_done_and_tools_hit_the_sim(tmp_path):
    from mast.agentruntime.testing import ScriptedModel

    from stmbench.harness.ic_driver import CONTINUE_TEXT, REPORT_TOOL, run_llm_episode
    from stmbench.harness.runtime_host import RuntimeHost

    w = _tunnelling_world(tmp_path, seed=11, time_scale=20.0)
    host = RuntimeHost(w, tmp_path / "out")
    host.start()
    try:
        model = ScriptedModel([{"tool": "GetBias"}, "先看一下状态，下一步扫图。",   # turn 1: tool, then stop
                               # dimensioned params are strings in MAST (LLM number-corruption defence)
                               {"tool": "SetBias", "args": {"bias_v": "50m"}}, "偏压已设。[DONE]"])  # turn 2
        n_before = len(host.dispatcher.call_log)
        r = run_llm_episode(host, "把偏压设到 50 mV。", model_id="scripted", mode="B0",
                            episode_id="t", max_model_calls=20, max_tool_calls=20, model=model)
        assert r.error is None, r.error
        assert r.outcome == "done" and r.turns == 2
        assert [e["name"] for e in r.events if e["kind"] == "tool_start"] == ["GetBias", "SetBias"]
        assert all(e["ok"] for e in r.events if e["kind"] == "tool_end"), r.events
        # the continue message reached the model on turn 2
        second = model.requests[2]          # requests: t1 call, t1 text, t2 call, t2 text
        assert any(CONTINUE_TEXT in str(getattr(m, "content", "")) for m in second.messages)
        # the harness tool is OFFERED to the model (survives the visibility narrowing), in B0 too
        assert REPORT_TOOL in model.last_tool_names(), model.last_tool_names()
        assert REPORT_TOOL in model.last_system_prompt()
        # the tool call went through the wire to the simulator
        assert len(host.dispatcher.call_log) > n_before
        assert abs(w.bias_v - 0.05) < 1e-9
        # no report ⇒ the diagnosis channel says "unknown", not "clean"
        assert r.diagnosis["junction"] is None and r.diagnosis["reports"] == []
        assert r.diagnosis["final_text"] == r.final_text
        assert r.sim_s > 0 and r.wire_cmds >= 2
        # ledger provenance of the clock protocol
        assert r.model_calls == 4 == model.call_count
        assert r.model_wall_s > 0 and r.clock_paused is _clock_can_pause(w.clock)
    finally:
        host.stop()


def test_sim_budget_stops_the_loop_after_the_first_tool(tmp_path):
    """max_sim_s is polled on every tool_end, not only between turns: with a budget of
    1 sim-second at 20× the first tool already overruns it, the driver sets ctx.abort and
    MAST's loop must return at the next hop — the scripted [DONE] is never requested."""
    from mast.agentruntime.testing import ScriptedModel

    from stmbench.harness.ic_driver import DONE_MARK, run_llm_episode
    from stmbench.harness.runtime_host import RuntimeHost

    w = _tunnelling_world(tmp_path, seed=12, time_scale=20.0)
    host = RuntimeHost(w, tmp_path / "out")
    host.start()
    try:
        model = ScriptedModel([{"tool": "GetBias"}, "看过状态了。[DONE]",
                               {"tool": "SetBias", "args": {"bias_v": "50m"}}, "偏压已设。[DONE]"])
        r = run_llm_episode(host, "把偏压设到 50 mV。", model_id="scripted", mode="B0",
                            episode_id="t-budget", max_model_calls=20, max_tool_calls=20,
                            max_sim_s=1.0, model=model)
        assert r.error is None, r.error
        assert r.outcome == "budget_sim", (r.outcome, r.stop_reason)
        assert r.turns == 1
        assert [e["name"] for e in r.events if e["kind"] == "tool_start"] == ["GetBias"]
        # the loop really stopped: the model was asked exactly once (the tool call), the
        # scripted [DONE] reply was never requested and the second tool never ran
        assert len(model.requests) == 1 and not model.exhausted
        assert DONE_MARK not in r.final_text and r.final_text == ""
        assert abs(w.bias_v - 0.05) > 1e-6, "SetBias ran after the budget was spent"
        assert r.sim_s >= 1.0, r.sim_s
        assert r.diagnosis["junction"] is None
    finally:
        host.stop()


def test_cmd_budget_and_legacy_diagnosis_capture(tmp_path):
    """max_wire_cmds counts wire commands from the episode start; a name-guessed
    report_tip_state call (a tool that does not exist) is still recorded for the ledger."""
    from mast.agentruntime.testing import ScriptedModel

    from stmbench.harness.ic_driver import _diagnosis_from, is_diagnosis_tool, run_llm_episode
    from stmbench.harness.runtime_host import RuntimeHost

    assert is_diagnosis_tool("report_tip_state") and is_diagnosis_tool("ReportTipState")
    assert not is_diagnosis_tool("GetBias") and not is_diagnosis_tool("AssessTipSharpness")
    d = _diagnosis_from([{"name": "report_tip_state", "args": {"junction": "contaminated"}},
                         {"name": "report_tip_state", "args": {"junction": "clean", "atomic": True}}], "x")
    assert d["junction"] == "clean" and d["frames_passed_atomic"] == 1
    assert _diagnosis_from([{"name": "report_tip_state", "args": {"note": "?"}}], "")["junction"] is None

    w = _tunnelling_world(tmp_path, seed=13, time_scale=20.0)
    host = RuntimeHost(w, tmp_path / "out")
    host.start()
    try:
        # an unknown tool name reaches the loop as a tool_start (then fails as unknown) — that
        # is exactly what a model "reporting" through a non-existent tool looks like
        model = ScriptedModel([{"tool": "report_tip_state", "args": {"junction": "dirty", "reason": "phi 1.2 eV"}},
                               {"tool": "GetBias"}, {"tool": "GetBias"}, {"tool": "GetBias"},
                               "全部看完。[DONE]"])
        r = run_llm_episode(host, "判断结面是否干净。", model_id="scripted", mode="B0",
                            episode_id="t-cmds", max_model_calls=20, max_tool_calls=20,
                            max_wire_cmds=1, model=model)
        assert r.error is None, r.error
        assert r.outcome == "budget_cmds", (r.outcome, r.stop_reason)
        assert r.turns == 1 and r.wire_cmds >= 1
        names = [e["name"] for e in r.events if e["kind"] == "tool_start"]
        # the tool whose end crosses the budget is the last one: MAST's IC loop issues a few
        # dozen wire probes of its own before the first tool (middleware), so a budget of 1
        # is spent at report_tip_state's end already; even if it were not, GetBias itself
        # costs >= 1 command — so a second GetBias can never run
        assert names[0] == "report_tip_state" and names.count("GetBias") <= 1, names
        assert not model.exhausted
        assert r.diagnosis["junction"] == "dirty"
        assert r.diagnosis["reports"][0]["args"]["reason"] == "phi 1.2 eV"
        assert len(r.diagnosis["reports"]) == 1
    finally:
        host.stop()


def test_report_tip_state_tool_records_once_and_acks(tmp_path):
    """The harness tool is real: it runs through MAST's tool stack, records the statement
    exactly once (not again from the tool_start event), answers the model with an ack."""
    from mast.agentruntime.testing import ScriptedModel

    from stmbench.harness.ic_driver import REPORT_TOOL, run_llm_episode
    from stmbench.harness.runtime_host import RuntimeHost

    w = _tunnelling_world(tmp_path, seed=14, time_scale=20.0)
    host = RuntimeHost(w, tmp_path / "out")
    host.start()
    try:
        args = {"junction": "clean", "atomic_resolution": True, "tip_state": "sharp", "reason": "φ 4.8 eV，晶格 0.29 nm"}
        model = ScriptedModel([{"tool": REPORT_TOOL, "args": args}, "报告完毕。[DONE]"])
        r = run_llm_episode(host, "判断结面与针尖。", model_id="scripted", mode="A",
                            episode_id="t-report", max_model_calls=20, max_tool_calls=20, model=model)
        assert r.error is None, r.error
        assert r.outcome == "done" and r.turns == 1
        ends = [e for e in r.events if e["kind"] == "tool_end" and e["name"] == REPORT_TOOL]
        assert len(ends) == 1 and ends[0]["ok"], ends
        assert r.diagnosis["junction"] == "clean" and r.diagnosis["tip_state"] == "sharp"
        assert r.diagnosis["frames_passed_atomic"] == 1
        assert len(r.diagnosis["reports"]) == 1
        assert r.diagnosis["reports"][0] == {"name": REPORT_TOOL, "turn": 1, "args": args}
        # the ack came back to the model as the paired tool message
        paired = model.requests[1].messages[-1]
        assert "已记录" in str(getattr(paired, "content", "")), paired
        # the report tool was the only tool the model called
        assert [e["name"] for e in r.events if e["kind"] == "tool_start"] == [REPORT_TOOL]
    finally:
        host.stop()


def test_marker_in_the_body_does_not_end_the_episode(tmp_path):
    from mast.agentruntime.testing import ScriptedModel

    from stmbench.harness.ic_driver import run_llm_episode
    from stmbench.harness.runtime_host import RuntimeHost

    w = _tunnelling_world(tmp_path, seed=15, time_scale=20.0)
    host = RuntimeHost(w, tmp_path / "out")
    host.start()
    try:
        model = ScriptedModel(["等扫完我会写 [DONE]，现在先看状态。",          # turn 1: marker in the body
                               {"tool": "GetBias"}, "看过了。[DONE]"])        # turn 2: marker at the end
        r = run_llm_episode(host, "看一下偏压。", model_id="scripted", mode="B0",
                            episode_id="t-marker", max_model_calls=20, max_tool_calls=20, model=model)
        assert r.error is None, r.error
        assert r.outcome == "done" and r.turns == 2
        assert r.turn_log[0]["note"] == "marker_in_body" and r.turn_log[0]["marker"] is None
        assert r.turn_log[1]["marker"] == "done" and r.turn_log[1]["note"] is None
    finally:
        host.stop()


def test_model_call_budget_is_exact(tmp_path):
    """max_model_calls=N ⇒ exactly N model invocations and N recorded — whether the cap
    falls between turns (nothing left ⇒ no turn is started) or inside a turn (MAST's loop
    counts the call that tripped its limit; the ledger clamps it)."""
    from mast.agentruntime.testing import ScriptedModel

    from stmbench.harness.ic_driver import run_llm_episode
    from stmbench.harness.runtime_host import RuntimeHost

    w = _tunnelling_world(tmp_path, seed=16, time_scale=20.0)
    host = RuntimeHost(w, tmp_path / "out")
    host.start()
    try:
        # between turns: 2 calls per turn, cap 4 ⇒ two turns, then 'budget' before turn 3
        model = ScriptedModel([{"tool": "GetBias"}, "看了。", {"tool": "SetBias", "args": {"bias_v": "50m"}},
                               "设了。", {"tool": "GetBias"}, "再看。[DONE]"])
        r = run_llm_episode(host, "看偏压。", model_id="scripted", mode="B0",
                            episode_id="t-cap-a", max_model_calls=4, max_tool_calls=20, model=model)
        assert r.error is None, r.error
        assert r.outcome == "budget", (r.outcome, r.stop_reason)
        assert r.turns == 2 and r.model_calls == 4 == model.call_count and not model.exhausted
        # inside a turn: tools only, cap 3 ⇒ the 4th request is never made
        model = ScriptedModel([{"tool": "GetBias"}, {"tool": "SetBias", "args": {"bias_v": "60m"}},
                               {"tool": "GetBias"}, {"tool": "SetBias", "args": {"bias_v": "70m"}}, "完。[DONE]"])
        r = run_llm_episode(host, "看偏压。", model_id="scripted", mode="B0",
                            episode_id="t-cap-b", max_model_calls=3, max_tool_calls=20, model=model)
        assert r.error is None, r.error
        assert r.outcome == "budget", (r.outcome, r.stop_reason)
        assert r.model_calls == 3 == model.call_count, (r.model_calls, model.call_count)
        assert r.tool_calls == 3
        # float32 on the wire: 60 mV comes back as 0.0599999987
        assert abs(w.bias_v - 0.06) < 1e-6, ("the 4th tool (SetBias 70m) must not have run", w.bias_v)
        # cap 0: not a single request
        model = ScriptedModel([{"tool": "GetBias"}, "x [DONE]"])
        r = run_llm_episode(host, "看偏压。", model_id="scripted", mode="B0",
                            episode_id="t-cap-c", max_model_calls=0, max_tool_calls=20, model=model)
        assert r.outcome == "budget" and r.turns == 0 and r.model_calls == 0 == model.call_count
    finally:
        host.stop()


class _SlowScripted:
    """A ScriptedModel that takes real wall time per reply (a slow provider)."""

    def __init__(self, inner, delay_s: float):
        self._inner = inner
        self._delay = delay_s

    def invoke(self, request):
        time.sleep(self._delay)
        return self._inner.invoke(request)

    def stream(self, request):
        time.sleep(self._delay)
        yield from self._inner.stream(request)

    @property
    def call_count(self):
        return self._inner.call_count


def test_sim_clock_stands_still_while_the_model_thinks(tmp_path):
    """Benchmark protocol: sim time advances only while tools run. A provider that takes
    0.15 s per reply must not cost the episode 3 sim-seconds at 20×."""
    from mast.agentruntime.testing import ScriptedModel

    from stmbench.harness.ic_driver import run_llm_episode
    from stmbench.harness.runtime_host import RuntimeHost

    w = _tunnelling_world(tmp_path, seed=17, time_scale=20.0)
    host = RuntimeHost(w, tmp_path / "out")
    host.start()
    try:
        delay = 0.15
        model = _SlowScripted(ScriptedModel([{"tool": "GetBias"}, "看过了。[DONE]"]), delay)
        wall0 = w.clock.wall()
        t0 = time.perf_counter()
        r = run_llm_episode(host, "看偏压。", model_id="scripted", mode="B0",
                            episode_id="t-clock", max_model_calls=20, max_tool_calls=20, model=model)
        elapsed = time.perf_counter() - t0
        clock_elapsed = w.clock.wall() - wall0
        assert r.error is None, r.error
        assert r.outcome == "done" and model.call_count == 2
        assert r.model_wall_s >= 2 * delay, r.model_wall_s
        if not _clock_can_pause(w.clock):
            pytest.skip("stmsim Clock has no pause()/resume() yet — the driver degrades to pass-through")
        assert r.clock_paused is True
        # the clock saw the episode MINUS the model's thinking time
        assert clock_elapsed <= elapsed - r.model_wall_s + 0.05, (clock_elapsed, elapsed, r.model_wall_s)
        assert clock_elapsed < elapsed - 2 * delay + 0.05
    finally:
        host.stop()
