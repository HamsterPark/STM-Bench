"""The model-alone loop (stmbench.agent): an episode with the instrument's own tools and nothing
else, recorded next to the instrument's ledger, replayable; the two model ports speak their
providers' formats (checked on fake clients, no network)."""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

from stmbench.agent import (AnthropicModel, ModelReply, OpenAICompatibleModel, ScriptedModel, ToolResult,
                            run_agent_episode)
from stmbench.instrument import verify

P4 = "P4_atom_positioning_cu111"
PNG = {"type": "image", "data": "iVBORw0KGgo=", "mimeType": "image/png"}


def _p4_turns():
    return [[("get_task", {})],
            [("scan", {"center_x_nm": 0.0, "center_y_nm": 0.0, "size_nm": 20.0, "pixels": 64})],
            "Let me think about where the atoms are.",                       # no call: one nudge
            [("set_bias", {"bias_v": 0.005}), ("set_setpoint", {"current_pa": 100000.0}),
             ("move_tip", {"x_nm": 4.0, "y_nm": 0.0, "speed_nm_s": 1.0})],
            [("set_setpoint", {"current_pa": 200.0}),
             ("report_result", {"claim_id": "target_site", "value": 4.0, "x_nm": 4.0, "y_nm": 0.0,
                                "frame": "frame:1", "evidence": ["frame:1"]})],
            [("finish", {})]]


def test_a_scripted_agent_runs_an_episode_whose_record_verifies_and_replays(tmp_path):
    from stmbench.replay.timeline import Ledger, build_timeline

    run = tmp_path / "run"
    model = ScriptedModel(_p4_turns())
    res = run_agent_episode(P4, 0, run, model)
    assert res.stop == "finished" and res.turns == 6 and res.tool_calls == 8
    assert model.nudges and len(model.nudges) == 1
    scan_result = model.received[1][0]
    assert not scan_result.is_error and any(b.get("type") == "image" for b in scan_result.blocks)

    driver = json.loads((run / "driver.json").read_text(encoding="utf-8"))
    assert driver["agent"]["provider"] == "scripted" and driver["stop"] == "finished"
    assert len(driver["prompt_sha256"]) == 64 and driver["nudges"] == 1
    starts = [e for e in driver["events"] if e["kind"] == "tool_start"]
    assert [e["name"] for e in starts] == ["get_task", "scan", "set_bias", "set_setpoint", "move_tip",
                                           "set_setpoint", "report_result", "finish"]
    assert all(isinstance(e.get("sim_s"), float) for e in driver["events"])
    assert len((run / "transcript.jsonl").read_text(encoding="utf-8").splitlines()) == 6
    assert verify(run)["ok"]

    tl = build_timeline(Ledger.load(run))
    move = next(s for s in tl["steps"] if s["name"] == "move_tip")
    assert "50 kΩ" in move["caption"] and "拖" in move["caption"]
    assert any("turn" in m["text"] for m in tl["messages"])


def test_the_loop_stops_at_its_limits_and_on_a_refusal(tmp_path):
    many = [[("status", {})] for _ in range(10)]
    res = run_agent_episode(P4, 0, tmp_path / "a", ScriptedModel(many), max_tool_calls=3)
    assert res.stop == "max_tool_calls" and res.tool_calls == 3

    res = run_agent_episode(P4, 0, tmp_path / "b", ScriptedModel(["just text"] * 10), max_nudges=2)
    assert res.stop == "no_tool_call" and res.turns == 3

    class Refuses(ScriptedModel):
        def turn(self, tools):
            return ModelReply(text="", stop="refusal", usage={"refusal_category": "cyber"})

    res = run_agent_episode(P4, 0, tmp_path / "c", Refuses([]))
    assert res.stop == "refusal"
    assert json.loads((tmp_path / "c" / "episode.json").read_text(encoding="utf-8"))["success"] is False
    with pytest.raises(FileExistsError):
        run_agent_episode(P4, 0, tmp_path / "c", ScriptedModel([]))


def test_a_tool_error_reaches_the_model_as_an_error_result(tmp_path):
    model = ScriptedModel([[("scan", {"center_x_nm": 0.0})], [("finish", {})]])        # center_y_nm missing
    run_agent_episode(P4, 0, tmp_path / "r", model)
    bad = model.received[0][0]
    assert bad.is_error and "refused" in bad.blocks[0]["text"]


# ── the Anthropic port, on a fake client ───────────────────────────────────
class _FakeStream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


class _FakeMessages:
    def __init__(self, msg):
        self.msg = msg
        self.requests = []

    def stream(self, **req):
        self.requests.append(req)
        return _FakeStream(self.msg)


def _claude_message(stop="tool_use", stop_details=None):
    content = [NS(type="thinking", thinking="The frame first.", signature="sig"),
               NS(type="text", text="I will scan."),
               NS(type="tool_use", id="toolu_1", name="scan", input={"center_x_nm": 0, "center_y_nm": 0, "size_nm": 20})]
    usage = NS(input_tokens=120, output_tokens=40, cache_read_input_tokens=100, cache_creation_input_tokens=0)
    return NS(content=content, stop_reason=stop, stop_details=stop_details, usage=usage, model="claude-opus-5-5")


TOOLS = [{"name": "scan", "description": "Image a frame.", "inputSchema": {"type": "object", "properties": {}}}]


def test_the_anthropic_port_builds_its_request_and_keeps_the_history_append_only():
    msgs = _FakeMessages(_claude_message())
    port = AnthropicModel("claude-opus-5-5", effort="high", client=NS(messages=msgs, beta=None))
    port.start("SYSTEM", "FIRST")
    reply = port.turn(TOOLS)
    req = msgs.requests[0]
    assert req["model"] == "claude-opus-5-5" and req["system"] == "SYSTEM"
    assert req["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert req["output_config"] == {"effort": "high"} and req["cache_control"] == {"type": "ephemeral"}
    assert req["tools"][0] == {"name": "scan", "description": "Image a frame.", "input_schema": {"type": "object", "properties": {}}}
    assert "betas" not in req and "fallbacks" not in req and "tool_choice" not in req
    assert [c.name for c in reply.tool_calls] == ["scan"] and reply.text == "I will scan."
    assert reply.thinking == "The frame first." and reply.usage["cache_read_input_tokens"] == 100
    assert port.messages[1]["content"] is msgs.msg.content         # the turn goes back as it came

    port.add_tool_results([ToolResult("toolu_1", [{"type": "text", "text": "{}"}, PNG]),
                           ToolResult("toolu_2", [{"type": "text", "text": "refused: no"}], True)])
    sent = port.messages[2]["content"]
    assert sent[0]["type"] == "tool_result" and sent[0]["tool_use_id"] == "toolu_1" and "is_error" not in sent[0]
    assert sent[0]["content"][1] == {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                                 "data": PNG["data"]}}
    assert sent[1]["is_error"] is True


def test_the_anthropic_port_uses_fallbacks_only_when_asked_and_reads_a_refusal():
    beta = _FakeMessages(_claude_message(stop="refusal", stop_details=NS(category="cyber")))
    port = AnthropicModel(client=NS(messages=None, beta=NS(messages=beta)), fallbacks=True)
    port.start("S", "F")
    reply = port.turn(TOOLS)
    assert beta.requests[0]["betas"] == ["server-side-fallback-2026-07-01"] and beta.requests[0]["fallbacks"] == "default"
    assert reply.stop == "refusal" and reply.usage["refusal_category"] == "cyber"


# ── the OpenAI-compatible port, on a fake transport ────────────────────────
def test_the_openai_compatible_port_speaks_chat_completions():
    posted = []

    def post(url, body):
        posted.append((url, json.loads(json.dumps(body))))
        return {"model": "some-model-2026", "choices": [{"finish_reason": "tool_calls", "message": {
            "content": "Scanning.", "reasoning_content": "Think.",
            "tool_calls": [{"id": "c1", "function": {"name": "scan", "arguments": "{\"size_nm\": 20}"}},
                           {"id": "c2", "function": {"name": "scan", "arguments": "{not json"}}]}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 9, "prompt_tokens_details": {"cached_tokens": 30}}}

    port = OpenAICompatibleModel("some-model", base_url="https://example.invalid/v1/", post=post)
    port.start("SYSTEM", "FIRST")
    reply = port.turn(TOOLS)
    url, body = posted[0]
    assert url == "https://example.invalid/v1/chat/completions" and body["tool_choice"] == "auto"
    assert body["tools"][0] == {"type": "function", "function": {"name": "scan", "description": "Image a frame.",
                                                                 "parameters": {"type": "object", "properties": {}}}}
    assert body["messages"][0] == {"role": "system", "content": "SYSTEM"}
    assert reply.tool_calls[0].args == {"size_nm": 20} and reply.tool_calls[1].error
    assert reply.thinking == "Think." and reply.usage["cache_read_input_tokens"] == 30
    assert reply.served_by == "some-model-2026"
    hist = port.messages[-1]
    assert hist["role"] == "assistant" and [t["id"] for t in hist["tool_calls"]] == ["c1", "c2"]
    assert "reasoning_content" not in hist

    port.add_tool_results([ToolResult("c1", [{"type": "text", "text": "saved"}, PNG]),
                           ToolResult("c2", [{"type": "text", "text": "refused: bad"}], True)])
    assert port.messages[-3] == {"role": "tool", "tool_call_id": "c1", "content": "saved"}
    img = port.messages[-1]
    assert img["role"] == "user" and img["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")

    blind = OpenAICompatibleModel("text-only", post=post, vision=False)
    blind.start("S", "F")
    blind.turn(TOOLS)
    blind.add_tool_results([ToolResult("c1", [{"type": "text", "text": "saved"}, PNG])])
    assert blind.messages[-1]["role"] == "tool" and "without images" in blind.messages[-1]["content"]
