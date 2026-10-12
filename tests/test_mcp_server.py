"""The instrument over MCP: the protocol subset, the tool table, images, refusals, and an
episode played end to end through JSON-RPC messages (in process and as a subprocess)."""
from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
from pathlib import Path

from stmbench.instrument import ACTIONS, Instrument, verify
from stmbench.instrument.mcp_server import EpisodeServer
from stmbench.instrument.render import read_png_gray

P4 = "P4_atom_positioning_cu111"


def _call(srv: EpisodeServer, name: str, args: dict | None = None, mid: int = 1) -> dict:
    resp = srv.handle({"jsonrpc": "2.0", "id": mid, "method": "tools/call",
                       "params": {"name": name, "arguments": args or {}}})
    assert "result" in resp, resp
    return resp["result"]


def _json(result: dict) -> dict:
    return json.loads(result["content"][0]["text"])


def test_initialize_and_the_tool_table():
    srv = EpisodeServer(Instrument(P4, 0))
    init = srv.handle({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                       "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t"}}})
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert "tools" in init["result"]["capabilities"]
    assert srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    tools = {t["name"]: t for t in srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]}
    for name in [*ACTIONS, "get_task", "status", "info", "list_data", "show_frame", "frame_values",
                 "get_spectrum", "report_result", "report_tip_state", "finish"]:
        assert name in tools and tools[name]["inputSchema"]["type"] == "object", name
    assert tools["report_result"]["inputSchema"]["properties"]["claim_id"]["enum"] == \
        ["target_site", "bystander_max_shift_nm"]
    assert srv.handle({"jsonrpc": "2.0", "id": 2, "method": "nope"})["error"]["code"] == -32601


def test_the_task_carries_the_rule_card_and_the_manual():
    text = _call(EpisodeServer(Instrument(P4, 0)), "get_task")["content"][0]["text"]
    assert "评分规则" in text and "Instrument API" in text and "move_tip" in text


def test_a_scan_returns_its_facts_and_its_image():
    srv = EpisodeServer(Instrument(P4, 0))
    res = _call(srv, "scan", {"center_x_nm": 0, "center_y_nm": 0, "size_nm": 20, "pixels": 64})
    assert not res["isError"]
    facts = _json(res)
    assert facts["frame"].endswith(".sxm") and facts["image"]["display_scale"] == 8
    img = read_png_gray(base64.b64decode(res["content"][1]["data"]))
    assert img.shape == (512, 512) and img.max() > img.min()
    vals = _json(_call(srv, "frame_values", {"frame": facts["frame"], "max_size": 16}))
    assert vals["shape"] == [16, 16] and vals["unit"] == "pm"
    shown = _call(srv, "show_frame", {"frame": facts["frame"], "channel": "current"})
    assert shown["content"][1]["mimeType"] == "image/png"


def test_a_refused_call_is_an_error_result_and_costs_nothing():
    inst = Instrument(P4, 0)
    srv = EpisodeServer(inst)
    res = _call(srv, "scan", {"center_x_nm": 0, "center_y_nm": 0, "size_nm": 20, "pixels": 3})
    assert res["isError"] and "outside" in res["content"][0]["text"]
    res = _call(srv, "show_frame", {"frame": "missing.sxm"})
    assert res["isError"]
    assert inst.used_s == 0.0 and inst.n_actions == 0


def test_an_episode_over_mcp_is_judged_and_rechecks(tmp_path):
    inst = Instrument(P4, 2, out_dir=tmp_path / "run")
    srv = EpisodeServer(inst)
    frame = _json(_call(srv, "scan", {"center_x_nm": 0, "center_y_nm": 0, "size_nm": 15, "pixels": 64}))["frame"]
    _call(srv, "report_result", {"claim_id": "target_site", "value": 0.5, "x_nm": 0.5, "y_nm": 0.5,
                                 "frame": frame, "evidence": [frame]})
    done = _call(srv, "finish")
    assert "over" in done["content"][0]["text"] and "success" not in done["content"][0]["text"]
    assert (tmp_path / "run" / "episode.json").exists()
    assert verify(tmp_path / "run")["ok"]
    assert _call(srv, "wait", {"seconds": 1})["isError"]


def test_closing_the_stream_finishes_the_episode(tmp_path):
    inst = Instrument(P4, 1, out_dir=tmp_path / "run")
    lines = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
             {"jsonrpc": "2.0", "method": "notifications/initialized"},
             {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
              "params": {"name": "set_bias", "arguments": {"bias_v": 0.1}}}]
    out = io.StringIO()
    EpisodeServer(inst).serve_stdio(io.StringIO("\n".join(json.dumps(m) for m in lines) + "\n\nnot json\n"), out)
    replies = [json.loads(x) for x in out.getvalue().splitlines()]
    assert [r.get("id") for r in replies] == [1, 2, None]          # the bad line gets a parse error
    assert replies[2]["error"]["code"] == -32700
    assert inst.finished and (tmp_path / "run" / "episode.json").exists()


def test_the_cli_serves_an_episode_on_stdio(tmp_path):
    msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "status", "arguments": {}}}]
    proc = subprocess.run([sys.executable, "-m", "stmbench.cli", "episode", "--scenario", P4, "--seed", "0",
                           "--out", str(tmp_path / "run")],
                          input="\n".join(json.dumps(m) for m in msgs) + "\n", capture_output=True, text=True,
                          timeout=120, env={**__import__("os").environ, "PYTHONUTF8": "1"})
    assert proc.returncode == 0, proc.stderr
    replies = [json.loads(x) for x in proc.stdout.splitlines() if x.strip()]
    assert replies[0]["result"]["serverInfo"]["name"] == "stm-bench-instrument"
    assert json.loads(replies[1]["result"]["content"][0]["text"])["instrument_time_s"] == 0.0
    assert (tmp_path / "run" / "episode.json").exists()
    v = subprocess.run([sys.executable, "-m", "stmbench.cli", "verify", str(tmp_path / "run")],
                       capture_output=True, text=True, timeout=120, env={**__import__("os").environ, "PYTHONUTF8": "1"})
    assert v.returncode == 0 and json.loads(v.stdout)["ok"], v.stdout + v.stderr
    assert Path(tmp_path / "run" / "actions.jsonl").exists()
