"""One episode with a language model at the instrument and nothing else.

The model gets the instrument's own MCP tools — the same table, the same replies and refusals
an MCP client sees (calls go through :meth:`EpisodeServer.handle`) — a short neutral prompt,
and no other tool: no code sandbox, no files, no helpers. It is the "model alone" condition.

    python -m stmbench.cli agent --scenario P4_atom_positioning_cu111 --seed 0 --out runs/p4_s0 \\
        --provider anthropic --model claude-opus-5-5 --effort high

The run directory holds the instrument's ledger (``episode.json``, ``actions.jsonl``, ...) plus
``driver.json``: the agent, the prompt and its hash, the limits, why the loop stopped, token
use, and every model turn and tool call stamped with instrument time (what a replay plays),
and ``transcript.jsonl``: each turn's content as the model returned it.

The loop ends when the model calls ``finish``, or at a limit (turns, tool calls, wall clock),
after repeated turns without a tool call, on a refusal, or on an error; the instrument is then
finished and judged as it stands.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..instrument import Instrument
from ..instrument.mcp_server import EpisodeServer
from ..replay.record import sim_stamped
from .models import ModelReply, ToolResult

SYSTEM_PROMPT = (
    "You are operating a simulated scanning tunnelling microscope (STM) on your own. The tools you are "
    "given are the instrument and its result channel; nothing else is available. Call get_task first: it "
    "gives the task, its scoring rules and the instrument manual. Actions change the instrument and spend "
    "instrument time, which has a budget; observations do not, and the instrument is paused while you "
    "think. Report results with report_result as the scoring rules describe, then call finish, which ends "
    "the episode; it does not tell you the score. Nobody will answer questions.")
FIRST_MESSAGE = "The instrument is ready."
NUDGE = ("No tool was called. Continue the experiment with the instrument's tools; when your results are "
         "reported, call finish.")
PREVIEW_CHARS = 2000


@dataclass
class AgentResult:
    run_dir: Path
    stop: str
    turns: int
    tool_calls: int
    summary: dict
    usage: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"run_dir": str(self.run_dir), "stop": self.stop, "turns": self.turns,
                "tool_calls": self.tool_calls, "usage": self.usage, **self.summary}


def prompt_sha256(tools: list[dict]) -> str:
    """One hash for everything the model is told before it acts: prompt, first message, nudge, tools."""
    blob = json.dumps({"system": SYSTEM_PROMPT, "first": FIRST_MESSAGE, "nudge": NUDGE, "tools": tools},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _text(s: str) -> dict:
    return {"type": "text", "text": s}


def _preview(blocks: list[dict]) -> str:
    texts = [str(b.get("text", "")) for b in blocks if b.get("type") == "text"]
    n_img = sum(1 for b in blocks if b.get("type") == "image")
    s = "\n".join(texts)
    if len(s) > PREVIEW_CHARS:
        s = s[:PREVIEW_CHARS] + f"… ({len(s)} chars)"
    return s + (f" [+{n_img} image]" if n_img else "")


def run_agent_episode(scenario: str | Path, seed: int, out_dir: str | Path, model: Any, *,
                      max_turns: int = 400, max_tool_calls: int = 1000, max_wall_s: float | None = None,
                      max_nudges: int = 3, log=None) -> AgentResult:
    """Run one episode with ``model`` (a port from :mod:`stmbench.agent.models`) at the controls."""
    out = Path(out_dir)
    if (out / "episode.json").exists() or (out / "driver.json").exists():
        raise FileExistsError(f"{out} already holds an episode; use a new directory")
    inst = Instrument(scenario, seed, out_dir=out)
    server = EpisodeServer(inst)
    tools = server.handle({"jsonrpc": "2.0", "id": 0, "method": "tools/list"})["result"]["tools"]
    events: list[dict] = []
    stamp = sim_stamped(inst.world, events.append)
    transcript = (out / "transcript.jsonl").open("w", encoding="utf-8")
    usage: dict[str, int] = {}
    turns = calls = nudges = 0
    stop: str | None = None
    last_text = ""
    t_start = time.time()
    model.start(SYSTEM_PROMPT, FIRST_MESSAGE)
    say = log or (lambda *a: None)
    try:
        while stop is None:
            if turns >= max_turns:
                stop = "max_turns"
                break
            if max_wall_s is not None and time.time() - t_start > max_wall_s:
                stop = "wall_clock"
                break
            t0 = time.time()
            reply: ModelReply = model.turn(tools)
            turns += 1
            for k, v in (reply.usage or {}).items():
                if isinstance(v, (int, float)):
                    usage[k] = usage.get(k, 0) + int(v)
            stamp({"kind": "model_call", "i": turns, "t": t0, "t_end": time.time(), "stop": reply.stop,
                   "served_by": reply.served_by, "usage": reply.usage})
            transcript.write(json.dumps({"turn": turns, "stop": reply.stop, "served_by": reply.served_by,
                                         "usage": reply.usage, "content": reply.raw}, ensure_ascii=False,
                                        default=str) + "\n")
            transcript.flush()
            if reply.thinking.strip():
                stamp({"kind": "thinking", "role": "assistant", "text": reply.thinking, "t": time.time()})
            if reply.text.strip():
                last_text = reply.text
                stamp({"kind": "message", "role": "assistant", "text": reply.text, "t": time.time()})
            say(f"turn {turns}: {len(reply.tool_calls)} call(s), stop={reply.stop}")
            if reply.stop == "refusal":
                stop = "refusal"
                break
            if not reply.tool_calls:
                nudges += 1
                if nudges > max_nudges:
                    stop = "no_tool_call"
                    break
                model.add_user_text(NUDGE)
                stamp({"kind": "message", "role": "user", "text": NUDGE, "t": time.time()})
                continue
            results: list[ToolResult] = []
            for call in reply.tool_calls:
                if calls >= max_tool_calls:
                    stop = "max_tool_calls"
                    results.append(ToolResult(call.id, [_text("refused: the session's tool-call limit is reached")], True))
                    continue
                calls += 1
                stamp({"kind": "tool_start", "tool_call_id": call.id, "name": call.name, "args": call.args,
                       "t": time.time()})
                if call.error:
                    blocks, is_error = [_text(f"refused: {call.error}")], True
                else:
                    resp = server.handle({"jsonrpc": "2.0", "id": calls, "method": "tools/call",
                                          "params": {"name": call.name, "arguments": call.args}})
                    if "error" in resp:
                        blocks, is_error = [_text(f"refused: {resp['error'].get('message')}")], True
                    else:
                        blocks, is_error = list(resp["result"]["content"]), bool(resp["result"].get("isError"))
                stamp({"kind": "tool_end", "tool_call_id": call.id, "name": call.name, "ok": not is_error,
                       "preview": _preview(blocks), "t": time.time()})
                results.append(ToolResult(call.id, blocks, is_error))
            model.add_tool_results(results)
            if inst.finished and stop is None:
                stop = "finished"
    except Exception as exc:  # noqa: BLE001 — the episode is judged as it stands, and the error recorded
        stop = f"error: {type(exc).__name__}: {exc}"
    finally:
        transcript.close()
        if not inst.finished:
            inst.finish()
        summary = inst.finish()                     # idempotent: the judged summary
        driver = {"agent": model.describe(), "prompt_sha256": prompt_sha256(tools), "system": SYSTEM_PROMPT,
                  "first_message": FIRST_MESSAGE, "nudge": NUDGE,
                  "limits": {"max_turns": max_turns, "max_tool_calls": max_tool_calls, "max_wall_s": max_wall_s,
                             "max_nudges": max_nudges},
                  "stop": stop, "turns": turns, "tool_calls": calls, "nudges": nudges,
                  "wall_s": round(time.time() - t_start, 3), "usage": usage, "events": events,
                  "final_text": last_text, "summary": summary}
        (out / "driver.json").write_text(json.dumps(driver, ensure_ascii=False, indent=1, default=str),
                                         encoding="utf-8")
    return AgentResult(run_dir=out, stop=str(stop), turns=turns, tool_calls=calls, summary=summary, usage=usage)
