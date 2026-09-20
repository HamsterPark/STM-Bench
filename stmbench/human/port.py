"""Mode H: a person at the keyboard is the model under test.

``HumanPort`` implements MAST's ``ChatModelPort`` (``invoke`` / ``stream``) the way a
provider adapter does, except that the reply comes from a browser instead of an API.
Everything else is the LLM path unchanged, because the port is handed to
``ic_driver.run_llm_episode`` exactly where a provider model would go:

* the same IC loop with the full hardware safety stack, the same tool surface as mode A,
  the same stub-for-an-unloaded-pack rule (``bind_all_names_mw``);
* the same ``ReportResult`` / ``ReportTipState`` channels — nothing is parsed out of what
  the person types, a claim never reported is never reproduced;
* the same budgets, polled after every tool, and the same ``[DONE]`` / ``[ABORT]`` end
  markers, which the person sends as a text reply;
* the same clock protocol: the driver wraps this port in ``ClockPausedPort`` as it wraps a
  provider, so sim time stands still while the person thinks and runs only while tools run.

What the person gets that a model does not: the frames and spectra the instrument saved,
rendered as pictures (``stmbench.human.render``). A model in mode A reads them through
the analysis skills; a person reads them with their eyes. That is the one asymmetry, and
the ledger records the mode so the two are never pooled.

``HumanSession`` is the state shared between the episode thread (where the loop calls
``invoke`` and blocks until an action arrives) and the HTTP threads (which read a snapshot
and enqueue actions). Everything in a snapshot is plain JSON.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..harness.ic_driver import ABORT_MARK, DONE_MARK, _budget_probe

#: how ``bind_all_names_mw.BindAllToolNames._stub`` prefixes the description of a tool whose
#: pack is not loaded yet (pinned by tests/test_human_port.py)
STUB_PREFIX = "[未加载·包"

PHASES = ("idle", "starting", "awaiting_action", "running", "finished", "error")


@dataclass
class Action:
    """One reply from the person: a tool call, or a text (the end markers are text)."""

    kind: str                       # "tool" | "text"
    name: str = ""
    args: dict = field(default_factory=dict)
    text: str = ""
    id: str = ""


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    out.append(str(block.get("text", "")))
            elif isinstance(block, str):
                out.append(block)
        return "".join(out)
    return str(content or "")


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str, ensure_ascii=False))


def message_record(m: Any) -> dict:
    """A langchain message as the person should read it: role, text, tool calls, pairing."""
    role = str(getattr(m, "type", None) or type(m).__name__.lower())
    rec: dict = {"role": role, "text": _text_of(getattr(m, "content", ""))}
    calls = getattr(m, "tool_calls", None)
    if calls:
        rec["tool_calls"] = [{"name": c.get("name"), "args": _jsonable(c.get("args") or {}),
                              "id": c.get("id")} for c in calls if isinstance(c, dict)]
    for k in ("tool_call_id", "name"):
        v = getattr(m, k, None)
        if v:
            rec[k] = str(v)
    return rec


def tool_record(t: Any) -> dict:
    desc = str(getattr(t, "description", "") or "")
    return {"name": str(getattr(t, "name", "") or ""), "description": desc,
            "schema": _jsonable(getattr(t, "schema", None) or {}),
            "stub": desc.startswith(STUB_PREFIX),
            "touches_instrument": bool(getattr(t, "touches_instrument", False))}


def strip_markers(text: str) -> str:
    """A note that rides on a tool call must not end the episode by accident."""
    return str(text or "").replace(DONE_MARK, "").replace(ABORT_MARK, "").strip()


class HumanSession:
    """Shared state of one mode-H episode (thread-safe; every reader gets a JSON snapshot)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.version = 0
        self.phase = "idle"
        self.scenario: dict = {}
        self.seed: int | None = None
        self.time_scale: float | None = None
        self.policy: str | None = None
        self.out: str | None = None
        self.started_wall: float | None = None
        self.finished_wall: float | None = None
        self.request: dict | None = None          # the last ModelRequest, as records
        self._messages: list[dict] = []           # the transcript of the last request
        self.events: list[dict] = []
        self.actions: "queue.Queue[Action | None]" = queue.Queue()
        self.cancelled = False
        self.host: Any = None
        self.session_dir: str | None = None       # where the instrument saves .sxm / .dat
        self.run_dir: str | None = None
        self.sim0: float | None = None
        self.cmds0: int | None = None
        self.max_sim_s: float | None = None
        self.max_cmds: int | None = None
        self.result: dict | None = None
        self.error: str | None = None
        self.traceback: str | None = None
        self.calls = 0                            # model calls = replies the person gave

    # ── lifecycle (episode thread / server) ──
    def _bump(self) -> None:
        self.version += 1
        self.cond.notify_all()

    def begin(self, scenario: dict, *, seed: int, time_scale: float | None, policy: str | None,
              out: str, max_sim_s: float | None, max_cmds: int | None) -> None:
        with self.lock:
            self.phase = "starting"
            self.scenario = dict(scenario)
            self.seed = int(seed)
            self.time_scale = time_scale
            self.policy = policy
            self.out = out
            self.max_sim_s = max_sim_s
            self.max_cmds = max_cmds
            self.started_wall = time.time()
            self._bump()

    def attach_host(self, host: Any) -> None:
        """``run_episode``'s ``on_host`` hook: the budget is counted from here."""
        sim, cmds = _budget_probe(host)
        with self.lock:
            self.host = host
            self.session_dir = str(getattr(getattr(host, "world", None), "session_dir", "") or "") or None
            self.run_dir = str(getattr(host, "out_dir", "") or "") or None
            self.sim0, self.cmds0 = sim, cmds
            self._bump()

    def finish(self, result: dict) -> None:
        with self.lock:
            self.result = _jsonable(result)
            self.phase = "finished"
            self.finished_wall = time.time()
            self.host = None
            self._bump()

    def fail(self, error: str, tb: str | None = None) -> None:
        with self.lock:
            self.error = error
            self.traceback = tb
            self.phase = "error"
            self.finished_wall = time.time()
            self.host = None
            self._bump()

    # ── the port side ──
    def publish_request(self, request: Any, call: int) -> None:
        tools = [tool_record(t) for t in (getattr(request, "tools", None) or [])]
        msgs = [message_record(m) for m in (getattr(request, "messages", None) or [])]
        with self.lock:
            self.calls = call
            self.request = {"call": call, "system_prompt": str(getattr(request, "system_prompt", "") or ""),
                            "tools": tools, "n_messages": len(msgs),
                            "state": _jsonable(getattr(request, "state", None) or {})}
            self._messages = msgs
            self.phase = "awaiting_action"
            self._bump()

    def wait_action(self) -> Action | None:
        """Block the episode thread until the person replies; ``None`` = cancelled."""
        while True:
            with self.lock:
                if self.cancelled:
                    return None
            try:
                action = self.actions.get(timeout=0.5)
            except queue.Empty:
                continue
            with self.lock:
                self.phase = "running"
                self._bump()
            return action

    # ── the browser side ──
    def submit(self, action: Action) -> None:
        with self.lock:
            if self.phase != "awaiting_action":
                raise RuntimeError(f"not waiting for an action (phase={self.phase})")
            self.phase = "running"          # taken; the port flips it again when the reply is consumed
            self.events.append({"kind": "human_action", "action": action.kind, "name": action.name,
                                "args": action.args, "text": action.text, "t": time.time(),
                                "i": len(self.events)})
            self._bump()
        self.actions.put(action)

    def cancel(self) -> None:
        with self.lock:
            self.cancelled = True
            self._bump()
        self.actions.put(None)

    def push_event(self, rec: dict) -> None:
        """``run_llm_episode``'s ``on_event``: the loop's tool_start / tool_end / message …"""
        with self.lock:
            r = dict(rec)
            r["i"] = len(self.events)
            self.events.append(r)
            self._bump()

    def note(self, text: str, **extra: Any) -> None:
        with self.lock:
            self.events.append({"kind": "note", "text": text, "t": time.time(), "i": len(self.events), **extra})
            self._bump()

    # ── reading ──
    def budget(self) -> dict:
        sim_used = cmds_used = None
        clock_paused = None
        host = self.host
        if host is not None and self.sim0 is not None:
            sim, cmds = _budget_probe(host)
            sim_used = (sim - self.sim0) if sim is not None else None
            cmds_used = (cmds - self.cmds0) if (cmds is not None and self.cmds0 is not None) else None
            clock = getattr(getattr(host, "world", None), "clock", None)
            clock_paused = bool(getattr(clock, "is_paused", False)) if clock is not None else None
        if self.result is not None:
            sim_used = self.result.get("sim_consumed_s", sim_used)
            sr = self.result.get("skill_result") or {}
            cmds_used = sr.get("wire_cmds", cmds_used)
        return {"sim_used_s": sim_used, "sim_max_s": self.max_sim_s, "cmds_used": cmds_used,
                "cmds_max": self.max_cmds, "clock_paused": clock_paused,
                "wall_s": ((self.finished_wall or time.time()) - self.started_wall) if self.started_wall else None}

    def snapshot(self, since: int = 0) -> dict:
        with self.lock:
            req = None
            if self.request is not None:
                req = {k: v for k, v in self.request.items() if k != "system_prompt"}
                req["system_prompt_chars"] = len(self.request.get("system_prompt", ""))
            return {"version": self.version, "phase": self.phase, "scenario": self.scenario,
                    "seed": self.seed, "time_scale": self.time_scale, "policy": self.policy,
                    "out": self.out, "run_dir": self.run_dir, "session_dir": self.session_dir,
                    "calls": self.calls, "cancelled": self.cancelled,
                    "request": req, "events": self.events[since:], "n_events": len(self.events),
                    "budget": self.budget(), "result": self.result, "error": self.error}

    def transcript(self) -> dict:
        with self.lock:
            return {"call": (self.request or {}).get("call"),
                    "system_prompt": (self.request or {}).get("system_prompt", ""),
                    "messages": list(self._messages)}

    def wait_change(self, version: int, timeout_s: float = 25.0) -> None:
        with self.cond:
            self.cond.wait_for(lambda: self.version != version, timeout=timeout_s)


class HumanPort:
    """``ChatModelPort`` whose replies come from :class:`HumanSession`."""

    def __init__(self, session: HumanSession):
        self.session = session
        self.calls = 0

    def invoke(self, request: Any):
        from langchain_core.messages import AIMessage
        from mast.agentruntime.model import ModelResponse

        self.calls += 1
        self.session.publish_request(request, self.calls)
        action = self.session.wait_action()
        if action is None:
            return ModelResponse(message=AIMessage(content=ABORT_MARK), text=ABORT_MARK)
        if action.kind != "tool":
            text = str(action.text or "")
            return ModelResponse(message=AIMessage(content=text), text=text)
        call = {"name": action.name, "args": dict(action.args or {}),
                "id": action.id or f"human-{self.calls}", "type": "tool_call"}
        note = strip_markers(action.text)
        return ModelResponse(message=AIMessage(content=note, tool_calls=[call]),
                             tool_calls=[call], text=note)

    def stream(self, request: Any):
        resp = self.invoke(request)
        if resp.text:
            yield resp.text
        return resp
