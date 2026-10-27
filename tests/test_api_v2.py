"""Instrument API version 2 (``stmbench.instrument.core.API_VERSION``): what an agent can call and
what a ledger holds, after two screen-only trials on P4 leaned on what the instrument gives away.

* ``pulse.width_ms``, ``poke.depth_nm`` and ``poke.dwell_s`` are required, with no default: the
  instrument does not choose how hard to condition the tip (one default poke mended a double tip
  at the first try, so the default was the answer);
* there is no ``report_tip_state``: no tool, endpoint, control or record, so nothing invites an
  agent to worry about its tip;
* a ledger or an export of another API version is refused by ``replay``, ``verify`` and ``grade``,
  saying that it needs the code of its own version."""
from __future__ import annotations

import ast
import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pytest

from stmbench.console import Console, make_server
from stmbench.instrument import ACTIONS, ActionError, Instrument
from stmbench.instrument.mcp_server import EpisodeServer

ROOT = Path(__file__).resolve().parent.parent
P4 = "P4_atom_positioning_cu111"
#: the parameters version 2 requires, with the bounds they keep
REQUIRED = {"pulse": {"width_ms": (0.1, 5000.0)}, "poke": {"depth_nm": (0.0, 20.0), "dwell_s": (0.0, 10.0)}}


def _mcp(srv: EpisodeServer, name: str, args: dict | None = None) -> dict:
    resp = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": name, "arguments": args or {}}})
    assert "result" in resp, resp
    return resp["result"]


@contextmanager
def _serving(console: Console):
    srv = make_server(console, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def _post(base: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(base + path, data=json.dumps(body or {}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


# ── no default to lean on ───────────────────────────────────────────────────
def test_the_conditioning_parameters_are_required_with_no_default_and_keep_their_bounds():
    manual = Instrument.manual().splitlines()
    for action, params in REQUIRED.items():
        schema = ACTIONS[action].schema()
        for name, (lo, hi) in params.items():
            p = next(q for q in ACTIONS[action].params if q.name == name)
            assert p.required and p.default is None, (action, name)
            assert (p.lo, p.hi) == (lo, hi), (action, name)
            prop = schema["properties"][name]
            assert name in schema["required"] and "default" not in prop, (action, name)
            assert (prop["minimum"], prop["maximum"]) == (lo, hi)
            line = next(x for x in manual if x.strip().startswith(f"{name} ("))
            assert "[required]" in line and "default" not in line, line
    # what stays optional stays so: the pulse holds Z unless told otherwise, the poke keeps the bias
    pulse = {p.name: p for p in ACTIONS["pulse"].params}
    poke = {p.name: p for p in ACTIONS["poke"].params}
    assert pulse["bias_v"].required and pulse["z_hold"].default is True and not pulse["z_hold"].required
    assert not poke["bias_v"].required and poke["bias_v"].default is None


@pytest.mark.parametrize("action,args,message", [
    ("pulse", {"bias_v": 3.0}, "pulse: width_ms is required"),
    ("pulse", {"bias_v": 3.0, "width_ms": None}, "pulse: width_ms is required"),
    ("poke", {}, "poke: depth_nm is required"),
    ("poke", {"bias_v": 0.1, "dwell_s": 0.5}, "poke: depth_nm is required"),
    ("poke", {"depth_nm": 0.4}, "poke: dwell_s is required"),
    ("poke", {"depth_nm": 0.4, "bias_v": 0.1}, "poke: dwell_s is required")])
def test_a_conditioning_call_without_its_parameters_is_refused_in_the_instruments_words(action, args, message):
    inst = Instrument(P4, 0)
    n_events = len(inst.world.events)
    with pytest.raises(ActionError) as exc:
        inst.act(action, **args)
    assert str(exc.value) == message
    assert inst.used_s == 0.0 and inst.n_actions == 0 and len(inst.world.events) == n_events   # the tip was not touched
    rec = inst.log[-1]
    assert rec["kind"] == "refused" and rec["name"] == action and rec["error"] == message and rec["args"] == args


def test_given_their_parameters_the_conditioning_actions_run_and_zero_is_a_value():
    inst = Instrument(P4, 0)
    r = inst.act("pulse", bias_v=3.0, width_ms=50.0)
    assert (r["bias_v"], r["width_ms"]) == (3.0, 50.0)
    r = inst.act("poke", depth_nm=0.4, dwell_s=0.5)
    assert (r["depth_nm"], r["dwell_s"]) == (0.4, 0.5)
    r = inst.act("poke", depth_nm=0.0, dwell_s=0.0)               # zero is given, not missing
    assert (r["depth_nm"], r["dwell_s"]) == (0.0, 0.0)
    assert inst.n_actions == 3 and [e["kind"] for e in inst.world.events if e["kind"] in ("pulse", "poke")] == \
        ["pulse", "poke", "poke"]


def test_an_mcp_agent_and_the_web_console_are_refused_the_same_way(tmp_path):
    srv = EpisodeServer(Instrument(P4, 0))
    tools = {t["name"]: t for t in srv.handle({"jsonrpc": "2.0", "id": 0, "method": "tools/list"})["result"]["tools"]}
    assert tools["pulse"]["inputSchema"]["required"] == ["bias_v", "width_ms"]
    assert tools["poke"]["inputSchema"]["required"] == ["depth_nm", "dwell_s"]
    res = _mcp(srv, "pulse", {"bias_v": 3.0})
    assert res["isError"] and res["content"][0]["text"] == "refused: pulse: width_ms is required"
    res = _mcp(srv, "poke", {"depth_nm": 0.4})
    assert res["isError"] and res["content"][0]["text"] == "refused: poke: dwell_s is required"
    assert srv.inst.used_s == 0.0 and srv.inst.n_actions == 0

    with _serving(Console(tmp_path / "run", P4, 0)) as base:
        assert _post(base, "/api/act", {"name": "pulse", "args": {"bias_v": 3.0}}) == \
            (400, {"error": "pulse: width_ms is required"})
        assert _post(base, "/api/act", {"name": "poke", "args": {}}) == (400, {"error": "poke: depth_nm is required"})
        code, out = _post(base, "/api/act", {"name": "pulse", "args": {"bias_v": 3.0, "width_ms": 50.0}})
        assert code == 200 and out["width_ms"] == 50.0
        _post(base, "/api/finish")


def _callers() -> list[tuple[Path, int, str, set[str], bool]]:
    """Every ``<x>.act("pulse" | "poke", ...)`` in the repository's Python: file, line, action, the
    keywords it passes and whether it also passes ``**kwargs``."""
    found = []
    for path in sorted([*(ROOT / "stmbench").rglob("*.py"), *(ROOT / "tests").rglob("*.py")]):
        if path.name == Path(__file__).name:
            continue                                           # this file calls without them on purpose
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "act"
                    and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value in REQUIRED):
                found.append((path, node.lineno, node.args[0].value, {k.arg for k in node.keywords if k.arg},
                              any(k.arg is None for k in node.keywords)))
    return found


def test_no_caller_in_the_repository_leans_on_a_default_that_is_gone():
    callers = _callers()
    assert len(callers) >= 4                                   # the baselines and the reference solvers at least
    for path, line, action, given, splat in callers:
        missing = set(REQUIRED[action]) - given
        assert splat or not missing, f"{path.relative_to(ROOT)}:{line} act({action!r}) does not pass {sorted(missing)}"
