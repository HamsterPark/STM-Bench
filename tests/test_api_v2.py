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

from stmbench.cli import main as cli_main
from stmbench.console import Console, make_server
from stmbench.console import server as console_server
from stmbench.instrument import ACTIONS, API_VERSION, ActionError, Instrument, VersionError, replay, verify
from stmbench.instrument.core import replay_records
from stmbench.instrument.export import ExportWriter, read_export
from stmbench.instrument.grade import grade_export
from stmbench.instrument.mcp_server import EpisodeServer
from stmbench.trackB.rules import RULES_VERSION

ROOT = Path(__file__).resolve().parent.parent
P4 = "P4_atom_positioning_cu111"
#: the parameters version 2 requires, with the bounds they keep
REQUIRED = {"pulse": {"width_ms": (0.1, 5000.0)}, "poke": {"depth_nm": (0.0, 20.0), "dwell_s": (0.0, 10.0)}}
#: the only source files that still name ``report_tip_state``: they read the ledgers of earlier versions
OLD_LEDGER_READERS = {"stmbench/replay/lay.py", "stmbench/replay/timeline.py"}


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


def _fetch(req: urllib.request.Request) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _get(base: str, path: str) -> tuple[int, dict]:
    return _fetch(urllib.request.Request(base + path))


def _post(base: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    return _fetch(urllib.request.Request(base + path, data=json.dumps(body or {}).encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST"))


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


# ── no tip diagnosis ────────────────────────────────────────────────────────
def test_no_agent_facing_surface_offers_report_tip_state(tmp_path):
    inst = Instrument(P4, 0, out_dir=tmp_path / "run")
    assert not hasattr(inst, "report_tip_state") and not hasattr(inst, "diagnosis")
    assert not any("tip_state" in name for name in ACTIONS)

    # the MCP tool table, and a call to the tool that used to be there
    srv = EpisodeServer(inst)
    listed = srv.handle({"jsonrpc": "2.0", "id": 0, "method": "tools/list"})["result"]["tools"]
    assert "report_tip_state" not in {t["name"] for t in listed}
    gone = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": "report_tip_state", "arguments": {"tip_state": "sharp"}}})
    assert gone["error"]["code"] == -32602 and "unknown tool" in gone["error"]["message"]

    # what an agent reads: the tools' own text, the task with its rule card (both languages) and the manual
    for text in (json.dumps(listed), inst.task("zh"), inst.task("en"), inst.manual(), _mcp(srv, "get_task")["content"][0]["text"]):
        assert "tip_state" not in text

    # the web console: the page, its endpoints, its object
    console = Console(tmp_path / "page", P4, 0)
    assert not hasattr(console, "tip_state")
    page = console_server.PAGE.read_text(encoding="utf-8")
    assert "tip_state" not in page and "tipdiag" not in page and "诊断" not in page
    with _serving(console) as base:
        assert _post(base, "/api/tip_state", {"tip_state": "sharp"})[0] == 404
        for path in ("/api/task", "/api/actions"):
            code, body = _get(base, path)
            assert code == 200 and "tip_state" not in json.dumps(body, ensure_ascii=False), path
        _post(base, "/api/finish")


def test_only_the_readers_of_older_ledgers_still_name_report_tip_state():
    named = {p.relative_to(ROOT).as_posix() for p in (ROOT / "stmbench").rglob("*")
             if p.suffix in {".py", ".html", ".js", ".json", ".yaml", ".md"} and p.is_file()
             and "report_tip_state" in p.read_text(encoding="utf-8")}
    assert named <= OLD_LEDGER_READERS, sorted(named - OLD_LEDGER_READERS)


def test_a_ledger_holds_no_diagnosis_and_replay_takes_none(tmp_path):
    run = tmp_path / "run"
    inst = Instrument(P4, 0, out_dir=run)
    inst.act("wait", seconds=5.0)
    inst.status()
    inst.finish()
    ep = json.loads((run / "episode.json").read_text(encoding="utf-8"))
    kinds = [json.loads(line)["kind"] for line in (run / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "diagnosis" not in ep and kinds == ["act", "observe", "finish"]
    # a record of that kind (an older instrument's) is not a call this one replays
    rec = {"i": 0, "kind": "diagnosis", "args": {"tip_state": "sharp", "junction": None, "note": ""}, "t_s": 0.0}
    replayed, mismatches = replay_records(P4, 0, [rec])
    assert mismatches == [{"i": 0, "name": "diagnosis", "error": "unknown record kind"}]
    assert replayed.n_actions == 0 and not hasattr(replayed, "diagnosis")


def test_the_readers_of_older_ledgers_still_read_a_tip_diagnosis(tmp_path):
    """A run directory of API version 1 holds ``diagnosis`` records and a ``diagnosis`` key. The
    replay for an audience and the report tables read it as before: the step is retold, and a row
    without the key is unknown, not a wrong answer."""
    from stmbench.replay.timeline import Ledger, build_timeline
    from stmbench.report.stats import diag_accuracy

    run = tmp_path / "run"
    inst = Instrument(P4, 0, out_dir=run)
    inst.act("wait", seconds=5.0)
    inst.finish()
    old = {"i": 2, "kind": "diagnosis", "args": {"tip_state": "sharp", "junction": None, "note": ""}, "t_s": 5.0}
    lines = (run / "actions.jsonl").read_text(encoding="utf-8").splitlines()
    lines.insert(1, json.dumps(old))                                      # before `finish`
    (run / "actions.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    ep = json.loads((run / "episode.json").read_text(encoding="utf-8"))
    ep.update(api_version=1, diagnosis=[old["args"]])
    (run / "episode.json").write_text(json.dumps(ep), encoding="utf-8")

    tl = build_timeline(Ledger.load(run))
    assert [s["name"] for s in tl["steps"]] == ["wait", "report_tip_state", "finish"]
    assert "sharp" in tl["steps"][1]["caption"]
    assert diag_accuracy([{"success": True}, {"diag_correct": None}]) == \
        {"diag_n": 0, "diag_correct_n": 0, "diag_abstain": 0, "diag_acc": None}


# ── the version, and the refusal of another one ─────────────────────────────
def _wait_run(tmp_path: Path) -> Path:
    """A finished run directory of this code: one wait."""
    run = tmp_path / "run"
    inst = Instrument(P4, 0, out_dir=run)
    inst.act("wait", seconds=5.0)
    inst.finish()
    return run


def _wait_export(tmp_path: Path) -> Path:
    """An export of this code with one wait in it (the episode never finishes: graded as it stands)."""
    inst = Instrument(P4, 0)
    exp = ExportWriter(inst, tmp_path / "export")
    inst.act("wait", seconds=5.0)
    exp.sync()
    return tmp_path / "export"


def _edit_json(path: Path, **keys) -> None:
    """Set keys of a JSON file; a value of ``...`` removes the key."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    for k, v in keys.items():
        if v is ...:
            doc.pop(k, None)
        else:
            doc[k] = v
    path.write_text(json.dumps(doc), encoding="utf-8")


def test_the_api_version_is_2_wherever_it_is_stated(tmp_path):
    assert API_VERSION == 2
    inst = Instrument(P4, 0, out_dir=tmp_path / "run")
    exp = ExportWriter(inst, tmp_path / "export")
    assert Instrument.manual().startswith("Instrument API v2: ")
    assert inst.task("en") and "Instrument API v2: " in _mcp(EpisodeServer(inst), "get_task")["content"][0]["text"]
    assert inst.info()["api_version"] == 2
    init = EpisodeServer(inst).handle({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                                      "params": {"protocolVersion": "2025-06-18"}})
    assert init["result"]["serverInfo"]["version"] == "api-2"
    inst.finish()
    exp.sync()
    ep = json.loads((tmp_path / "run" / "episode.json").read_text(encoding="utf-8"))
    assert ep["api_version"] == 2 and ep["rules_version"] == RULES_VERSION
    session = read_export(tmp_path / "export")["session"]
    assert session["api_version"] == 2 and session["rules_version"] == RULES_VERSION


def test_a_ledger_of_this_version_re_checks(tmp_path):
    run = _wait_run(tmp_path)
    assert verify(run)["ok"] and replay(run).n_actions == 1


@pytest.mark.parametrize("tag,said", [(1, "is of api_version 1"), (3, "is of api_version 3"),
                                      ("2", 'is of api_version "2"'), (True, "is of api_version true"),
                                      (None, "carries no api_version"), (..., "carries no api_version")])
def test_replay_and_verify_refuse_a_ledger_of_another_api_version(tmp_path, tag, said):
    run = _wait_run(tmp_path)
    _edit_json(run / "episode.json", api_version=tag)
    for refuse in (replay, verify):
        with pytest.raises(VersionError) as exc:
            refuse(run)
        msg = str(exc.value)
        assert isinstance(exc.value, ValueError) and msg.startswith("the ledger in ") and said in msg
        assert "this code is api_version 2" in msg and "the code of its own API version" in msg
        assert "needs the code that wrote it" in msg


def test_a_version_1_ledger_is_refused_before_anything_is_replayed(tmp_path):
    """Its calls are not this version's: replayed anyway, a v1 poke without a dwell would only
    be a refusal deep in the replay, a v1 diagnosis an unknown record."""
    run = _wait_run(tmp_path)
    _edit_json(run / "episode.json", api_version=1)
    with (run / "actions.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"i": 3, "kind": "diagnosis", "args": {"tip_state": "sharp"}, "t_s": 5.0}) + "\n")
        fh.write(json.dumps({"i": 4, "kind": "act", "name": "poke", "params": {}, "result": {}}) + "\n")
    with pytest.raises(VersionError, match="api_version 1"):
        replay(run)


def test_the_verify_command_exits_2_with_the_reason_for_a_ledger_of_another_version(tmp_path, capsys):
    run = _wait_run(tmp_path)
    assert cli_main(["verify", str(run)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    _edit_json(run / "episode.json", api_version=1)
    assert cli_main(["verify", str(run)]) == 2
    cap = capsys.readouterr()
    assert cap.out == "" and cap.err.startswith("verify: the ledger in ")
    assert "api_version 1" in cap.err and "the code of its own API version" in cap.err


def test_grade_refuses_an_export_of_version_1_in_the_same_words(tmp_path, capsys):
    path = _wait_export(tmp_path)
    assert grade_export(path, P4, 0)["replay"]["consistent"] is True            # this version grades
    _edit_json(path / "session.json", api_version=1, rules_version=1)           # what a version-1 export says
    with pytest.raises(VersionError) as exc:
        grade_export(path, P4, 0)
    msg = str(exc.value)
    assert msg.startswith("the export is of api_version 1; this code is api_version 2.")
    assert "the code of its own API version" in msg and "needs the code that wrote it" in msg
    # the same words as the ledger's refusal after the part that names what was refused
    run = _wait_run(tmp_path)
    _edit_json(run / "episode.json", api_version=1)
    with pytest.raises(VersionError) as ledger:
        replay(run)
    assert str(ledger.value).split("; this code", 1)[1] == msg.split("; this code", 1)[1]
    # and the command, which exits 2 and says it on stderr
    assert cli_main(["grade", str(path), "--scenario", P4, "--seed", "0"]) == 2
    cap = capsys.readouterr()
    assert cap.out == "" and cap.err.startswith("grade: the export is of api_version 1")
    assert "the code of its own API version" in cap.err


def test_grade_names_the_api_version_before_the_rules_version(tmp_path):
    """A version-1 export is of both: the first thing it is told is which code can replay it."""
    path = _wait_export(tmp_path)
    _edit_json(path / "session.json", api_version=1, rules_version=RULES_VERSION - 1)
    with pytest.raises(ValueError, match="api_version 1"):
        grade_export(path, P4, 0)
    _edit_json(path / "session.json", api_version=API_VERSION)                  # only the rules differ
    with pytest.raises(ValueError, match="rules_version") as exc:
        grade_export(path, P4, 0)
    assert not isinstance(exc.value, VersionError)


def test_text_out_of_an_export_is_clipped_before_the_refusal_repeats_it(tmp_path):
    path = _wait_export(tmp_path)
    _edit_json(path / "session.json", api_version="A" * 5000 + "\n\x1b[31m")
    with pytest.raises(VersionError) as exc:
        grade_export(path, P4, 0)
    assert len(str(exc.value)) < 400 and "A" * 100 not in str(exc.value)
    assert "\n" not in str(exc.value) and "\x1b" not in str(exc.value)
