"""The desktop console's one door to the instrument (stmbench.gui.session; no Qt): the same calls
through it and through the MCP server give the same episode; the export holds what the ledger holds,
refusals included; the sealed mode's seed appears nowhere an agent could read; and nothing in
``stmbench/gui`` can reach the truth, the verdict or a network."""
from __future__ import annotations

import ast
import io
import json
import os
import re
from pathlib import Path

import pytest

import stmbench.gui
from stmbench.gui import GuiSession, SpecError, make_private_dir, read_spec
from stmbench.gui import fmt, jobs
from stmbench.instrument import ActionError, Instrument
from stmbench.instrument.export import read_export
from stmbench.instrument.mcp_server import EpisodeServer

P4 = "P4_atom_positioning_cu111"
#: a seed whose digits cannot turn up in a record by chance
SEED = 913275
SCAN = {"center_x_nm": 0.0, "center_y_nm": 0.0, "size_nm": 8.0, "pixels": 32}
MOVES = [("scan", SCAN), ("set_bias", {"bias_v": 0.005}), ("set_setpoint", {"current_pa": 100000.0}),
         ("move_tip", {"x_nm": 1.0, "y_nm": 0.0, "speed_nm_s": 1.0}), ("set_setpoint", {"current_pa": 200.0}),
         ("scan", SCAN)]


def _session(tmp_path, seed=0, name="run") -> GuiSession:
    return GuiSession(P4, seed, run_dir=tmp_path / name, export_dir=tmp_path / f"{name}_export")


def test_the_same_calls_through_the_session_and_the_mcp_server_give_the_same_episode(tmp_path):
    sess = _session(tmp_path, name="gui")
    frames = [sess.act(n, a).get("frame") for n, a in MOVES]
    report = {"claim_id": "target_site", "value": 1.0, "x_nm": 1.0, "y_nm": 0.0, "frame": frames[-1],
              "evidence": [frames[-1]]}
    sess.status()
    sess.data()
    sess.frame(frames[-1])
    sess.report({**report, "evidence": frames[-1]})                 # a comma-separated string is cleaned
    assert sess.finish() is None and sess.finished
    mcp = EpisodeServer(Instrument(P4, 0, out_dir=tmp_path / "mcp"))
    for i, (n, a) in enumerate(MOVES):
        mcp.handle({"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": n, "arguments": a}})
    mcp.handle({"jsonrpc": "2.0", "id": 90, "method": "tools/call",
                "params": {"name": "report_result", "arguments": report}})
    mcp.handle({"jsonrpc": "2.0", "id": 91, "method": "tools/call", "params": {"name": "finish", "arguments": {}}})
    gui, agent = (json.loads((tmp_path / d / "episode.json").read_text(encoding="utf-8")) for d in ("gui", "mcp"))
    assert gui["digest"] == agent["digest"] and gui["verdict"] == agent["verdict"]
    assert gui["instrument_time_s"] == agent["instrument_time_s"]


def test_the_export_equals_the_ledger_and_holds_a_refusal(tmp_path):
    sess = _session(tmp_path)
    frame = sess.act("scan", SCAN)["frame"]
    with pytest.raises(ActionError, match="center_y_nm is required"):
        sess.act("scan", {"center_x_nm": 0.0})
    # the refusal is on record at once, not at the end
    assert [r["kind"] for r in read_export(tmp_path / "run_export")["records"]] == ["act", "refused"]
    sess.status()
    sess.act("sts", {"v_start": -0.1, "v_end": 0.1, "points": 16})
    sess.data()
    sess.frame(frame, "current", "backward")
    sess.spectrum(sess.data()["spectra"][0]["spectrum"])
    sess.report({"claim_id": "target_site", "value": 1.0, "x_nm": 1.0, "y_nm": 0.0, "frame": frame, "evidence": [frame]})
    sess.finish()
    got = read_export(tmp_path / "run_export")
    ledger = [json.loads(line) for line in (tmp_path / "run" / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert got["records"] == ledger and not got["torn"] and got["session"]["finished"]
    assert {"act", "refused", "observe", "report", "finish"} <= {r["kind"] for r in ledger}


def test_a_new_run_directory_and_a_new_export_directory_are_required(tmp_path):
    GuiSession(P4, 0, run_dir=tmp_path / "run", export_dir=tmp_path / "run" / "export")    # the dev mode's layout
    with pytest.raises(FileExistsError):
        GuiSession(P4, 0, run_dir=tmp_path / "run", export_dir=tmp_path / "other_export")
    with pytest.raises(FileExistsError):                                 # an export directory holds one episode
        GuiSession(P4, 0, run_dir=tmp_path / "again", export_dir=tmp_path / "run" / "export")


def test_the_report_form_is_cleaned_as_the_web_console_cleans_it(tmp_path):
    sess = _session(tmp_path)
    frame = sess.act("scan", SCAN)["frame"]
    with pytest.raises(ActionError, match="value is required"):
        sess.report({"claim_id": "target_site", "value": "", "x_nm": "1"})
    with pytest.raises(ActionError, match="unknown claim ''"):
        sess.report({"value": "1.0"})
    with pytest.raises(ValueError, match="could not convert"):
        sess.report({"claim_id": "target_site", "value": "1.0", "x_nm": "abc"})
    msg = sess.report({"claim_id": "target_site", "value": "1.5", "x_nm": "1.5", "y_nm": " 0.25 ", "frame": frame,
                       "evidence": f"{frame}，{frame} , ", "unit": "nm", "sigma": "", "note": ""})
    assert msg.startswith("recorded target_site = 1.5")
    rec = [r for r in read_export(tmp_path / "run_export")["records"] if r["kind"] == "report"][-1]["args"]
    assert rec["evidence"] == [frame, frame] and rec["x_nm"] == 1.5 and rec["y_nm"] == 0.25
    assert rec["sigma"] is None and rec["note"] == ""


def test_the_claims_are_the_ones_to_report_as_the_rule_card_states_them(tmp_path):
    sess = _session(tmp_path)
    claims = sess.claims()
    assert [c["id"] for c in claims] == ["target_site"]               # the constraint is not reported
    c = claims[0]
    assert c["position"] and c["unit"] == "nm" and c["kind"] == "position" and c["say"]
    assert "target_site" in sess.task_text() and "move_tip" in sess.manual()


def test_a_sealed_session_leaves_its_seed_nowhere_an_agent_could_read(tmp_path):
    private = make_private_dir(tmp_path / "private")
    sess = GuiSession(P4, SEED, run_dir=private, export_dir=tmp_path / "export")
    texts = [sess.task_text(), sess.manual(), json.dumps(sess.claims(), ensure_ascii=False)]
    frame = sess.act("scan", SCAN)["frame"]
    texts.append(json.dumps(sess.status()))
    texts.append(json.dumps(sess.data()))
    sess.act("sts", {"v_start": -0.1, "v_end": 0.1, "points": 16})
    with pytest.raises(ActionError):
        sess.act("move_tip", {"x_nm": 1e6, "y_nm": 0.0})
    sess.report({"claim_id": "target_site", "value": 1.0, "x_nm": 1.0, "y_nm": 0.0, "frame": frame, "evidence": [frame]})
    sess.finish()
    assert all(str(SEED) not in t for t in texts)
    files = sorted(p for p in (tmp_path / "export").rglob("*") if p.is_file())
    assert {p.name for p in files} >= {"session.json", "actions.jsonl", "digests.jsonl", "report.json"}
    for p in files:
        assert str(SEED) not in p.read_text(encoding="utf-8"), p.name
    # the private directory does hold it: that is what it is private for
    assert str(SEED) in (private / "episode.json").read_text(encoding="utf-8")


def test_the_session_exposes_nothing_of_the_episodes_truth(tmp_path):
    sess = _session(tmp_path, seed=SEED)
    public = {n for n in dir(sess) if not n.startswith("_")}
    assert public == {"act", "claims", "data", "finish", "finished", "frame", "info", "lang", "manual", "report",
                      "scenario_id", "spectrum", "status", "task_text"}
    assert str(SEED) not in json.dumps({n: repr(getattr(sess, n)) for n in public if not callable(getattr(sess, n))})


def test_a_private_directory_is_new_and_closed_to_others(tmp_path):
    p = make_private_dir(tmp_path / "a" / "private")
    assert p.is_dir()
    if os.name == "posix":
        assert p.stat().st_mode & 0o777 == 0o700
    (p / "x").write_text("x")
    with pytest.raises(FileExistsError):
        make_private_dir(p)


# ── the sealed mode's one line ──
@pytest.mark.parametrize("line", [f'{{"scenario": "{P4}", "seed": 7}}', f'  {{"seed": 0, "scenario": "{P4}"}} \n',
                                  '{"scenario": "P1_barth1990_au111", "seed": 2147483648}'])
def test_read_spec_accepts_one_json_line(line):
    scenario, seed = read_spec(io.StringIO(line + "\nanything after the first line is not read\n"))
    assert scenario in line and seed == json.loads(line)["seed"]


@pytest.mark.parametrize("line", ["", "\n", "not json", "[]", "7", "null", '{"scenario": "%s"}' % P4,
                                  '{"seed": 3}', '{"scenario": "%s", "seed": 913275, "out": "x"}' % P4,
                                  '{"scenario": "%s", "seed": 913275.0}' % P4,
                                  '{"scenario": "%s", "seed": "913275"}' % P4,
                                  '{"scenario": "%s", "seed": true}' % P4,
                                  '{"scenario": "%s", "seed": -913275}' % P4,
                                  '{"scenario": "nosuch", "seed": 913275}',
                                  '{"scenario": "../trackB/scenarios/%s", "seed": 913275}' % P4,
                                  '{"scenario": 4, "seed": 913275}'])
def test_read_spec_rejects_junk_without_repeating_it(line):
    with pytest.raises(SpecError) as exc:
        read_spec(io.StringIO(line))
    assert "913275" not in str(exc.value)


# ── a source scan: no network, no truth ──
_FORBIDDEN_MODULES = {"socket", "socketserver", "http", "xmlrpc", "ssl", "asyncio", "webbrowser", "selectors",
                      "urllib.request", "multiprocessing.connection", "PySide6.QtNetwork", "PySide6.QtWebSockets",
                      "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebChannel",
                      "PySide6.QtRemoteObjects", "PySide6.QtDBus"}
_FORBIDDEN_QT_CLASSES = re.compile(r"^(QLocal(Server|Socket)|QTcp(Server|Socket)|QUdpSocket|QAbstractSocket|"
                                   r"QSslSocket|QWebSocket(Server)?|QNetwork\w*|QDBus\w*)$")
#: what the window must never read off the instrument (``seed`` is a parameter name in the session, never an attribute)
_FORBIDDEN_NAMES = {"world", "truth", "truth_before", "truth_after", "verdict", "hidden"}
_FORBIDDEN_ATTRS = _FORBIDDEN_NAMES | {"seed"}


def _gui_sources() -> list[Path]:
    return sorted(p for p in Path(stmbench.gui.__file__).parent.rglob("*.py"))


def _module_hits(mod: str) -> bool:
    return any(mod == bad or mod.startswith(bad + ".") for bad in _FORBIDDEN_MODULES)


def test_nothing_in_the_gui_package_can_reach_a_network_or_the_episodes_truth():
    sources = _gui_sources()
    assert len(sources) >= 5
    problems = []
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        docstrings = {id(n.body[0]) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                                                             ast.AsyncFunctionDef))
                      and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        for node in ast.walk(tree):
            where = f"{path.name}:{getattr(node, 'lineno', 0)}"
            if isinstance(node, ast.Import):
                problems += [f"{where} imports {a.name}" for a in node.names if _module_hits(a.name)]
            elif isinstance(node, ast.ImportFrom):
                mod = ("." * node.level) + (node.module or "")
                if node.level == 0 and _module_hits(node.module or ""):
                    problems.append(f"{where} imports from {mod}")
                problems += [f"{where} imports {a.name}" for a in node.names if _FORBIDDEN_QT_CLASSES.match(a.name)]
            elif isinstance(node, ast.Attribute):
                if node.attr in _FORBIDDEN_ATTRS or _FORBIDDEN_QT_CLASSES.match(node.attr):
                    problems.append(f"{where} reads .{node.attr}")
            elif isinstance(node, ast.Name):
                if node.id in _FORBIDDEN_NAMES or _FORBIDDEN_QT_CLASSES.match(node.id):
                    problems.append(f"{where} uses {node.id}")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                if node.value in ("episode.json", "tip_timeline.json") or "episode.json" in node.value:
                    problems.append(f"{where} names {node.value!r}")
    assert not problems, "\n".join(problems)


def test_the_package_imports_without_qt_and_names_no_seed_in_its_exports():
    assert set(stmbench.gui.__all__) == {"GuiSession", "SpecError", "make_private_dir", "read_spec"}


# ── what the window writes ──
def test_readings_are_written_in_the_units_of_the_fields_that_set_them():
    st = {"bias_v": 0.05, "setpoint_pa": 200.0, "feedback": True, "current_pa": 198.31, "z_nm": -1.2345,
          "current_full_scale_na": 10.0, "tip_nm": [1.0, -2.5], "withdrawn": False, "lockin": False,
          "oscillating": None, "drift_comp": {"on": False, "v_nm_per_min": [0.0, 0.0], "vz_pm_per_min": 0.0}}
    texts = dict(fmt.reading_items(st))
    assert texts == {"bias": "0.050 V", "setpoint": "200 pA", "feedback": "ON", "current": "198.3 pA",
                     "z": "-1.234 nm", "range": "10 nA", "tip": "(1.000, -2.500) nm", "withdrawn": "no",
                     "lockin": "OFF", "oscillation": "n/a", "drift": "OFF"}
    # bias is always V (the millivolt kept: the field takes 0.050), setpoint and current always pA, never nA
    assert [fmt.volts(v) for v in (1.2, 0.05, -1.25, 0.0125, 0.0, -1e-9, 10.0, -0.001)] == [
        "1.200 V", "0.050 V", "-1.250 V", "0.0125 V", "0.000 V", "0.000 V", "10.000 V", "-0.001 V"]
    assert [fmt.picoamps(v) for v in (1.5e5, 1500.0, 0.1, 198.31, 12345.6, 999.96, -50.0)] == [
        "150000 pA", "1500 pA", "0.1 pA", "198.3 pA", "12346 pA", "1000 pA", "-50 pA"]
    # a switch the rig does not have is n/a, a drift compensation that is on shows what it was given
    assert fmt.on_off(None) == "n/a" and fmt.on_off(True, "zh") == "开 ON"
    on = {"on": True, "v_nm_per_min": [1.5, -2.25], "vz_pm_per_min": 30.0}
    assert fmt.drift_comp_text(on) == "ON (1.5, -2.25) nm/min, 30 pm/min" and fmt.drift_comp_text(on, "zh").startswith("开 ON (")
    assert fmt.drift_comp_text({**on, "vz_pm_per_min": 0.0}) == "ON (1.5, -2.25) nm/min"
    assert fmt.drift_comp_text(None) == "n/a" and fmt.drift_comp_text({**on, "on": False}, "zh") == "关 OFF"
    assert fmt.hms(845.4) == "0:14:05" and fmt.hms(5400) == "1:30:00" and fmt.hms(None) == "0:00:00"
    assert "e-" not in " ".join(texts.values())


def test_the_switches_and_the_range_are_read_off_status_by_the_action_that_sets_them():
    st = {"feedback": True, "lockin": False, "oscillating": None, "current_full_scale_na": 10.0,
          "drift_comp": {"on": True, "v_nm_per_min": [1.0, 0.0], "vz_pm_per_min": 0.0}}
    assert fmt.instrument_state(st) == {"feedback": True, "lockin": False, "pll": None, "drift_comp": True,
                                        "set_current_range": 10.0}
    old_status = {"feedback": False, "lockin": True, "oscillating": True}               # no drift_comp, no range
    assert fmt.instrument_state(old_status) == {"feedback": False, "lockin": True, "pll": True, "drift_comp": None,
                                                "set_current_range": None}


def test_a_reply_is_summarised_in_the_instruments_words():
    scan = {"frame": "sim005.sxm", "duration_s": 300.2, "size_nm": 30.0, "instrument_time_s": 312.5, "budget_s": 5400.0}
    assert fmt.reply_summary("scan", scan) == "scan → sim005.sxm (300.2 s)"
    assert fmt.reply_summary("sts", {"spectrum": "sim006.dat", "points": 16, "duration_s": 2.5}) == "sts → sim006.dat (2.5 s)"
    move = {"tip_nm": [4.0, 0.0], "distance_nm": 4.0, "current_spikes": 0, "instrument_time_s": 9.0, "budget_s": None}
    assert fmt.reply_summary("move_tip", move) == "move_tip → tip_nm=(4, 0), distance_nm=4, current_spikes=0"
    assert fmt.reply_summary("withdraw", {"instrument_time_s": 1.0}) == "withdraw → done"
    assert json.loads(fmt.reply_full(move)) == move


def test_a_field_is_sent_as_typed_unless_it_is_a_number():
    assert fmt.parse_number("256") == 256 and isinstance(fmt.parse_number("256"), int)
    assert fmt.parse_number(" -0.5 ") == -0.5 and fmt.parse_number("1e-3") == 1e-3 and fmt.parse_number(".5") == 0.5
    for text in ("abc", "nan", "inf", "1_000", "１２", "1,5", "--1", "0x10", "1e", "."):
        assert fmt.parse_number(text) == text, text


def test_equal_length_numeric_lists_of_a_reply_are_found_whatever_the_action():
    trace = {"t_s": list(range(10)), "z_pm": [float(i) ** 2 for i in range(10)], "short": [1.0] * 4, "flags": [True] * 10,
             "frame": "x", "tip_nm": [1.0, 2.0]}
    assert set(fmt.numeric_lists(trace)) == {"t_s", "z_pm"}
    assert fmt.numeric_lists({"a": list(range(8)), "b": list(range(9))}) == {"b": [float(i) for i in range(9)]} \
        or len(fmt.numeric_lists({"a": list(range(8)), "b": list(range(9))})) == 1
    assert fmt.numeric_lists("scan done") == {} and fmt.numeric_lists({"tip_nm": [1.0, 2.0]}) == {}


# ── the jobs the worker runs ──
def test_an_action_job_reads_what_the_window_shows_next(tmp_path):
    sess = _session(tmp_path)
    out = jobs.run_act(sess, "scan", SCAN)
    assert out.ok and out.frame["frame"] == out.reply["frame"] and out.frame["channel"] == "z"
    assert out.status["actions"] == 1 and [f["frame"] for f in out.data["frames"]] == [out.reply["frame"]]
    out = jobs.run_act(sess, "sts", {"v_start": -0.1, "v_end": 0.1, "points": 16})
    assert out.spectrum["name"] == out.reply["spectrum"] and out.frame is None and len(out.data["spectra"]) == 1
    out = jobs.run_act(sess, "set_bias", {"bias_v": 0.02})
    assert out.ok and out.frame is None and out.spectrum is None and out.data is None and out.status["bias_v"] == 0.02
    refused = jobs.run_act(sess, "move_tip", {"x_nm": 1e6, "y_nm": 0.0})
    assert refused.refused and not refused.ok and refused.text == "move_tip: outside the piezo range"
    assert refused.status is None and refused.reply is None
    typed = jobs.run_act(sess, "set_bias", {"bias_v": "abc"})            # a field sent as typed is refused in the instrument's words
    assert typed.refused and typed.text == "set_bias: bias_v must be a number"
    assert jobs.run_view_frame(sess, "nosuch.sxm", "z", "forward").text == "no frame 'nosuch.sxm'"
    assert jobs.run_view_frame(sess, out.frame["frame"] if out.frame else sess.data()["frames"][0]["frame"],
                               "didv", "forward").refused
    done = jobs.run_finish(sess)
    assert done.finished and done.ok and sess.finished
    assert jobs.run_act(sess, "wait", {"seconds": 1}).text == "the episode is over"


# ── the command line ──
@pytest.mark.parametrize("argv", [
    ["gui"], ["gui", "--scenario", P4], ["gui", "--out", "x"],
    ["gui", "--sealed"], ["gui", "--sealed", "--private-dir", "a"], ["gui", "--sealed", "--export-dir", "b"],
    ["gui", "--sealed", "--private-dir", "a", "--export-dir", "b", "--seed", "7"],
    ["gui", "--sealed", "--private-dir", "a", "--export-dir", "b", "--scenario", P4],
    ["gui", "--sealed", "--private-dir", "a", "--export-dir", "b", "--out", "x"],
    ["gui", "--scenario", P4, "--out", "x", "--private-dir", "a"],
    ["gui", "--scenario", P4, "--out", "x", "--export-dir", "b"]])
def test_the_gui_command_refuses_a_bad_combination_of_options_before_it_loads_qt(argv, capsys):
    from stmbench import cli
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code not in (0, None) and "913275" not in str(exc.value)


def test_the_gui_command_is_listed_with_its_options(capsys):
    from stmbench import cli
    with pytest.raises(SystemExit) as exc:
        cli.main(["gui", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--scenario", "--seed", "--out", "--lang", "--ui-scale", "--sealed", "--private-dir", "--export-dir"):
        assert flag in out


def _without_qt(code: str):
    """Run ``code`` in a fresh interpreter in which importing PySide6 fails, as it does where the gui extra is not installed."""
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    return subprocess.run([sys.executable, "-c", "import sys; sys.modules['PySide6'] = None\n" + code], cwd=root,
                          capture_output=True, text=True, timeout=300)


def test_the_base_package_needs_no_qt_and_the_gui_command_says_which_extra_to_install():
    ok = _without_qt("import stmbench.gui, stmbench.cli; print('imported')")
    assert ok.returncode == 0 and ok.stdout.strip() == "imported", ok.stderr
    out = _without_qt("from stmbench import cli; cli.main(['gui', '--scenario', 'P4_atom_positioning_cu111', '--out', 'x'])")
    assert out.returncode != 0 and "stm-bench[gui]" in out.stderr and "Traceback" not in out.stderr, out.stderr


def test_the_task_follows_the_sessions_language_and_the_claims_carry_both_wordings(tmp_path):
    zh = GuiSession(P4, 0, run_dir=tmp_path / "zh", export_dir=tmp_path / "zh_export", lang="zh")
    en = GuiSession(P4, 0, run_dir=tmp_path / "en", export_dir=tmp_path / "en_export", lang="en")
    assert zh.task_text() == Instrument(P4, 0).task(lang="zh") and "【评分规则 v1】" in zh.task_text()
    assert en.task_text() == Instrument(P4, 0).task(lang="en") and "[Scoring rules v1]" in en.task_text()
    assert en.claims()[0]["say_en"].startswith("The final position of the atom you moved") and zh.claims()[0]["say"]
    assert GuiSession(P4, 0, run_dir=tmp_path / "x", export_dir=tmp_path / "x_export", lang="fr").lang == "zh"


def test_the_instruments_facts_are_an_observation_that_costs_no_time(tmp_path):
    sess = _session(tmp_path)
    t0 = sess.status()["instrument_time_s"]
    info = sess.info()
    assert info["piezo_xy_range_nm"][0] > 0 and info["bias_range_v"] == [-10.0, 10.0] and info["material"] == "Cu(111)"
    assert "qplus" not in info and sess.status()["instrument_time_s"] == t0
    recs = read_export(tmp_path / "run_export")["records"]
    assert [r["what"] for r in recs if r["kind"] == "observe"] == ["status", "info", "status"]
    qp = GuiSession("P5_sader_jarvis_qplus", 0, run_dir=tmp_path / "p5", export_dir=tmp_path / "p5_export").info()["qplus"]
    assert set(qp) == {"f0_hz", "k_n_per_m", "q"} and qp["k_n_per_m"] > 0


def test_the_facts_are_written_in_the_units_they_are_registered_in():
    info = {"piezo_xy_range_nm": [2669.0, 2600.0], "piezo_z_range_nm": 339.0, "bias_range_v": [-10.0, 10.0],
            "current_ranges_na": [10000.0, 1000.0, 0.1], "budget_s": 5400.0, "temperature_k": 4.36, "material": "Cu(111)",
            "qplus": {"f0_hz": 34182.312, "k_n_per_m": 1773.666, "q": 18309.9}}
    assert fmt.facts_items(info) == [
        ("fact_material", "Cu(111)"), ("fact_temperature", "4.36 K"), ("fact_piezo_xy", "2669 × 2600 nm"),
        ("fact_piezo_z", "339 nm"), ("fact_bias_range", "-10 … 10 V"), ("fact_current_ranges", "10000, 1000, 0.1 nA"),
        ("fact_budget", "1:30:00"), ("fact_qplus_f0", "34182.312 Hz"), ("fact_qplus_k", "1773.666 N/m"),
        ("fact_qplus_q", "18309.9")]
    assert fmt.facts_items({**info, "budget_s": None, "qplus": None})[-1] == ("fact_budget", "-")
