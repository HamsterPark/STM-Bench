"""Grading an export (stmbench.instrument.grade). The host holds the scenario and the seed the
machine of the episode never had: it replays the recorded calls on its own instrument and judges
the replay, and the replay has to be the recorded episode (every reply, instrument time and
hash), or the episode scores 0. Also the pieces around it: ``replay_records``, the ``grade``
command, the OSWorld-Science adapter and the files of the grader image."""
from __future__ import annotations

import ast
import importlib.util
import json
import math
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from stmbench.cli import main as cli_main
from stmbench.instrument import API_VERSION, ActionError, Instrument, replay
from stmbench.instrument.core import replay_records
from stmbench.instrument.export import EXPORT_VERSION, ExportWriter, read_export
from stmbench.instrument.grade import _digest_mismatches, grade_export
from stmbench.trackB.rules import RULES_VERSION

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy" / "osworld"
P4 = "P4_atom_positioning_cu111"
P5 = "P5_sader_jarvis_qplus"
#: a seed whose digits cannot turn up in a result by chance
SEED = 913275
#: what a result must never carry: the world's truth, the verdict's per-claim detail, the seed
SECRET_KEYS = {"seed", "truth", "truth_before", "truth_after", "verdict", "details", "claims", "thresholds"}


def _keys(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k
            yield from _keys(v)
    elif isinstance(o, list):
        for v in o:
            yield from _keys(v)


def _go(inst: Instrument, sx_m: float, sy_m: float) -> None:
    """Move the tip to a place on the sample (m), through the drift."""
    ox, oy, _ = inst.world.drift_offset(inst.world.clock.sim())
    inst.act("move_tip", x_nm=(sx_m - ox) * 1e9, y_nm=(sy_m - oy) * 1e9)


def _p5_episode(path: Path, *, reports: int = 3, finish: bool = True) -> Instrument:
    """A P5 episode that reproduces the paper (the truth only navigates, as in test_instrument),
    with a frame, a refused call and observations in its log, and its export in ``path``."""
    inst = Instrument(P5, 0)
    exp = ExportWriter(inst, path)
    w = inst.world
    reg = w.surface.site.adatoms
    atom = min(reg.atoms.values(), key=lambda a: math.hypot(*reg.site_xy(a.i, a.j, a.sub)))
    ax, ay = reg.site_xy(atom.i, atom.j, atom.sub)
    inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=5.0, pixels=32, line_time_s=0.05)
    with pytest.raises(ActionError):                     # refused: in the log, never replayed
        inst.act("scan", center_x_nm=0.0)
    inst.act("pll", on=True, amplitude_pm=50.0)
    exp.sync()
    _go(inst, ax, ay)
    on_atom = inst.act("zspec", sweep_nm=0.5, z_offset_nm=0.3, points=256)["spectrum"]
    inst.status()                                        # observations: in the log, never replayed
    inst.spectrum(on_atom)
    exp.sync()
    _go(inst, ax + 6e-9, ay)
    clean = inst.act("zspec", sweep_nm=0.5, z_offset_nm=0.3, points=256)["spectrum"]
    truth = w.truth()
    for c in inst.scenario.claims[:reports]:
        inst.report(c["id"], float(truth["force"][c["truth"].split(".", 1)[1]]), evidence=[on_atom, clean])
    exp.sync()
    if finish:
        inst.finish()
        exp.sync()
    return inst


def _wait_export(path: Path, scenario: str = P4, seed: int = 0) -> Instrument:
    """The smallest export there is: one wait."""
    inst = Instrument(scenario, seed)
    exp = ExportWriter(inst, path)
    inst.act("wait", seconds=30.0)
    exp.sync()
    return inst


def _rewrite(path: Path, edit) -> None:
    """Apply ``edit`` to the export's log records (a list of dicts, in place) and write them back."""
    p = path / "actions.jsonl"
    recs = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    edit(recs)
    p.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")


def _edit_session(path: Path, **kw) -> None:
    p = path / "session.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    s.update(kw)
    p.write_text(json.dumps(s), encoding="utf-8")


@pytest.fixture(scope="module")
def p5(tmp_path_factory):
    """A finished P5 episode that succeeded, its export, and the instrument's own summary."""
    path = tmp_path_factory.mktemp("p5") / "export"
    inst = _p5_episode(path)
    return path, inst.finish()


@pytest.fixture(scope="module")
def graded(p5):
    return grade_export(p5[0], P5, 0)


# ── the replay is the episode ───────────────────────────────────────────────
def test_an_export_grades_as_the_episode_it_came_from(p5, graded):
    path, own = p5
    assert own["success"] and own["claims_verified"] == own["claims_total"] == 3     # a real success
    assert graded["success"] is True and graded["partial"] == own["partial"] == 1.0 and graded["score"] == 1.0
    for k in ("claims_total", "claims_verified", "instrument_time_s", "actions"):
        assert graded[k] == own[k], k
    assert graded["finished_in_episode"] is True and graded["scenario_id"] == P5
    assert graded["rules_version"] == RULES_VERSION and graded["export_version"] == EXPORT_VERSION
    records = read_export(path)["records"]
    assert {r["kind"] for r in records} >= {"act", "refused", "observe", "report", "finish"}
    assert graded["replay"] == {"consistent": True, "result_mismatches": [], "digest_mismatches": [],
                                "torn": False, "records": len(records)}
    assert graded["note"] and "\n" not in graded["note"] and graded["note"].endswith("replay consistent")


def test_an_unfinished_export_is_judged_as_the_instrument_would_be_where_it_stopped(tmp_path):
    """The agent ran out of steps after two of the three answers: no ``finish`` in the export."""
    inst = _p5_episode(tmp_path / "export", reports=2, finish=False)
    own = inst.finish()                                  # the same instrument, judged as it stands
    assert own["claims_verified"] == 2 and own["claims_total"] == 3
    res = grade_export(tmp_path / "export", P5, 0)
    assert res["finished_in_episode"] is False and res["replay"]["consistent"]
    assert res["success"] is False and res["partial"] == own["partial"] == pytest.approx(2 / 3)
    assert res["score"] == own["partial"]
    for k in ("claims_total", "claims_verified", "instrument_time_s", "actions"):
        assert res[k] == own[k], k
    assert "never reached finish" in res["note"]


def test_a_whole_p4_episode_replays_bit_for_bit(tmp_path):
    """The pilot task end to end, as the scripted baseline plays it (only what an agent sees): about
    ninety actions, atoms dragged, tens of frames. Replayed by a grader that has only the records, it
    is the same episode: every reply, time and hash, and so the same verdict."""
    from stmbench.baselines import p4 as baseline
    from stmbench.baselines._common import Blind

    seed = 450
    inst = Instrument(P4, seed)
    exp = ExportWriter(inst, tmp_path / "export")
    got = baseline.play(Blind(inst), {"atoms_tried": 1})
    if got is not None:
        x, y = got["xy"]
        inst.report("target_site", x, unit="nm", x_nm=x, y_nm=y, frame=got["frame"]["frame"],
                    evidence=[got["frame"]["frame"]])
    own = inst.finish()
    exp.sync()
    assert own["actions"] > 50 and len(inst.frames) > 5
    res = grade_export(tmp_path / "export", P4, seed)
    assert res["replay"]["consistent"], res["replay"]
    assert res["replay"]["digest_mismatches"] == [] and res["replay"]["result_mismatches"] == []
    for k in ("success", "partial", "claims_total", "claims_verified", "instrument_time_s", "actions"):
        assert res[k] == own[k], k
    assert res["score"] == own["partial"]


def test_an_episode_over_its_instrument_time_budget_is_graded_and_said_so(tmp_path):
    """The overrun rule is the judge's: the replay spends the time again and ends over budget."""
    inst = Instrument(P4, 0)
    exp = ExportWriter(inst, tmp_path / "export")
    inst.act("wait", seconds=7200.0)                      # the budget is 1.5 h of instrument time
    with pytest.raises(ActionError, match="budget"):
        inst.act("wait", seconds=1.0)
    own = inst.finish()
    exp.sync()
    res = grade_export(tmp_path / "export", P4, 0)
    assert own["budget_overrun"] and res["replay"]["consistent"]
    assert res["success"] is False and res["instrument_time_s"] == own["instrument_time_s"] == 7200.0
    assert "over the instrument-time budget" in res["note"]


def test_the_result_holds_no_seed_truth_or_claim_detail(tmp_path):
    inst = Instrument(P4, SEED)
    exp = ExportWriter(inst, tmp_path / "export")
    frame = inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=8.0, pixels=32)["frame"]
    inst.act("sts", v_start=-0.1, v_end=0.1, points=32)
    inst.report("target_site", 1.0, x_nm=1.0, y_nm=0.0, frame=frame, evidence=[frame])
    inst.finish()
    exp.sync()
    res = grade_export(tmp_path / "export", P4, SEED)
    assert res["replay"]["consistent"] and res["claims_total"] == 2
    text = json.dumps(res)
    assert str(SEED) not in text
    assert not SECRET_KEYS & set(_keys(res))


def test_a_tampered_act_result_is_caught(p5, graded, tmp_path):
    """The export says an action replied otherwise than the replay's instrument does."""
    path = tmp_path / "export"
    shutil.copytree(p5[0], path)
    hit: dict = {}

    def edit(recs):
        rec = next(r for r in recs if r["kind"] == "act" and r["name"] == "zspec")
        rec["result"]["duration_s"] += 1.0
        hit.update(i=rec["i"])
    _rewrite(path, edit)
    want = [{"i": hit["i"], "name": "zspec", "field": "duration_s"}]
    strict = grade_export(path, P5, 0)
    assert strict["replay"]["consistent"] is False and strict["replay"]["result_mismatches"] == want
    assert strict["replay"]["digest_mismatches"] == []
    assert strict["score"] == 0.0 and strict["success"] is False and "scored 0 (strict)" in strict["note"]
    # the replay was not the recorded episode: what it judged is unknown to the result, not zero
    assert [strict[k] for k in ("partial", "claims_verified", "instrument_time_s", "actions")] == [None] * 4
    assert strict["claims_total"] == 3 and "claims verified" not in strict["note"]
    lax = grade_export(path, P5, 0, strict=False)                 # the replay's own grade, difference reported
    assert lax["replay"]["result_mismatches"] == want and lax["replay"]["consistent"] is False
    assert lax["score"] == graded["score"] == 1.0 and lax["success"] is True
    assert lax["partial"] == 1.0 and lax["claims_verified"] == 3 and "not strict" in lax["note"]


def test_a_wrong_seed_is_not_the_recorded_episode(p5):
    """The export carries no seed; the hashes of what was measured are what a wrong one disagrees with."""
    res = grade_export(p5[0], P5, 1)
    assert res["replay"]["consistent"] is False and res["replay"]["digest_mismatches"]
    assert res["score"] == 0.0 and res["success"] is False and res["partial"] is None


def test_a_tampered_missing_or_extra_digest_is_caught(p5, tmp_path):
    path, _ = p5
    export = read_export(path)
    records, lines = export["records"], export["digests"]
    assert {k for ln in lines for k in ln} >= {"frame", "spectrum"}
    inst, mismatches = replay_records(P5, 0, records)
    assert mismatches == [] and _digest_mismatches(inst, lines, records) == []

    def with_(edit):
        out = [json.loads(json.dumps(ln)) for ln in lines]
        edit(out)
        return _digest_mismatches(inst, out, records)

    frame = next(ln for ln in lines if "frame" in ln)
    spec = next(ln for ln in lines if "spectrum" in ln)
    # one array's hash edited: that file, and which array
    got = with_(lambda out: out[lines.index(spec)]["digest"].update(i="0" * 64))
    assert got == [{"i": spec["i"], "spectrum": spec["spectrum"], "reason": "the hashes differ", "arrays": ["i"]}]
    # a line dropped: the file the replay saved has none
    got = with_(lambda out: out.pop(lines.index(frame)))
    assert got == [{"i": frame["i"], "frame": frame["frame"], "reason": "the export has no digest line for it"}]
    # a line for a file the replay never saved
    got = with_(lambda out: out.append({"i": 3, "frame": "sim999.sxm", "digest": {"z_fwd": "0" * 64}}))
    assert got == [{"i": 3, "frame": "sim999.sxm", "reason": "the replay saved no such file"}]
    # a line that names nothing
    nameless = with_(lambda out: out.append({"i": 3, "digest": {}}))
    assert nameless == [{"i": 3, "reason": "a digest line names no frame or spectrum"}]
    # and through the whole grade
    p = tmp_path / "export"
    shutil.copytree(path, p)
    text = (p / "digests.jsonl").read_text(encoding="utf-8")
    (p / "digests.jsonl").write_text(text.replace(frame["digest"]["z_fwd"], "0" * 64), encoding="utf-8")
    res = grade_export(p, P5, 0)
    assert res["replay"]["result_mismatches"] == [] and res["score"] == 0.0 and not res["success"]
    assert res["replay"]["digest_mismatches"] == [{"i": frame["i"], "frame": frame["frame"],
                                                   "reason": "the hashes differ", "arrays": ["z_fwd"]}]


# ── not an export of this scenario: an error, not a score ───────────────────
def test_a_wrong_pairing_or_version_is_an_error_not_a_score(tmp_path):
    base = tmp_path / "base"
    _wait_export(base)
    assert grade_export(base, P4, 0)["replay"]["consistent"]

    with pytest.raises(ValueError, match="scenario"):
        grade_export(base, P5, 0)
    with pytest.raises(ValueError, match="no scenario"):
        grade_export(base, "P9_nope", 0)

    def broken(**kw):
        p = tmp_path / ("x%d" % len(list(tmp_path.iterdir())))
        shutil.copytree(base, p)
        _edit_session(p, **kw)
        return p
    for kw, needle in (({"rules_version": 0}, "rules_version"), ({"api_version": API_VERSION - 1}, "api_version"),
                       ({"version": 2}, "version-1"), ({"format": "other"}, "version-1"),
                       ({"records": 5}, "counts 5 records")):
        with pytest.raises(ValueError, match=needle):
            grade_export(broken(**kw), P4, 0)
    # not an export at all
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="cannot read the export"):
        grade_export(empty, P4, 0)
    with pytest.raises(ValueError, match="cannot read the export"):
        grade_export(tmp_path / "missing", P4, 0)
    for junk in ("not json", "[]", '"x"'):
        p = tmp_path / ("junk%d" % len(junk))
        shutil.copytree(base, p)
        (p / "session.json").write_text(junk, encoding="utf-8")
        with pytest.raises(ValueError):
            grade_export(p, P4, 0)
    p = tmp_path / "lines"
    shutil.copytree(base, p)
    (p / "actions.jsonl").write_text("[1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not an export directory"):
        grade_export(p, P4, 0)


def test_a_log_cut_off_mid_write_is_reported_and_graded_as_it_stands(tmp_path):
    path = tmp_path / "export"
    _wait_export(path)
    whole = grade_export(path, P4, 0)
    with (path / "actions.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"i": 1, "kind": "act", "na')
    cut = grade_export(path, P4, 0)
    assert cut["replay"]["torn"] is True and cut["replay"]["consistent"] and cut["replay"]["records"] == 1
    assert "cut off" in cut["note"] and not whole["replay"]["torn"]
    assert {k: v for k, v in cut.items() if k not in ("replay", "note")} == \
           {k: v for k, v in whole.items() if k not in ("replay", "note")}


def test_text_out_of_the_export_is_clipped_before_a_result_repeats_it(tmp_path):
    path = tmp_path / "export"
    _wait_export(path)
    nasty = "scan\n\x1b[31m" + "A" * 5000
    _rewrite(path, lambda recs: recs.append({"i": 1, "kind": "act", "name": nasty, "params": {}, "result": {}}))
    res = grade_export(path, P4, 0)
    (m,) = res["replay"]["result_mismatches"]
    assert m["i"] == 1 and len(m["name"]) <= 120 and len(m["error"]) <= 120
    assert "\n" not in m["name"] and "\x1b" not in m["name"] and "\n" not in m["error"]
    assert len(json.dumps(res)) < 3000 and "A" * 200 not in json.dumps(res)


# ── replay_records ──────────────────────────────────────────────────────────
def _wait(i: int, seconds: float, t0: float) -> dict:
    return {"i": i, "kind": "act", "name": "wait", "params": {"seconds": seconds}, "t0_s": t0,
            "t1_s": t0 + seconds, "result": {"waited_s": seconds}}


def test_replay_records_runs_the_calls_and_skips_what_changed_nothing():
    recs = [_wait(0, 30.0, 0.0),
            {"i": 1, "kind": "observe", "what": "status", "args": {}, "t_s": 30.0},
            {"i": 2, "kind": "refused", "name": "scan", "args": {}, "error": "x", "t_s": 30.0},
            {"i": 3, "kind": "finish", "t_s": 30.0},
            {"i": 4, "kind": "observe", "what": "status", "args": {}, "t_s": 30.0},
            _wait(5, 99.0, 30.0),                           # after finish the instrument takes no act:
            {"i": 6, "kind": "report", "args": {"claim_id": "target_site", "value": 1.0}}]  # nor a report
    inst, mismatches = replay_records(P4, 0, recs)
    assert mismatches == [{"i": 5, "name": "wait", "error": "a call after finish"},
                          {"i": 6, "name": "report", "error": "a call after finish"}]
    assert inst.finished and inst.n_actions == 1 and inst.used_s == pytest.approx(30.0)   # nothing after finish is run
    assert inst.verdict.success is False                    # judged as it stands: nothing reported
    assert replay_records(P4, 0, recs[:4])[1] == []


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
            {"i": 6, "kind": "report", "args": {"claim_id": "target_site", "value": 1.0, "nonsense": 1}},
            {"i": 7, "kind": "mystery"},
            "junk",
            {"i": 9, "kind": "act", "name": "wait", "params": {"seconds": 1.0}, "result": {"waited_s": 1.0},
             "t0_s": 0.0, "t1_s": 1.0}]                     # the replay goes on after them
    inst, mismatches = replay_records(P4, 0, recs)
    assert [(m["i"], m.get("name")) for m in mismatches] == [
        (0, "scan"), (1, "report"), (2, "wait"), (3, None), (4, "report"), (5, "report"), (6, "report"),
        (7, "mystery"), (None, None)]
    assert all("error" in m and "field" not in m for m in mismatches)
    assert "outside" in mismatches[0]["error"] and "unknown claim" in mismatches[1]["error"]
    assert inst.n_actions == 1 and inst.finished


def test_replay_of_a_ledger_still_raises_a_call_the_replay_refuses(tmp_path):
    """``replay`` is ``replay_records`` that raises: a ledger that cannot be replayed is not
    silently replayed to the end."""
    rec = {"i": 0, "kind": "act", "name": "scan",
           "params": {"center_x_nm": 0.0, "center_y_nm": 0.0, "size_nm": 10.0, "pixels": 7}, "result": {}}
    (tmp_path / "episode.json").write_text(json.dumps({"api_version": API_VERSION, "scenario_id": P4, "seed": 0}),
                                           encoding="utf-8")
    (tmp_path / "actions.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    with pytest.raises(ActionError, match="outside"):
        replay(tmp_path)


# ── the grade command ───────────────────────────────────────────────────────
def _cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "stmbench.cli", *map(str, args)], capture_output=True, text=True,
                          timeout=600, cwd=ROOT, env={**__import__("os").environ, "PYTHONUTF8": "1"})


def test_the_grade_command_prints_the_result_and_exits_0(p5, graded, tmp_path):
    out = tmp_path / "result.json"
    proc = _cli("grade", p5[0], "--scenario", P5, "--seed", 0, "--out", out)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == graded                 # stdout is the result and nothing else
    assert json.loads(out.read_text(encoding="utf-8")) == graded


def test_the_grade_command_exits_2_when_the_input_is_not_a_gradable_export(tmp_path, capsys):
    base = tmp_path / "base"
    _wait_export(base)
    assert cli_main(["grade", str(base), "--scenario", P5, "--seed", "0"]) == 2
    cap = capsys.readouterr()
    assert cap.out == "" and cap.err.startswith("grade: ") and "scenario" in cap.err
    assert cli_main(["grade", str(tmp_path / "missing"), "--scenario", P4, "--seed", "0"]) == 2
    assert "cannot read the export" in capsys.readouterr().err
    assert cli_main(["grade", str(base), "--scenario", "nope", "--seed", "0"]) == 2
    assert "no scenario" in capsys.readouterr().err
    _edit_session(base, rules_version=0)
    assert cli_main(["grade", str(base), "--scenario", P4, "--seed", "0"]) == 2
    assert "rules_version" in capsys.readouterr().err


def test_no_strict_keeps_the_replays_score_and_the_difference(tmp_path, capsys):
    path = tmp_path / "export"
    _wait_export(path)
    _rewrite(path, lambda recs: recs[0]["result"].update(waited_s=31.0))
    out = tmp_path / "result.json"
    assert cli_main(["grade", str(path), "--scenario", P4, "--seed", "0", "--no-strict", "--out", str(out)]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["replay"]["consistent"] is False and "not strict" in res["note"]
    assert res["replay"]["result_mismatches"] == [{"i": 0, "name": "wait", "field": "waited_s"}]
    assert json.loads(out.read_text(encoding="utf-8")) == res
    assert cli_main(["grade", str(path), "--scenario", P4, "--seed", "0"]) == 0     # strict: graded, scored 0
    strict = json.loads(capsys.readouterr().out)
    assert strict["score"] == 0.0 and strict["partial"] is None and strict["replay"] == res["replay"]
    assert res["partial"] == 0.0 and res["actions"] == 1


# ── the OSWorld-Science adapter ─────────────────────────────────────────────
def _adapter():
    spec = importlib.util.spec_from_file_location("stm_metric_under_test", DEPLOY / "stm_metric.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                              # by path, as the harness loads it
    return mod


def _oracle(tmp_path: Path, **gt) -> dict:
    d = tmp_path / "oracle"
    d.mkdir(exist_ok=True)
    (d / "gt.json").write_text(json.dumps({"scenario": P5, "seed": 0, "grader_image": "stmbench-grader:test", **gt}),
                               encoding="utf-8")
    return {"type": "rule", "rules": {"oracle_dir": str(d)}}


def test_the_adapter_returns_the_score_of_the_grade_command(p5, graded, tmp_path, monkeypatch):
    mod = _adapter()
    monkeypatch.chdir(ROOT)                                   # the command imports the worktree's package
    monkeypatch.setenv(mod.CMD_ENV, f"{shlex.quote(Path(sys.executable).as_posix())} -m stmbench.cli")
    got = mod.grade_episode(str(p5[0] / "actions.jsonl"), _oracle(tmp_path))
    assert got == (graded["score"], graded["note"]) and got[0] == 1.0
    # a failed grade is a score with its reason, never an exception
    score, note = mod.grade_episode(str(p5[0] / "actions.jsonl"), _oracle(tmp_path, scenario=P4))
    assert score == 0.0 and note.startswith("grader failed: exit 2: grade: the export is of scenario")
    no_gt = {"type": "rule", "rules": {"oracle_dir": str(tmp_path)}}                # no gt.json in there
    score, note = mod.grade_episode(str(p5[0] / "actions.jsonl"), no_gt)
    assert score == 0.0 and note.startswith("grader failed: ") and "gt.json" in note
    monkeypatch.setenv(mod.CMD_ENV, "stmbench-no-such-program")
    assert mod.grade_episode(str(p5[0] / "actions.jsonl"), _oracle(tmp_path))[0] == 0.0


def test_the_adapter_runs_the_grader_image_without_network_on_a_read_only_export(tmp_path, monkeypatch):
    mod = _adapter()
    monkeypatch.delenv(mod.CMD_ENV, raising=False)
    export = tmp_path / "export"
    export.mkdir()
    (export / "actions.jsonl").write_text("", encoding="utf-8")
    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"score": 0.5, "note": "half"}), stderr="")
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    assert mod.grade_episode(str(export / "actions.jsonl"), _oracle(tmp_path, seed=7)) == (0.5, "half")
    cmd, kw = calls[0]
    name = cmd[cmd.index("--name") + 1]
    assert cmd == ["docker", "run", "--rm", "--name", name, "--network", "none",
                   "-v", f"{export.resolve()}:/export:ro", "stmbench-grader:test",
                   "grade", "/export", "--scenario", P5, "--seed", "7"]
    assert kw["timeout"] == 30 * 60

    def out_of_range(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"score": 2.0, "note": "x"}), stderr="")
    monkeypatch.setattr(mod.subprocess, "run", out_of_range)
    assert mod.grade_episode(str(export / "actions.jsonl"), _oracle(tmp_path))[1].startswith("grader failed: ")
    for stdout in ("", "not json", '{"note": "no score"}'):
        monkeypatch.setattr(mod.subprocess, "run",
                            lambda cmd, _out=stdout, **kw: subprocess.CompletedProcess(cmd, 0, _out, ""))
        assert mod.grade_episode(str(export / "actions.jsonl"), _oracle(tmp_path))[0] == 0.0

    def failing(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 125, stdout="", stderr="docker: Unable to find image\nsecond line")
    monkeypatch.setattr(mod.subprocess, "run", failing)
    assert mod.grade_episode(str(export / "actions.jsonl"), _oracle(tmp_path)) == \
        (0.0, "grader failed: exit 125: second line")


def test_a_grader_that_times_out_is_a_zero_and_its_container_is_killed(tmp_path, monkeypatch):
    mod = _adapter()
    monkeypatch.delenv(mod.CMD_ENV, raising=False)
    (tmp_path / "actions.jsonl").write_text("", encoding="utf-8")
    calls = []

    def hung(cmd, **kw):
        calls.append(cmd)
        if cmd[1] == "run":
            raise subprocess.TimeoutExpired(cmd, kw["timeout"])
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(mod.subprocess, "run", hung)
    assert mod.grade_episode(str(tmp_path / "actions.jsonl"), _oracle(tmp_path)) == \
        (0.0, "grader failed: no result within 30 min")
    assert calls[1] == ["docker", "kill", calls[0][calls[0].index("--name") + 1]]


def test_the_adapter_is_standard_library_only_and_python_3_11_syntax():
    src = (DEPLOY / "stm_metric.py").read_text(encoding="utf-8")
    tree = ast.parse(src, feature_version=(3, 11))
    names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert names <= set(sys.stdlib_module_names) and "stmbench" not in names


# ── the grader image ────────────────────────────────────────────────────────
def test_the_grader_image_copies_what_exists_and_pins_what_the_replay_needs():
    """The image is not built in the suite (it needs Docker): what can be checked is that its
    inputs exist, are not ignored by the build context, and pin the numerics."""
    docker = (DEPLOY / "grader" / "Dockerfile").read_text(encoding="utf-8")
    ignore = [ln.strip() for ln in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
              if ln.strip() and not ln.startswith("#")]
    sources = []
    for ln in docker.splitlines():
        if ln.startswith("COPY "):
            *srcs, _dest = ln.split()[1:]
            sources += srcs
    assert {"pyproject.toml", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "stmsim", "stmbench",
            "deploy/osworld/grader/requirements.txt"} <= set(sources)
    for s in sources:
        assert (ROOT / s).exists(), s
        assert s not in ignore and not any(s.startswith(p.rstrip("/") + "/") for p in ignore if "*" not in p), s
    for must in ("output", ".venv*", "runs", ".git"):
        assert must in ignore, must
    pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([0-9][0-9A-Za-z.]*)\s*$",
                           (DEPLOY / "grader" / "requirements.txt").read_text(encoding="utf-8"), re.M))
    assert set(pins) == {"numpy", "scipy", "PyYAML"}
    assert 'ENTRYPOINT ["python", "-m", "stmbench.cli"]' in docker and "FROM ${PYTHON_IMAGE}" in docker
    assert "ARG PYTHON_IMAGE=python:3.13-slim" in docker and "pip install --no-deps ." in docker
    assert re.search(r"^USER (?!root)\S+", docker, re.M)
