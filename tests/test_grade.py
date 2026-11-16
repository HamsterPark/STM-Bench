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
from stmbench.instrument.export import EXPORT_VERSION, ExportDamaged, ExportWriter, read_export
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
                                "hashes_equal": True, "rounding_differences": [],
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


def _without_fp(text: str) -> str:
    """A digests.jsonl as a version-1 export wrote it: hashes only."""
    out = []
    for line in text.splitlines():
        rec = json.loads(line)
        rec.pop("fp", None)
        out.append(json.dumps(rec))
    return "\n".join(out) + "\n"


def _digest_lines(path, edit):
    p = path / "digests.jsonl"
    recs = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    edit(recs)
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")


def test_a_rounding_difference_is_consistent_and_a_real_one_is_not(p5, tmp_path):
    """Version 2: equal hashes are the exact test; unequal hashes are decided by the fingerprints."""
    path, _ = p5
    assert read_export(path)["session"]["version"] == EXPORT_VERSION == 2
    assert all("fp" in ln for ln in read_export(path)["digests"])
    base = grade_export(path, P5, 0)
    assert base["replay"]["hashes_equal"] is True and base["replay"]["rounding_differences"] == []

    def hash_and(edit, name):
        p = tmp_path / name
        shutil.copytree(path, p)

        def both(recs):
            frame = next(r for r in recs if "frame" in r)
            frame["digest"]["z_fwd"] = "0" * 64                     # another machine's last bits: other hash...
            edit(frame["fp"]["z_fwd"])                              # ...and the fingerprint as that machine saw it
        _digest_lines(p, both)
        return grade_export(p, P5, 0)

    # a difference of 1e-12 of the scale: rounding. Consistent, scored as replayed, and said so.
    def nudge(fp):
        fp["mean"] += 1e-12 * max(abs(fp["min"]), abs(fp["max"]))
    ok = hash_and(nudge, "rounded")
    assert ok["replay"]["consistent"] and ok["score"] == base["score"] and ok["replay"]["digest_mismatches"] == []
    assert ok["replay"]["hashes_equal"] is False
    (diff,) = ok["replay"]["rounding_differences"]
    assert diff["array"] == "z_fwd" and 0 < diff["max_rel_diff"] <= 1e-9 and "rounding differences" in ok["note"]

    # 1e-6 of the scale: a different measurement
    def shift(fp):
        fp["mean"] += 1e-6 * max(abs(fp["min"]), abs(fp["max"]))
    bad = hash_and(shift, "real")
    assert not bad["replay"]["consistent"] and bad["score"] == 0.0
    (m,) = bad["replay"]["digest_mismatches"]
    assert m["arrays"] == ["z_fwd"] and "beyond rounding" in m["reason"] and "1e-06" in m["reason"]

    # shape and non-finite values are exact
    def mask(fp):
        fp["nonfinite"] += 1
    assert not hash_and(mask, "mask")["replay"]["consistent"]

    # a version-1 export (no fingerprints) with another hash: a mismatch, as before
    p = tmp_path / "v1"
    shutil.copytree(path, p)
    _edit_session(p, version=1)
    (p / "digests.jsonl").write_text(_without_fp((p / "digests.jsonl").read_text(encoding="utf-8")), encoding="utf-8")
    assert grade_export(p, P5, 0)["replay"]["consistent"]
    _digest_lines(p, lambda recs: next(r for r in recs if "frame" in r)["digest"].update(z_fwd="0" * 64))
    assert grade_export(p, P5, 0)["replay"]["digest_mismatches"][0]["reason"] == "the hashes differ"


def test_a_tampered_missing_or_extra_digest_is_caught(p5, tmp_path):
    path, _ = p5
    export = read_export(path)
    records, lines = export["records"], export["digests"]
    assert {k for ln in lines for k in ln} >= {"frame", "spectrum"}
    inst, mismatches = replay_records(P5, 0, records)
    assert mismatches == [] and _digest_mismatches(inst, lines, records) == []

    def with_(edit):
        out = [json.loads(json.dumps(ln)) for ln in lines]
        for ln in out:
            ln.pop("fp", None)                   # no fingerprint: unequal hashes are a mismatch, as in version 1
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
    (p / "digests.jsonl").write_text(_without_fp(text.replace(frame["digest"]["z_fwd"], "0" * 64)), encoding="utf-8")
    res = grade_export(p, P5, 0)
    assert res["replay"]["result_mismatches"] == [] and res["score"] == 0.0 and not res["success"]
    assert res["replay"]["digest_mismatches"] == [{"i": frame["i"], "frame": frame["frame"],
                                                   "reason": "the hashes differ", "arrays": ["z_fwd"]}]


def test_an_export_with_moves_grades_with_no_mismatch_and_a_changed_trace_is_one(tmp_path):
    """Every move's trace is a measurement the replay has to reproduce: the grade finds them
    equal, and finds an edited hash of one array of one trace."""
    inst = Instrument(P4, 0)
    exp = ExportWriter(inst, tmp_path / "e")
    inst.act("move_tip", x_nm=3.0, y_nm=0.0, speed_nm_s=20.0)
    inst.act("move_tip", x_nm=0.0, y_nm=0.0)
    inst.act("withdraw")
    inst.act("move_tip", x_nm=5.0, y_nm=0.0)                    # a withdrawn tip's trace is one too
    exp.sync()
    res = grade_export(tmp_path / "e", P4, 0)
    assert res["replay"]["consistent"] and res["replay"]["digest_mismatches"] == [] and res["replay"]["result_mismatches"] == []
    lines = read_export(tmp_path / "e")["digests"]
    assert len(lines) == 3 and all(set(x["digest"]) == {"t", "dist", "z_abs", "i"} for x in lines)
    p = tmp_path / "e" / "digests.jsonl"
    p.write_text(_without_fp(p.read_text(encoding="utf-8").replace(lines[1]["digest"]["z_abs"], "0" * 64)),
                 encoding="utf-8")
    bad = grade_export(tmp_path / "e", P4, 0)
    assert bad["score"] == 0.0 and bad["replay"]["digest_mismatches"] == [
        {"i": lines[1]["i"], "spectrum": lines[1]["spectrum"], "reason": "the hashes differ", "arrays": ["z_abs"]}]


def test_a_trace_missing_from_the_digests_is_traced_to_the_move_that_made_it(tmp_path):
    """A move's reply names its trace under ``trace``, not ``spectrum``: the mismatch for a trace the
    export has no digest line for still says which record made it."""
    inst = Instrument(P4, 0)
    exp = ExportWriter(inst, tmp_path / "e")
    inst.act("move_tip", x_nm=3.0, y_nm=0.0, speed_nm_s=20.0)
    inst.act("move_tip", x_nm=0.0, y_nm=0.0)
    exp.sync()
    export = read_export(tmp_path / "e")
    records, lines = export["records"], export["digests"]
    replayed, mismatches = replay_records(P4, 0, records)
    assert mismatches == [] and _digest_mismatches(replayed, lines, records) == [] and len(lines) == 2
    moves = [r["i"] for r in records if r.get("kind") == "act" and r.get("name") == "move_tip"]
    assert moves == [lines[0]["i"], lines[1]["i"]]
    assert _digest_mismatches(replayed, lines[:1], records) == [
        {"i": moves[1], "spectrum": lines[1]["spectrum"], "reason": "the export has no digest line for it"}]


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
    # a mismatch of setup: a plain ValueError (exit 2), not a damaged export
    for kw, needle in (({"rules_version": 0}, "rules_version"), ({"api_version": API_VERSION - 1}, "api_version"),
                       ({"version": 3}, "version-1 or 2")):
        with pytest.raises(ValueError, match=needle) as err:
            grade_export(broken(**kw), P4, 0)
        assert not isinstance(err.value, ExportDamaged)
    # a damaged submission: ExportDamaged (a ValueError), which the grade command scores 0
    for kw, needle in (({"format": "other"}, "not that of a"), ({"records": 5}, "counts 5 records"),
                       ({"scenario_id": None}, "lacks a scenario"), ({"api_version": None}, "lacks a scenario"),
                       ({"rules_version": "2"}, "lacks a scenario"), ({"version": "2"}, "not that of a")):
        with pytest.raises(ExportDamaged, match=needle):
            grade_export(broken(**kw), P4, 0)
    # not an export at all
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ExportDamaged, match="missing or cannot be read"):
        grade_export(empty, P4, 0)
    with pytest.raises(ExportDamaged, match="missing or cannot be read"):
        grade_export(tmp_path / "missing", P4, 0)
    for junk in ("not json", "[]", '"x"'):
        p = tmp_path / ("junk%d" % len(junk))
        shutil.copytree(base, p)
        (p / "session.json").write_text(junk, encoding="utf-8")
        with pytest.raises(ExportDamaged):
            grade_export(p, P4, 0)
    p = tmp_path / "lines"
    shutil.copytree(base, p)
    (p / "actions.jsonl").write_text("[1]\n", encoding="utf-8")
    with pytest.raises(ExportDamaged, match="a record is missing or out of order"):
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
    assert cli_main(["grade", str(base), "--scenario", "nope", "--seed", "0"]) == 2
    assert "no scenario" in capsys.readouterr().err
    _edit_session(base, rules_version=0)
    assert cli_main(["grade", str(base), "--scenario", P4, "--seed", "0"]) == 2
    assert "rules_version" in capsys.readouterr().err
    _edit_session(base, rules_version=RULES_VERSION, version=3)
    assert cli_main(["grade", str(base), "--scenario", P4, "--seed", "0"]) == 2          # another export version
    assert "version-1 or 2" in capsys.readouterr().err


def _damage_cases(tmp_path):
    """Exports damaged in the ways an agent's machine could leave them: name -> directory."""
    base = tmp_path / "base"
    _wait_export(base)

    def copy(name):
        p = tmp_path / name
        shutil.copytree(base, p)
        return p
    out = {}
    out["session counts more records than the log holds"] = copy("count")
    _edit_session(out["session counts more records than the log holds"], records=9)
    p = out["a record missing"] = copy("gap")
    recs = [json.loads(x) for x in (p / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
    recs[0]["i"] = 5
    (p / "actions.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    p = out["actions.jsonl missing"] = copy("noactions")
    (p / "actions.jsonl").unlink()
    p = out["session.json missing"] = copy("nosession")
    (p / "session.json").unlink()
    p = out["a file that is not JSON"] = copy("notjson")
    (p / "session.json").write_text("not json", encoding="utf-8")
    p = out["a line of the log that is not JSON"] = copy("badline")
    (p / "actions.jsonl").write_text("{broken\n{\"i\": 0}\n", encoding="utf-8")
    out["not an export directory"] = tmp_path / "nothing-here"
    return out


def test_a_damaged_export_is_scored_0_with_exit_0(tmp_path, capsys):
    """A submission that cannot be verified is not a reproduction: score 0, never exit 2 (the adapter's pending)."""
    for why, path in _damage_cases(tmp_path).items():
        assert cli_main(["grade", str(path), "--scenario", P4, "--seed", "0"]) == 0, why
        cap = capsys.readouterr()
        res = json.loads(cap.out)
        assert res["score"] == 0.0 and res["success"] is False, why
        assert all(res[k] is None for k in ("partial", "claims_verified", "instrument_time_s", "actions")), why
        assert res["replay"]["consistent"] is False and res["replay"]["damaged"], why
        assert res["note"].startswith("the export is damaged: ") and res["note"].endswith("; scored 0"), why
        assert res["scenario_id"] == P4 and str(path) not in cap.out, why          # no directory in the result


def test_a_damaged_export_scores_0_through_the_adapter_and_a_mismatch_is_pending(tmp_path, monkeypatch):
    mod = _adapter()
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv(mod.CMD_ENV, f"{shlex.quote(Path(sys.executable).as_posix())} -m stmbench.cli")
    cases = _damage_cases(tmp_path)
    for why in ("session counts more records than the log holds", "a record missing", "session.json missing",
                "a file that is not JSON", "a line of the log that is not JSON"):
        got = mod.grade_episode(str(cases[why] / "actions.jsonl"), _oracle(tmp_path, scenario=P4))
        assert isinstance(got, tuple) and got[0] == 0.0 and got[1].startswith("the export is damaged"), (why, got)
    # a mismatch of setup is still the grader's: pending
    base = tmp_path / "setup"
    _wait_export(base)
    _edit_session(base, version=3)
    assert _pending(mod.grade_episode(str(base / "actions.jsonl"), _oracle(tmp_path, scenario=P4))).startswith(
        "grader fault: exit 2: grade: ")


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


def _pending(got) -> str:
    """The reason of a grader-side fault: a dict whose ``pending`` the runner's scoring reads (score null, not 0)."""
    assert isinstance(got, dict) and got["score"] == 0.0 and got["pending"] and got["note"] == got["pending"], got
    return got["pending"]


def test_the_adapter_returns_the_score_of_the_grade_command(p5, graded, tmp_path, monkeypatch):
    mod = _adapter()
    monkeypatch.chdir(ROOT)                                   # the command imports the worktree's package
    monkeypatch.setenv(mod.CMD_ENV, f"{shlex.quote(Path(sys.executable).as_posix())} -m stmbench.cli")
    got = mod.grade_episode(str(p5[0] / "actions.jsonl"), _oracle(tmp_path))
    assert got == (graded["score"], graded["note"]) and got[0] == 1.0
    # a fault on the grading side is "pending" (no score, no verdict), never an exception and never a zero
    got = mod.grade_episode(str(p5[0] / "actions.jsonl"), _oracle(tmp_path, scenario=P4))
    assert _pending(got).startswith("grader fault: exit 2: grade: the export is of scenario")
    no_gt = {"type": "rule", "rules": {"oracle_dir": str(tmp_path)}}                # no gt.json in there
    assert "gt.json" in _pending(mod.grade_episode(str(p5[0] / "actions.jsonl"), no_gt))
    monkeypatch.setenv(mod.CMD_ENV, "stmbench-no-such-program")
    assert _pending(mod.grade_episode(str(p5[0] / "actions.jsonl"), _oracle(tmp_path))).startswith(
        "grader fault: the grader could not be started")


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
    assert _pending(mod.grade_episode(str(export / "actions.jsonl"), _oracle(tmp_path))) == \
        "grader fault: the grader returned a score outside the unit interval"
    for stdout in ("", "not json", '{"note": "no score"}', '{"score": "x", "note": "y"}', "[]"):
        monkeypatch.setattr(mod.subprocess, "run",
                            lambda cmd, _out=stdout, **kw: subprocess.CompletedProcess(cmd, 0, _out, ""))
        assert _pending(mod.grade_episode(str(export / "actions.jsonl"), _oracle(tmp_path))) == \
            "grader fault: the grader's output is not a result"

    def failing(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 125, stdout="", stderr="docker: Unable to find image\nsecond line")
    monkeypatch.setattr(mod.subprocess, "run", failing)
    assert _pending(mod.grade_episode(str(export / "actions.jsonl"), _oracle(tmp_path))) == \
        "grader fault: exit 125: second line"

    def no_docker(cmd, **kw):
        raise FileNotFoundError(2, "No such file or directory: 'docker'")
    monkeypatch.setattr(mod.subprocess, "run", no_docker)
    assert _pending(mod.grade_episode(str(export / "actions.jsonl"), _oracle(tmp_path))).startswith(
        "grader fault: the grader could not be started")


def test_a_grader_that_times_out_is_pending_and_its_container_is_killed(tmp_path, monkeypatch):
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
    assert _pending(mod.grade_episode(str(tmp_path / "actions.jsonl"), _oracle(tmp_path))) == \
        "grader fault: no result within 30 min"
    assert calls[1] == ["docker", "kill", calls[0][calls[0].index("--name") + 1]]


def _fake_grader(tmp_path: Path, body: str) -> str:
    """A command prefix that stands in for ``python -m stmbench.cli``: a script run as ``<script> grade <dir> ...``."""
    script = tmp_path / "fake_grader.py"
    script.write_text("import sys\n" + body, encoding="utf-8")
    return f"{shlex.quote(Path(sys.executable).as_posix())} {shlex.quote(script.as_posix())}"


@pytest.mark.parametrize("seed", [913275, 7, 0])
def test_no_seed_reaches_what_the_adapter_returns(tmp_path, monkeypatch, seed):
    """The seed is on the grader's command line; whatever the grader prints about it must not come back."""
    mod = _adapter()
    (tmp_path / "actions.jsonl").write_text("", encoding="utf-8")
    oracle = _oracle(tmp_path, seed=seed)
    monkeypatch.setenv(mod.CMD_ENV, _fake_grader(tmp_path, "print('seed ' + sys.argv[-1] + ' is wrong', file=sys.stderr)\n"
                                                           "sys.exit(1)\n"))
    reason = _pending(mod.grade_episode(str(tmp_path / "actions.jsonl"), oracle))
    assert reason == "grader fault: exit 1: seed <seed> is wrong"
    # the whole command line, and a note of a successful grade, are scrubbed as well
    monkeypatch.setenv(mod.CMD_ENV, _fake_grader(
        tmp_path, "import json\nprint(json.dumps({'score': 0.5, 'note': 'argv ' + ' '.join(sys.argv[1:])}))\n"))
    score, note = mod.grade_episode(str(tmp_path / "actions.jsonl"), oracle)
    assert score == 0.5 and note.endswith("--seed <seed>") and f"--seed {seed}" not in note
    monkeypatch.setenv(mod.CMD_ENV, _fake_grader(tmp_path, f"print('x{seed}y' * 100, file=sys.stderr)\nsys.exit(2)\n"))
    reason = _pending(mod.grade_episode(str(tmp_path / "actions.jsonl"), oracle))
    assert "<seed>" in reason
    if seed > 1000:
        assert str(seed) not in reason


def test_an_inconsistent_replay_is_a_real_zero_not_a_pending(tmp_path, monkeypatch):
    mod = _adapter()
    (tmp_path / "actions.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setenv(mod.CMD_ENV, _fake_grader(
        tmp_path, "import json\nprint(json.dumps({'score': 0.0, 'note': 'replay differs from the recorded episode'}))\n"))
    assert mod.grade_episode(str(tmp_path / "actions.jsonl"), _oracle(tmp_path)) == \
        (0.0, "replay differs from the recorded episode")


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
