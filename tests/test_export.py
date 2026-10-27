"""The export directory (stmbench.instrument.export): what leaves a screen agent's machine to be
graded. It is written as the episode happens, holds the records the run directory's ledger holds
and the hashes of every saved frame and spectrum, and nothing the agent may not see."""
from __future__ import annotations

import json

import pytest

from stmbench.instrument import ActionError, Instrument
from stmbench.instrument.export import ExportWriter, read_export

P4 = "P4_atom_positioning_cu111"
#: a seed whose digits cannot turn up in a record by chance
SEED = 913275
SCAN = {"center_x_nm": 0.0, "center_y_nm": 0.0, "size_nm": 8.0, "pixels": 32}
SECRET_KEYS = {"truth", "truth_before", "truth_after", "verdict", "success", "partial", "hidden", "seed"}


def _keys(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k
            yield from _keys(v)
    elif isinstance(o, list):
        for v in o:
            yield from _keys(v)


def _episode(tmp_path, finish: bool = True):
    inst = Instrument(P4, SEED, out_dir=tmp_path / "run")
    exp = ExportWriter(inst, tmp_path / "export")
    frame = inst.act("scan", **SCAN)["frame"]
    exp.sync()
    with pytest.raises(ActionError):
        inst.act("scan", center_x_nm=0.0)
    exp.sync()
    inst.status()
    inst.frame(frame)
    exp.sync()
    inst.act("sts", v_start=-0.1, v_end=0.1, points=32)
    exp.sync()
    inst.report("target_site", 1.0, x_nm=1.0, y_nm=0.0, frame=frame, evidence=[frame])
    exp.sync()
    if finish:
        inst.finish()
        exp.sync()
    return inst


def test_the_export_holds_the_ledgers_records_and_the_hashes_of_what_was_measured(tmp_path):
    inst = _episode(tmp_path)
    got = read_export(tmp_path / "export")
    ledger = [json.loads(line) for line in (tmp_path / "run" / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert got["records"] == ledger and not got["torn"]
    assert [r["kind"] for r in ledger] == ["act", "refused", "observe", "observe", "act", "report", "finish"]
    s = got["session"]
    assert s["finished"] and s["scenario_id"] == P4 and s["records"] == len(ledger)
    d = inst.digest()
    assert {x["frame"]: x["digest"] for x in got["digests"] if "frame" in x} == d["frames"]
    assert {x["spectrum"]: x["digest"] for x in got["digests"] if "spectrum" in x} == d["spectra"]
    assert all(ledger[x["i"]]["kind"] == "act" for x in got["digests"])
    report = json.loads((tmp_path / "export" / "report.json").read_text(encoding="utf-8"))
    assert report == json.loads((tmp_path / "run" / "report.json").read_text(encoding="utf-8"))


def test_the_export_carries_no_seed_truth_or_verdict(tmp_path):
    _episode(tmp_path)
    for p in sorted((tmp_path / "export").iterdir()):
        text = p.read_text(encoding="utf-8")
        assert str(SEED) not in text, p.name
        docs = [json.loads(line) for line in text.splitlines()] if p.suffix == ".jsonl" else [json.loads(text)]
        assert not SECRET_KEYS & set(_keys(docs)), p.name
    # the private ledger beside it does hold them
    assert str(SEED) in (tmp_path / "run" / "episode.json").read_text(encoding="utf-8")


def test_an_episode_that_never_finishes_is_on_record_as_it_stands(tmp_path):
    inst = _episode(tmp_path, finish=False)
    got = read_export(tmp_path / "export")
    assert len(got["records"]) == len(inst.log) and got["records"][-1]["kind"] == "report"
    assert not got["session"]["finished"]
    assert json.loads((tmp_path / "export" / "report.json").read_text(encoding="utf-8")) == inst.sink.folded()


def test_a_line_cut_off_mid_write_is_left_out_and_said_so(tmp_path):
    _episode(tmp_path, finish=False)
    p = tmp_path / "export" / "actions.jsonl"
    clean = p.read_text(encoding="utf-8")
    whole = read_export(tmp_path / "export")["records"]
    p.write_text(clean + '{"i": 7, "kind": "act", "na', encoding="utf-8")
    got = read_export(tmp_path / "export")
    assert got["torn"] and got["records"] == whole
    # a damaged line before the last is not a cut-off write
    lines = clean.splitlines()
    lines[1] = lines[1][:10]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_export(tmp_path / "export")
    # nor is a record missing from the middle
    p.write_text("\n".join(clean.splitlines()[:1] + clean.splitlines()[2:]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_export(tmp_path / "export")


def test_an_export_directory_holds_one_episode(tmp_path):
    _episode(tmp_path, finish=False)
    with pytest.raises(FileExistsError):
        ExportWriter(Instrument(P4, 1), tmp_path / "export")


def test_a_moves_trace_is_kept_like_a_spectrum_and_has_its_digest_line(tmp_path):
    """``move_tip`` replies with the file name of the trace it recorded, not with a ``spectrum``
    key: the export writes its digest line all the same, and the grader re-checks it."""
    inst = Instrument(P4, SEED, out_dir=tmp_path / "run")
    exp = ExportWriter(inst, tmp_path / "export")
    r = inst.act("move_tip", x_nm=2.0, y_nm=1.0, speed_nm_s=50.0)
    exp.sync()
    inst.act("move_tip", x_nm=0.0, y_nm=0.0, speed_nm_s=50.0)
    exp.sync()
    got = read_export(tmp_path / "export")
    lines = [x for x in got["digests"] if x.get("spectrum") == r["trace"]]
    assert len(lines) == 1 and lines[0]["i"] == 0
    assert lines[0]["digest"] == inst.spectrum_digest(r["trace"]) and set(lines[0]["digest"]) == {"t", "dist", "z_abs", "i"}
    assert {x["spectrum"] for x in got["digests"]} == set(inst.spectra) and len(inst.spectra) == 2
