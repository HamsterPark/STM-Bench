"""Track A manifests on a synthetic index (never the 105k-row parquet):

* §3.6 de-duplication keeps the non-copy file and neighbour gaps are recomputed after it;
* frame references are relative to the raw mirror root (``rel_path`` + context), the absolute
  ``path`` is meta; frames outside the root are dropped and counted, a wrong root refuses;
* T1 takes only restart-save-ON months, only frames with a full head, context strictly
  earlier, label = the operator stopped THIS frame, and declares no next-frame input;
* T2 puts the next frame's box in the current frame's coordinates (scan angle undone) and
  carries the frame height so a non-square frame scales ``v`` by its own height;
* T3 stay / relocate / long_stop, the day's last frame has no label;
* T4 material from COMMENT or file name, COMMENT itself absent, working points collapsed;
* the schema rejects tampering; the split is a pure function of the group;
* the CLI builds from a directory of the three inputs with ``--limit`` and ``--raw-root``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from stmbench.trackA import manifests as M  # noqa: E402
from stmbench.trackA import schema as S  # noqa: E402
from stmbench.trackA.render import resolve_path  # noqa: E402
from stmbench.trackA.score import normalize_manifest  # noqa: E402

RAW = "F:/mirror/corpus"


def _frame(path, inst, t, *, size_nm=20.0, h_nm=None, x_nm=0.0, y_nm=0.0, acq_frac=1.0, ny=256, nx=256, angle=0.0,
           bias=0.1, comment="", t_fwd=0.5, size_bytes=1000, temp="4.5"):
    return dict(path=path, inst=inst, t=t, size_nm=size_nm, h_nm=size_nm if h_nm is None else h_nm, x_nm=x_nm,
                y_nm=y_nm, acq_frac=acq_frac, ny=ny, nx=nx, angle=angle, bias=bias, comment=comment, t_fwd=t_fwd,
                size_bytes=size_bytes, temp=temp)


def _p(inst, day, name):
    return f"{RAW}\\{inst} SPM data\\{day.replace('-', chr(92))}\\{name}"


def _tables(rows, auto):
    """The three inputs (index parquet, sequence parquet, autosave csv) from ``_frame`` rows."""
    ix = pd.DataFrame({
        "path": [r["path"] for r in rows],
        "nx": [float(r["nx"]) for r in rows], "ny": [float(r["ny"]) for r in rows],
        "range_x_m": [r["size_nm"] * 1e-9 for r in rows], "range_y_m": [r["h_nm"] * 1e-9 for r in rows],
        "off_x_m": [r["x_nm"] * 1e-9 for r in rows], "off_y_m": [r["y_nm"] * 1e-9 for r in rows],
        "angle_deg": [float(r["angle"]) for r in rows], "bias_v": [r["bias"] for r in rows],
        "acq_frac": [r["acq_frac"] for r in rows], "acq_rows": [round(r["acq_frac"] * r["ny"]) for r in rows],
        "t_fwd_s": [r["t_fwd"] for r in rows], "comment": [r["comment"] for r in rows],
        "rec_temp": [r["temp"] for r in rows], "z_ctrl_on": ["log Current"] * len(rows),
        "setpoint": ["1"] * len(rows),                                       # the index's mislabelled ON flag
        "size": [r["size_bytes"] for r in rows],
    })
    seq = pd.DataFrame({
        "path": [r["path"] for r in rows], "group": [r["inst"] for r in rows],
        "year": [int(r["t"][:4]) for r in rows], "t": pd.to_datetime([r["t"] for r in rows]),
        "early": [r["acq_frac"] < 0.1 for r in rows], "aborted": [r["acq_frac"] < 1 for r in rows],
    })
    return ix, seq, auto


AUTO = pd.DataFrame({"group": ["corpus:Rig_B", "corpus:Rig_A"], "ym": ["2024-10", "2020-07"], "n": [10, 5],
                     "autosave": ["ON", "off"]})


def synthetic():
    """A Rig_B day with restart-save ON and a Rig_A day with it OFF."""
    d1, d2 = "2024-10-15", "2020-07-01"
    rows = [
        # Rig_B: overview → zoom (T2), glance run (T1), relocation, long stop, rotated zoom, day end
        _frame(_p("Rig_B", d1, "Ag0001.sxm"), "Rig_B", f"{d1} 09:00:00", size_nm=100, comment="Ag(111)"),
        _frame(_p("Rig_B", d1, "Ag0001b.sxm"), "Rig_B", f"{d1} 09:00:00", size_nm=100, comment="Ag(111)",
               size_bytes=1002),                                                    # double-saved twin (§3.6)
        _frame(_p("Rig_B", d1, "Ag0002.sxm"), "Rig_B", f"{d1} 09:05:00", size_nm=20, x_nm=10, y_nm=-20),
        _frame(_p("Rig_B", d1, "Ag0003.sxm"), "Rig_B", f"{d1} 09:12:00", size_nm=20, x_nm=10, y_nm=-20, acq_frac=0.02),
        _frame(_p("Rig_B", d1, "Ag0004.sxm"), "Rig_B", f"{d1} 09:12:20", size_nm=20, x_nm=10, y_nm=-20, acq_frac=0.05),
        _frame(_p("Rig_B", d1, "Ag0005.sxm"), "Rig_B", f"{d1} 09:12:50", size_nm=20, x_nm=10, y_nm=-20),
        _frame(_p("Rig_B", d1 + " - 副本", "Ag0005.sxm"), "Rig_B", f"{d1} 09:12:50", size_nm=20, x_nm=10, y_nm=-20,
               size_bytes=1001),                                                    # copy-folder twin
        _frame(_p("Rig_B", d1, "Ag0006.sxm"), "Rig_B", f"{d1} 09:20:00", size_nm=20, x_nm=500, y_nm=500),
        _frame(_p("Rig_B", d1, "Ag0007.sxm"), "Rig_B", f"{d1} 10:30:00", size_nm=20, x_nm=500, y_nm=500, angle=90),
        _frame(_p("Rig_B", d1, "Ag0008.sxm"), "Rig_B", f"{d1} 10:35:00", size_nm=5, x_nm=505, y_nm=500, angle=90),
        # Rig_A, regime OFF: a glance run that must NOT become T1 samples; T4 material rows
        _frame(_p("Rig_A", d2, "20200701-001.sxm"), "Rig_A", f"{d2} 10:00:00", size_nm=50, comment="Cu(111)", bias=0.5),
        _frame(_p("Rig_A", d2, "20200701-002.sxm"), "Rig_A", f"{d2} 10:03:00", size_nm=50, comment="Cu(111)", bias=0.5,
               acq_frac=0.06),
        _frame(_p("Rig_A", d2, "20200701-003.sxm"), "Rig_A", f"{d2} 10:04:00", size_nm=50, comment="Cu(111)", bias=0.5),
        _frame(_p("Rig_A", d2, "20200701-004.sxm"), "Rig_A", f"{d2} 11:00:00", size_nm=20, x_nm=300, comment="B/Ag(111)",
               bias=-0.2),
        _frame(_p("Rig_A", d2, "Ag111-mica0007.sxm"), "Rig_A", f"{d2} 12:00:00", size_nm=20, x_nm=600, comment="",
               bias=-0.2),
        _frame(_p("Rig_A", d2, "20200701-005.sxm"), "Rig_A", f"{d2} 12:30:00", size_nm=20, x_nm=900,
               comment="20 mV 5 pA -40pm", bias=0.02),                             # parameters, no material
        _frame(_p("Rig_A", "1904-01-01", "junk.sxm"), "Rig_A", "1904-01-01 08:00:00"),  # REC_DATE missing
    ]
    return _tables(rows, AUTO)


@pytest.fixture(scope="module")
def frames():
    ix, seq, auto = synthetic()
    return M.prepare_frames(ix, seq, auto, raw_root=RAW)


def _row(df, name):
    hit = df[df["path"].str.endswith(name)]
    assert len(hit) == 1, name
    return hit.iloc[0]


# ── frames ───────────────────────────────────────────────────────────────────
def test_dedup_keeps_the_original_and_recomputes_neighbours(frames):
    df, info = frames
    assert info["n_duplicates_dropped"] == 2
    assert not df["path"].str.contains("副本").any()
    assert not df["path"].str.endswith("Ag0001b.sxm").any()          # lexically later twin dropped
    assert not (df["year"] < 2000).any()                                # 1904 sentinel dropped
    f4 = _row(df, "Ag0004.sxm")
    assert f4["dt_next_s"] == 30.0 and f4["move_next_frac"] == 0.0       # the twin no longer sits in between
    assert f4["run_id"] == _row(df, "Ag0003.sxm")["run_id"] == _row(df, "Ag0005.sxm")["run_id"]
    assert _row(df, "Ag0002.sxm")["run_id"] != f4["run_id"]              # 420 s gap breaks the run
    assert _row(df, "Ag0005.sxm")["run_end_kind"] == "relocate" and bool(_row(df, "Ag0005.sxm")["run_last_complete"])
    assert _row(df, "Ag0008.sxm")["run_end_kind"] == "day_end"
    assert _row(df, "Ag0002.sxm")["run_end_kind"] == "pause"
    assert dict(zip(df["path"].map(lambda p: p[-11:]), df["abort_kind"]))[_row(df, "Ag0003.sxm")["path"][-11:]] == "glance"
    assert df["rel_path"].str.startswith("Rig_").all()                  # raw root stripped, '/' separators
    assert info["raw_root"] == RAW and info["n_outside_raw_root"] == 0 and info["index_common_root"] == RAW
    assert (df["height_nm"] == df["size_nm"]).all()                       # square synthetic frames
    assert df["group_key"].map(S.split_of).equals(df["split"])
    assert (df.loc[df["instrument"] == "Rig_B", "autosave_regime"] == "on").all()
    assert (df.loc[df["instrument"] == "Rig_A", "autosave_regime"] == "off").all()


def test_frames_outside_the_raw_root_are_dropped_and_a_wrong_root_refuses():
    ix, seq, auto = synthetic()
    with pytest.raises(ValueError, match="raw mirror root"):
        M.prepare_frames(ix, seq, auto, raw_root="Q:/elsewhere")
    # a root that covers one instrument only: the other instrument's frames cannot be resolved → dropped, counted
    df, info = M.prepare_frames(ix, seq, auto, raw_root=RAW + "/Rig_B SPM data")
    assert set(df["instrument"]) == {"Rig_B"} and info["n_outside_raw_root"] == 6     # the six dated Rig_A frames
    assert df["rel_path"].str.startswith("2024/10/15/").all() and info["raw_root"] == RAW + "/Rig_B SPM data"
    # case / separator differences on the root do not matter
    df2, _ = M.prepare_frames(ix, seq, auto, raw_root=RAW.upper().replace("/", "\\") + "\\")
    assert len(df2) == len(M.prepare_frames(ix, seq, auto, raw_root=RAW)[0])


# ── T1 ───────────────────────────────────────────────────────────────────────
def test_t1_samples_regime_head_rows_context_and_label(frames):
    df, _ = frames
    t1 = M.build_t1(df)
    assert set(t1["instrument"]) == {"Rig_B"}                              # Rig_A's run: regime off
    names = t1["path"].map(lambda p: Path(p).name).tolist()
    assert "Ag0003.sxm" not in names                                      # 5 acquired rows < head_rows
    assert set(names) == {"Ag0001.sxm", "Ag0002.sxm", "Ag0004.sxm", "Ag0005.sxm", "Ag0006.sxm", "Ag0007.sxm", "Ag0008.sxm"}
    r4 = _row(t1, "Ag0004.sxm")
    r5 = _row(t1, "Ag0005.sxm")
    assert bool(r4["stopped"]) and not bool(r5["stopped"])                 # the label is about THIS frame
    assert r4["step_idx"] == 1 and r5["step_idx"] == 2 and r4["head_rows"] == S.T1_HEAD_ROWS
    assert r4["rel_path"] == "Rig_B SPM data/2024/10/15/Ag0004.sxm"
    assert json.loads(r4["context_paths_json"]) == ["Rig_B SPM data/2024/10/15/Ag0003.sxm"]   # relative, like rel_path
    assert json.loads(r5["context_dt_s_json"]) == [50.0, 30.0]           # oldest first, seconds to this start
    assert (t1["head_rows"] == S.T1_HEAD_ROWS).all()
    assert S.validate(t1, S.T1) == []


def test_t1_spec_declares_nothing_about_the_future_as_input():
    for spec in S.SPECS.values():
        assert not (set(spec.input_columns) & spec.leak_forbidden)
    assert "step_idx" in S.T1.input_columns                               # the negative-control baseline needs it
    assert "rel_path" in S.T1.input_columns and S.T1.column("path").role == S.ROLE_META   # consumers open rel_path
    for leak in ("acq_frac", "acq_rows", "dt_next_s", "move_next_frac", "run_len", "run_end_kind", "next_path"):
        assert S.T1.column(leak).role == S.ROLE_META
    with pytest.raises(ValueError, match="leak-forbidden"):
        S.ManifestSpec("X", "x", S.T1.cols + (S.Col("dt_next_s2", "float", S.ROLE_INPUT, ""),),
                       frozenset({"dt_next_s2"}), "")


def test_manifest_references_resolve_under_the_raw_mirror_root(frames, monkeypatch):
    df, _ = frames
    monkeypatch.setenv("STM_BENCH_RAW", RAW)
    r5 = normalize_manifest(M.build_t1(df))
    r5 = next(r for r in r5 if r["rel_path"].endswith("Ag0005.sxm"))
    assert r5["input_paths"] == ["Rig_B SPM data/2024/10/15/Ag0003.sxm", "Rig_B SPM data/2024/10/15/Ag0004.sxm",
                                 "Rig_B SPM data/2024/10/15/Ag0005.sxm"]
    assert [resolve_path(p) for p in r5["input_paths"]] == [
        Path(RAW) / "Rig_B SPM data/2024/10/15" / n for n in ("Ag0003.sxm", "Ag0004.sxm", "Ag0005.sxm")]
    assert Path(r5["path"]) == Path(RAW) / "Rig_B SPM data/2024/10/15/Ag0005.sxm"   # the absolute meta agrees


# ── T2 ───────────────────────────────────────────────────────────────────────
def test_t2_box_in_current_frame_coordinates(frames):
    df, _ = frames
    t2 = M.build_t2(df)
    names = set(t2["path"].map(lambda p: Path(p).name))
    assert names == {"Ag0001.sxm", "Ag0007.sxm"}
    a = _row(t2, "Ag0001.sxm")                                           # 100 nm → 20 nm at (+10, −20) nm
    assert a["next_u"] == pytest.approx(0.10) and a["next_v"] == pytest.approx(-0.20)
    assert a["next_w_rel"] == pytest.approx(0.2) and bool(a["zoom_in"])
    b = _row(t2, "Ag0007.sxm")                                           # frame rotated 90°: +x lab → −v frame
    assert b["next_u"] == pytest.approx(0.0, abs=1e-9) and b["next_v"] == pytest.approx(-0.25)
    assert b["next_w_rel"] == pytest.approx(0.25) and b["next_angle_delta_deg"] == 0.0
    assert json.loads(b["context_paths_json"])[-1].endswith("Ag0006.sxm")
    assert all(not p.startswith(RAW) for p in json.loads(b["context_paths_json"]))   # relative
    assert len(json.loads(b["context_paths_json"])) == S.CONTEXT_FRAMES
    assert S.validate(t2, S.T2) == []


def test_t2_non_square_frame_scales_v_by_its_own_height():
    d = "2024-10-16"
    rows = [
        _frame(_p("Rig_B", d, "Wd0001.sxm"), "Rig_B", f"{d} 09:00:00", size_nm=100, h_nm=50, nx=256, ny=128, comment="Ag(111)"),
        _frame(_p("Rig_B", d, "Wd0002.sxm"), "Rig_B", f"{d} 09:05:00", size_nm=20, h_nm=10, nx=128, ny=64, x_nm=10, y_nm=-10),
    ]
    ix, seq, auto = _tables(rows, AUTO)
    df, _ = M.prepare_frames(ix, seq, auto, raw_root=RAW)
    t2 = M.build_t2(df)
    assert len(t2) == 1
    a = t2.iloc[0]
    assert a["size_nm"] == pytest.approx(100.0) and a["height_nm"] == pytest.approx(50.0)
    assert a["nx"] == 256 and a["ny"] == 128
    assert a["next_u"] == pytest.approx(0.10)                              # 10 nm of a 100 nm width
    assert a["next_v"] == pytest.approx(-0.20)                             # −10 nm of a 50 nm HEIGHT, not −0.10
    assert a["next_w_rel"] == pytest.approx(0.2) and a["next_h_rel"] == pytest.approx(0.2)
    assert S.validate(t2, S.T2) == []
    n = normalize_manifest(t2)[0]
    assert n["label"] == pytest.approx([10.0, -10.0])                     # back in nm along each axis
    assert n["range_nm"] == pytest.approx(100.0) and n["height_nm"] == pytest.approx(50.0)
    assert n["box_nm"] == pytest.approx(20.0) and n["box_h_nm"] == pytest.approx(10.0)
    assert n["input_paths"] == ["Rig_B SPM data/2024/10/16/Wd0001.sxm"]
    # a square frame of the same width gives the same label from a different next_v
    sq = [dict(rows[0], h_nm=100, ny=256), dict(rows[1], h_nm=20, ny=128)]
    ix, seq, auto = _tables(sq, AUTO)
    t2s = M.build_t2(M.prepare_frames(ix, seq, auto, raw_root=RAW)[0])
    assert t2s.iloc[0]["next_v"] == pytest.approx(-0.10)
    assert normalize_manifest(t2s)[0]["label"] == pytest.approx([10.0, -10.0])


# ── T3 ───────────────────────────────────────────────────────────────────────
def test_t3_next_action_classes(frames):
    df, _ = frames
    t3 = M.build_t3(df)
    act = dict(zip(t3["path"].map(lambda p: Path(p).name), zip(t3["next_action"], t3["next_action_fine"])))
    assert act["Ag0001.sxm"] == ("stay", "zoom")
    assert act["Ag0002.sxm"] == ("stay", "restart_same")
    assert act["Ag0005.sxm"] == ("relocate", "relocate")
    assert act["Ag0006.sxm"] == ("long_stop", "long_stop")
    assert "Ag0008.sxm" not in act and "20200701-005.sxm" not in act     # last frame of the day: no next
    assert bool(_row(t3, "Ag0005.sxm")["relocate"]) and not bool(_row(t3, "Ag0002.sxm")["relocate"])
    assert S.validate(t3, S.T3) == []
    assert M.next_action(100.0, 0.5, 1.0) == ("stay", "pan")
    # three-class is what the scorer reads by default; the binary fold on request
    labs = {r["rel_path"].rsplit("/", 1)[-1]: r["label"] for r in normalize_manifest(t3)}
    assert labs["Ag0006.sxm"] == "long_stop" and labs["Ag0005.sxm"] == "relocate" and labs["Ag0002.sxm"] == "stay"
    labs2 = {r["rel_path"].rsplit("/", 1)[-1]: r["label"] for r in normalize_manifest(t3, t3_binary=True)}
    assert labs2["Ag0006.sxm"] == "relocate" and labs2["Ag0002.sxm"] == "stay"


# ── T4 ───────────────────────────────────────────────────────────────────────
def test_t4_material_without_comment_and_collapsed_working_points(frames):
    df, _ = frames
    t4 = M.build_t4(df)
    assert "comment" not in t4.columns and "path" not in t4.columns and "rel_path" not in t4.columns
    cu = t4[t4["material"] == "Cu(111)"]
    assert len(cu) == 1 and cu.iloc[0]["n_frames"] == 3 and cu.iloc[0]["n_complete"] == 2
    assert cu.iloc[0]["bias_v"] == 0.5 and cu.iloc[0]["speed_nm_s"] == pytest.approx(100.0)
    assert cu.iloc[0]["genre"] == "clean_metal" and cu.iloc[0]["rec_temp_k"] == 4.5
    ag = t4[t4["material"] == "Ag(111)"].sort_values("first_path")
    assert set(ag["adsorbate"]) == {"B", ""}
    assert set(ag["genre"]) == {"molecule_on_metal", "clean_metal"}
    assert ag["first_path"].str.endswith("Ag111-mica0007.sxm").any()  # material from the file name
    assert not t4["first_path"].str.endswith("20200701-005.sxm").any()   # parameters-only comment ⇒ unknown
    assert t4["setpoint_a"].isna().all()                                  # headers not read
    assert S.validate(t4, S.T4) == []


def test_parse_material_cases():
    assert M.parse_material("Br-TEB/Cu(111) Anneal at 400K") == ("Cu(111)", "Br-TEB", "molecule_on_metal")
    assert M.parse_material("B Ag(100) LHe STM W tip") == ("Ag(100)", "B", "molecule_on_metal")
    assert M.parse_material("BCN/Cu#2") == ("Cu", "BCN", "molecule_on_metal")
    assert M.parse_material("Si/HOPG grow at RT") == ("HOPG", "Si", "layered")
    assert M.parse_material("New Ag(111)/mica") == ("Ag(111)", "", "clean_metal")
    assert M.parse_material("Si(111)") == ("Si(111)", "", "semiconductor")
    assert M.parse_material("LHe2 STM W tip") == ("unknown", "", "unknown")
    assert M.parse_material("", "Ag111-mica0003.sxm") == ("Ag(111)", "", "clean_metal")


def test_read_setpoint_from_raw_header(tmp_path):
    hdr = (b":NANONIS_VERSION:\n2\n:BIAS:\n            1.000E-1\n:Z-CONTROLLER:\n"
           b"\tName\ton\tSetpoint\tP-gain\tI-gain\tT-const\n"
           b"\tlog Current\t1\t3.000E-11 A\t3.000E-12 m\t5.000E-8 m/s\t6.000E-5 s\n"
           b":COMMENT:\n20 mV 5 pA\n:SCANIT_END:\n\n\x1a\x04")
    p = tmp_path / "x.sxm"
    p.write_bytes(hdr + b"\x00" * 64)
    assert M.read_setpoint_a(str(p)) == pytest.approx(3.0e-11)
    (tmp_path / "y.sxm").write_bytes(b":BIAS:\n1\n\x1a\x04")
    assert M.read_setpoint_a(str(tmp_path / "y.sxm")) is None
    assert M.read_setpoint_a(str(tmp_path / "missing.sxm")) is None


# ── schema / split ───────────────────────────────────────────────────────────
def test_validate_rejects_tampering(frames):
    df, _ = frames
    t3 = M.build_t3(df)
    assert S.validate(t3.drop(columns=["relocate"]), S.T3) == ["missing columns: ['relocate']"]
    bad = t3.copy()
    bad.loc[bad.index[0], "split"] = "test" if bad.iloc[0]["split"] != "test" else "train"
    assert any("straddle" in p or "disagree" in p for p in S.validate(bad, S.T3))
    bad = t3.copy()
    bad["next_action"] = "elsewhere"
    assert any("outside" in p for p in S.validate(bad, S.T3))
    bad = t3.copy()
    bad["extra"] = 1
    assert any("undeclared" in p for p in S.validate(bad, S.T3))
    bad = t3.copy()
    bad.loc[:, "sample_id"] = bad["sample_id"].iloc[0]
    assert any("duplicates" in p for p in S.validate(bad, S.T3))
    with pytest.raises(ValueError, match="T3 manifest invalid"):
        S.assert_valid(bad, S.T3)


def test_split_is_deterministic_and_close_to_the_fractions():
    keys = [f"Rig_B:2024-{m:02d}-{d:02d}" for m in range(1, 13) for d in range(1, 29)]
    a = [S.split_of(k) for k in keys]
    assert a == [S.split_of(k) for k in keys]
    counts = {s: a.count(s) / len(a) for s in S.SPLITS}
    for s in S.SPLITS:
        assert abs(counts[s] - S.SPLIT_FRACS[s]) < 0.12
    assert S.split_of("Rig_B:2024-10-15") in S.SPLITS


# ── CLI ──────────────────────────────────────────────────────────────────────
def test_cli_build_with_limit(tmp_path):
    ix, seq, auto = synthetic()
    idx = tmp_path / "index"
    idx.mkdir()
    ix.to_parquet(idx / M.INDEX_FILE, index=False)
    seq.to_parquet(idx / M.SEQUENCE_FILE, index=False)
    auto.to_csv(idx / M.AUTOSAVE_FILE, index=False)
    out = tmp_path / "trackA"
    assert M.main(["build", "--index-dir", str(idx), "--out", str(out), "--limit", "50", "--raw-root", RAW]) == 0
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert set(meta["tasks"]) == {"T1", "T2", "T3", "T4"}
    for task, spec in S.SPECS.items():
        df = pd.read_parquet(out / f"{task}.parquet")
        assert S.validate(df, spec) == [], task
        assert len(df) == meta["tasks"][task]["rows"] > 0
    assert meta["frames"]["n_duplicates_dropped"] == 2
    assert meta["frames"]["raw_root"] == RAW and meta["frames"]["n_outside_raw_root"] == 0
    assert meta["thresholds"]["T1_HEAD_ROWS"] == S.T1_HEAD_ROWS
    assert "rel_path" in meta["tasks"]["T1"]["inputs"] and "path" not in meta["tasks"]["T1"]["inputs"]
    # --limit keeps the first N rows per instrument: 3 per instrument leaves nothing for T1's run
    out2 = tmp_path / "small"
    assert M.main(["build", "--index-dir", str(idx), "--out", str(out2), "--limit", "3", "--tasks", "T3",
                   "--raw-root", RAW]) == 0
    t3 = pd.read_parquet(out2 / "T3.parquet")
    assert len(t3) <= 4 and (out2 / "T3.parquet").exists() and not (out2 / "T1.parquet").exists()
    # the wrong root is refused, not silently worked around
    with pytest.raises(ValueError, match="raw mirror root"):
        M.main(["build", "--index-dir", str(idx), "--out", str(tmp_path / "bad"), "--raw-root", "Q:/elsewhere"])
