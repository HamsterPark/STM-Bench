"""Plain-text copies of the saved data (stmbench.gui.datacopy, ``--data-dir``): P2 (a dispersion fit) and P5 (an inversion of
Delta-f(z) curves) cannot be done through the console alone, which only shows values. The copies are exactly what the window
can show, in its units, with a whitelisted header; nothing of the drift, the tip's state or anything private."""
from __future__ import annotations

import math
import os
import re
import stat
import time
from pathlib import Path

import numpy as np
import pytest

from stmbench.gui import datacopy as dc
from stmbench.gui.session import GuiSession
from stmbench.render import pixel_to_scan_nm

P5, P3 = "P5_sader_jarvis_qplus", "P3_corral_cu111"
#: a seed whose digits cannot turn up in a record by chance
SEED = 913275


def parse(path: Path) -> tuple[dict[str, str], list[str]]:
    """The ``# key: value`` header and the rest of a file, line by line."""
    head, rest = {}, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            key, _, value = line[2:].partition(": ")
            head[key] = value
        else:
            rest.append(line)
    return head, rest


@pytest.fixture
def copies(tmp_path):
    """A P5 session with a copier, a scan (32 px, lock-in off), an STS, a Z spectrum and a move trace done."""
    sess = GuiSession(P5, SEED, run_dir=tmp_path / "run", export_dir=tmp_path / "export", lang="en")
    out = tmp_path / "data"
    copier = dc.DataCopier(sess, out)
    replies = {}
    replies["scan"] = sess.act("scan", {"center_x_nm": 1.0, "center_y_nm": -2.0, "size_nm": 8.0, "pixels": 32, "angle_deg": 15.0})
    replies["sts"] = sess.act("sts", {"v_start": -0.2, "v_end": 0.2, "points": 16})
    replies["zspec"] = sess.act("zspec", {"sweep_nm": 0.5, "points": 20})
    replies["move"] = sess.act("move_tip", {"x_nm": 1.0, "y_nm": 1.0})
    junctions = {replies["sts"]["spectrum"]: {"v_start": -0.2, "v_end": 0.2, "bias_v": 0.02, "setpoint_pa": 50.0},
                 replies["zspec"]["spectrum"]: {"sweep_nm": 0.5, "bias_v": 0.02, "setpoint_pa": 50.0}}
    for r in replies.values():
        copier.save(r, junctions)
    assert copier.wait()
    yield sess, out, replies, copier
    copier.close()


def test_one_file_per_spectrum_or_trace_and_one_per_frame_channel_and_direction(copies):
    sess, out, replies, copier = copies
    assert copier.errors == []
    names = sorted(p.name for p in out.iterdir())
    base_sts = replies["sts"]["spectrum"].rsplit(".", 1)[0]
    base_z = replies["zspec"]["spectrum"].rsplit(".", 1)[0]
    base_move = replies["move"]["trace"].rsplit(".", 1)[0]
    frame = replies["scan"]["frame"].rsplit(".", 1)[0]
    assert names == sorted([f"{base_sts}.csv", f"{base_z}.csv", f"{base_move}.csv", f"{frame}_z_forward.csv",
                            f"{frame}_z_backward.csv", f"{frame}_current_forward.csv", f"{frame}_current_backward.csv"])


def test_a_spectrum_file_says_what_the_window_says_and_holds_its_columns_in_the_windows_units(copies):
    sess, out, replies, _ = copies
    name = replies["sts"]["spectrum"]
    head, rest = parse(out / (name.rsplit(".", 1)[0] + ".csv"))
    assert head["source"] == name and head["kind"] == "sts" and head["points"] == "16"
    assert head["sweep"] == "sweep -0.200 to 0.200 V" and float(head["v_start_v"]) == -0.2 and float(head["setpoint_pa"]) == 50.0
    spec = sess.spectrum(name)
    assert float(head["tip_x_nm"]) == pytest.approx(spec["tip_nm"][0]) and float(head["time_s"]) == pytest.approx(spec["t_s"])
    assert rest[0] == "V,I (pA),dI/dV (nS)" and len(rest) == 17
    rows = np.array([[float(x) for x in r.split(",")] for r in rest[1:]])
    assert np.allclose(rows[:, 0], spec["v"]) and np.allclose(rows[:, 1], np.asarray(spec["i"]) * 1e12, rtol=1e-8)
    assert np.allclose(rows[:, 2], np.asarray(spec["didv"]) * 1e9, rtol=1e-8)


def test_a_z_spectrum_has_its_columns_and_a_move_trace_its_path(copies):
    sess, out, replies, _ = copies
    z = replies["zspec"]["spectrum"]
    head, rest = parse(out / (z.rsplit(".", 1)[0] + ".csv"))
    assert head["kind"] == "zspec" and float(head["sweep_nm"]) == 0.5
    assert rest[0] == "I (pA),z rel (pm),z (nm),Δf (Hz),amplitude (pm)" or "Δf (Hz)" in rest[0]
    assert len(rest) == 21
    t = replies["move"]["trace"]
    head, rest = parse(out / (t.rsplit(".", 1)[0] + ".csv"))
    assert head["source"] == t and head["kind"] == "move" and float(head["to_x_nm"]) == 1.0 and head["feedback"] in ("on", "off")
    assert float(head["speed_nm_s"]) == 293.0 and rest[0].startswith("I (pA)") or "distance (nm)" in rest[0]
    n = int(head["points"])
    assert len(rest) == n + 1


def test_a_frame_file_has_the_matrix_as_acquired_in_the_windows_units_and_its_pixel_mapping_agrees_with_the_judges(copies):
    sess, out, replies, _ = copies
    name = replies["scan"]["frame"]
    base = name.rsplit(".", 1)[0]
    for channel, unit, factor in (("z", "pm", 1e12), ("current", "pA", 1e12)):
        for direction in ("forward", "backward"):
            head, rest = parse(out / f"{base}_{channel}_{direction}.csv")
            frame = sess.frame(name, channel, direction)
            assert head["source"] == name and head["channel"] == channel and head["direction"] == direction
            assert head["unit"] == unit and head["values"].startswith("as acquired (no flattening)")
            assert (head["rows"], head["columns"], head["pixels"]) == ("32", "32", "32")
            mat = np.array([[float(x) for x in r.split(",")] for r in rest])
            assert mat.shape == (32, 32) and np.allclose(mat, frame["array"] * factor, rtol=1e-8)
    head, _ = parse(out / f"{base}_z_forward.csv")
    assert (float(head["centre_x_nm"]), float(head["centre_y_nm"]), float(head["size_nm"]), float(head["angle_deg"])) == (1.0, -2.0, 8.0, 15.0)
    assert float(head["line_time_s"]) > 0 and "start_time_s" in head and "start_time" in head
    # the stated mapping, evaluated from the header's own numbers, is pixel_to_scan_nm's
    cx, cy, size, a = (float(head[k]) for k in ("centre_x_nm", "centre_y_nm", "size_nm", "angle_deg"))
    nx = ny = int(head["pixels"])
    geom = {"cx_nm": cx, "cy_nm": cy, "w_nm": size, "h_nm": size * ny / nx, "angle_deg": a, "nx": nx, "ny": ny}
    assert head["pixel_mapping"] == dc.PIXEL_MAPPING and "row 0 is the top" in head["pixel_mapping"]
    for col, row in ((0, 0), (31, 0), (0, 31), (31, 31), (10, 20)):
        u = -size / 2 + (col + 0.5) * size / nx
        v = (size * ny / nx) / 2 - (row + 0.5) * (size * ny / nx) / ny
        r = math.radians(a)
        mine = (cx + math.cos(r) * u + math.sin(r) * v, cy - math.sin(r) * u + math.cos(r) * v)
        assert mine == pytest.approx(pixel_to_scan_nm(geom, col, row), abs=1e-12)
    assert pixel_to_scan_nm(geom, 0, 0)[1] > pixel_to_scan_nm(geom, 0, 31)[1] - 1e-9 or a != 0     # row 0 is the high-v edge


def test_only_whitelisted_header_keys_are_written_and_nothing_private_is_in_any_byte(copies, tmp_path):
    sess, out, _, _ = copies
    files = sorted(out.iterdir())
    assert len(files) == 7
    forbidden = [b"row_offsets", b"apex", b"drift", b"seed", b"truth", b"hidden", b"verdict", b"tip_state", b"episode",
                 b"tolerance", str(SEED).encode()]
    for p in files:
        raw = p.read_bytes()
        for bad in forbidden:
            assert bad not in raw.lower() if bad.isalpha() or b"_" in bad else bad not in raw, (p.name, bad)
        head, _ = parse(p)
        assert set(head) <= dc.HEADER_KEYS, (p.name, set(head) - dc.HEADER_KEYS)
    with pytest.raises(KeyError):
        dc._header([("row_offsets_nm", "x")], dc.FRAME_KEYS)               # the writer itself refuses a key off the list


def test_the_viewers_columns_and_units_are_the_copies(copies):
    from stmbench.gui.center import SPECTRUM_COLUMNS
    from stmbench.gui.widgets.frame_view import _DISPLAY
    assert dc.COLUMNS == SPECTRUM_COLUMNS and dc.FRAME_UNITS == _DISPLAY


# ── writing ──
def test_a_file_appears_whole_with_mode_0644_and_no_temporary_file_is_left(tmp_path):
    d = tmp_path / "new" / "data"
    assert dc.write_atomically(d, "a.csv", "# source: a\nx\n") is True
    assert (d / "a.csv").read_text() == "# source: a\nx\n"
    assert [p.name for p in d.iterdir()] == ["a.csv"]
    if os.name == "posix":
        assert stat.S_IMODE((d / "a.csv").stat().st_mode) == 0o644 and stat.S_IMODE(d.stat().st_mode) == 0o755
    assert dc.write_atomically(d, "a.csv", "changed") is False                   # saved data does not change
    assert (d / "a.csv").read_text() == "# source: a\nx\n"


def test_a_symlink_is_never_followed(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("keep")
    try:
        (tmp_path / "link").symlink_to(real, target_is_directory=True)
        (real / "evil.csv").symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("no symlinks here")
    with pytest.raises(OSError):
        dc.write_atomically(tmp_path / "link", "a.csv", "x")                       # the directory is a symlink
    with pytest.raises(OSError):
        dc.write_atomically(real, "evil.csv", "overwritten")                       # the target is a symlink
    assert secret.read_text() == "keep"


def test_a_name_that_could_leave_the_directory_is_not_written():
    assert dc.base_name("../../x y.dat") == "x y" and dc.base_name("a/b.sxm") == "b"      # only the file name is used
    for bad in ("", ".hidden", "a\x00b", " lead", "x" * 200):
        with pytest.raises(ValueError):
            dc.base_name(bad + ".dat")


# ── the option ──
def test_the_data_dir_option_is_refused_inside_a_record(tmp_path):
    import argparse
    from stmbench import cli
    base = dict(scenario="P3_corral_cu111", seed=0, lang="en", ui_scale=1.0, sealed=False, private_dir=None, export_dir=None,
                ready_file=None)
    for out, data in ((tmp_path / "run", tmp_path / "run"), (tmp_path / "run", tmp_path / "run" / "copies"),
                      (tmp_path / "run" / "export", tmp_path / "run")):
        with pytest.raises(SystemExit, match="data-dir"):
            cli.cmd_gui(argparse.Namespace(**base, out=str(out), data_dir=str(data)))
    sealed = dict(base, scenario=None, seed=None, sealed=True, out=None, private_dir=str(tmp_path / "p"), export_dir=str(tmp_path / "e"))
    with pytest.raises(SystemExit, match="data-dir"):
        cli.cmd_gui(argparse.Namespace(**sealed, data_dir=str(tmp_path / "e" / "csv")))


def test_writing_the_copies_leaves_the_export_gradable_and_replay_consistent(copies, tmp_path):
    """The copies read the session (observations: no instrument time, no random number), so the export, which is the graded
    record, still replays to the same measurements."""
    from stmbench.instrument.grade import grade_export
    sess, out, replies, copier = copies
    assert copier.wait() and len(list(out.iterdir())) == 7
    sess.finish()
    res = grade_export(tmp_path / "export", P5, SEED)
    assert res["replay"]["consistent"] and res["finished_in_episode"] is True
    assert not any(p.parent == tmp_path / "export" for p in out.iterdir())           # the copies are not in the export
