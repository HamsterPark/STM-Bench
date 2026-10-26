"""The desktop console's window (stmbench.gui.app), run offscreen and driven the way a screen agent
drives it: by clicks and keystrokes at widget positions, never by calling a slot. The window must lay
out every action of the table where it can be reached without scrolling, run the same instrument an MCP
agent gets (the same words, refusals included), keep the sealed mode's seed out of every widget and
file, and never wait for a click to be answered before taking the next."""
from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ["QT_QPA_PLATFORM"] = "offscreen"                 # never a visible window

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QCloseEvent, QImage  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (QAbstractButton, QApplication, QComboBox, QLabel, QLineEdit, QListWidget,  # noqa: E402
                               QPlainTextEdit, QPushButton, QTableWidget, QTabBar, QTextBrowser, QWidget)

from stmbench.gui import layout, shots, style  # noqa: E402
from stmbench.gui.app import MainWindow  # noqa: E402
from stmbench.gui.panels import token  # noqa: E402
from stmbench.gui.session import GuiSession, make_private_dir, read_spec  # noqa: E402
from stmbench.instrument import ACTIONS  # noqa: E402
from stmbench.instrument.export import read_export  # noqa: E402
from stmbench.render import pixel_to_scan_nm  # noqa: E402

P4 = "P4_atom_positioning_cu111"
#: a seed whose digits cannot turn up in a record by chance
SEED = 913275
SMALL = {"act.scan.center_x_nm": "0", "act.scan.center_y_nm": "0", "act.scan.size_nm": "8", "act.scan.pixels": "32"}


@pytest.fixture(scope="module")
def app():
    """The application with the console's look; the look is taken off again afterwards, because the
    application is shared with every other Qt test in the process (the viewer widgets' tests pin pixel
    sizes that depend on the default font)."""
    a = QApplication.instance() or QApplication(["test"])
    before = (a.font(), a.palette(), a.styleSheet(), a.style().objectName())
    style.apply(a, 1.0)
    yield a
    style.SCALE = 1.0
    a.setStyleSheet(before[2])
    a.setStyle(before[3])
    a.setPalette(before[1])
    a.setFont(before[0])


@pytest.fixture
def open_window(app, tmp_path):
    """A factory of windows on 1920x1053, shut down afterwards (their worker threads must end)."""
    made: list[MainWindow] = []

    def make(lang: str = "zh", seed: int = 0, sealed: bool = False, name: str = "run") -> MainWindow:
        if sealed:
            sess = GuiSession(P4, seed, run_dir=make_private_dir(tmp_path / "private"), export_dir=tmp_path / "export",
                              lang=lang)
        else:
            sess = GuiSession(P4, seed, run_dir=tmp_path / name, export_dir=tmp_path / name / "export", lang=lang)
        win = MainWindow(sess, lang=lang, sealed=sealed)
        win.resize(*shots.WINDOW)
        win.show()
        made.append(win)
        shots.settle(win)
        return win

    yield make
    for w in made:
        w.shutdown(finish=False)
        w.deleteLater()
    app.processEvents()


def open_right(win: MainWindow, key: str) -> None:
    bar = win.right.tabBar()
    shots.click(win, bar, bar.tabRect(("operate", "spectroscopy").index(key)).center())


def open_left(win: MainWindow, key: str) -> None:
    bar = win.sidebar.tabBar()
    shots.click(win, bar, bar.tabRect(win.sidebar.KEYS.index(key)).center())


def run_button(win: MainWindow, name: str) -> None:
    shots.click(win, shots.widget(win, name))


def scan_small(win: MainWindow) -> str:
    shots.fill(win, SMALL)
    run_button(win, "run.scan")
    shots.settle(win)
    return win.viewed[1]


def records(tmp_path: Path, name: str = "run") -> list[dict]:
    return read_export(tmp_path / name / "export")["records"]


def acts(recs: list[dict]) -> list[str]:
    return [r["name"] for r in recs if r["kind"] == "act"]


# ── the layout ──
@pytest.mark.parametrize("lang", ["zh", "en"])
def test_every_action_has_its_controls_at_the_agreed_names_and_each_can_be_clicked(open_window, lang):
    win = open_window(lang)
    for tab, groups in layout.TABS:
        open_right(win, tab)
        for g in groups:
            for line in g.lines:
                for c in line:
                    spec = ACTIONS[c.action]
                    params = {p.name: p for p in spec.params}
                    if isinstance(c, layout.F):
                        p = params[c.param]
                        if p.kind == "boolean":
                            names = [f"act.{c.action}.{c.param}.yes", f"act.{c.action}.{c.param}.no"]
                        elif p.choices:
                            names = [f"act.{c.action}.{c.param}.{token(ch)}" for ch in p.choices]
                        else:
                            names = [f"act.{c.action}.{c.param}"]
                    elif isinstance(c, layout.Run):
                        names = [f"run.{c.action}"]
                    elif isinstance(c, layout.OnOff):
                        names = [f"run.{c.action}.on", f"run.{c.action}.off", f"act.{c.action}.on"]
                    else:
                        names = [f"run.{c.action}.{token(ch)}" for ch in params[c.param].choices] \
                                + [f"act.{c.action}.{c.param}"]
                    for n in names:
                        w = shots.widget(win, n)
                        shots.hit(win, w)                       # visible, inside the window, not covered
                        if isinstance(w, (QPushButton, QLineEdit)):
                            assert w.height() >= 32, (n, w.height())
    # every parameter of every action is addressed
    for action, spec in ACTIONS.items():
        for p in spec.params:
            assert win.findChild(QWidget, f"act.{action}.{p.name}") is not None, (action, p.name)


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_the_operate_and_spectroscopy_tabs_show_everything_without_scrolling_at_1920x1053(open_window, lang):
    win = open_window(lang)
    assert (win.width(), win.height()) == shots.WINDOW
    for tab, panel in (("operate", win.operate), ("spectroscopy", win.spectroscopy)):
        open_right(win, tab)
        win.wait_idle()
        assert not panel.verticalScrollBar().isVisible() and not panel.horizontalScrollBar().isVisible(), tab
        vp = panel.viewport().rect()
        for w in panel.widget().findChildren(QWidget):
            if w.isVisible() and (w.objectName().startswith(("act.", "run.", "reply.", "lbl.", "group."))):
                inside = w.rect().translated(w.mapTo(panel.viewport(), w.rect().topLeft()))
                assert vp.contains(inside), (tab, w.objectName(), inside, vp)
    # no part of the window asks for more than the screen gives, so Qt never widens it
    assert win.minimumSizeHint().width() <= shots.WINDOW[0]
    open_right(win, "operate")                                   # the tab with every action P4 uses is the first one
    assert win.right.currentIndex() == 0


def test_every_control_is_big_enough_to_hit_and_text_is_large(open_window):
    win = open_window("zh")
    for tab in ("operate", "spectroscopy"):
        open_right(win, tab)
        for w in win.right.currentWidget().findChildren(QWidget):
            n = w.objectName()
            if isinstance(w, (QPushButton, QLineEdit)) and n.startswith(("act.", "run.")):
                assert w.height() >= 32, (n, w.height())
    for w in win.findChildren(QLineEdit):
        assert w.font().pixelSize() >= 17, w.objectName()
    for w in win.findChildren(QLabel):
        if w.objectName().startswith(("lbl.", "reply.")):
            assert w.font().pixelSize() >= 16 or w.property("role") == "label", w.objectName()


def test_ui_scale_multiplies_every_size(app):
    try:
        style.apply(app, 1.5)
        assert style.px(10) == 15 and style.sans_font(18).pixelSize() == 27
    finally:
        style.apply(app, 1.0)
    assert style.px(10) == 10


# ── operating the instrument ──
def test_a_scan_started_by_a_click_shows_its_frame(open_window, tmp_path):
    win = open_window("en")
    name = scan_small(win)
    assert win.viewed == ("frame", name) and name.endswith(".sxm")
    fv = win.center.frame_view
    assert fv.image_rect().width() > 100
    assert f"✓ scan → {name}" in win.status.plain_text()
    assert shots.widget(win, "reply.scan").full_text().startswith(f"✓ scan → {name}")
    assert shots.widget(win, "data.frames").count() == 1 and name in shots.widget(win, "data.frames").item(0).text()
    assert shots.widget(win, "thumb.0").isVisible() and shots.widget(win, "thumb.0").name == name
    assert "Instrument time 0:00:38 / 1:30:00" in win.status.plain_text()     # 32 lines, two ways, 0.586 s each
    recs = records(tmp_path)
    scan = next(r for r in recs if r["kind"] == "act")
    assert scan["name"] == "scan" and scan["params"]["pixels"] == 32 and scan["params"]["size_nm"] == 8.0
    assert scan["params"]["direction"] == "down" and scan["result"]["frame"] == name       # the selector's visible state
    assert {"status", "list_data", "frame"} <= {r["what"] for r in recs if r["kind"] == "observe"}   # the reads are on record


def test_set_bias_set_setpoint_and_move_tip_clicked_back_to_back_run_in_that_order(open_window, tmp_path):
    win = open_window("en")
    shots.fill(win, {"act.set_bias.bias_v": "0.005", "act.set_setpoint.current_pa": "100000",
                     "act.move_tip.x_nm": "1", "act.move_tip.y_nm": "0", "act.move_tip.speed_nm_s": "1"})
    for name in ("run.set_bias", "run.set_setpoint", "run.move_tip"):
        b = shots.widget(win, name)
        shots.click(win, b)                                      # no waiting between the clicks
        assert b.isEnabled()                                     # a busy instrument does not disable a button
    assert not win.is_idle() or win.wait_idle()
    shots.settle(win)
    assert acts(records(tmp_path)) == ["set_bias", "set_setpoint", "move_tip"]
    text = shots.widget(win, "log.view").toPlainText()
    assert text.index("move_tip") < text.index("set_setpoint") < text.index("set_bias")      # newest first
    assert "Bias 5.0 mV" in win.status.plain_text() and "Setpoint 100 nA" in win.status.plain_text()


def test_a_refusal_shows_the_instruments_words_everywhere(open_window, tmp_path):
    win = open_window("en")
    shots.fill(win, {"act.move_tip.x_nm": "1000000", "act.move_tip.y_nm": "0"})
    run_button(win, "run.move_tip")
    shots.settle(win)
    words = "refused: move_tip: outside the piezo range"
    assert win.status.result.full_text() == "✗ " + words
    assert win.status.result.property("tone") == "bad"
    assert shots.widget(win, "reply.tip").full_text() == "✗ " + words and shots.widget(win, "reply.tip").property("tone") == "bad"
    assert "move_tip: outside the piezo range" in shots.widget(win, "log.view").toPlainText()
    assert not acts(records(tmp_path))                          # nothing happened on the instrument
    # an empty required field sends nothing, a field that is not a number is sent as typed: both are refused,
    # and the instrument puts a refusal it makes before acting on record
    run_button(win, "run.scan")
    shots.settle(win)
    assert win.status.result.full_text() == "✗ refused: scan: center_x_nm is required"
    assert [r["kind"] for r in records(tmp_path) if r["kind"] != "observe"] == ["refused"]
    shots.fill(win, {"act.set_bias.bias_v": "abc"})
    run_button(win, "run.set_bias")
    shots.settle(win)
    assert win.status.result.full_text() == "✗ refused: set_bias: bias_v must be a number"
    sent = [r for r in records(tmp_path) if r["kind"] == "refused"][-1]
    assert sent["args"] == {"bias_v": "abc"}


def test_buttons_selectors_and_on_off_send_what_they_show(open_window, tmp_path):
    win = open_window("en")
    shots.click(win, shots.widget(win, "act.scan.direction.up"))
    shots.click(win, shots.widget(win, "act.pulse.z_hold.no"))
    shots.fill(win, {"act.pulse.bias_v": "1.0", **SMALL})
    run_button(win, "run.scan")
    run_button(win, "run.pulse")
    run_button(win, "run.feedback.off")
    run_button(win, "run.set_current_range.1")
    shots.fill(win, {"act.drift_comp.vx_nm_per_min": "1.5"})
    run_button(win, "run.drift_comp.on")
    run_button(win, "run.withdraw")
    run_button(win, "run.coarse_move")                           # no direction chosen: the instrument says so
    shots.settle(win)
    by = {r["name"]: r for r in records(tmp_path) if r["kind"] == "act"}
    assert by["scan"]["params"]["direction"] == "up" and by["pulse"]["params"]["z_hold"] is False
    assert by["pulse"]["params"]["bias_v"] == 1.0 and by["feedback"]["params"] == {"on": False}
    assert by["set_current_range"]["params"] == {"full_scale_na": 1.0}
    assert by["drift_comp"]["params"]["on"] is True and by["drift_comp"]["params"]["vx_nm_per_min"] == 1.5
    assert "withdraw" in by
    assert win.status.result.full_text() == "✗ refused: coarse_move: direction is required"
    plain = win.status.plain_text()
    assert "Z feedback OFF" in plain and "Range 1 nA" in plain and "Withdrawn yes" in plain


def test_enter_in_a_field_runs_its_groups_main_action_and_values_stay(open_window, tmp_path):
    win = open_window("en")
    f = shots.widget(win, "act.set_bias.bias_v")
    shots.type_into(win, f, "0.02")
    QTest.keyClick(f, Qt.Key.Key_Return)
    shots.settle(win)
    assert acts(records(tmp_path)) == ["set_bias"] and f.text() == "0.02"
    open_right(win, "spectroscopy")
    a = shots.widget(win, "act.lockin.amplitude_mv")
    shots.type_into(win, a, "7")
    QTest.keyClick(a, Qt.Key.Key_Return)                         # Enter in the amplitude field: lock-in on
    shots.settle(win)
    last = [r for r in records(tmp_path) if r["kind"] == "act"][-1]
    assert last["name"] == "lockin" and last["params"] == {"on": True, "amplitude_mv": 7.0}


def test_clicking_a_text_field_selects_its_whole_content(open_window):
    win = open_window("en")
    f = shots.widget(win, "act.set_bias.bias_v")
    shots.type_into(win, f, "0.5")
    shots.click(win, shots.widget(win, "act.set_setpoint.current_pa"))
    shots.click(win, f)
    assert f.selectedText() == "0.5"
    QTest.keyClicks(f, "0.7")
    assert f.text() == "0.7"                                    # typing replaced it


def test_the_task_the_manual_and_the_claims_are_what_every_agent_is_told(open_window):
    win = open_window("en")
    task, manual = win.session.task_text(), win.session.manual()
    assert shots.widget(win, "task.text").toPlainText() == task and "target_site" in task and "评分规则" in task
    assert shots.widget(win, "manual.text").toPlainText() == manual and "move_tip" in manual
    assert win.windowTitle() == f"STM-Bench console · {P4}"


def test_the_status_strip_reads_in_units_and_counts_instrument_time(open_window):
    win = open_window("en")
    plain = win.status.plain_text()
    for want in ("Bias 50.0 mV", "Setpoint 200 pA", "Z feedback ON", "Range 10 nA", "Withdrawn no", "Lock-in OFF",
                 "0:00:00 / 1:30:00", "Actions 0", P4, "READY"):
        assert want in plain, want
    assert "e-" not in plain
    scan_small(win)
    assert "Actions 1" in win.status.plain_text() and win.status.bar.value() > 0
    zh = open_window("zh", name="run_zh")
    assert "偏压 Bias 50.0 mV" in zh.status.plain_text() and "就绪 READY" in zh.status.plain_text()


# ── the report and the end ──
def test_a_report_and_the_end_through_the_report_tab(open_window, tmp_path):
    win = open_window("en")
    name = scan_small(win)
    open_left(win, "report")
    rep = win.sidebar.report
    run_button(win, "report.submit")                             # nothing filled in: the instrument says what is missing
    shots.settle(win)
    assert rep.reply.text() == "✗ refused: report: value is required" and rep.reply.property("tone") == "bad"
    shots.fill(win, {"report.value": "1"})
    run_button(win, "report.submit")                             # no claim chosen
    shots.settle(win)
    assert "unknown claim ''" in rep.reply.text()
    assert not rep.fields["unit"].text()                         # nothing is pre-filled before a claim is chosen
    shots.click(win, shots.widget(win, "report.claim.target_site"))
    assert rep.fields["unit"].text() == "nm"                     # except the unit
    shots.fill(win, {"report.value": "1.5", "report.x_nm": "1.5", "report.y_nm": "-0.25", "report.frame": name})
    run_button(win, "report.viewed")                             # appends the name of what the viewer shows
    assert rep.fields["evidence"].text() == name
    run_button(win, "report.viewed")
    assert rep.fields["evidence"].text() == name                 # not twice
    run_button(win, "report.submit")
    shots.settle(win)
    assert rep.reply.text().startswith("✓ recorded target_site = 1.5") and rep.reply.property("tone") == "ok"
    assert json.loads((tmp_path / "run" / "export" / "report.json").read_text(encoding="utf-8"))["claims"]["target_site"]["value"] == 1.5
    sent = [r for r in records(tmp_path) if r["kind"] == "report"][-1]["args"]
    assert sent["claim_id"] == "target_site" and sent["frame"] == name and sent["evidence"] == [name]
    # finishing takes two clicks and no dialog
    fb, ask = shots.widget(win, "finish.begin"), win.sidebar.report.finish_ask
    assert not ask.isVisible()
    run_button(win, "finish.begin")
    assert ask.isVisible() and not fb.isVisible() and not win.session.finished
    run_button(win, "finish.cancel")
    assert fb.isVisible() and not ask.isVisible() and not win.session.finished
    run_button(win, "finish.begin")
    run_button(win, "finish.confirm")
    shots.settle(win)
    assert win.session.finished and "The episode is over; the record is written." in shots.widget(win, "finish.message").text()
    assert "FINISHED" in win.status.plain_text() and win.status.result.full_text().endswith("the record is written.")
    # every action is refused afterwards, in the instrument's words, and the window shows no score
    open_right(win, "operate")
    shots.fill(win, {"act.wait.seconds": "1"})
    run_button(win, "run.wait")
    shots.settle(win)
    assert win.status.result.full_text() == "✗ refused: the episode is over"
    assert json.loads((tmp_path / "run" / "episode.json").read_text(encoding="utf-8"))["success"] in (True, False)
    assert read_export(tmp_path / "run" / "export")["session"]["finished"]
    texts = _all_text(win)
    assert not any(w in texts.lower() for w in ("verdict", "success", "truth", "partial"))


def test_closing_the_dev_window_finishes_the_episode(open_window, tmp_path):
    win = open_window("en")
    scan_small(win)
    assert not (tmp_path / "run" / "episode.json").exists()
    assert win.close()
    assert (tmp_path / "run" / "episode.json").is_file() and win.session.finished
    assert read_export(tmp_path / "run" / "export")["session"]["finished"]


# ── the viewer ──
def test_markers_are_read_off_the_frame_and_copied_into_forms_by_their_buttons(open_window):
    win = open_window("en")
    name = scan_small(win)
    fv = win.center.frame_view
    for col, row in ((10.0, 12.0), (20.0, 8.0)):
        shots.click(win, fv, fv.pixel_to_widget(col, row).toPoint())
    markers = fv.markers()
    assert len(markers) == 2 and fv.selected_marker() == markers[-1]
    geom = {"cx_nm": 0.0, "cy_nm": 0.0, "w_nm": 8.0, "h_nm": 8.0, "angle_deg": 0.0, "nx": 32, "ny": 32}
    for m, (col, row) in zip(markers, ((10.0, 12.0), (20.0, 8.0))):     # the clicked pixel, through the judge's mapping
        x, y = pixel_to_scan_nm(geom, col, row)
        assert m["x_nm"] == pytest.approx(x, abs=0.02) and m["y_nm"] == pytest.approx(y, abs=0.02)
        assert m["frame"] == name
    table = shots.widget(win, "markers.table")
    assert table.rowCount() == 2 and table.item(1, 1).text() == f"{markers[1]['x_nm']:.3f}"
    assert not table.item(0, 5).text()                           # the first marker has no previous one
    assert float(table.item(1, 5).text()) == pytest.approx(math.hypot(markers[1]["x_nm"] - markers[0]["x_nm"],
                                                                       markers[1]["y_nm"] - markers[0]["y_nm"]), abs=1e-3)
    shots.click(win, table.viewport(), table.visualItemRect(table.item(0, 0)).center())
    assert fv.selected_marker() == markers[0]
    run_button(win, "markers.tip")
    assert shots.widget(win, "act.move_tip.x_nm").text() == f"{markers[0]['x_nm']:.3f}"
    assert shots.widget(win, "act.move_tip.y_nm").text() == f"{markers[0]['y_nm']:.3f}"
    run_button(win, "markers.scan")
    assert shots.widget(win, "act.scan.center_x_nm").text() == f"{markers[0]['x_nm']:.3f}"
    open_right(win, "spectroscopy")
    run_button(win, "markers.tip")
    assert win.right.currentWidget() is win.operate              # the tab that holds the field it filled
    run_button(win, "markers.report")
    assert win.sidebar.tab_key() == "report"
    f = win.sidebar.report.fields
    assert (f["x_nm"].text(), f["y_nm"].text(), f["frame"].text()) == (
        f"{markers[0]['x_nm']:.3f}", f"{markers[0]['y_nm']:.3f}", name)
    assert not f["value"].text()                                 # nothing else is filled for the reader
    # no marker, no copy: the window says so rather than guessing
    run_button(win, "view.clear_markers")
    assert not fv.markers() and table.rowCount() == 0
    run_button(win, "markers.tip")
    assert "No marker selected" in win.status.result.full_text()


def test_a_profile_is_drawn_by_dragging_with_the_profile_tool(open_window):
    win = open_window("en")
    scan_small(win)
    fv = win.center.frame_view
    run_button(win, "view.tool.profile")
    shots.drag(win, fv, fv.pixel_to_widget(5, 16).toPoint(), fv.pixel_to_widget(26, 16).toPoint())
    assert not fv.markers()                                      # a drag with the profile tool places no marker
    plot = shots.widget(win, "view.profile")
    assert plot.isVisible()
    run_button(win, "view.tool.marker")


def test_frames_are_reached_by_the_data_list_and_the_thumbnails_and_channels_by_buttons(open_window, tmp_path):
    win = open_window("en")
    a = scan_small(win)
    shots.fill(win, {"act.scan.size_nm": "6"})
    run_button(win, "run.scan")
    shots.settle(win)
    b = win.viewed[1]
    assert a != b
    data = shots.widget(win, "data.frames")
    assert data.count() == 2 and b in data.item(0).text() and a in data.item(1).text()       # newest first
    open_left(win, "data")
    shots.click(win, data.viewport(), data.visualItemRect(data.item(1)).center())
    shots.settle(win)
    assert win.viewed == ("frame", a)
    shots.click(win, shots.widget(win, "thumb.0"))
    shots.settle(win)
    assert win.viewed == ("frame", b)
    run_button(win, "view.direction.backward")
    run_button(win, "view.channel.current")
    shots.settle(win)
    assert (b, "current", "backward") in win._cache and win.center.current_channel() == "current"
    assert not shots.widget(win, "view.channel.didv").isEnabled()      # this frame has no lock-in map
    run_button(win, "view.flatten.plane")
    run_button(win, "view.flatten.none")
    run_button(win, "view.flatten.line")
    observed = [r["args"] for r in records(tmp_path) if r["kind"] == "observe" and r["what"] == "frame"]
    assert {"name": b, "channel": "current", "direction": "backward"} in observed       # the read is in the ledger


def test_a_spectrum_is_plotted_with_selectable_columns(open_window):
    win = open_window("en")
    open_right(win, "spectroscopy")
    shots.fill(win, {"act.sts.v_start": "-0.3", "act.sts.v_end": "0.3", "act.sts.points": "32"})
    run_button(win, "run.sts")
    shots.settle(win)
    kind, name = win.viewed
    assert kind == "spectrum" and name.endswith(".dat") and win.center.stack.currentIndex() == 1
    assert win.center.current_spectrum_axes() == ("v", "didv")
    run_button(win, "view.spec_y.i")
    assert win.center.current_spectrum_axes() == ("v", "i")
    assert shots.widget(win, "data.spectra").count() == 1 and name in shots.widget(win, "data.spectra").item(0).text()
    assert win.status.result.full_text().startswith(f"✓ sts → {name}")


def test_the_log_lists_every_call_newest_first_and_marks_a_call_that_has_not_run(open_window):
    win = open_window("en")
    log = win.sidebar.log
    log.add(999, "wait", {"seconds": 5}, 12.0)                  # queued, never run
    assert "queued" in log.plain() and "wait seconds=5" in log.plain()
    log.done(999, False, "refused: x", 12.0)
    shots.fill(win, {"act.set_bias.bias_v": "0.01"})
    run_button(win, "run.set_bias")
    shots.settle(win)
    text = log.plain()
    assert text.index("set_bias") < text.index("wait") and "bias_v=0.01" in text and "✓" in text and "✗ refused: x" in text


# ── the sealed mode ──
def _all_text(win: QWidget) -> str:
    out = [win.windowTitle()]
    for w in win.findChildren(QWidget):
        out.append(w.toolTip())
        out.append(w.statusTip())
        if isinstance(w, QLabel):
            out.append(w.text())
        if isinstance(w, QAbstractButton):
            out.append(w.text())
        if isinstance(w, QLineEdit):
            out += [w.text(), w.placeholderText()]
        if isinstance(w, (QPlainTextEdit, QTextBrowser)):
            out.append(w.toPlainText())
        if isinstance(w, QComboBox):
            out += [w.itemText(i) for i in range(w.count())]
        if isinstance(w, QListWidget):
            out += [w.item(i).text() for i in range(w.count())]
        if isinstance(w, QTableWidget):
            out += [w.item(r, c).text() for r in range(w.rowCount()) for c in range(w.columnCount()) if w.item(r, c)]
        if isinstance(w, QTabBar):
            out += [w.tabText(i) for i in range(w.count())]
    return "\n".join(out)


def test_in_the_sealed_mode_the_seed_is_in_no_widget_title_or_file_and_close_does_nothing(open_window, tmp_path, capfd):
    spec = io.StringIO(json.dumps({"scenario": P4, "seed": SEED}) + "\n")
    scenario, seed = read_spec(spec)
    assert (scenario, seed) == (P4, SEED)
    win = open_window("zh", seed=SEED, sealed=True)
    name = scan_small(win)
    shots.fill(win, {"act.move_tip.x_nm": "1000000", "act.move_tip.y_nm": "0"})
    run_button(win, "run.move_tip")
    open_right(win, "spectroscopy")
    shots.fill(win, {"act.sts.v_start": "-0.2", "act.sts.v_end": "0.2", "act.sts.points": "16"})
    run_button(win, "run.sts")
    shots.settle(win)
    open_left(win, "report")
    shots.click(win, shots.widget(win, "report.claim.target_site"))
    shots.fill(win, {"report.value": "1", "report.x_nm": "1", "report.y_nm": "0", "report.frame": name})
    run_button(win, "report.viewed")
    run_button(win, "report.submit")
    shots.settle(win)
    for tab in win.sidebar.KEYS:
        open_left(win, tab)
        assert str(SEED) not in _all_text(win), tab
    assert str(SEED) not in win.windowTitle() and "seed" not in _all_text(win).lower()
    for p in sorted((tmp_path / "export").rglob("*")):
        if p.is_file():
            assert str(SEED) not in p.read_text(encoding="utf-8"), p.name
    out, err = capfd.readouterr()
    assert str(SEED) not in out + err
    # the close button does nothing but say so in the status strip
    ev = QCloseEvent()
    QApplication.sendEvent(win, ev)
    assert not ev.isAccepted() and win.isVisible() and not win.session.finished
    assert "stays open" in win.status.result.full_text()
    win.showMaximized()
    assert win.isMaximized()
    if os.name == "posix":
        assert (tmp_path / "private").stat().st_mode & 0o777 == 0o700


# ── the screenshots ──
def test_the_screenshot_script_renders_every_state_at_both_sizes(app, tmp_path):
    saved = shots.render_states(tmp_path / "png", "en", work=tmp_path / "work")
    names = sorted(p.name for p in saved)
    assert len(names) == 16 and any(n.startswith("a_started") for n in names) and any(n.startswith("e_refusal") for n in names)
    for p in saved:
        img = QImage(str(p))
        assert (img.width(), img.height()) == (shots.DOWNSCALED if p.stem.endswith("1280x720") else shots.SCREEN), p.name
    app.processEvents()
