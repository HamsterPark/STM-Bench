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
from stmbench.instrument.core import SCENARIO_DIR, validate  # noqa: E402
from stmbench.instrument.export import read_export  # noqa: E402
from stmbench.render import pixel_to_scan_nm  # noqa: E402

P4 = "P4_atom_positioning_cu111"
P5 = "P5_sader_jarvis_qplus"
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

    def make(lang: str = "zh", seed: int = 0, sealed: bool = False, name: str = "run", scenario: str = P4) -> MainWindow:
        if sealed:
            sess = GuiSession(scenario, seed, run_dir=make_private_dir(tmp_path / "private"),
                              export_dir=tmp_path / "export", lang=lang)
        else:
            sess = GuiSession(scenario, seed, run_dir=tmp_path / name, export_dir=tmp_path / name / "export", lang=lang)
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
    reply = shots.widget(win, "reply.tip")
    assert reply.full_text() == "✗ " + words and reply.property("tone") == "bad"
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
    en, zh = open_window("en"), open_window("zh", name="run_zh")
    task, manual = en.session.task_text(), en.session.manual()
    assert shots.widget(en, "task.text").toPlainText() == task and manual == zh.session.manual()
    assert "[Scoring rules v1]" in task and "ReportResult" in task and "评分规则" not in task       # the English card
    assert "【评分规则 v1】" in shots.widget(zh, "task.text").toPlainText()                          # the Chinese one
    assert shots.widget(en, "manual.text").toPlainText() == manual and "move_tip" in manual
    assert en.windowTitle() == f"STM-Bench console · {P4}"
    # the report form shows the rule card's line about the claim in the window's language
    def claim_texts(w: MainWindow) -> str:
        return " ".join(label.text() for label in w.sidebar.report.findChildren(QLabel))
    assert "write the same x_nm in the value field" in claim_texts(en) and "value 写同一个 x_nm" in claim_texts(zh)


def test_the_report_form_names_its_fields_as_reportresult_and_the_rule_card_name_them(open_window):
    for lang in ("en", "zh"):
        win = open_window(lang, name=f"run_{lang}")
        rep = win.sidebar.report
        assert rep.title.text().endswith("(ReportResult)") and ("Report a result" in rep.title.text())
        for key in ("value", "x_nm", "y_nm", "frame", "evidence", "unit", "sigma", "note"):
            assert shots.widget(win, f"report.label.{key}").text().split(" ")[0] == key         # the exact name comes first
            assert shots.widget(win, f"report.{key}") is rep.fields[key]


def test_the_instruments_facts_head_the_manual_tab(open_window, tmp_path):
    win = open_window("en")
    info = win.session.info()
    facts = win.sidebar.facts.facts()
    rx, ry = info["piezo_xy_range_nm"]
    assert facts["Piezo XY range"] == f"{rx:g} × {ry:g} nm" and facts["Piezo Z range"] == f"{info['piezo_z_range_nm']:g} nm"
    assert facts["Bias range"] == "-10 … 10 V" and facts["Material"] == "Cu(111)" and facts["Time budget"] == "1:30:00"
    assert facts["Current ranges (full scale)"] == "10000, 1000, 100, 10, 1, 0.1 nA"
    assert facts["Temperature"] == f"{info['temperature_k']:g} K" and not [k for k in facts if k.startswith("qPlus")]
    boot = [r["what"] for r in records(tmp_path) if r["kind"] == "observe"][:3]
    assert boot == ["status", "info", "list_data"]                  # read once, at the start, as an observation
    open_left(win, "manual")
    table, text = shots.widget(win, "facts.table"), shots.widget(win, "manual.text")
    shots.hit(win, table)
    assert table.mapTo(win, table.rect().topLeft()).y() < text.mapTo(win, text.rect().topLeft()).y()      # above the manual
    lab, val = win.sidebar.facts.rows[0]
    assert lab.font().pixelSize() >= 16 and val.font().pixelSize() >= 16 and len(win.sidebar.facts.rows) == 7
    assert all(shots.hit(win, w) for pair in win.sidebar.facts.rows for w in pair)       # every cell on screen, none scrolled off
    assert table.mapTo(win, table.rect().bottomLeft()).y() < text.mapTo(win, text.rect().topLeft()).y() + 1
    zh = open_window("zh", name="run_zh")
    assert "压电 XY 量程 Piezo XY range" in " ".join(zh.sidebar.facts.facts())


def test_a_qplus_rigs_sensor_constants_are_among_the_facts(open_window):
    win = open_window("en", scenario=P5)
    q = win.session.info()["qplus"]
    facts = win.sidebar.facts.facts()
    assert facts["qPlus k"] == f"{q['k_n_per_m']:.9g} N/m" and facts["qPlus f0"] == f"{q['f0_hz']:.9g} Hz"
    assert facts["qPlus Q"] == f"{q['q']:.9g}"


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
    folded = json.loads((tmp_path / "run" / "export" / "report.json").read_text(encoding="utf-8"))
    assert folded["claims"]["target_site"]["value"] == 1.5
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
    texts = _all_text(win, own_only=True)
    assert not any(w in texts.lower() for w in ("verdict", "success", "truth", "partial", "score"))


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
    name = scan_small(win)
    fv = win.center.frame_view
    got: list[dict] = []
    fv.profile_changed.connect(got.append)
    run_button(win, "view.tool.profile")
    shots.drag(win, fv, fv.pixel_to_widget(5, 16).toPoint(), fv.pixel_to_widget(26, 16).toPoint())
    assert not fv.markers()                                      # a drag with the profile tool places no marker
    p = got[-1]
    geom = {"cx_nm": 0.0, "cy_nm": 0.0, "w_nm": 8.0, "h_nm": 8.0, "angle_deg": 0.0, "nx": 32, "ny": 32}
    (x0, y0), (x1, y1) = pixel_to_scan_nm(geom, 5, 16), pixel_to_scan_nm(geom, 26, 16)
    assert p["frame"] == name and p["unit"] == "pm" and len(p["values"]) == len(p["s_nm"]) >= 20
    assert (p["x0_nm"], p["y0_nm"], p["x1_nm"], p["y1_nm"]) == pytest.approx((x0, y0, x1, y1), abs=0.02)
    assert p["s_nm"][-1] == pytest.approx(math.hypot(x1 - x0, y1 - y0), abs=0.02)
    run_button(win, "view.clear_markers")                        # clears the line as well: the small plot is emptied
    assert got[-1] == {}
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
def _all_text(win: QWidget, own_only: bool = False) -> str:
    """Every text a widget of the window shows. ``own_only`` leaves out the task and the manual, which are the
    scenario's and the instrument's words (the rule card says "successful" of an overrun)."""
    out = [win.windowTitle()]
    for w in win.findChildren(QWidget):
        if own_only and w.objectName() in ("task.text", "manual.text"):
            continue
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


# ── the entry point ──
def test_run_refuses_what_it_cannot_open_without_repeating_the_seed(app, tmp_path, capsys):
    from stmbench.gui.app import run
    junk = json.dumps({"scenario": P4, "seed": SEED, "extra": 1})
    assert run(sealed=True, private_dir=tmp_path / "p", export_dir=tmp_path / "e", stdin=io.StringIO(junk)) == 2
    assert run(sealed=True, private_dir=tmp_path / "p", export_dir=tmp_path / "e", stdin=io.StringIO("not json")) == 2
    taken = tmp_path / "taken"
    taken.mkdir()
    (taken / "x").write_text("x")
    good = json.dumps({"scenario": P4, "seed": SEED})
    assert run(sealed=True, private_dir=taken, export_dir=tmp_path / "e", stdin=io.StringIO(good)) == 2
    assert run(scenario=P4, seed=SEED, out=taken, lang="en") == 2                # a run directory holds one episode
    err = capsys.readouterr().err
    assert err.count("error:") == 4 and str(SEED) not in err
    assert not (tmp_path / "p").exists() and not (tmp_path / "e").exists()


def test_the_sealed_command_runs_as_a_process_that_stays_open_and_keeps_the_seed_to_itself(tmp_path):
    """The whole path of the virtual machine's launch: the spec on stdin, a window that never closes by
    itself, an export that fills as the program starts, and the seed nowhere it could be read."""
    import subprocess
    import sys
    import time

    export, private = tmp_path / "export", tmp_path / "private"
    argv = [sys.executable, "-m", "stmbench.cli", "gui", "--sealed", "--private-dir", str(private),
            "--export-dir", str(export), "--lang", "en"]
    assert str(SEED) not in " ".join(argv)
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env=env, cwd=Path(__file__).resolve().parents[1])
    try:
        proc.stdin.write(json.dumps({"scenario": P4, "seed": SEED}) + "\n")
        proc.stdin.flush()
        end = time.monotonic() + 120
        seen = False
        while time.monotonic() < end and proc.poll() is None and not seen:
            time.sleep(0.5)
            try:
                seen = {r.get("what") for r in read_export(export)["records"]} >= {"status", "list_data"}
            except (OSError, ValueError, KeyError):
                pass
        assert seen, "the window did not start: " + (proc.stderr.read() if proc.poll() is not None else "")
        time.sleep(1.0)
        assert proc.poll() is None                               # a sealed window stays open until it is killed
    finally:
        proc.kill()
        out, err = proc.communicate(timeout=60)
    assert str(SEED) not in out + err
    for p in export.rglob("*"):
        if p.is_file():
            assert str(SEED) not in p.read_text(encoding="utf-8"), p.name
    assert not (private / "episode.json").exists()               # killed, not finished: nothing judged


# ── every parameter, every action ──
#: one call per action (two for those that switch something), every parameter given a value other than
#: its default; the order keeps the tip in tunnelling until the end
EVERY = [
    ("set_bias", {"bias_v": 0.03}),
    ("set_setpoint", {"current_pa": 60.0}),
    ("set_current_range", {"full_scale_na": 100.0}),
    ("feedback", {"on": False}),
    ("feedback", {"on": True}),
    ("scan", {"center_x_nm": 1.5, "center_y_nm": -2.0, "size_nm": 6.0, "pixels": 24, "angle_deg": 10.0,
              "line_time_s": 0.05, "direction": "up"}),
    ("move_tip", {"x_nm": 0.5, "y_nm": 0.25, "speed_nm_s": 50.0}),
    ("pulse", {"bias_v": 1.2, "width_ms": 20.0, "z_hold": False}),
    ("poke", {"depth_nm": 0.1, "bias_v": 0.05, "dwell_s": 0.2}),
    ("sts", {"v_start": -0.2, "v_end": 0.2, "points": 16, "settle_ms": 2.0, "integration_ms": 3.0,
             "z_offset_nm": 0.1, "backward": False}),
    ("pll", {"on": True, "amplitude_pm": 80.0}),
    ("zspec", {"sweep_nm": 0.1, "points": 16, "z_offset_nm": 0.05, "sweeps": 2, "settle_ms": 2.0,
               "integration_ms": 3.0, "backward": False}),
    ("pll", {"on": False}),
    ("lockin", {"on": True, "amplitude_mv": 5.0}),
    ("lockin", {"on": False}),
    ("drift_comp", {"on": True, "vx_nm_per_min": 0.5, "vy_nm_per_min": -0.5, "vz_pm_per_min": 2.0}),
    ("coarse_move", {"direction": "Y+", "steps": 1}),
    ("wait", {"seconds": 2.0}),
    ("withdraw", {}),
    ("approach", {"timeout_s": 30.0}),
]


def drive(win: MainWindow, action: str, args: dict, held: dict) -> dict:
    """Set the controls the way a reader would, click the action's button, and return what the fields hold
    (a field keeps its value after a run, so a later click sends it again)."""
    spec = ACTIONS[action]
    cells = [c for _, c in layout.cells() if c.action == action]
    onoff = any(isinstance(c, layout.OnOff) for c in cells)
    mine = held.setdefault(action, {})
    for p in spec.params:
        if p.name == "on" or p.name not in args:
            continue
        v = args[p.name]
        choices = any(isinstance(c, layout.Choices) and c.param == p.name for c in cells)
        if p.kind == "boolean":
            shots.click(win, shots.widget(win, f"act.{action}.{p.name}.{'yes' if v else 'no'}"))
        elif choices:
            pass                                                  # the button that runs it is the selector
        elif p.choices is not None:
            shots.click(win, shots.widget(win, f"act.{action}.{p.name}.{token(v)}"))
        else:
            shots.type_into(win, shots.widget(win, f"act.{action}.{p.name}"), str(v))
        mine[p.name] = v
    sent = dict(mine)
    if onoff:
        sent = {"on": args["on"], **sent}
        shots.click(win, shots.widget(win, f"run.{action}.{'on' if args['on'] else 'off'}"))
    elif any(isinstance(c, layout.Choices) for c in cells):
        (c,) = [c for c in cells if isinstance(c, layout.Choices)]
        sent = {c.param: args[c.param]}
        shots.click(win, shots.widget(win, f"run.{action}.{token(args[c.param])}"))
    else:
        shots.click(win, shots.widget(win, f"run.{action}"))
    shots.settle(win)
    return sent


def test_every_parameter_of_every_action_reaches_the_instrument_as_set_on_the_screen(open_window, tmp_path):
    win = open_window("en", scenario=P5)
    assert {a for a, _ in EVERY} == set(ACTIONS)                        # a new action must be added here
    typed = {(a, k) for a, args in EVERY for k in args}
    assert {(a, p.name) for a, s in ACTIONS.items() for p in s.params} <= typed     # and each parameter of it
    held: dict = {}
    for action, args in EVERY:
        tab = "spectroscopy" if action in ("sts", "zspec", "pll", "lockin") else "operate"
        open_right(win, tab)
        sent = drive(win, action, args, held)
        assert win.status.result.full_text().startswith("✓"), (action, win.status.result.full_text())
        done = [r for r in records(tmp_path) if r["kind"] == "act"][-1]
        assert done["name"] == action
        assert done["params"] == validate(ACTIONS[action], sent), action


def test_a_frame_taken_with_the_lock_in_on_offers_the_didv_channel(open_window):
    win = open_window("en")
    open_right(win, "spectroscopy")
    shots.fill(win, {"act.lockin.amplitude_mv": "5"})
    run_button(win, "run.lockin.on")
    shots.settle(win)
    open_right(win, "operate")
    name = scan_small(win)
    assert "Lock-in ON" in win.status.plain_text()
    didv = shots.widget(win, "view.channel.didv")
    assert didv.isEnabled()
    shots.click(win, didv)
    shots.settle(win)
    assert (name, "didv", "forward") in win._cache and win.center.current_channel() == "didv"
    assert "dI/dV" in shots.widget(win, "data.frames").item(0).text()


@pytest.mark.parametrize("scenario", sorted(p.stem for p in SCENARIO_DIR.glob("P*.yaml")))
def test_every_paper_scenario_opens_with_its_reportable_claims_and_a_report_form_that_fits(open_window, scenario):
    win = open_window("zh", scenario=scenario)
    claims = win.session.claims()
    assert claims and set(win.sidebar.report.radios) == {c["id"] for c in claims}
    open_left(win, "report")
    page = win.sidebar.report
    for name in ("report.submit", "finish.begin", "report.value", "report.evidence", "report.note"):
        w = shots.widget(win, name)
        shots.hit(win, w)                                        # reachable without scrolling
        assert w.mapTo(page, w.rect().bottomLeft()).y() <= page.height()
    for r in page.radios.values():
        shots.hit(win, r)


def test_the_window_shrinks_to_fit_a_screen_its_fonts_would_overflow_and_not_otherwise(app):
    from PySide6.QtCore import QSize

    from stmbench.gui.app import fit_scale
    try:
        assert fit_scale(app, "zh", 1.0, QSize(*shots.WINDOW)) == 1.0           # the screen it was tuned for
        small = fit_scale(app, "zh", 1.0, QSize(1500, 900))
        assert 0.7 <= small < 1.0 and style.SCALE == small                       # left applied, for the window to use
        assert fit_scale(app, "zh", small, QSize(1500, 900)) == small           # and it fits: asking again changes nothing
        assert fit_scale(app, "en", 1.3, QSize(2560, 1400)) == 1.3              # a larger scale on a larger screen is kept
    finally:
        style.apply(app, 1.0)


# ── a whole episode ──
def drive_exact(win: MainWindow, action: str, params: dict) -> None:
    """Make every control of the action show what the call holds (clearing a field the call leaves out),
    then click its button."""
    cells = [c for _, c in layout.cells() if c.action == action]
    open_right(win, "spectroscopy" if action in ("sts", "zspec", "pll", "lockin") else "operate")
    for p in ACTIONS[action].params:
        if p.name == "on" or any(isinstance(c, layout.Choices) and c.param == p.name for c in cells):
            continue
        if p.kind == "boolean" or p.choices is not None:
            if p.name in params:
                v = params[p.name]
                suffix = ("yes" if v else "no") if p.kind == "boolean" else token(v)
                shots.click(win, shots.widget(win, f"act.{action}.{p.name}.{suffix}"))
            continue
        w = shots.widget(win, f"act.{action}.{p.name}")
        if p.name in params:
            shots.type_into(win, w, repr(params[p.name]))
        elif w.text():
            shots.click(win, w)
            QTest.keyClick(w, Qt.Key.Key_Delete)
    if any(isinstance(c, layout.OnOff) for c in cells):
        run_button(win, f"run.{action}.{'on' if params['on'] else 'off'}")
    elif any(isinstance(c, layout.Choices) for c in cells):
        (c,) = [c for c in cells if isinstance(c, layout.Choices)]
        run_button(win, f"run.{action}.{token(params[c.param])}")
    else:
        run_button(win, f"run.{action}")


def test_a_whole_p4_episode_played_on_the_screen_measures_and_scores_as_the_same_calls_do(open_window, tmp_path):
    """The calls of the scripted baseline's episode for seed 3 (12 scans, 14 tip moves, pokes, a pulse,
    drift compensation ...), once straight into the instrument and once by clicks and keystrokes: the
    same measurements, the same verdict, the same instrument time."""
    from stmbench.instrument import Instrument

    fixture = json.loads((Path(__file__).parent / "fixtures" / "p4_seed3_baseline_actions.json").read_text(encoding="utf-8"))
    steps, seed = fixture["steps"], fixture["seed"]
    inst = Instrument(P4, seed, out_dir=tmp_path / "direct")
    for s in steps:
        if s["kind"] == "act":
            inst.act(s["name"], **s["params"])
        elif s["kind"] == "report":
            a = dict(s["args"])
            inst.report(a.pop("claim_id"), a.pop("value"), **a)
        else:
            inst.finish()
    win = open_window("zh", seed=seed, name="screen")
    for s in steps:
        if s["kind"] == "act":
            drive_exact(win, s["name"], s["params"])
            shots.settle(win)
            assert win.status.result.full_text().startswith("✓"), (s, win.status.result.full_text())
        elif s["kind"] == "report":
            a = s["args"]
            open_left(win, "report")
            run_button(win, f"report.claim.{a['claim_id']}")
            shots.fill(win, {"report.value": repr(a["value"]), "report.x_nm": repr(a["x_nm"]), "report.y_nm": repr(a["y_nm"]),
                             "report.frame": a["frame"], "report.evidence": ", ".join(a["evidence"])})
            run_button(win, "report.submit")
            shots.settle(win)
            assert win.sidebar.report.reply.text().startswith("✓ recorded target_site")
        else:
            run_button(win, "finish.begin")
            run_button(win, "finish.confirm")
            shots.settle(win)
    assert win.session.finished
    direct, screen = (json.loads((tmp_path / d / "episode.json").read_text(encoding="utf-8")) for d in ("direct", "screen"))
    assert screen["digest"] == direct["digest"] and screen["verdict"] == direct["verdict"]
    assert screen["instrument_time_s"] == direct["instrument_time_s"] and screen["success"] is True


def test_no_tooltip_comes_up_under_the_pointer(open_window):
    """The pointer rests on the control just clicked; a tooltip would cover the next one in the next screenshot."""
    from PySide6.QtCore import QEvent, QPoint
    from PySide6.QtGui import QHelpEvent
    from PySide6.QtWidgets import QToolTip
    win = open_window("en")
    b = shots.widget(win, "run.scan")
    assert not any(w.toolTip() for w in win.findChildren(QWidget))       # the window sets none ...
    b.setToolTip("this would cover the next control")                    # ... and one set anyway is never shown
    QApplication.sendEvent(b, QHelpEvent(QEvent.Type.ToolTip, QPoint(5, 5), b.mapToGlobal(QPoint(5, 5))))
    QApplication.processEvents()
    assert not QToolTip.isVisible()
