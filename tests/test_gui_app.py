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
import re
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ["QT_QPA_PLATFORM"] = "offscreen"                 # never a visible window

from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt  # noqa: E402
from PySide6.QtGui import QCloseEvent, QImage  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (QAbstractButton, QApplication, QComboBox, QLabel, QLineEdit, QListWidget,  # noqa: E402
                               QPlainTextEdit, QPushButton, QTableWidget, QTabBar, QTextBrowser, QWidget)

from stmbench.gui import fmt, layout, shots, style  # noqa: E402
from stmbench.gui.app import MainWindow  # noqa: E402
from stmbench.gui.labels import one  # noqa: E402
from stmbench.gui.panels import token  # noqa: E402
from stmbench.gui.session import GuiSession, make_private_dir, read_spec  # noqa: E402
from stmbench.instrument import ACTIONS  # noqa: E402
from stmbench.instrument.core import SCENARIO_DIR, validate  # noqa: E402
from stmbench.instrument.export import read_export  # noqa: E402
from stmbench.render import pixel_to_scan_nm  # noqa: E402
from stmbench.trackB.rules import RULES_VERSION  # noqa: E402

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


@pytest.fixture(params=["default", "dejavu"])
def fonts(request, app):
    """The window's text in the fonts it was tuned with, and again in DejaVu Sans / Sans Mono: what a Linux
    desktop without anything else (the virtual machine) falls back to, wider and with another line height.
    matplotlib carries the files; without them the second run is skipped. Requested before ``open_window``
    so that the window is made in these fonts."""
    if request.param == "default":
        yield "default"
        return
    mpl = pytest.importorskip("matplotlib")
    from PySide6.QtGui import QFontDatabase
    folder = Path(mpl.get_data_path()) / "fonts" / "ttf"
    names = ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf")
    if not all((folder / n).is_file() for n in names):
        pytest.skip("no DejaVu fonts to test with")
    ids = [QFontDatabase.addApplicationFont(str(folder / n)) for n in names]
    sans, mono = list(style.SANS), list(style.MONO)
    style.SANS[:] = ["DejaVu Sans", *sans]
    style.MONO[:] = ["DejaVu Sans Mono", *mono]
    style.apply(app, 1.0)
    try:
        yield "dejavu"
    finally:
        style.SANS[:] = sans
        style.MONO[:] = mono
        style.apply(app, 1.0)
        for i in ids:
            QFontDatabase.removeApplicationFont(i)


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
    # really delete them now: a window left until the event loop turns stays in the application, and every
    # later change of the style sheet (a fixture that swaps the fonts, fit_scale) restyles all of them
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def open_right(win: MainWindow, key: str) -> None:
    bar = win.right.tabBar()
    shots.click(win, bar, bar.tabRect(("operate", "spectroscopy").index(key)).center())


def open_left(win: MainWindow, key: str) -> None:
    bar = win.sidebar.tabBar()
    shots.click(win, bar, bar.tabRect(win.sidebar.KEYS.index(key)).center())


def run_button(win: MainWindow, name: str) -> None:
    shots.click(win, shots.widget(win, name))


def acknowledge(win: MainWindow) -> None:
    """The reader's click on the button of the band that a refusal puts on the strip: operator calls are accepted
    again. A test that clicks one refusal after another does this in between."""
    assert win.worker.held and shots.widget(win, "refusal.band").isVisible()
    run_button(win, "refusal.ack")
    QApplication.processEvents()
    assert not win.worker.held and not shots.widget(win, "refusal.band").isVisible()


def pause(seconds: float) -> None:
    """A wait inside a step, as the screen harness makes it: the event loop runs and the worker thread gets its turns
    (``QTest.qWait`` does not release the interpreter lock, so the worker would stand still)."""
    end = time.monotonic() + seconds
    QApplication.processEvents()
    while time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.005)


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
            if w.isVisible() and (w.objectName().startswith(("act.", "run.", "lbl.", "group."))):
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
        if w.objectName().startswith(("lbl.", "recent.")):
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
    assert shots.widget(win, "recent.0").text().startswith(f"✓ scan → {name}")
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
    assert "Bias 0.005 V" in win.status.plain_text() and "Setpoint 100000 pA" in win.status.plain_text()   # never mV or nA


def test_a_refusal_shows_the_instruments_words_everywhere(open_window, tmp_path):
    win = open_window("en")
    shots.fill(win, {"act.move_tip.x_nm": "1000000", "act.move_tip.y_nm": "0"})
    run_button(win, "run.move_tip")
    shots.settle(win)
    words = "refused: move_tip: outside the piezo range"
    assert win.status.result.full_text() == "✗ " + words
    assert win.status.result.property("tone") == "bad"
    newest = shots.widget(win, "recent.0")
    assert newest.text() == "✗ " + words and newest.property("tone") == "bad"
    assert "move_tip: outside the piezo range" in shots.widget(win, "log.view").toPlainText()
    assert not acts(records(tmp_path))                          # nothing happened on the instrument
    # an empty required field sends nothing, a field that is not a number is sent as typed: both are refused,
    # and every refusal is on record, the instrument's own range check above included
    acknowledge(win)                                            # a refusal holds the next call until it is acknowledged
    run_button(win, "run.scan")
    shots.settle(win)
    assert win.status.result.full_text() == "✗ refused: scan: center_x_nm is required"
    assert [r["kind"] for r in records(tmp_path) if r["kind"] != "observe"] == ["refused", "refused"]
    acknowledge(win)
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
    shots.fill(win, {"act.pulse.bias_v": "1.0", "act.pulse.width_ms": "20", **SMALL})     # the width: required in API v2
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
    assert by["pulse"]["params"]["bias_v"] == 1.0 and by["pulse"]["params"]["width_ms"] == 20.0
    assert by["feedback"]["params"] == {"on": False}
    assert by["set_current_range"]["params"] == {"full_scale_na": 1.0}
    assert by["drift_comp"]["params"]["on"] is True and by["drift_comp"]["params"]["vx_nm_per_min"] == 1.5
    assert "withdraw" in by
    assert win.status.result.full_text() == "✗ refused: coarse_move: direction is required"
    plain = win.status.plain_text()
    assert "Z feedback OFF" in plain and "Range 1 nA" in plain and "Withdrawn yes" in plain


# ── required parameters ──
def reshape(monkeypatch, action: str, name: str, **changes) -> None:
    """Change one parameter of the action table (``stmbench.instrument.ACTIONS``, which the window and the instrument
    both read) for the length of a test."""
    spec = ACTIONS[action]
    params = tuple(replace(p, **changes) if p.name == name else p for p in spec.params)
    monkeypatch.setitem(ACTIONS, action, replace(spec, params=params))


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_every_field_is_starred_hinted_and_left_empty_as_the_action_table_says(open_window, lang):
    """Each text field's red star and grey hint come from its parameter in ``ACTIONS``: "required" and a star where
    the parameter is required, "default 50" where it has a default, "optional" otherwise; every field starts empty,
    because a default is the instrument's to apply, not the window's. The pulse's width and the poke's depth and
    dwell are required with no default (API version 2): they are starred, hinted "required" and empty like the rest."""
    win = open_window(lang)
    starred = hinted = 0
    for _, c in layout.cells():
        if not isinstance(c, layout.F):
            continue
        p = next(p for p in ACTIONS[c.action].params if p.name == c.param)
        label = shots.widget(win, f"lbl.{c.action}.{c.param}")
        assert ("*" in label.text()) == p.required, (c.action, c.param)
        starred += p.required
        w = shots.widget(win, f"act.{c.action}.{c.param}")
        if isinstance(w, QLineEdit):
            want = "required" if p.required else f"default {fmt.num(p.default)}" if p.default is not None else "optional"
            assert w.placeholderText() == want and w.text() == "", (c.action, c.param, w.placeholderText())
            hinted += 1
    assert starred >= 13 and hinted > 25                         # the table's required fields, and the text fields
    for name in ("pulse.width_ms", "poke.depth_nm", "poke.dwell_s"):
        action, param = name.split(".")
        p = next(p for p in ACTIONS[action].params if p.name == param)
        assert p.required and p.default is None, name
        assert shots.widget(win, f"act.{name}").placeholderText() == "required"
        assert "*" in shots.widget(win, f"lbl.{name}").text()
    assert shots.widget(win, "act.poke.bias_v").placeholderText() == "optional"       # still not required


def test_a_field_follows_the_action_table_when_it_changes_not_a_list_of_names(open_window, monkeypatch, tmp_path):
    """The window knows no parameter by name: make a required one optional with a default, one that had a default
    required, and the star, the hint and what the instrument is sent follow."""
    reshape(monkeypatch, "wait", "seconds", required=False, default=1.5)           # was required
    reshape(monkeypatch, "scan", "angle_deg", required=True, default=None)          # had a default of 0
    reshape(monkeypatch, "poke", "dwell_s", required=False, default=0.5)            # as API version 1 had it
    win = open_window("en")
    assert "*" not in shots.widget(win, "lbl.wait.seconds").text()
    assert shots.widget(win, "act.wait.seconds").placeholderText() == "default 1.5"
    assert "*" in shots.widget(win, "lbl.scan.angle_deg").text()
    assert shots.widget(win, "act.scan.angle_deg").placeholderText() == "required"
    assert "*" not in shots.widget(win, "lbl.poke.dwell_s").text()
    assert shots.widget(win, "act.poke.dwell_s").placeholderText() == "default 0.5"
    run_button(win, "run.wait")                                  # empty: nothing sent, the instrument applies its default
    shots.fill(win, SMALL)
    run_button(win, "run.scan")                                  # the angle left empty
    shots.settle(win)
    assert win.status.result.full_text() == "✗ refused: scan: angle_deg is required"
    acknowledge(win)
    shots.fill(win, {"act.scan.angle_deg": "0"})
    run_button(win, "run.scan")
    shots.settle(win)
    done = {r["name"]: r["params"] for r in records(tmp_path) if r["kind"] == "act"}
    assert done["wait"] == {"seconds": 1.5} and done["scan"]["angle_deg"] == 0.0


# ── the ranges ──
def numeric_fields():
    """Every text field of the panels that takes a number: ``(tab, cell, parameter)``."""
    for tab, groups in layout.TABS:
        for g in groups:
            for line in g.lines:
                for c in line:
                    if isinstance(c, layout.F):
                        p = next(q for q in ACTIONS[c.action].params if q.name == c.param)
                        if p.kind in ("number", "integer") and p.choices is None:
                            yield tab, c, p


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_field_last_clicked_tells_its_range_from_the_action_table_beside_the_tabs(fonts, open_window, lang):
    """The screen agent learnt the 100000 pA setpoint limit only from a refusal, in all three trials: the console showed the
    range of no field. The line beside the tabs describes the field last clicked, in the field's own unit, with the table's
    numbers, whole in the widest fonts, with no hover and no height of its own."""
    win = open_window(lang)
    line = shots.widget(win, "field.help")
    assert line.text() == ""                                                  # no field clicked yet: nothing to say
    where = {n: shots.widget(win, n).mapTo(win, shots.widget(win, n).rect().topLeft())
             for n in ("run.set_setpoint", "recent.box", "right.tabs")}
    ranged = 0
    for tab, c, p in numeric_fields():
        open_right(win, tab)
        shots.click(win, shots.widget(win, f"act.{c.action}.{c.param}"))
        shots.hit(win, line)                                                   # on screen, whichever tab is open
        text = line.full_text()
        assert text and line.text() == text, (c.action, c.param, text, line.text())            # whole, not elided
        assert c.en in text and (f"({c.unit})" in text) == bool(c.unit), (c.action, c.param, text)
        want = fmt.range_text(p.lo, p.hi)
        if want:
            ranged += 1
            assert text.endswith(f"{one('field_range', lang)} {want}"), (c.action, c.param, text)
        else:
            assert one("field_range", lang) not in text, (c.action, c.param, text)
        assert {n: shots.widget(win, n).mapTo(win, shots.widget(win, n).rect().topLeft()) for n in where} == where
    assert ranged >= 25
    open_right(win, "operate")
    shots.click(win, shots.widget(win, "act.set_setpoint.current_pa"))
    assert line.full_text().endswith("0.1 … 100000")
    run_button(win, "run.set_bias")                                          # a click on a button leaves it: the field is the last clicked
    assert line.full_text().endswith("0.1 … 100000")
    shots.click(win, shots.widget(win, "act.move_tip.speed_nm_s"))
    assert line.full_text().endswith("0.01 … 10000")


def test_a_range_is_the_action_tables_and_the_field_still_sends_what_was_typed(open_window, monkeypatch, tmp_path):
    reshape(monkeypatch, "set_setpoint", "current_pa", lo=2.5, hi=5000.0)
    reshape(monkeypatch, "scan", "angle_deg", lo=None)
    win = open_window("en")
    shots.click(win, shots.widget(win, "act.set_setpoint.current_pa"))
    assert shots.widget(win, "field.help").full_text() == "setpoint (pA): range 2.5 … 5000"
    shots.click(win, shots.widget(win, "act.scan.angle_deg"))
    assert shots.widget(win, "field.help").full_text() == "angle (°): range ≤ 180"        # one end open
    shots.type_into(win, shots.widget(win, "act.set_setpoint.current_pa"), "9000")      # outside it: no clamping, no check here
    run_button(win, "run.set_setpoint")
    shots.settle(win)
    assert win.status.result.full_text() == "✗ refused: set_setpoint: current_pa = 9000.0 is outside [2.5, 5000.0]"
    sent = [r for r in records(tmp_path) if r["kind"] == "refused"][-1]
    assert sent["args"] == {"current_pa": 9000}


def test_the_trials_refusal_leaves_the_range_of_the_field_it_came_from_on_screen(open_window):
    win = open_window("en")
    shots.fill(win, {"act.set_setpoint.current_pa": "200000"})
    run_button(win, "run.set_setpoint")
    shots.settle(win)
    acknowledge(win)
    assert shots.widget(win, "field.help").full_text() == "setpoint (pA): range 0.1 … 100000"
    assert "outside [0.1, 100000.0]" in win.status.result.full_text()


def test_with_the_conditioning_parameters_required_an_empty_field_sends_nothing_and_the_instrument_refuses(
        open_window, tmp_path):
    win = open_window("en")
    assert win.operate.collect("pulse") == {"z_hold": True} and win.operate.collect("poke") == {}    # empty: not sent
    shots.fill(win, {"act.pulse.bias_v": "1.0"})
    run_button(win, "run.pulse")                                 # the width left empty
    shots.settle(win)
    assert win.status.result.full_text() == "✗ refused: pulse: width_ms is required"
    refused = [r for r in records(tmp_path) if r["kind"] == "refused"]
    assert refused[-1]["name"] == "pulse" and refused[-1]["args"] == {"bias_v": 1.0, "z_hold": True}    # no 50 filled in
    acknowledge(win)
    run_button(win, "run.poke")
    shots.settle(win)
    assert win.status.result.full_text() == "✗ refused: poke: depth_nm is required"
    acknowledge(win)
    shots.fill(win, {"act.poke.depth_nm": "0.1"})
    run_button(win, "run.poke")
    shots.settle(win)
    assert win.status.result.full_text() == "✗ refused: poke: dwell_s is required"
    assert win.operate.collect("poke") == {"depth_nm": 0.1}
    assert not acts(records(tmp_path))                           # nothing reached the instrument but the refusals
    acknowledge(win)
    shots.fill(win, {"act.pulse.width_ms": "20", "act.poke.dwell_s": "0.2"})
    run_button(win, "run.pulse")
    run_button(win, "run.poke")
    shots.settle(win)
    done = {r["name"]: r["params"] for r in records(tmp_path) if r["kind"] == "act"}
    assert done == {"pulse": {"bias_v": 1.0, "width_ms": 20.0, "z_hold": True},
                    "poke": {"depth_nm": 0.1, "dwell_s": 0.2}}


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
    assert f"[Scoring rules v{RULES_VERSION}]" in task and "ReportResult" in task        # the English card
    assert "评分规则" not in task
    assert f"【评分规则 v{RULES_VERSION}】" in shots.widget(zh, "task.text").toPlainText()  # the Chinese one
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


def manual_lines(win: MainWindow) -> tuple[int, int]:
    """``(visual lines in all, lines a page shows)`` of the Manual tab's text as it is laid out now."""
    bar = shots.widget(win, "manual.text").verticalScrollBar()
    return bar.maximum() + bar.pageStep(), bar.pageStep()


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_manuals_wide_view_reads_in_two_screens_and_gives_the_window_back_when_it_is_left(fonts, open_window, lang):
    """Screen trials 4 and 5 spent five to seven steps scrolling the manual: a pane of 27 lines in a column of 360 px, where
    most of its 64 lines wrap two or three times (about 100 lines in all). The Manual tab's wide view takes the whole
    window under the status strip, puts the instrument's facts beside the text and shows the same text, byte for byte, in
    two screens or fewer. Leaving the tab, or the button, gives the window back."""
    win = open_window(lang)
    manual = win.session.manual()
    open_left(win, "manual")
    for _ in range(3):                                                          # the layout settles in a few turns of the loop
        QApplication.processEvents()
    text = shots.widget(win, "manual.text")
    toggle = shots.widget(win, "manual.wide")
    narrow_lines, narrow_page = manual_lines(win)
    assert narrow_lines > 3 * narrow_page                                       # in the column it is three screens and more
    left_w, strip_h = win.sidebar.width(), win.status.height()
    shots.hit(win, toggle)
    assert toggle.text().endswith("Wide view") and toggle.width() >= 56 and toggle.height() >= 32
    run_button(win, "manual.wide")
    for _ in range(3):
        QApplication.processEvents()
    assert not win.center.isVisible() and not shots.widget(win, "recent.box").isVisible() and not win.right.isVisible()
    assert win.sidebar.width() > 1800 and win.status.height() == strip_h and win.status.isVisible()      # the strip stays
    assert win.width() == shots.WINDOW[0]                                        # and the window did not grow to make room
    assert text.toPlainText() == manual                                          # the manual an MCP agent gets, byte for byte
    total, page = manual_lines(win)
    assert total <= 2 * page, (total, page)                                      # two screens or fewer, in these fonts too
    table = shots.widget(win, "facts.table")
    shots.hit(win, table)
    shots.hit(win, text)
    here = lambda w: w.mapTo(win, w.rect().topLeft())                           # noqa: E731
    assert here(table).x() + table.width() <= here(text).x()                     # beside the text, not above it
    assert here(text).y() <= here(table).y() < here(text).y() + text.height()
    assert toggle.text().endswith("Narrow view")
    # a wheel notch is three lines, in this view and in the narrow one
    bar = text.verticalScrollBar()
    shots.wheel(win, text, delta=-120)
    assert bar.value() == 3
    shots.wheel(win, text, delta=-120)
    shots.wheel(win, text, delta=-120)
    assert bar.value() == min(9, bar.maximum())
    shots.wheel(win, text, delta=120 * 5)
    assert bar.value() == 0
    run_button(win, "manual.wide")                                               # the button gives the window back
    for _ in range(3):
        QApplication.processEvents()
    assert win.center.isVisible() and win.right.isVisible() and win.sidebar.width() == left_w and toggle.text().endswith("Wide view")
    assert win.width() == shots.WINDOW[0] and manual_lines(win) == (narrow_lines, narrow_page)
    run_button(win, "manual.wide")                                               # and so does leaving the tab
    open_left(win, "report")
    for _ in range(3):
        QApplication.processEvents()
    assert win.center.isVisible() and win.right.isVisible() and win.sidebar.width() == left_w
    assert win.width() == shots.WINDOW[0]
    open_left(win, "manual")
    for _ in range(3):
        QApplication.processEvents()
    assert toggle.text().endswith("Wide view") and win.sidebar.width() == left_w     # it opens narrow again
    # in the column too, a notch is three lines
    bar = text.verticalScrollBar()
    shots.wheel(win, text, delta=-120)
    assert bar.value() == 3
    shots.wheel(win, text, delta=-120 * 4)
    assert bar.value() == 15


def test_a_qplus_rigs_sensor_constants_are_among_the_facts(open_window):
    win = open_window("en", scenario=P5)
    q = win.session.info()["qplus"]
    facts = win.sidebar.facts.facts()
    assert facts["qPlus k"] == f"{q['k_n_per_m']:.9g} N/m" and facts["qPlus f0"] == f"{q['f0_hz']:.9g} Hz"
    assert facts["qPlus Q"] == f"{q['q']:.9g}"


def test_the_status_strip_reads_in_units_and_counts_instrument_time(open_window):
    win = open_window("en")
    plain = win.status.plain_text()
    for want in ("Bias 0.050 V", "Setpoint 200 pA", "Z feedback ON", "Range 10 nA", "Withdrawn no", "Lock-in OFF",
                 "Oscillation n/a", "Drift comp. OFF", "0:00:00 / 1:30:00", "Actions 0", P4, "READY"):
        assert want in plain, want
    assert "e-" not in plain
    scan_small(win)
    assert "Actions 1" in win.status.plain_text() and win.status.bar.value() > 0
    zh = open_window("zh", name="run_zh")
    assert "偏压 0.050 V" in zh.status.plain_text() and "就绪 READY" in zh.status.plain_text()


# ── the report and the end ──
def test_a_picked_claim_stays_picked_when_it_is_clicked_again_with_one_claim_or_several(open_window):
    win = open_window("en")                                              # P4 has one reportable claim: picked from the start
    open_left(win, "report")
    rep = win.sidebar.report
    assert list(rep.radios) == ["target_site"] and rep.claim_id() == "target_site"
    for click in ("radio", "radio", "its line", "its line", "radio"):       # the trial's second click took it back
        if click == "radio":
            run_button(win, "report.claim.target_site")
        else:
            shots.click(win, shots.widget(win, "report.say.target_site"))
        assert rep.claim_id() == "target_site" and shots.widget(win, "report.claim.target_site").isChecked()
    assert rep.fields["unit"].text() == "nm"
    # several claims (P1): one at a time, and the one that is picked stays picked
    win = open_window("en", name="run_p1", scenario="P1_barth1990_au111")
    open_left(win, "report")
    rep = win.sidebar.report
    ids = list(rep.radios)
    assert len(ids) == 4 and rep.claim_id() == ""
    for cid in (ids[0], ids[0], ids[2], ids[2], ids[1]):
        run_button(win, f"report.claim.{cid}")
        assert rep.claim_id() == cid and sum(r.isChecked() for r in rep.radios.values()) == 1
    shots.click(win, shots.widget(win, f"report.say.{ids[3]}"))                 # a click on a claim's own line picks it
    assert rep.claim_id() == ids[3] and rep.fields["unit"].text() == "deg"
    shots.click(win, shots.widget(win, f"report.say.{ids[3]}"))
    assert rep.claim_id() == ids[3]


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_shown_file_button_adds_the_name_of_what_the_viewer_shows_or_says_there_is_none(open_window, lang):
    win = open_window(lang)
    open_left(win, "report")
    rep = win.sidebar.report
    button = shots.widget(win, "report.viewed")
    assert button.text() == ("+ shown file" if lang == "en" else "+ 显示中的文件")
    evidence = rep.fields["evidence"]
    run_button(win, "report.viewed")                                      # nothing shown yet
    assert evidence.text() == ""
    tone, text = win.recent.entries()[0]
    assert tone == "note" and "no frame or spectrum is shown" in text
    first = scan_small(win)
    open_left(win, "report")
    run_button(win, "report.viewed")
    assert evidence.text() == first
    run_button(win, "report.viewed")                                      # again: the name is not added twice
    assert evidence.text() == first
    open_right(win, "spectroscopy")
    shots.fill(win, {"act.sts.v_start": "-0.2", "act.sts.v_end": "0.2", "act.sts.points": "16"})
    run_button(win, "run.sts")
    shots.settle(win)
    spectrum = win.viewed[1]
    assert win.viewed[0] == "spectrum" and spectrum.endswith(".dat")
    open_left(win, "report")
    run_button(win, "report.viewed")                                      # the file in the viewer now is the spectrum
    assert evidence.text() == f"{first}, {spectrum}"
    win.request_frame(first)                                              # back to the frame, by the thumbnail's way
    shots.settle(win)
    evidence.setText("")
    run_button(win, "report.viewed")
    assert evidence.text() == first


def test_the_report_form_marks_what_the_picked_claim_needs_and_enter_sends_it(open_window, tmp_path):
    win = open_window("en", scenario="P1_barth1990_au111")
    open_left(win, "report")
    rep = win.sidebar.report
    starred = lambda: {k for k, lab in rep.labels.items() if lab.text().endswith("*</span>")}       # noqa: E731
    assert starred() == {"value"}                                         # no claim yet: a value is always needed
    run_button(win, "report.claim.stripe_period_nm")                     # a number about the whole surface
    assert starred() == {"value"} and rep.required_fields() == {"value"}
    run_button(win, "report.claim.stripe_orientation_deg")               # an angle at a place: where, and in which frame
    assert starred() == {"value", "x_nm", "y_nm", "frame"}
    run_button(win, "report.claim.stripe_period_nm")                     # and back
    assert starred() == {"value"}
    assert shots.widget(win, "report.label.x_nm").text().startswith("x_nm ")      # the name still comes first
    # Enter in any field sends the report, as the button does
    run_button(win, "report.claim.stripe_orientation_deg")
    shots.fill(win, {"report.x_nm": "1"})
    QTest.keyClick(shots.widget(win, "report.x_nm"), Qt.Key.Key_Return)        # no value: the instrument says so
    shots.settle(win)
    assert rep.reply.text() == "✗ refused: report: value is required"
    acknowledge(win)
    shots.fill(win, {"report.value": "30", "report.y_nm": "2", "report.frame": "sim001.sxm", "report.evidence": "sim001.sxm"})
    QTest.keyClick(shots.widget(win, "report.evidence"), Qt.Key.Key_Enter)    # the keypad's Enter too
    shots.settle(win)
    assert rep.reply.text().startswith("✓ recorded stripe_orientation_deg = 30")
    assert [r["kind"] for r in records(tmp_path) if r["kind"] == "report"] == ["report"]       # the one that was complete


def test_a_report_and_the_end_through_the_report_tab(open_window, tmp_path):
    win = open_window("en")
    name = scan_small(win)
    open_left(win, "report")
    rep = win.sidebar.report
    assert rep.claim_id() == "target_site" and rep.fields["unit"].text() == "nm"     # P4 has one claim to report: it is picked
    assert not rep.fields["value"].text() and not rep.fields["evidence"].text()      # and nothing else is filled in for the reader
    run_button(win, "report.submit")                             # nothing filled in: the instrument says what is missing
    shots.settle(win)
    assert rep.reply.text() == "✗ refused: report: value is required" and rep.reply.property("tone") == "bad"
    acknowledge(win)
    shots.click(win, shots.widget(win, "report.claim.target_site"))                # clicking the picked claim again keeps it
    assert rep.claim_id() == "target_site" and rep.fields["unit"].text() == "nm"
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


def test_with_several_claims_to_report_none_is_picked_and_a_report_without_one_is_refused(open_window):
    win = open_window("en", scenario="P1_barth1990_au111")
    open_left(win, "report")
    rep = win.sidebar.report
    assert len(rep.radios) == 4 and rep.claim_id() == "" and not rep.fields["unit"].text()      # nothing is chosen for the reader
    shots.fill(win, {"report.value": "1"})
    run_button(win, "report.submit")
    shots.settle(win)
    assert "unknown claim ''" in rep.reply.text()


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_note_shows_its_whole_text_and_is_sent_whole_in_the_widest_fonts_too(fonts, open_window, lang, tmp_path):
    """Screen trial 4: "the narrow note field cannot show its text": the note was a line edit a third of the form wide.
    It now has the width of the form, wraps, and has the height of four lines."""
    win = open_window(lang)
    scan = scan_small(win)
    open_left(win, "report")
    rep = win.sidebar.report
    note = shots.widget(win, "report.note")
    assert note.width() >= 0.9 * shots.widget(win, "report.evidence").width()
    notes = {"en": ("TARGET NOT REACHED: atom is within 0.2 nm of its start site; not moved +4 nm",    # the note of that trial
                    "a longer one: " + "word " * 12 + "end"),
             "zh": ("目标未到达：原子离出发的格点不到 0.2 nm，没有移动 +4 nm", "较长的一条：" + "说明" * 8)}[lang]
    for text in notes:
        if lang == "zh":                                                       # QTest cannot type Chinese: the harness sends key events by hand
            shots.click(win, note)
            note.setText(text)
        else:
            shots.type_into(win, note, text)
        assert rep.fields["note"].text() == text
        block = note.document().firstBlock()
        assert note.document().documentLayout().blockBoundingRect(block).height() <= note.viewport().height()   # all of it in view
        assert not note.verticalScrollBar().isVisible()
        shots.hit(win, note)
    shots.fill(win, {"report.value": "1.5", "report.x_nm": "1.5", "report.y_nm": "0", "report.frame": scan,
                     "report.evidence": scan})
    shots.click(win, note)
    QTest.keyClick(note, Qt.Key.Key_Return)                                    # Enter in the note sends the report, as in every field
    shots.settle(win)
    sent = [r for r in records(tmp_path) if r["kind"] == "report"][-1]["args"]
    assert sent["note"] == text and "\n" not in sent["note"]


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
    assert not table.item(0, 6).text()                           # the first marker has no previous one
    assert float(table.item(1, 6).text()) == pytest.approx(math.hypot(markers[1]["x_nm"] - markers[0]["x_nm"],
                                                                       markers[1]["y_nm"] - markers[0]["y_nm"]), abs=1e-3)
    assert table.item(0, 5).text() == "0:00:00"                  # when the frame it was placed on began
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
    assert win.recent.entries()[0] == ("note", "No marker selected.")


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_a_marker_is_deleted_by_a_button_and_the_table_scrolls_to_the_newest(fonts, open_window, lang):
    win = open_window(lang)
    scan_small(win)
    fv = win.center.frame_view
    table = shots.widget(win, "markers.table")
    row = table.verticalHeader().defaultSectionSize()
    assert table.horizontalHeader().height() >= 23                              # 15 px or more in the 1280x720 picture
    assert table.viewport().height() == 6 * row                                 # six whole rows: none is cut by the header
    for i in range(6):
        shots.click(win, fv, fv.pixel_to_widget(3 + 4 * i, 6 + 5 * (i % 3)).toPoint())
    assert table.rowCount() == 6 and not table.verticalScrollBar().isVisible()
    for i in range(6, 9):
        shots.click(win, fv, fv.pixel_to_widget(3 + 3 * i, 5 + 4 * (i % 4)).toPoint())
    assert table.rowCount() == 9
    bar = table.verticalScrollBar()
    assert bar.isVisible() and bar.maximum() > 0                                 # more rows than fit: it scrolls
    newest = table.visualItemRect(table.item(8, 0))
    assert table.viewport().rect().contains(newest)                              # the newest, the selected, is in view
    # scrolled to the end or the start, the first row shown is whole under the header and the last one ends the view
    for value in (bar.maximum(), 0, bar.maximum() // 2):
        bar.setValue(value)
        first, last = table.rowAt(0), table.rowAt(table.viewport().height() - 1)
        assert table.rowViewportPosition(first) == 0
        assert table.rowViewportPosition(last) + row == table.viewport().height()
    bar.setValue(bar.maximum())
    assert not table.horizontalScrollBar().isVisible()                           # the seven columns fit beside the scroll bar
    assert table.horizontalHeader().length() <= table.viewport().width()
    xs = [m["x_nm"] for m in fv.markers()]
    bar.setValue(0)
    shots.click(win, table.viewport(), table.visualItemRect(table.item(2, 1)).center())      # select the third
    assert win.center.selected_row() == 2 and fv.selected_marker()["x_nm"] == xs[2]
    shots.hit(win, shots.widget(win, "markers.delete"))
    run_button(win, "markers.delete")
    assert [m["x_nm"] for m in fv.markers()] == xs[:2] + xs[3:] and table.rowCount() == 8
    assert [table.item(r, 0).text() for r in range(8)] == [str(r + 1) for r in range(8)]    # numbered again
    assert win.center.selected_row() == 7                                        # the selection moves to the newest
    assert float(table.item(7, 1).text()) == pytest.approx(xs[-1], abs=5e-4)
    run_button(win, "view.clear_markers")
    run_button(win, "markers.delete")                                            # none to delete: said, not ignored
    tone, text = win.recent.entries()[0]
    assert tone == "note" and text.endswith("No marker selected.") and table.rowCount() == 0


def test_each_marker_shows_when_the_frame_it_was_placed_on_began_as_the_data_tab_gives_it(open_window):
    win = open_window("en")
    first = scan_small(win)
    shots.fill(win, {"act.scan.center_x_nm": "1", "act.scan.center_y_nm": "1", "act.scan.size_nm": "6", "act.scan.pixels": "24"})
    run_button(win, "run.scan")
    shots.settle(win)
    second = win.viewed[1]
    assert first != second
    fv = win.center.frame_view
    table = shots.widget(win, "markers.table")
    headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
    assert headers == ["#", "x nm", "y nm", "value", "frame", "start", "to prev. nm"]
    shots.click(win, fv, fv.pixel_to_widget(12, 12).toPoint())                  # on the second frame
    win.request_frame(first)
    shots.settle(win)
    shots.click(win, fv, fv.pixel_to_widget(10, 10).toPoint())                  # and on the first
    starts = {f["frame"]: f["t_start_s"] for f in win._data["frames"]}
    assert starts[first] != starts[second]
    from stmbench.gui import fmt
    assert [(table.item(r, 4).text(), table.item(r, 5).text()) for r in range(2)] == [
        (second, fmt.hms(starts[second])), (first, fmt.hms(starts[first]))]
    data = shots.widget(win, "data.frames")                                        # the same h:mm:ss the Data tab shows
    assert fmt.hms(starts[second]) in data.item(0).text() and fmt.hms(starts[first]) in data.item(1).text()


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_viewer_toolbar_has_a_contrast_switch_that_starts_on_auto_and_holds_for_the_next_frame(open_window, lang):
    import numpy as np
    win = open_window(lang)
    scan_small(win)
    fv = win.center.frame_view
    assert checked(win, "view.contrast.auto") and not checked(win, "view.contrast.full") and fv.contrast() == "auto"
    shown = fv._shown
    auto_lo, auto_hi = shown.lo, shown.hi
    values = shown.values.copy()
    run_button(win, "view.contrast.full")
    assert checked(win, "view.contrast.full") and fv.contrast() == "full"
    assert shown.lo == pytest.approx(np.nanmin(values)) and shown.hi == pytest.approx(np.nanmax(values))
    assert shown.lo <= auto_lo and shown.hi >= auto_hi                          # the whole range holds the mapped one
    assert np.array_equal(shown.values, values)                                  # what is read off the image is the same
    shots.fill(win, {"act.scan.center_x_nm": "1", "act.scan.center_y_nm": "1"})
    run_button(win, "run.scan")
    shots.settle(win)
    assert fv.contrast() == "full" and checked(win, "view.contrast.full")        # the next frame is shown in full too
    assert fv._shown is not shown and fv._shown.hi == pytest.approx(np.nanmax(fv._shown.values))
    run_button(win, "view.contrast.auto")
    assert fv.contrast() == "auto" and fv._shown.hi < np.nanmax(fv._shown.values) + 1e-9
    caption = shots.widget(win, "view.contrast.full").text()
    assert caption == ("full" if lang == "en" else "全幅")                        # the viewer's toolbar names its buttons in one language
    assert shots.widget(win, "view.contrast.visible").text() == ("visible" if lang == "en" else "可见区")
    for key in ("view.contrast.auto", "view.contrast.full", "view.contrast.visible", "view.channel.z", "view.flatten.line",
                "view.tool.marker", "view.tool.pan", "view.zoom.1", "view.zoom.8"):
        assert shots.widget(win, key).width() >= 56


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_visible_contrast_button_sets_the_scale_by_the_part_in_view(open_window, lang):
    import numpy as np
    win = open_window(lang)
    scan_small(win)
    fv = win.center.frame_view
    run_button(win, "view.contrast.full")
    full = (fv._shown.lo, fv._shown.hi)
    run_button(win, "view.contrast.visible")
    assert checked(win, "view.contrast.visible") and not checked(win, "view.contrast.full") and fv.contrast() == "visible"
    assert (fv._shown.lo, fv._shown.hi) == full                                   # the whole frame in view: the same as full
    run_button(win, "view.zoom.8")
    r0, r1, c0, c1 = fv._region()
    part = fv._shown.values[r0:r1, c0:c1]
    assert fv._shown.lo == pytest.approx(np.nanmin(part)) and fv._shown.hi == pytest.approx(np.nanmax(part))
    assert fv._shown.lo >= full[0] - 1e-9 and fv._shown.hi <= full[1] + 1e-9      # a part of the frame: inside its range
    run_button(win, "view.zoom.1")
    assert (fv._shown.lo, fv._shown.hi) == full


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_zoom_buttons_the_wheel_and_the_pan_tool_move_the_view_and_a_new_frame_is_shown_whole(open_window, lang):
    from PySide6.QtCore import QPoint, QPointF
    win = open_window(lang)
    first = scan_small(win)
    fv = win.center.frame_view
    assert checked(win, "view.zoom.1") and fv.zoom() == 1
    for level in (2, 4, 8, 1):                                                  # the buttons, ending on Fit
        run_button(win, f"view.zoom.{level}")
        assert fv.zoom() == level and checked(win, f"view.zoom.{level}")
    assert shots.widget(win, "view.zoom.1").text() == ("Fit" if lang == "en" else "全图")
    # the wheel, through the window: in about the pointer, and the buttons show where it is
    r = fv.image_rect()
    spot = QPoint(r.x() + r.width() // 3, r.y() + r.height() // 2)
    under = fv._point_at(QPointF(spot))
    for want in (2, 4):
        shots.wheel(win, fv, spot)
        assert fv.zoom() == want and checked(win, f"view.zoom.{want}")
        now = fv._point_at(QPointF(spot))
        assert now["x_nm"] == pytest.approx(under["x_nm"], abs=1e-6) and now["y_nm"] == pytest.approx(under["y_nm"], abs=1e-6)
    shots.wheel(win, fv, spot, -120)
    assert fv.zoom() == 2 and checked(win, "view.zoom.2")
    # a marker placed on the zoomed image is at the scan position of the pixel under the click
    geom = {"cx_nm": 0.0, "cy_nm": 0.0, "w_nm": 8.0, "h_nm": 8.0, "angle_deg": 0.0, "nx": 32, "ny": 32}
    click = QPoint(r.x() + r.width() * 3 // 5, r.y() + r.height() * 2 // 5)
    shots.click(win, fv, click)
    (m,) = fv.markers()
    p = fv._point_at(QPointF(click))
    x, y = pixel_to_scan_nm(geom, p["col_f"], p["row_f"])
    assert m["x_nm"] == pytest.approx(x, abs=1e-6) and m["y_nm"] == pytest.approx(y, abs=1e-6)
    assert shots.widget(win, "markers.table").item(0, 1).text() == f"{x:.3f}"
    # the Pan tool carries the image; the marker tool is the one that places markers
    run_button(win, "view.tool.pan")
    c0, r0, _, _ = fv.view_rect()
    shots.drag(win, fv, QPoint(r.x() + 200, r.y() + 200), QPoint(r.x() + 140, r.y() + 170))
    c1, r1, _, _ = fv.view_rect()
    assert c1 > c0 and r1 > r0 and len(fv.markers()) == 1                       # dragged left and up: the view moves right and down
    run_button(win, "view.tool.marker")
    # Fit: the whole frame
    run_button(win, "view.zoom.1")
    assert fv.view_rect() == (0.0, 0.0, 32.0, 32.0) and checked(win, "view.zoom.1")
    # another channel of the same frame keeps the zoom, the next frame is shown whole
    run_button(win, "view.zoom.4")
    run_button(win, "view.channel.current")
    shots.settle(win)
    assert win.viewed == ("frame", first) and fv.zoom() == 4
    shots.fill(win, {"act.scan.center_x_nm": "1", "act.scan.center_y_nm": "1"})
    run_button(win, "run.scan")
    shots.settle(win)
    assert win.viewed[1] != first and fv.zoom() == 1 and checked(win, "view.zoom.1")


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


# ── move traces ──
def make_trace(name: str = "move_0001.dat", *, frm=(-6.891, -3.818), to=(-2.887, -3.895), speed: float = 0.25,
               n: int = 400, bias: float = 0.005, setpoint: float = 100000.0, t_s: float = 612.0,
               feedback: bool = True, index: int = 1) -> dict:
    """A move trace as ``Instrument.spectrum(name)`` returns it for ``kind`` "move" (``t`` s, ``dist`` m, ``z_abs`` m,
    ``i`` A, and the move's own conditions), made by hand so that the window is tested against the record's layout and
    not against one version of the physics: a corrugation, one atom-sized step 52 % of the way along with a burst of
    current there, and noise."""
    rng = np.random.default_rng(7)
    dist = np.linspace(0.0, math.hypot(to[0] - frm[0], to[1] - frm[1]) * 1e-9, n)
    z = -1.2e-9 + 6e-12 * np.sin(2 * np.pi * dist / 0.255e-9) + 0.4e-12 * rng.standard_normal(n)
    i = setpoint * 1e-12 * (1.0 + 0.04 * rng.standard_normal(n))
    k = int(0.52 * n)
    z[k:] += 14e-12
    i[k:k + 4] *= 3.0
    return {"name": name, "kind": "move", "index": index, "tip_nm": list(to), "from_nm": list(frm), "to_nm": list(to),
            "speed_nm_s": speed, "bias_v": bias, "setpoint_pa": setpoint, "feedback": feedback, "t_s": t_s,
            "t": dist / (speed * 1e-9), "dist": dist, "z_abs": z, "i": i}


class TraceServer:
    """Makes a window's session answer ``move_tip`` as the instrument will once it records traces: the reply names
    the ``trace`` (and has no ``current_spikes``), the data list carries it among the spectra with ``kind`` "move",
    and ``spectrum(name)`` returns its record. The tip still moves on the instrument that is there; the traces are
    served in the order they were given."""

    def __init__(self, win: MainWindow, *traces: dict):
        self.pending, self.served = list(traces), []
        s = win.session
        self._act, self._data, self._spectrum = s.act, s.data, s.spectrum
        s.act, s.data, s.spectrum = self.act, self.data, self.spectrum

    def act(self, name, args=None):
        reply = self._act(name, args)
        if name == "move_tip":
            self.served.append(self.pending.pop(0))
            reply = {**{k: v for k, v in reply.items() if k != "current_spikes"},
                     "trace": self.served[-1]["name"], "duration_s": 16.0}
        return reply

    def data(self):
        out = self._data()
        # the instrument records traces now: the ones it kept are replaced by the ones served here
        out["spectra"] = [*(s for s in out["spectra"] if s.get("kind") != "move"),
                          *({"spectrum": t["name"], "kind": "move", "tip_nm": t["tip_nm"],
                             "points": len(t["t"]), "t_s": t["t_s"]} for t in self.served)]
        return out

    def spectrum(self, name):
        return next((t for t in self.served if t["name"] == name), None) or self._spectrum(name)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_a_finished_tip_move_shows_its_trace_like_a_spectrum_and_the_data_tab_lists_it_as_a_move_trace(
        fonts, open_window, lang):
    win = open_window(lang)
    trace = make_trace()
    TraceServer(win, trace)
    frame = scan_small(win)
    shots.fill(win, {"act.move_tip.x_nm": "-2.887", "act.move_tip.y_nm": "-3.895", "act.move_tip.speed_nm_s": "0.25"})
    run_button(win, "run.move_tip")
    shots.settle(win)
    # what an STS does: the record is shown, in the viewer's plot, open on Z against the distance along the move
    assert win.viewed == ("spectrum", trace["name"]) and win.center.stack.currentIndex() == 1
    assert win.center.toolbars.currentIndex() == 1 and not any(b.isChecked() for b in win.center.thumbs)
    assert win.center.current_spectrum_axes() == ("dist", "z_abs")
    plot = win.center.plot
    assert np.allclose(plot._x, trace["dist"] * 1e9) and np.allclose(plot._series[0][1], trace["z_abs"] * 1e9)
    assert plot._x_label == ("距离 distance (nm)" if lang == "zh" else "distance (nm)") and plot._y_label == "z (nm)"
    assert plot._title == trace["name"] and plot._message == ""
    # its own columns: where along the move or when, to read Z or the current at; none of a spectrum's
    assert [b.objectName() for b in win.center.spec_x.buttons()] == ["view.spec_x.dist", "view.spec_x.t"]
    assert [b.objectName() for b in win.center.spec_y.buttons()] == ["view.spec_y.z_abs", "view.spec_y.i"]
    assert shots.widget(win, "view.spec_x.dist").isChecked() and shots.widget(win, "view.spec_y.z_abs").isChecked()
    for name in ("view.spec_x.t", "view.spec_x.dist", "view.spec_y.i", "view.spec_y.z_abs"):
        shots.hit(win, shots.widget(win, name))
    run_button(win, "view.spec_x.t")
    run_button(win, "view.spec_y.i")
    assert win.center.current_spectrum_axes() == ("t", "i")
    assert np.allclose(plot._x, trace["t"]) and np.allclose(plot._series[0][1], trace["i"] * 1e12)
    assert plot._x_label == ("时间 time (s)" if lang == "zh" else "time (s)") and plot._y_label == "I (pA)"
    # the list: a heading of its own that names it, not among the spectra
    data = win.sidebar.data
    assert data.traces.count() == 1 and data.spectra.count() == 0 and data.frames.count() == 1
    assert data.traces_title.text() == ("移动轨迹 Move traces (1)" if lang == "zh" else "Move traces (1)")
    assert data.spectra_title.text() == ("谱 Spectra (0)" if lang == "zh" else "Spectra (0)")
    item = data.traces.item(0).text().split(chr(10))
    assert item == [trace["name"], "move trace · 0:10:12", "to (-2.887, -3.895) nm · 400 points"]
    assert data.traces.currentRow() == 0                                     # the item on show is the selected one
    # the strip and the recent results say what a spectrum's say: the file a report may cite, and how long it took
    assert win.status.result.full_text() == f"✓ move_tip → {trace['name']} (16 s)"
    assert shots.widget(win, "recent.0").text() == f"✓ move_tip → {trace['name']} (16 s)"
    # the two lines above the plot carry the move's own conditions, whole, in the window's fonts
    info = shots.widget(win, "view.info")
    lines = info.text().split(chr(10))
    assert lines == [f"{trace['name']}   at 0:10:12 · 0.005 V · 100000 pA · feedback ON",
                     "move · (-6.891, -3.818) → (-2.887, -3.895) nm · 0.25 nm/s · 400 points"]
    lines_fit(info, lines, info.width())
    # nothing found, counted or marked: the window says nothing of hops over what the instrument recorded
    shown = "\n".join(_all_text(w) for w in (win.center, win.status, win.recent, data))
    assert not re.search(r"\bhops?\b|spike", shown, re.IGNORECASE)
    # the trace can be cited as evidence, and the frame and the trace are each one click from the other
    open_left(win, "report")
    run_button(win, "report.viewed")
    assert shots.widget(win, "report.evidence").text() == trace["name"]
    shots.click(win, shots.widget(win, "thumb.0"))
    shots.settle(win)
    assert win.viewed == ("frame", frame) and win.center.stack.currentIndex() == 0 and not data.traces.selectedItems()
    open_left(win, "data")
    shots.click(win, data.traces.viewport(), data.traces.visualItemRect(data.traces.item(0)).center())
    shots.settle(win)
    assert win.viewed == ("spectrum", trace["name"]) and win.center.current_spectrum_axes() == ("dist", "z_abs")
    assert data.traces.currentRow() == 0


def test_every_move_makes_a_trace_the_newest_first_in_the_list_and_a_refused_move_makes_none(open_window):
    win = open_window("en")
    first, second = make_trace("move_0001.dat", index=1), make_trace("move_0002.dat", index=2, t_s=700.0,
                                                                     frm=(-2.887, -3.895), to=(0.0, 0.0))
    TraceServer(win, first, second)
    shots.fill(win, {"act.move_tip.x_nm": "1", "act.move_tip.y_nm": "0", "act.move_tip.speed_nm_s": "100"})
    run_button(win, "run.move_tip")
    shots.settle(win)
    shots.fill(win, {"act.move_tip.x_nm": "2", "act.move_tip.y_nm": "0"})
    run_button(win, "run.move_tip")
    shots.settle(win)
    traces = win.sidebar.data.traces
    assert [traces.item(r).text().split(chr(10))[0] for r in range(traces.count())] == ["move_0002.dat", "move_0001.dat"]
    assert win.viewed == ("spectrum", "move_0002.dat") and traces.currentRow() == 0
    shots.fill(win, {"act.move_tip.x_nm": "1000000", "act.move_tip.y_nm": "0"})          # outside the piezo range
    run_button(win, "run.move_tip")
    shots.settle(win)
    assert win.status.result.full_text() == "✗ refused: move_tip: outside the piezo range"
    assert traces.count() == 2 and win.viewed == ("spectrum", "move_0002.dat")             # nothing new, nothing replaced


def test_a_move_trace_of_two_points_is_plotted_and_the_longest_info_fits_the_narrowest_centre_column(fonts, open_window):
    win = open_window("en")
    center = win.center
    center.show_spectrum(make_trace("move_0003.dat", n=2))
    assert center.plot._x.size == 2 and center.plot._message == "" and center.current_spectrum_axes() == ("dist", "z_abs")
    assert center.info.text().endswith("· 2 points")
    longest = make_trace("move_0123.dat", frm=(-150.12, -150.12), to=(150.12, 150.12), speed=10000.0, n=12345,
                         bias=-10.0, setpoint=100000.0, t_s=5399.0, feedback=False, index=123)
    center.show_spectrum(longest)
    lines = center.info.text().split(chr(10))
    assert len(lines) == 2 and lines[0].endswith("feedback OFF") and lines[1].endswith("12345 points")
    lines_fit(center.info, lines, center.minimumSizeHint().width())
    empty = {**make_trace(n=2), "dist": np.zeros(1), "t": np.zeros(1)}                    # no column to read it along
    center.show_spectrum(empty)
    assert center.plot._series == [] and center.plot._message == "Nothing shown."


def test_a_move_traces_columns_are_its_own_and_a_spectrums_are_as_they_were():
    from stmbench.gui import center
    trace = make_trace()
    assert center.axis_columns(trace) == (["dist", "t"], ["z_abs", "i"]) and center.default_axes(trace) == ("dist", "z_abs")
    v = np.linspace(-0.5, 0.5, 16)
    sts = {"name": "sts0001.dat", "kind": "sts", "tip_nm": [0.0, 0.0], "t_s": 1.0, "v": v, "i": v * 1e-10, "didv": v ** 2}
    assert center.axis_columns(sts) == (["v", "i", "didv"], ["v", "i", "didv"]) and center.default_axes(sts) == ("v", "didv")
    short_sts = {**sts, "v": v[:4], "i": v[:4], "didv": v[:4]}                  # a spectrum is eight points or more ...
    assert center.spectrum_columns(short_sts) == []
    two = {**trace, **{k: trace[k][:2] for k in ("dist", "t", "z_abs", "i")}}      # ... and a move trace's two
    assert center.spectrum_columns(two) == ["i", "z_abs", "dist", "t"]
    zspec = {"name": "z0001.dat", "kind": "zspec", "tip_nm": [0.0, 0.0], "t_s": 2.0, "z_rel": v, "i": v, "df": v, "amp": v, "z_abs": v}
    assert center.default_axes(zspec)[0] == "z_rel" and center.axis_columns(zspec)[0] == center.axis_columns(zspec)[1]
    assert [center.column_label(c) for c in ("v", "i", "didv", "z_rel", "z_abs", "df", "amp")] == [
        "V", "I (pA)", "dI/dV (nS)", "z rel (pm)", "z (nm)", "Δf (Hz)", "amplitude (pm)"]
    assert [center.column_label(c, lang) for c in ("dist", "t") for lang in ("en", "zh")] == [
        "distance (nm)", "距离 distance (nm)", "time (s)", "时间 time (s)"]


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
    assert len(names) == 26 and any(n.startswith("a_started") for n in names) and any(n.startswith("e_refusal") for n in names)
    assert any(n.startswith("i_busy") for n in names) and any(n.startswith("l_zoomed") for n in names)
    for p in saved:
        img = QImage(str(p))
        assert (img.width(), img.height()) == (shots.DOWNSCALED if p.stem.endswith("1280x720") else shots.SCREEN), p.name
    # the pictures are of the full-screen window (the one an agent is shown): no desktop bar above it, so
    # the first row of pixels is the status strip's frame, not the bar's grey
    top = QImage(str(next(p for p in saved if p.name == "a_started_en.png"))).pixelColor(960, 2)
    assert top.name() != "#2e3436"
    maximized = shots.render_states(tmp_path / "png_dev", "en", work=tmp_path / "work_dev", fullscreen=False)
    dev_top = QImage(str(next(p for p in maximized if p.name == "a_started_en.png"))).pixelColor(960, 2)
    assert dev_top.name() == "#2e3436"                                      # a maximized window leaves the 27 px bar
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


@pytest.mark.parametrize("lang", ["zh", "en"])
@pytest.mark.parametrize("scenario", sorted(p.stem for p in SCENARIO_DIR.glob("P*.yaml")))
def test_every_paper_scenario_opens_with_its_reportable_claims_and_a_report_form_that_fits(fonts, open_window, scenario, lang):
    """In both languages, in the window's fonts and in DejaVu (what the virtual machine has): the longest claims (P1's four,
    in English) leave the form whole and on screen, none of its controls over another, and no scroll bar to reach it. The
    form is four rows tall, the note among them, because P1 in English leaves the page 23 px at 1920x1053 (1 in DejaVu)."""
    win = open_window(lang, scenario=scenario)
    claims = win.session.claims()
    assert claims and set(win.sidebar.report.radios) == {c["id"] for c in claims}
    open_left(win, "report")
    win.wait_idle()
    QApplication.processEvents()
    page = win.sidebar.report
    assert not win.sidebar.report_scroll.verticalScrollBar().isVisible()
    controls = [*page.radios.values(), *page.fields.values(), page.viewed_button, page.submit, page.finish_begin]
    rects = {}
    for w in controls:
        shots.hit(win, w)                                        # reachable without scrolling
        rects[w] = w.rect().translated(w.mapTo(win, w.rect().topLeft()))
        assert w.mapTo(page, w.rect().bottomLeft()).y() <= page.height()
    for i, a in enumerate(controls):
        for b in controls[i + 1:]:
            assert not rects[a].intersects(rects[b]), (a.objectName(), b.objectName())


def test_the_window_shrinks_to_fit_a_screen_its_fonts_would_overflow_and_not_otherwise(app):
    from PySide6.QtCore import QSize

    from stmbench.gui.app import fit_scale
    try:
        assert fit_scale(app, "zh", 1.0, QSize(*shots.WINDOW)) == 1.0           # the screen it was tuned for
        # a maximized window under GNOME's top bar and its own 37 px title bar (1016 of the 1080), and the sealed
        # mode's full-screen window (all 1080, no frame)
        for lang in ("en", "zh"):
            assert fit_scale(app, lang, 1.0, QSize(1920, 1080 - 27)) == 1.0, lang
            assert fit_scale(app, lang, 1.0, QSize(1920, 1080), 0) == 1.0, lang
        small = fit_scale(app, "zh", 1.0, QSize(1500, 900))
        assert 0.7 <= small < 1.0 and style.SCALE == small                       # left applied, for the window to use
        assert fit_scale(app, "zh", small, QSize(1500, 900)) == small           # and it fits: asking again changes nothing
        assert fit_scale(app, "en", 1.3, QSize(2560, 1400)) == 1.3              # a larger scale on a larger screen is kept
    finally:
        style.apply(app, 1.0)


# ── the recent results ──
def test_the_last_five_results_stay_on_screen_and_a_refusal_is_not_pushed_out_by_later_successes(open_window):
    win = open_window("en")
    shots.fill(win, {"act.set_bias.bias_v": "abc", "act.set_setpoint.current_pa": "100", "act.wait.seconds": "1"})
    for name in ("run.set_bias", "run.set_setpoint", "run.wait"):          # three steps: a refusal, then two successes
        run_button(win, name)                                               # (calls after a refusal do not run until it is acknowledged)
        shots.settle(win)
        if win.worker.held:
            acknowledge(win)
    rows = [shots.widget(win, f"recent.{i}") for i in range(3)]
    assert [r.text() for r in rows] == ["✓ wait → waited_s=1", "✓ set_setpoint → setpoint_pa=100",
                                        "✗ refused: set_bias: bias_v must be a number"]
    assert [r.property("tone") for r in rows] == ["ok", "ok", "bad"]        # green, green, red: newest first
    for r in rows:
        shots.hit(win, r)                                                   # all three are in view
    assert not shots.widget(win, "recent.3").isVisible()                    # no row for a call that was not made
    for _ in range(2):                                                      # five calls in all: the refusal is the oldest
        run_button(win, "run.wait")
    shots.settle(win)
    bad = shots.widget(win, "recent.4")
    assert bad.text().startswith("✗ refused: set_bias") and bad.property("tone") == "bad"
    shots.hit(win, bad)
    run_button(win, "run.wait")                                             # a sixth call: the list holds five
    shots.settle(win)
    assert len(win.recent.entries()) == 5 and not any(t == "bad" for t, _ in win.recent.entries())
    assert win.status.result.full_text().startswith("✓ wait")               # the strip still says the last one


def test_the_recent_results_are_under_both_right_tabs_and_do_not_move_between_them(open_window):
    win = open_window("en")
    where = {}
    for tab in ("operate", "spectroscopy", "operate"):
        open_right(win, tab)
        box = shots.widget(win, "recent.box")
        shots.hit(win, box)
        assert win.rect().contains(box.rect().translated(box.mapTo(win, box.rect().topLeft())))
        where[tab] = box.mapTo(win, box.rect().topLeft())
    assert where["operate"] == where["spectroscopy"]
    tabs = win.right
    assert box.mapTo(win, box.rect().topLeft()).y() >= tabs.mapTo(win, tabs.rect().bottomLeft()).y()


def test_a_long_result_wraps_whole_and_the_list_keeps_its_place(open_window):
    win = open_window("en")
    before = shots.widget(win, "recent.box").geometry()
    long = "refused: coarse_move: " + "the tip is in tunnelling and the coarse positioner would crash it; " * 3
    win.recent.push("bad", long)
    win.recent.push("ok", "wait → waited_s=1")
    QApplication.processEvents()
    row = shots.widget(win, "recent.1")
    assert row.wordWrap() and row.text() == "✗ " + long and "…" not in row.text()      # whole, not elided
    assert row.height() > 2 * row.fontMetrics().height()                                  # on several lines
    assert shots.widget(win, "recent.box").geometry() == before                           # nothing else moved
    assert win.recent.scroll.height() >= 6 * win.recent.rows[0].fontMetrics().height()    # room for a wrapped row


# ── hit targets ──
@pytest.mark.parametrize("lang", ["en", "zh"])
def test_every_button_of_a_row_of_choices_is_wide_enough_to_hit_in_a_downscaled_picture(fonts, open_window, lang):
    """56 px at full size is 37 px in the 1280x720 picture a model is shown; the viewer's Z button was 22 there."""
    win = open_window(lang)
    scan_small(win)
    open_right(win, "spectroscopy")
    shots.fill(win, {"act.sts.v_start": "-0.5", "act.sts.v_end": "0.5", "act.sts.points": "32"})
    run_button(win, "run.sts")
    shots.settle(win)                                                          # the spectrum's axis selectors exist now
    seen = 0
    for view in ("spectrum", "frame"):
        if view == "frame":
            win.request_frame(win._data["frames"][-1]["frame"])
            shots.settle(win)
        for tab in ("operate", "spectroscopy"):
            open_right(win, tab)
            for b in win.findChildren(QPushButton):
                if b.property("seg") and b.isVisible():
                    seen += 1
                    assert b.width() >= 56 and b.height() >= 32, (b.objectName(), b.width(), b.height())
    assert seen > 60                                                          # the panels' and the viewer's, both views
    assert shots.widget(win, "view.channel.z").width() >= 56                   # the one that was missed


# ── nothing is cut off ──
def lines_fit(label_or_list, lines: list[str], width: int, font=None) -> None:
    from PySide6.QtGui import QFontMetrics
    fm = QFontMetrics(font or label_or_list.font())
    for line in lines:
        assert fm.horizontalAdvance(line) <= width, (label_or_list.objectName(), line, fm.horizontalAdvance(line), width)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_what_the_viewer_the_profile_and_the_data_tab_say_is_not_cut_off(fonts, open_window, lang):
    win = open_window(lang)
    # a rotated frame away from the origin: the longest numbers the geometry line can carry
    shots.fill(win, {"act.scan.center_x_nm": "-12.534", "act.scan.center_y_nm": "-5.607", "act.scan.size_nm": "8.25",
                     "act.scan.pixels": "80", "act.scan.angle_deg": "30"})
    run_button(win, "run.scan")
    shots.settle(win)
    open_right(win, "spectroscopy")
    shots.fill(win, {"act.sts.v_start": "-0.6", "act.sts.v_end": "0.6", "act.sts.points": "64"})
    run_button(win, "run.sts")
    shots.settle(win)
    info = shots.widget(win, "view.info")
    assert "…" not in info.text() and len(info.text().split(chr(10))) == 2
    lines_fit(info, info.text().split(chr(10)), info.width())                         # the spectrum's two lines
    win.request_frame(win.viewed[1] if win.viewed[0] == "frame" else win._data["frames"][-1]["frame"])
    shots.settle(win)
    assert win.viewed[0] == "frame"
    lines_fit(info, info.text().split(chr(10)), info.width())                         # and the frame's
    assert "angle 30°" in info.text() and "centre (-12.534, -5.607) nm" in info.text()
    fv = win.center.frame_view
    shots.click(win, shots.widget(win, "view.tool.profile"))
    shots.drag(win, fv, fv.pixel_to_widget(5, 5).toPoint(), fv.pixel_to_widget(70, 60).toPoint())
    plot = win.center.small_plot
    reg, bold = plot._fonts()
    title = plot._title
    assert title.startswith("A→B ") and "…" not in title
    lines_fit(plot, [title], plot.width() - 2 * 8, bold)                                # the title is whole
    open_left(win, "data")
    for lst in (shots.widget(win, "data.frames"), shots.widget(win, "data.spectra")):
        assert lst.count() == 1 and not lst.horizontalScrollBar().isVisible()
        text = lst.item(0).text().split(chr(10))
        assert len(text) == 3                                                           # three lines, each whole
        lines_fit(lst, text, lst.viewport().width() - 2 * style.px(6) - 2)
    assert "centre (-12.534, -5.607) nm" in shots.widget(win, "data.frames").item(0).text()


# ── the lists of the Data tab ──
def small_scans(win: MainWindow, n: int) -> list[str]:
    """``n`` small frames at different centres, in the order they were made (the list shows the newest first)."""
    names = []
    for i in range(n):
        shots.fill(win, {**SMALL, "act.scan.pixels": "16", "act.scan.center_x_nm": str(i)})
        run_button(win, "run.scan")
        shots.settle(win)
        names.append(win.viewed[1])
    return names


def first_in_view(lst: QListWidget) -> int:
    """The row of the entry at the top edge of the list, cut or whole."""
    return lst.indexAt(QPoint(6, 1)).row()


def notches(win: MainWindow, lst: QListWidget, n: int) -> None:
    """``n`` wheel notches over the list, down when positive (the harness's direction: a turn towards the reader)."""
    for _ in range(abs(n)):
        shots.wheel(win, lst, delta=-120 if n > 0 else 120)


def test_a_wheel_notch_moves_the_frame_list_by_one_entry_and_the_clicked_entry_stays_in_view(open_window):
    """Screen trial 5, steps 43-48: in the Data tab one wheel notch jumped several entries and eight notches landed at the
    very end, so reaching sim005 among fifteen frames took three steps."""
    win = open_window("en")
    names = small_scans(win, 15)
    open_left(win, "data")
    lst = shots.widget(win, "data.frames")
    assert lst.count() == 15 and lst.verticalScrollBar().maximum() > 3 * lst.visualItemRect(lst.item(0)).height()
    assert first_in_view(lst) == 0 and lst.item(0).text().startswith(names[-1])             # the newest first
    notches(win, lst, 1)
    assert first_in_view(lst) == 1 and lst.visualItemRect(lst.item(1)).top() == 0           # one entry, its top at the edge
    notches(win, lst, 3)
    assert first_in_view(lst) == 4
    notches(win, lst, -1)
    assert first_in_view(lst) == 3
    notches(win, lst, -10)
    assert first_in_view(lst) == 0 and lst.verticalScrollBar().value() == 0                  # up to the first, no further
    notches(win, lst, 8)                                                                     # the eight of the trial
    assert first_in_view(lst) == 8
    notches(win, lst, 60)                                                                    # far beyond the end: it stops there
    end = lst.verticalScrollBar().value()
    assert end == lst.verticalScrollBar().maximum()
    seen = []
    for _ in range(3):                                                                       # and from the end, one entry per notch
        seen.append(first_in_view(lst))
        notches(win, lst, -1)
    assert seen[0] - seen[1] == seen[1] - seen[2] == 1 and first_in_view(lst) == seen[2] - 1
    # a click on an entry that is cut by the bottom edge shows it whole, and shows its frame
    notches(win, lst, -20)
    vp = lst.viewport().rect()
    cut = next(r for r in range(lst.count()) if lst.visualItemRect(lst.item(r)).top() < vp.height() < lst.visualItemRect(lst.item(r)).bottom())
    rect = lst.visualItemRect(lst.item(cut))
    shots.click(win, lst.viewport(), QPoint(rect.center().x(), (rect.top() + vp.height()) // 2))
    shots.settle(win)
    assert vp.contains(lst.visualItemRect(lst.item(cut))), (lst.visualItemRect(lst.item(cut)), vp)
    assert win.viewed == ("frame", names[len(names) - 1 - cut])
    # a control that is held down, or the wheel over the tab bar, does nothing to the list
    before = lst.verticalScrollBar().value()
    shots.wheel(win, win.sidebar.tabBar(), delta=-120)
    assert lst.verticalScrollBar().value() == before


def test_the_spectra_and_the_move_traces_lists_step_one_entry_per_notch_too(open_window):
    win = open_window("en")
    open_right(win, "spectroscopy")
    for i in range(7):
        shots.fill(win, {"act.sts.v_start": "-0.5", "act.sts.v_end": f"{0.5 + i / 10}", "act.sts.points": "8"})
        run_button(win, "run.sts")
        shots.settle(win)
    open_right(win, "operate")
    for i in range(7):
        shots.fill(win, {"act.move_tip.x_nm": str(i + 1), "act.move_tip.y_nm": "0", "act.move_tip.speed_nm_s": "100"})
        run_button(win, "run.move_tip")
        shots.settle(win)
    open_left(win, "data")
    for name, n in (("data.spectra", 7), ("data.traces", 7)):
        lst = shots.widget(win, name)
        assert lst.count() == n
        notches(win, lst, 50)
        last_first = first_in_view(lst)                             # the furthest the list goes
        notches(win, lst, -50)
        assert first_in_view(lst) == 0 and last_first >= 1          # seven entries do not fit in the list
        for k in range(1, min(3, last_first) + 1):
            notches(win, lst, 1)
            assert first_in_view(lst) == k, (name, k)
        notches(win, lst, -10)
        assert first_in_view(lst) == 0


# ── text fields ──
def press_release(win: MainWindow, w: QWidget, dx: int = 0, steps: int = 0) -> None:
    """A click on ``w`` the way the screen harness and a real pointer make it: through the window, the pointer
    arriving first, then press, ``steps`` moves ``dx`` px away with the button down, release."""
    from PySide6.QtCore import QPoint
    left, none = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    qwin = win.windowHandle()
    c = w.mapTo(win, w.rect().center())
    QTest.mouseMove(qwin, c)
    QTest.mousePress(qwin, left, none, c)
    for i in range(1, steps + 1):
        QTest.mouseMove(qwin, QPoint(c.x() + dx * i // steps, c.y()))
    QTest.mouseRelease(qwin, left, none, QPoint(c.x() + dx, c.y()))
    QApplication.processEvents()


def test_every_click_in_a_field_selects_all_of_it_so_that_click_then_type_replaces(open_window):
    win = open_window("en")
    open_left(win, "report")
    fields = ["act.scan.size_nm", "act.scan.pixels", "act.move_tip.x_nm", "report.value", "report.evidence", "report.note"]
    for name in fields:
        w = shots.widget(win, name)
        for prior in ("unfocused", "focused with the whole text selected", "caret in the middle", "caret at the end"):
            for label, args in (("a click", (0, 0)), ("a click with a wobble", (3, 1)), ("a small drag", (30, 5)),
                                ("a pointer that moves a lot", (-40, 8))):
                w.setText("4012")
                w.clearFocus()
                if prior != "unfocused":
                    w.setFocus()
                    w.selectAll() if prior.startswith("focused") else w.setCursorPosition(2 if "middle" in prior else 4)
                    if not prior.startswith("focused"):
                        w.deselect()
                press_release(win, w, *args)
                assert w.selectedText() == "4012", (name, prior, label)
                QTest.keyClicks(w, "7")
                assert w.text() == "7", (name, prior, label, w.text())
    # a second click while the first one's selection is still there, then typing, as in the screen trial
    w = shots.widget(win, "act.scan.size_nm")
    w.setText("60")
    press_release(win, w)
    press_release(win, w)
    QTest.keyClicks(w, "16")
    assert w.text() == "16"
    # the double click selects it all too, and the keyboard still moves the caret (Home, then a digit goes first)
    w.setText("25")
    QTest.mouseDClick(win.windowHandle(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, w.mapTo(win, w.rect().center()))
    assert w.selectedText() == "25"
    QTest.keyClick(w, Qt.Key.Key_Home)
    QTest.keyClicks(w, "1")
    assert w.text() == "125"


def test_a_click_in_a_field_selects_all_of_it_and_the_selection_is_in_the_picture(open_window):
    """Screen trial 4 said that clicking a field does not select its text, so that every edit needed ctrl+a first. Checked with
    the harness's own sequence (the pointer arrives, then press and release through the window): it does, the field has the
    focus, and the highlight is drawn in the picture the agent is shown, not only held by the field."""
    from PySide6.QtGui import QColor
    win = open_window("en")
    f = shots.widget(win, "act.set_bias.bias_v")
    f.setText("0.050")
    press_release(win, f)
    assert f.selectedText() == "0.050" and f.hasFocus()
    img = win.grab().toImage()
    r = f.rect().translated(f.mapTo(win, f.rect().topLeft()))
    accent = QColor(style.COLORS["accent"])
    hits = sum(1 for x in range(r.left() + 3, r.right() - 3) for y in range(r.top() + 3, r.bottom() - 3)
               if img.pixelColor(x, y) == accent)
    assert hits > 200, hits
    QTest.keyClicks(f, "7")
    assert f.text() == "7"                                                      # typing replaced it: no ctrl+a first


# ── the state the buttons show ──
def checked(win: MainWindow, name: str) -> bool:
    return shots.widget(win, name).isChecked()


def test_on_off_pairs_and_the_range_row_show_the_state_the_instrument_reports(open_window):
    win = open_window("en")
    assert checked(win, "run.feedback.on") and not checked(win, "run.feedback.off")            # tunnelling, loop closed
    assert checked(win, "run.set_current_range.10") and not checked(win, "run.set_current_range.1")
    assert not checked(win, "run.drift_comp.on") and checked(win, "run.drift_comp.off")
    run_button(win, "run.feedback.off")
    shots.settle(win)
    assert checked(win, "run.feedback.off") and not checked(win, "run.feedback.on")
    assert "Z feedback OFF" in win.status.plain_text()
    run_button(win, "run.feedback.on")
    run_button(win, "run.set_current_range.1")
    shots.settle(win)
    assert checked(win, "run.feedback.on") and checked(win, "run.set_current_range.1")
    assert not checked(win, "run.set_current_range.10") and "Range 1 nA" in win.status.plain_text()
    shots.fill(win, {"act.drift_comp.vx_nm_per_min": "0.5", "act.drift_comp.vy_nm_per_min": "-0.25"})
    run_button(win, "run.drift_comp.on")                                   # the compensation was not in status() before
    shots.settle(win)
    assert checked(win, "run.drift_comp.on") and not checked(win, "run.drift_comp.off")
    assert "Drift comp. ON (0.5, -0.25) nm/min" in win.status.plain_text()
    run_button(win, "run.drift_comp.off")
    shots.settle(win)
    assert checked(win, "run.drift_comp.off") and "Drift comp. OFF" in win.status.plain_text()
    open_right(win, "spectroscopy")
    assert checked(win, "run.lockin.off")
    run_button(win, "run.lockin.on")
    shots.settle(win)
    assert checked(win, "run.lockin.on") and not checked(win, "run.lockin.off") and "Lock-in ON" in win.status.plain_text()


def test_a_switch_the_rig_does_not_have_shows_neither_state_and_a_refused_click_does_not_stick(open_window):
    win = open_window("en")
    open_right(win, "spectroscopy")
    assert not checked(win, "run.pll.on") and not checked(win, "run.pll.off")          # P4 has no qPlus sensor
    assert "Oscillation n/a" in win.status.plain_text()
    run_button(win, "run.pll.on")                                                       # the instrument refuses
    shots.settle(win)
    assert win.status.result.full_text() == "✗ refused: pll: this instrument has no qPlus sensor"
    assert not checked(win, "run.pll.on") and not checked(win, "run.pll.off")          # the click did not leave it selected
    qp = open_window("en", name="run_qplus", scenario=P5)
    open_right(qp, "spectroscopy")
    assert checked(qp, "run.pll.off") and "Oscillation OFF" in qp.status.plain_text()
    run_button(qp, "run.pll.on")
    shots.settle(qp)
    assert checked(qp, "run.pll.on") and not checked(qp, "run.pll.off") and "Oscillation ON" in qp.status.plain_text()


# ── the screens it opens on ──
@pytest.mark.parametrize("height", [1080, 1016])
@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_window_fits_a_full_screen_and_a_maximized_window_under_a_title_bar(app, fonts, open_window, lang, height):
    """1080: the sealed mode's window is the screen. 1016: a maximized dev window under GNOME's 27 px top bar
    and its 37 px title bar. Nothing scrolls, nothing is hidden, in the window's fonts and in DejaVu, in both
    languages. With Chinese beside English on every label the Operate tab needs 745 px in DejaVu against the right
    column's set 704: the column takes the width from the centre one, and the scale stays at 1.0 where it was
    shrunk to 0.733 and the tabs still overflowed."""
    from PySide6.QtCore import QSize

    from stmbench.gui.app import fit_scale
    screen = QSize(1920, 1080) if height == 1080 else QSize(1920, 1053)           # the dev window leaves the top bar
    assert fit_scale(app, lang, 1.0, screen, 0 if height == 1080 else 37) == 1.0
    win = open_window(lang)
    win.resize(1920, height)
    shots.settle(win)
    QApplication.processEvents()
    need = max(p.widget().minimumSizeHint().width() for p in (win.operate, win.spectroscopy)) + 2 * style.px(3)
    assert win.right.width() == max(704, need) and win.sidebar.width() == 360          # a column is as wide as it needs
    assert win.center.width() >= win.center.minimumSizeHint().width()                 # and the centre one is not squeezed
    for tab, panel in (("operate", win.operate), ("spectroscopy", win.spectroscopy)):
        open_right(win, tab)
        win.wait_idle()
        assert not panel.verticalScrollBar().isVisible() and not panel.horizontalScrollBar().isVisible(), (tab, height)
        vp = panel.viewport().rect()
        for w in panel.widget().findChildren(QWidget):
            if w.isVisible() and w.objectName().startswith(("act.", "run.", "lbl.", "group.")):
                inside = w.rect().translated(w.mapTo(panel.viewport(), w.rect().topLeft()))
                assert vp.contains(inside), (tab, height, w.objectName())
    box = shots.widget(win, "recent.box")
    assert win.rect().contains(box.rect().translated(box.mapTo(win, box.rect().topLeft())))     # the list is on screen
    assert win.status.readings.height() < 2 * win.status.readings.fontMetrics().height()           # one line of readings
    open_left(win, "report")
    QApplication.processEvents()                  # the first show: the bar of the construction-time size is hidden in the next turn
    assert not win.sidebar.report_scroll.verticalScrollBar().isVisible()
    assert win.center.frame_view.height() >= 400                                                  # the image keeps its room


def test_the_side_columns_are_360_and_704_px_wide_and_one_that_needs_more_takes_it_from_the_centre(
        open_window, monkeypatch):
    """The columns of a 1920 px window are 360, 824 and 704 px; a side column its contents need more of is given
    that, and the centre column keeps the rest (whatever the fonts, so the needs are put in by hand here)."""
    from stmbench.gui import app as gui_app
    win = open_window("en")
    assert (win.sidebar.width(), win.center.width(), win.right.width()) == (360, 824, 704)
    monkeypatch.setattr(gui_app, "column_needs", lambda sidebar, panels: (330, 760))
    wide = open_window("en", name="wide")
    assert (wide.sidebar.width(), wide.center.width(), wide.right.width()) == (360, 768, 760)
    monkeypatch.setattr(gui_app, "column_needs", lambda sidebar, panels: (400, 740))
    both_sides = open_window("en", name="both")
    assert (both_sides.sidebar.width(), both_sides.center.width(), both_sides.right.width()) == (400, 748, 740)
    assert gui_app.column_widths(330, 704) == (360, 704) and gui_app.column_widths(361, 705) == (361, 705)


def test_the_window_shrinks_by_what_the_screen_cannot_hold_and_not_by_what_a_wider_column_needs(app, monkeypatch):
    """A column that is too narrow for its contents is not mended by shrinking every size (the column and its
    contents shrink together): with the right column needing 960 px the three columns are 60 to 110 px wider than
    the 1920 px screen, and the scale is 0.9 to 0.99 - what the screen lacks - not the shrink by 0.73 at every pass
    that a ratio of the column to its contents asks for, which never fits."""
    from PySide6.QtCore import QSize

    from stmbench.gui import app as gui_app
    try:
        monkeypatch.setattr(gui_app, "column_needs",
                            lambda sidebar, panels: (round(330 * style.SCALE), round(960 * style.SCALE)))
        scale = gui_app.fit_scale(app, "en", 1.0, QSize(1920, 1080), 0)
        assert 0.9 <= scale < 0.99 and style.SCALE == scale                          # left applied, as it says
        assert gui_app.fit_scale(app, "en", scale, QSize(1920, 1080), 0) == scale    # and it fits
        assert gui_app.fit_scale(app, "en", 1.0, QSize(2560, 1440), 0) == 1.0        # a screen with the room: no shrink
    finally:
        style.apply(app, 1.0)


def test_a_sealed_window_opens_full_screen_and_a_dev_window_maximized(open_window, tmp_path):
    from stmbench.gui.app import show_window
    sealed = open_window("en", sealed=True, seed=SEED)
    dev = open_window("en", name="dev")
    sealed.hide()
    dev.hide()
    show_window(sealed)
    assert sealed.isFullScreen() and not sealed.isMaximized()
    show_window(dev)
    assert dev.isMaximized() and not dev.isFullScreen()
    sealed.showNormal()
    dev.showNormal()


# ── a busy instrument ──
class Gate:
    """A job that holds the worker until the test opens the gate: a scan that is "running" for as long as
    the test wants, whatever the speed of the machine."""

    def __init__(self, win: MainWindow, kind: str = "act", label: str = "scan", args: dict | None = None,
                 refuse: bool = False):
        import threading

        from stmbench.gui.jobs import Outcome
        from stmbench.gui.worker import Job
        self.event = threading.Event()
        self.args = args if args is not None else {"center_x_nm": 0, "center_y_nm": 0, "size_nm": 40, "pixels": 1024}

        def hold(_session):
            self.event.wait(60)
            if refuse:
                return Outcome(kind, label, dict(self.args), ok=False, refused=True, text=f"{label}: held by the test")
            return Outcome(kind, label, dict(self.args), ok=True, reply={"held": True})

        self.win = win
        win._submit(Job(kind, label, dict(self.args), hold))
        end = time.monotonic() + 10
        # until the window has heard that it runs: the worker sets it running a moment before it says so
        while win.status.result.property("tone") != "busy" and time.monotonic() < end:
            QApplication.processEvents()
            time.sleep(0.005)
        QApplication.processEvents()

    def open(self) -> None:
        self.event.set()
        shots.settle(self.win)


def test_the_strip_says_what_runs_and_how_many_wait_and_the_viewer_says_a_scan_is_running(open_window):
    win = open_window("en")
    gate = Gate(win)
    try:
        want = "running: scan centre (0, 0) size 40 nm, 1024 px"
        assert win.status.result.full_text() == want and win.status.result.property("tone") == "busy"
        assert "BUSY: scan" in win.status.plain_text()
        over = shots.widget(win, "view.busy")
        assert over.isVisible() and over.text() == "scan running…"
        assert over.geometry() == win.center.stack.rect()                      # over the whole viewer
        assert not shots.widget(win, "status.cancel").isVisible()               # nothing waits: nothing to cancel
        shots.fill(win, {"act.set_bias.bias_v": "0.1", "act.set_setpoint.current_pa": "100"})
        run_button(win, "run.set_bias")
        run_button(win, "run.set_setpoint")                                     # clicked while the instrument is busy
        QApplication.processEvents()
        assert win.status.result.full_text() == want + " (+2 queued)"
        assert shots.widget(win, "status.cancel").isVisible()
        shots.hit(win, shots.widget(win, "status.cancel"))
    finally:
        gate.open()
    assert win.status.result.full_text().startswith("✓ set_setpoint")           # idle again: the last result
    assert not shots.widget(win, "view.busy").isVisible()
    assert not shots.widget(win, "status.cancel").isVisible()
    assert win.recent.entries()[2] == ("ok", "scan → held=true")


def test_a_scan_that_waits_behind_another_call_is_shown_as_running_too(open_window):
    win = open_window("en")
    gate = Gate(win, "act", "wait", {"seconds": 5})
    try:
        assert not shots.widget(win, "view.busy").isVisible()                  # a wait is not a scan
        shots.fill(win, SMALL)
        run_button(win, "run.scan")
        QApplication.processEvents()
        assert shots.widget(win, "view.busy").isVisible()                      # queued: it will be the next thing running
        assert win.status.result.full_text() == "running: wait seconds=5 (+1 queued)"
    finally:
        gate.open()
    assert not shots.widget(win, "view.busy").isVisible() and win.viewed is not None


def test_cancel_queued_drops_what_has_not_started_and_nothing_else(open_window, tmp_path):
    win = open_window("en")
    gate = Gate(win, "act", "wait", {"seconds": 5})
    try:
        shots.fill(win, {"act.set_bias.bias_v": "0.1", "act.set_setpoint.current_pa": "100", **SMALL})
        for name in ("run.set_bias", "run.set_setpoint", "run.scan"):
            run_button(win, name)
        QApplication.processEvents()
        assert win.worker.waiting == 3 and shots.widget(win, "view.busy").isVisible()    # a scan is among them
        run_button(win, "status.cancel")
        QApplication.processEvents()
        assert win.worker.waiting == 0 and win.worker.running is not None         # the running one is not touched
        assert win.recent.entries()[0] == ("note", "cancelled queued: set_bias, set_setpoint, scan")
        assert not shots.widget(win, "status.cancel").isVisible()
        assert not shots.widget(win, "view.busy").isVisible()                   # the scan will not run
        assert win.status.result.full_text() == "running: wait seconds=5"
        text = shots.widget(win, "log.view").toPlainText()
        assert text.count("cancelled") == 3 and "queued" not in text and "set_bias" in text
    finally:
        gate.open()
    assert win.is_idle() and win.worker.idle
    assert not acts(records(tmp_path))                                           # none of the three reached the instrument
    run_button(win, "run.set_bias")                                               # and the window works as before
    shots.settle(win)
    assert acts(records(tmp_path)) == ["set_bias"]


def test_a_click_on_a_viewer_that_shows_no_frame_leaves_a_note_and_a_frame_takes_markers(open_window):
    win = open_window("en")
    fv = win.center.frame_view
    shots.click(win, fv)
    assert win.recent.entries()[0] == ("note", "no frame shown: nothing marked")
    assert shots.widget(win, "recent.0").property("tone") == "note" and not fv.markers()
    gate = Gate(win)                                                              # the overlay lets the pointer through
    try:
        shots.click(win, fv)
        assert len([e for e in win.recent.entries() if e[1].startswith("no frame")]) == 2
    finally:
        gate.open()
    scan_small(win)
    notes = lambda: [e for e in win.recent.entries() if e[1].startswith("no frame")]        # noqa: E731
    before = notes()
    shots.click(win, fv, fv.pixel_to_widget(16, 16).toPoint())
    assert len(fv.markers()) == 1 and notes() == before                           # a frame takes the click: no note


# ── a refused call ends its step ──
def test_a_refused_call_takes_the_calls_of_its_step_that_were_waiting_with_it(open_window, tmp_path):
    win = open_window("en")
    gate = Gate(win, "act", "wait", {"seconds": 5})                   # holds the worker, so that one step's clicks queue
    try:
        shots.fill(win, {"act.set_bias.bias_v": "abc", "act.set_setpoint.current_pa": "100", "act.wait.seconds": "1"})
        for name in ("run.set_bias", "run.set_setpoint", "run.wait"):
            run_button(win, name)
        QApplication.processEvents()
        assert win.worker.waiting == 3
    finally:
        gate.open()
    # the first of the three was refused: the two behind it did not run, and the strip says what was refused
    assert win.status.result.full_text() == "✗ refused: set_bias: bias_v must be a number"
    assert win.status.result.property("tone") == "bad"
    assert acts(records(tmp_path)) == [] and win.is_idle()
    log = shots.widget(win, "log.view").toPlainText()
    assert log.count("not run: an earlier call was refused") == 2 and "queued" not in log
    top, second = win.recent.entries()[:2]
    assert top == ("note", "not run: an earlier call was refused: set_setpoint, wait")
    assert second == ("bad", "refused: set_bias: bias_v must be a number")
    assert shots.widget(win, "recent.1").property("tone") == "bad"
    band = win.status.plain_text()                                      # the band says what was refused and what did not run
    assert "✗ REFUSED set_bias bias_v=abc — set_bias: bias_v must be a number" in band
    assert "Not run: set_setpoint, wait" in band and "HELD" in band
    acknowledge(win)                                                    # the next step is not touched
    run_button(win, "run.wait")
    shots.settle(win)
    assert acts(records(tmp_path)) == ["wait"] and win.status.result.full_text().startswith("✓ wait")


def test_a_refusal_made_by_the_instruments_own_checks_ends_the_step_too(open_window, tmp_path):
    win = open_window("en")
    gate = Gate(win, "act", "wait", {"seconds": 5})
    try:
        shots.fill(win, {"act.move_tip.x_nm": "1000000", "act.move_tip.y_nm": "0", "act.set_setpoint.current_pa": "100"})
        run_button(win, "run.move_tip")
        run_button(win, "run.set_setpoint")
        QApplication.processEvents()
    finally:
        gate.open()
    assert win.status.result.full_text() == "✗ refused: move_tip: outside the piezo range"
    recs = records(tmp_path)
    assert [r["kind"] for r in recs if r["kind"] != "observe"] == ["refused"]      # the refusal is on record, set_setpoint is not
    assert win.recent.entries()[0] == ("note", "not run: an earlier call was refused: set_setpoint")


def test_a_refused_report_stops_the_end_clicked_behind_it_and_the_button_comes_back(open_window, tmp_path):
    win = open_window("en")
    gate = Gate(win, "act", "wait", {"seconds": 5})
    try:
        open_left(win, "report")
        run_button(win, "report.submit")                                  # no claim, no value: refused
        run_button(win, "finish.begin")
        run_button(win, "finish.confirm")                                 # the end, asked for in the same step
        QApplication.processEvents()
        assert win.worker.waiting == 2
    finally:
        gate.open()
    assert win.status.result.full_text().startswith("✗ refused: report:")
    assert not win.session.finished and win.recent.entries()[0][1].startswith("not run: an earlier call was refused: finish")
    assert shots.widget(win, "finish.begin").isVisible() and not shots.widget(win, "finish.confirm").isVisible()


def refuse_set_bias(win: MainWindow) -> None:
    """One refused call: Set bias with a field that is not a number."""
    shots.fill(win, {"act.set_bias.bias_v": "abc"})
    run_button(win, "run.set_bias")
    shots.settle(win)
    assert win.worker.held


def test_the_trials_sequence_set_setpoint_over_the_limit_a_wait_then_move_tip_moves_nothing(open_window, tmp_path):
    """Screen trials 4 (step 17) and 5 (step 19): Set setpoint 200000 pA is refused (the limit is 100000), the agent's
    own wait of four seconds follows, then Move tip. The three-second guard had run out during the wait and the drag
    ran at the old setpoint, a wasted attempt each time. The console now holds until the refusal is acknowledged,
    whatever the wait; there is no timer to run out."""
    win = open_window("en")
    shots.fill(win, {"act.set_setpoint.current_pa": "200000", "act.move_tip.x_nm": "4", "act.move_tip.y_nm": "0",
                     "act.move_tip.speed_nm_s": "2"})
    run_button(win, "run.set_setpoint")
    pause(4.0)                                                        # the wait of the step: longer than the old guard
    words = "set_setpoint: current_pa = 200000.0 is outside [0.1, 100000.0]"
    assert win.worker.held and win.status.result.full_text() == "✗ refused: " + words
    run_button(win, "run.move_tip")
    shots.settle(win)
    assert acts(records(tmp_path)) == []                              # the drag did not run, at any setpoint
    assert "Tip (0.000, 0.000) nm" in win.status.plain_text()
    # what the screenshot after the step shows: a red band with the words, the call and what did not run, the pill
    # saying HELD, and the button that acknowledges it
    plain = win.status.plain_text()
    assert f"✗ REFUSED set_setpoint current_pa=200000 — {words}" in plain
    assert "Not run: move_tip" in plain and "HELD" in plain and "held until this is acknowledged" in plain
    for name in ("refusal.band", "refusal.text", "refusal.ack"):
        shots.hit(win, shots.widget(win, name))
    assert shots.widget(win, "status.state").property("tone") == "held"
    assert win.recent.entries()[0] == ("note", "not run: the refusal is not acknowledged: move_tip")
    assert win.recent.entries()[1] == ("bad", "refused: " + words)
    log = shots.widget(win, "log.view").toPlainText()
    assert "move_tip x_nm=4, y_nm=0, speed_nm_s=2" in log and "not run: the refusal is not acknowledged" in log
    # every click turned away is listed, in the same growing note and on the band, and the refusal stays on the strip
    pause(1.0)
    run_button(win, "run.move_tip")
    run_button(win, "run.wait")
    shots.settle(win)
    assert acts(records(tmp_path)) == []
    assert win.recent.entries()[0] == ("note", "not run: the refusal is not acknowledged: move_tip, move_tip, wait")
    assert "Not run: move_tip, move_tip, wait" in win.status.plain_text()
    assert win.status.result.full_text() == "✗ refused: " + words and win.status.result.property("tone") == "bad"
    # the reader's acknowledgement lifts the hold, and only that: the next click runs
    acknowledge(win)
    assert "HELD" not in win.status.plain_text() and "Not run" not in win.status.plain_text()
    run_button(win, "run.move_tip")
    shots.settle(win)
    assert acts(records(tmp_path)) == ["move_tip"] and win.status.result.full_text().startswith("✓ move_tip")


def test_the_hold_has_no_timer_in_it():
    import inspect

    from stmbench.gui import worker
    assert not hasattr(worker, "GUARD_S") and not hasattr(worker.Worker, "guard_s")
    assert list(inspect.signature(worker.Worker.__init__).parameters) == ["self", "session", "parent"]
    assert "monotonic" not in inspect.getsource(worker) and not hasattr(worker, "time")


def test_reads_tab_changes_and_the_viewers_tools_are_not_held_and_the_close_still_ends_the_episode(open_window, tmp_path):
    win = open_window("en")
    name = scan_small(win)
    open_right(win, "spectroscopy")
    shots.fill(win, {"act.sts.v_start": "-0.5", "act.sts.v_end": "0.5", "act.sts.points": "16"})
    run_button(win, "run.sts")
    shots.settle(win)
    spectrum = win.viewed[1]
    open_right(win, "operate")
    refuse_set_bias(win)
    # a read (the current channel of the frame is not in the cache: the instrument is asked), a spectrum, the tabs
    # and the viewer's own tools all work while the console holds
    win.request_frame(name)
    shots.settle(win)
    run_button(win, "view.channel.current")
    shots.settle(win)
    assert win.viewed == ("frame", name) and win.center.current_channel() == "current"
    open_left(win, "data")
    spectra = shots.widget(win, "data.spectra")
    shots.click(win, spectra.viewport(), spectra.visualItemRect(spectra.item(0)).center())
    shots.settle(win)
    assert win.viewed == ("spectrum", spectrum)
    win.request_frame(name)
    shots.settle(win)
    assert win.viewed == ("frame", name)
    for tab in ("report", "log", "task", "manual"):
        open_left(win, tab)
    open_right(win, "spectroscopy")
    for b in ("view.tool.profile", "view.flatten.plane", "view.zoom.2", "view.tool.marker", "view.clear_markers"):
        run_button(win, b)
    assert win.worker.held and shots.widget(win, "refusal.band").isVisible()        # nothing of this lifted it
    # the end of the episode, when the window is closed, runs while the console holds
    assert win.close()
    assert win.session.finished and (tmp_path / "run" / "episode.json").is_file()
    assert read_export(tmp_path / "run" / "export")["session"]["finished"]


def test_nothing_turned_away_reaches_the_record_and_the_acknowledgement_is_in_no_record_but_the_log(open_window, tmp_path):
    win = open_window("en")
    shots.fill(win, {"act.set_setpoint.current_pa": "200000", "act.move_tip.x_nm": "4", "act.move_tip.y_nm": "0",
                     "act.wait.seconds": "1"})
    run_button(win, "run.set_setpoint")
    shots.settle(win)
    for name in ("run.move_tip", "run.wait", "run.scan"):
        run_button(win, name)
    shots.settle(win)
    open_left(win, "report")
    run_button(win, "report.submit")
    shots.settle(win)
    acknowledge(win)
    run_button(win, "run.wait")
    shots.settle(win)
    assert "acknowledged the refusal of set_setpoint current_pa=200000" in shots.widget(win, "log.view").toPlainText()
    win.shutdown(finish=True)
    recs = records(tmp_path)
    assert [(r["kind"], r.get("name")) for r in recs if r["kind"] != "observe"] == [
        ("refused", "set_setpoint"), ("act", "wait"), ("finish", None)]            # the four turned-away clicks are not here
    assert [r["name"] for r in recs if r["kind"] == "act"] == ["wait"]
    for folder in (tmp_path / "run" / "export", tmp_path / "run"):
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix in (".json", ".jsonl", ".txt", ".log"):
                text = p.read_text(encoding="utf-8").lower()
                assert "acknowledg" not in text and "turned away" not in text and "not acknowledged" not in text, p.name
    ledger = [json.loads(line) for line in (tmp_path / "run" / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["name"] for r in ledger if r["kind"] == "act"] == ["wait"]


def test_an_episode_with_a_refusal_held_in_it_replays_and_grades_as_the_calls_that_ran(open_window, tmp_path):
    """Nothing turned away reaches the instrument, so an episode played through the window with a held refusal in it
    replays to the same measurements: the grading host, which holds the seed, finds the export consistent and the
    episode finished, and counts as actions the calls that ran."""
    from stmbench.instrument.grade import grade_export
    win = open_window("en", seed=3, name="held")
    shots.fill(win, {"act.set_setpoint.current_pa": "200000", "act.move_tip.x_nm": "4", "act.move_tip.y_nm": "0"})
    run_button(win, "run.set_setpoint")
    shots.settle(win)
    run_button(win, "run.move_tip")                                             # turned away: the setpoint was refused
    shots.settle(win)
    acknowledge(win)
    scan_small(win)
    win.shutdown(finish=True)
    export = tmp_path / "held" / "export"
    graded = grade_export(export, P4, 3)
    assert graded["replay"]["consistent"] and graded["finished_in_episode"] is True
    assert graded["actions"] == 1                                               # the scan: not the refusal, not the click turned away
    assert not grade_export(export, P4, 4)["replay"]["consistent"]              # and it is this seed's world that it replays


def test_the_band_takes_the_place_of_two_lines_so_nothing_moves_and_its_button_is_clear_of_cancel_queued(open_window):
    win = open_window("en")
    strip = win.status
    h0 = strip.height()
    corner = lambda n: shots.widget(win, n).mapTo(win, shots.widget(win, n).rect().topLeft())        # noqa: E731
    where = {n: corner(n) for n in ("run.set_setpoint", "run.scan", "recent.box", "right.tabs", "sidebar")}
    gate = Gate(win, "act", "wait", {"seconds": 5}, refuse=True)             # a refusal that comes when the test says
    try:
        shots.fill(win, {"act.wait.seconds": "1"})
        run_button(win, "run.wait")                                         # queued behind it: Cancel queued is there
        QApplication.processEvents()
        cancel = shots.widget(win, "status.cancel")
        shots.hit(win, cancel)
        cancel_rect = cancel.rect().translated(cancel.mapTo(win, cancel.rect().topLeft()))
    finally:
        gate.open()
    assert win.worker.held
    ack = shots.widget(win, "refusal.ack")
    ack_rect = ack.rect().translated(ack.mapTo(win, ack.rect().topLeft()))
    assert not ack_rect.intersects(cancel_rect), (ack_rect, cancel_rect)       # a click aimed at Cancel queued is not an ack
    assert ack.width() >= 56 and ack.height() >= 32
    assert strip.height() == h0                                              # the band is as tall as the two lines it replaces
    assert {n: corner(n) for n in where} == where
    assert not shots.widget(win, "status.readings").isVisible() and not shots.widget(win, "status.result_row").isVisible()
    for tab in ("spectroscopy", "operate"):                                  # on screen whichever tab is open
        open_right(win, tab)
        shots.hit(win, ack)
    for tab in ("report", "data", "log", "manual", "task"):
        open_left(win, tab)
        shots.hit(win, ack)
    # where the button is, no control is when nothing is refused: the readings are there
    centre = ack.mapTo(win, ack.rect().center())
    acknowledge(win)
    node = win.childAt(centre)
    while node is not None and not isinstance(node, (QPushButton, QLineEdit)):
        node = node.parentWidget()
    assert node is None, node and node.objectName()
    assert strip.height() == h0 and shots.widget(win, "status.readings").isVisible()
    assert {n: corner(n) for n in where} == where


def test_a_report_and_the_end_clicked_while_the_console_holds_are_not_run_and_the_end_button_comes_back(open_window, tmp_path):
    win = open_window("en")
    name = scan_small(win)
    refuse_set_bias(win)
    open_left(win, "report")
    run_button(win, "report.claim.target_site")
    shots.fill(win, {"report.value": "1.5", "report.x_nm": "1.5", "report.y_nm": "0", "report.frame": name,
                     "report.evidence": name})
    run_button(win, "report.submit")
    run_button(win, "finish.begin")
    run_button(win, "finish.confirm")
    shots.settle(win)
    assert not win.session.finished and shots.widget(win, "finish.begin").isVisible()
    assert win.sidebar.report.reply.text() == "✗ not run: the refusal is not acknowledged"
    assert not [r for r in records(tmp_path) if r["kind"] in ("report", "finish")]
    assert "Not run: report_result, finish" in win.status.plain_text()
    acknowledge(win)
    run_button(win, "report.submit")
    shots.settle(win)
    assert win.sidebar.report.reply.text().startswith("✓ recorded target_site")
    run_button(win, "finish.begin")
    run_button(win, "finish.confirm")
    shots.settle(win)
    assert win.session.finished


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_band_shows_a_long_refusal_and_a_long_list_whole_in_the_widest_fonts_too(fonts, open_window, lang):
    from stmbench.gui.jobs import Outcome
    from stmbench.gui.worker import Job
    win = open_window(lang)
    h0 = win.status.height()
    long = ("move_tip: the tip is in tunnelling and the coarse positioner would crash it; " * 2).strip()
    args = {"x_nm": 1.0, "y_nm": 2.0, "speed_nm_s": 3.0}

    def refuse(_s):
        return Outcome("act", "move_tip", dict(args), ok=False, refused=True, text=long)
    win._submit(Job("act", "move_tip", dict(args), refuse))
    shots.settle(win)
    for _ in range(3):                                                        # the layout settles in a few turns of the loop
        QApplication.processEvents()
    assert win.worker.held
    text = shots.widget(win, "refusal.text")
    assert long in win.status.plain_text()
    assert text.heightForWidth(text.width()) <= text.height()                  # every line of it is shown
    for n in range(12):                                                       # a long step: eight calls are named, then a count
        run_button(win, "run.wait" if n % 2 else "run.set_bias")
    shots.settle(win)
    for _ in range(3):
        QApplication.processEvents()
    plain = win.status.plain_text()
    assert ("Not run" in plain or "未执行" in plain) and "… +4" in plain
    assert text.heightForWidth(text.width()) <= text.height()
    assert win.recent.entries()[0][1].count(", ") == 11                       # the note in the recent results names all twelve
    shots.hit(win, shots.widget(win, "refusal.ack"))
    assert win.status.height() >= h0


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
    # the export the window wrote while it was played is what the grading host replays: it must be the episode
    from stmbench.instrument.grade import grade_export
    graded = grade_export(tmp_path / "screen" / "export", P4, seed)
    assert graded["replay"]["consistent"] and graded["success"] is True and graded["finished_in_episode"] is True
    assert graded["claims_verified"] == graded["claims_total"] == 2
    assert graded["instrument_time_s"] == pytest.approx(direct["instrument_time_s"], abs=1e-3)   # the grade rounds to ms
    assert graded["actions"] == direct["actions"] == 56
    assert not grade_export(tmp_path / "screen" / "export", P4, seed + 1)["replay"]["consistent"]   # another seed's world


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
