"""The usage hint in the viewer's readout line: while the line has no reading it says how the viewer is used.

Written after screen trial 6 (2026-11-06), the first agent that knew only the screen: it never hovered over or
clicked the image and read every coordinate off the axis ticks, missing its first atom by 0.02-0.03 nm. The user
decided on 2026-11-16 that the empty line shows a grey hint. It is about the viewer, not the experiment.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from stmbench.gui import shots, style  # noqa: E402
from stmbench.gui.labels import one  # noqa: E402
from tests.gui_helpers import app, pixel_point, run_button, scan  # noqa: E402, F401
from tests.test_gui_app import fonts, make_trace, open_window  # noqa: E402, F401

LONGEST = "pixel col 1023, row 1023   centre x -150.123 nm   y -150.123 nm   -1234.56 pm"     # the widest reading
TOOL_KEYS = {"marker": "hint_image", "profile": "hint_image_profile", "pan": "hint_image_pan"}


def is_hint(win, key: str = "hint_image") -> bool:
    r = win.center.readout
    return r.full_text() == one(key, win.lang) and r.property("tone") == "note"


def colour(win) -> QColor:
    r = win.center.readout
    return r.palette().color(r.foregroundRole())


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_hint_shows_at_start_gives_way_to_the_reading_and_returns_when_the_pointer_leaves(open_window, lang):
    win = open_window(lang)
    c = win.center
    assert c.readout.full_text() == "" and not c.readout.property("tone")     # no frame yet: a click marks nothing
    scan(win)
    assert c.readout.full_text() == ("悬停读坐标和数值，点击放标记，滚轮以指针处为中心缩放" if lang == "zh" else
                                     "Hover to read; click to place a marker; wheel to zoom about the pointer")
    assert is_hint(win)                                                  # with the first frame, and again with a new one
    height = c.readout.height()
    QTest.mouseMove(c.frame_view, pixel_point(win, 10, 12))
    assert c.readout.full_text().startswith("pixel col 10, row 12") and c.readout.property("tone") == ""
    assert c.readout.height() == height                                  # nothing moved
    QApplication.sendEvent(c.frame_view, QEvent(QEvent.Type.Leave))
    assert is_hint(win)
    QTest.mouseMove(c.frame_view, pixel_point(win, 10, 12))
    QTest.mouseMove(c.frame_view, QPoint(1, 1))                          # off the image, onto its margin
    assert is_hint(win)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_hint_is_muted_and_the_reading_is_not(open_window, lang):
    win = open_window(lang)
    scan(win)
    assert colour(win).name() == QColor(style.COLORS["muted"]).name()
    QTest.mouseMove(win.center.frame_view, pixel_point(win, 10, 12))
    assert colour(win).name() == QColor(style.COLORS["ink"]).name()


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_hint_names_what_the_picked_tool_does(open_window, lang):
    win = open_window(lang)
    run_button(win, "view.tool.profile")
    assert win.center.readout.full_text() == ""                          # no frame yet
    scan(win)
    for tool, key in TOOL_KEYS.items():
        run_button(win, f"view.tool.{tool}")
        assert is_hint(win, key), tool
    run_button(win, "view.tool.marker")
    run_button(win, "view.tool.pan")
    assert is_hint(win, "hint_image_pan")
    QTest.mouseMove(win.center.frame_view, pixel_point(win, 10, 12))
    run_button(win, "view.tool.marker")                                  # a reading on show is not replaced
    assert win.center.readout.full_text().startswith("pixel col")


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_a_spectrum_or_trace_has_the_plot_hint_and_a_new_frame_brings_the_image_one_back(open_window, lang):
    win = open_window(lang)
    scan(win)
    win.center.show_spectrum(make_trace("move_0001.dat", n=20))
    assert is_hint(win, "hint_plot")
    win.center.show_spectrum({**make_trace(n=2), "dist": np.zeros(1), "t": np.zeros(1)})     # nothing to plot
    assert is_hint(win, "hint_plot")
    scan(win)
    assert is_hint(win)


def test_the_sealed_mode_shows_the_hint_too_from_its_first_frame(open_window):
    win = open_window("en", sealed=True)
    assert win.center.readout.full_text() == ""
    scan(win)
    assert is_hint(win)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_no_hint_is_elided_and_none_is_wider_than_the_longest_reading(fonts, open_window, lang):
    """The window is the narrowest one the tests open (1920 px; DejaVu's wider fonts in the second run): every hint
    shows whole, and none needs more width than the longest reading, which the viewer already has to fit."""
    win = open_window(lang)
    scan(win)
    c = win.center
    longest = c.readout.fontMetrics().horizontalAdvance(LONGEST)
    for tool, key in TOOL_KEYS.items():
        run_button(win, f"view.tool.{tool}")
        shots.settle(win)
        assert c.readout.text() == one(key, lang) and "…" not in c.readout.text(), key
        assert c.readout.fontMetrics().horizontalAdvance(one(key, lang)) <= longest, key
    c.show_spectrum(make_trace("move_0001.dat", n=20))
    shots.settle(win)
    assert c.readout.text() == one("hint_plot", lang)
    assert c.readout.fontMetrics().horizontalAdvance(one("hint_plot", lang)) <= longest
