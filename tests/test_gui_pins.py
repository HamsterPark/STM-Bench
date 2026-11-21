"""Readings pinned on a plot (``PlotView``): a click pins the reading at the nearest data point, several clicks several
pins, each numbered and listed in the numbers the hover readout gives; the viewer's Clear plot readings button and the next
data remove them. Nothing is found: the pins are where the reader clicked.

Written for the round after screen trial 3 (step 44: the trace's hover readout gives exact values, "but only one per
step", because the screenshot comes after the last event).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Offscreen Qt on Windows has no fonts unless it is told where they are: every glyph becomes a box
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", str(Path(os.environ.get("WINDIR", "")) / "Fonts"))

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QFont, QFontMetrics, QImage, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from stmbench.gui.widgets import PlotView  # noqa: E402
from stmbench.gui.widgets import plot_view as pv  # noqa: E402

NAMES = ["dI/dV (nS)", "dI/dV back (nS)", "I (pA)"]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def make_plot(n: int = 201, size=(640, 340), series: int = 1) -> tuple[PlotView, np.ndarray, list[np.ndarray]]:
    plot = PlotView()
    font = QFont("Arial")
    font.setPixelSize(16)
    plot.setFont(font)
    plot.resize(*size)
    x = np.linspace(-1.0, 1.0, n)
    ys = [5.0 + 2.0 * np.exp(-((x + 0.43) / 0.08) ** 2) + 0.7 * k * x for k in range(series)]
    plot.set_series(x, list(zip(NAMES, ys)), "Bias (V)", "dI/dV (nS)", "spectrum")
    return plot, x, ys


def click(plot, pos, button=Qt.MouseButton.LeftButton):
    for kind, buttons in ((QEvent.Type.MouseButtonPress, button), (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton)):
        QApplication.sendEvent(plot, QMouseEvent(kind, QPointF(pos), QPointF(plot.mapToGlobal(QPointF(pos).toPoint())), button,
                                                 buttons, Qt.KeyboardModifier.NoModifier))


def move(plot, pos):
    QApplication.sendEvent(plot, QMouseEvent(QEvent.Type.MouseMove, QPointF(pos), QPointF(plot.mapToGlobal(QPointF(pos).toPoint())),
                                             Qt.MouseButton.NoButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))


def rgb(plot) -> np.ndarray:
    img = plot.grab().toImage().convertToFormat(QImage.Format.Format_RGB32)
    buf = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)
    return buf[:, :img.width(), [2, 1, 0]].copy()


def reading(i, x, ys, names=NAMES):
    return {"index": i, "x": float(x[i]), "values": [[nm, float(y[i])] for nm, y in zip(names, ys)]}


def test_several_clicks_pin_several_readings_in_order_each_reported_and_a_repeat_adds_nothing(app):
    plot, x, ys = make_plot()
    got = []
    plot.point_picked.connect(got.append)
    for i in (46, 120, 170):
        click(plot, plot.point_position(i))
    assert [p["index"] for p in plot.pinned_points()] == [46, 120, 170]
    assert plot.pinned_points() == [reading(i, x, ys, NAMES[:1]) for i in (46, 120, 170)]
    assert got == plot.pinned_points() and plot.pinned_point() == reading(170, x, ys, NAMES[:1])
    click(plot, plot.point_position(46))                                       # a pinned point again: no second pin
    assert [p["index"] for p in plot.pinned_points()] == [46, 120, 170]
    assert plot.pinned_point()["index"] == 46 and len(got) == 4                # ... but it is the last one clicked, and reported
    for i in (10, 20, 30, 40):
        click(plot, plot.point_position(i))                                    # beyond the most that are kept: the oldest go
    assert len(plot.pinned_points()) == pv.MAX_PINS == 6
    assert [p["index"] for p in plot.pinned_points()] == [120, 170, 10, 20, 30, 40]        # 46 and nothing else was dropped
    # right or middle clicks, and clicks off the axes, pin nothing
    n = len(plot.pinned_points())
    r = plot.plot_rect()
    click(plot, plot.point_position(60), Qt.MouseButton.RightButton)
    click(plot, QPointF(r.left() - 4, r.center().y()))
    assert len(plot.pinned_points()) == n


def test_the_readout_lists_every_pin_in_the_numbers_the_hover_gives_and_clear_pins_empties_it(app):
    plot, x, ys = make_plot(series=2, size=(980, 340))                          # wide enough for every field of every line
    fresh = rgb(plot)
    for i in (46, 120, 170):
        click(plot, plot.point_position(i))
    pins = [(str(k + 1), i) for k, i in enumerate((46, 120, 170))]
    mfm = QFontMetrics(plot._mono_font())
    lines = plot._list_lines(pins, mfm, plot.plot_rect(), 6)
    assert len(lines) == 3
    for (tag, i), line in zip(pins, lines):
        assert line.startswith(tag + "  ")
        fields = plot._fields(i)
        for _colour, label, number in fields:
            assert f"{label} = " in line and number in line                    # the hover's label and the hover's digits
    assert len({len(line) for line in lines}) == 1                             # aligned: the columns have one width
    assert not np.array_equal(rgb(plot), fresh)
    plot.clear_pins()
    assert plot.pinned_points() == [] and plot.pinned_point() is None
    assert plot.cursor_point()["index"] == 170                                 # the pointer is still over the last point it pinned
    leave = QEvent(QEvent.Type.Leave)
    QApplication.sendEvent(plot, leave)
    assert plot.cursor_point() is None
    assert np.array_equal(rgb(plot), fresh)                                    # nothing left over: no line, dot, number or box
    plot.clear_pins()                                                          # and again: nothing to do
    click(plot, plot.point_position(30))
    plot.set_series(x, [(NAMES[0], ys[0])], "Bias (V)", "dI/dV (nS)", "spectrum")
    assert plot.pinned_points() == []                                          # new data removes them as it did the pin


def test_the_hover_reads_the_pointers_point_as_a_last_line_when_it_is_not_pinned_and_pins_stay(app):
    plot, x, ys = make_plot()
    for i in (46, 120):
        click(plot, plot.point_position(i))
    move(plot, plot.point_position(150))
    assert plot.cursor_point() == reading(150, x, ys, NAMES[:1]) and [p["index"] for p in plot.pinned_points()] == [46, 120]
    entries = [("1", 46), ("2", 120), (">", 150)]
    lines = plot._list_lines(entries, QFontMetrics(plot._mono_font()), plot.plot_rect(), 6)
    assert lines[2].startswith(">  ") and f"{plot._fields(150)[1][2]}" in lines[2]
    QApplication.sendEvent(plot, QEvent(QEvent.Type.Leave))
    assert plot.cursor_point()["index"] == 120                                 # the last pin, as the single pin was


def dot_visible(plot, shot, i: int, k: int) -> bool:
    colour = pv.SERIES_COLORS[k]
    pt = plot.point_position(i, k)
    c = np.array([int(colour[j:j + 2], 16) for j in (1, 3, 5)])
    cx, cy = int(round(pt.x())), int(round(pt.y()))
    return bool((np.abs(shot[cy - 3:cy + 4, cx - 3:cx + 4].astype(int) - c).sum(axis=2) < 40).any())


def test_every_pin_is_drawn_a_line_and_a_dot_on_the_curve_and_neither_the_box_nor_a_number_hides_one(app):
    """The box takes the corner that hides the fewest dots, and a number stands where it covers none: with the
    pins where a corner is clear, every dot of every pin shows."""
    for pins in ((46, 120, 170), (150, 160, 170), (10, 20, 30), (120, 140, 170)):
        plot, x, ys = make_plot(series=2)
        for i in pins:
            click(plot, plot.point_position(i))
        shot = rgb(plot)
        for i in pins:
            for k in (0, 1):
                assert dot_visible(plot, shot, i, k), f"the dot of the pin at {i}, curve {k}, is hidden (pins {pins})"
            col = int(round(plot.point_position(i).x()))
            r = plot.plot_rect()
            line = shot[int(r.top()) + 2:int(r.bottom()) - 2, col - 1:col + 2]
            assert (line.sum(axis=2) < 150).any(axis=1).mean() > 0.2           # the dark line down the plot there


def test_crowded_pins_keep_their_lines_and_their_numbers_in_the_list_when_no_corner_is_clear(app):
    plot, x, ys = make_plot(series=2)
    pins = (60, 75, 90, 105)                                                   # spread over the curves: every corner hides something
    for i in pins:
        click(plot, plot.point_position(i))
    shot = rgb(plot)
    assert sum(dot_visible(plot, shot, i, k) for i in pins for k in (0, 1)) >= 4          # the box hides some of eight, not most
    lines = plot._list_lines([(str(n + 1), i) for n, i in enumerate(pins)], QFontMetrics(plot._mono_font()), r_ := plot.plot_rect(), 6)
    assert len(lines) == 4 and r_.width() > 0                                           # and still says all four, in numbers
    r = plot.plot_rect()
    for i in pins:
        col = int(round(plot.point_position(i).x()))
        assert (shot[int(r.top()) + 2:int(r.bottom()) - 2, col - 1:col + 2].sum(axis=2) < 150).any()


def test_a_small_plot_lists_what_fits_and_a_tiny_one_draws_without_failing(app):
    plot, x, ys = make_plot(size=(330, 150))                                    # the line profile beside the marker table
    for i in (20, 60, 100, 140, 180):
        click(plot, plot.point_position(i))
    assert len(plot.pinned_points()) == 5
    assert rgb(plot).shape == (150, 330, 3)
    tiny = PlotView()
    tiny.resize(120, 60)
    tiny.set_series([0, 1, 2, 3], [("a", [1, 3, 2, 4])], "x", "y")
    for i in (0, 1, 2):
        click(tiny, tiny.point_position(i))
    assert rgb(tiny).shape == (60, 120, 3) and len(tiny.pinned_points()) == 3
    narrow = PlotView()
    narrow.resize(200, 300)
    narrow.set_series(np.linspace(0, 1, 50), [("a very long series label indeed", np.linspace(0, 1, 50) ** 2)],
                      "a long x label (V)", "pA")
    for i in (5, 25, 45):
        click(narrow, narrow.point_position(i))
    mfm = QFontMetrics(narrow._mono_font())
    lines = narrow._list_lines([("1", 5), ("2", 25), ("3", 45)], mfm, narrow.plot_rect(), 6)
    assert len(lines) == 3 and all("=" not in t or "a very long" not in t for t in lines)    # the long label gave way
    assert all(mfm.horizontalAdvance(t) + 12 <= narrow.plot_rect().width() - 12 for t in lines)    # and no line is wider than the axes
