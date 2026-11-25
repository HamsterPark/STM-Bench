"""The plot's wheel zoom, and where its readout box goes.

Written for the round after screen trial 23 (2026-11-25): the agent tried the wheel on a spectrum (the viewer's hint says
the wheel zooms the image about the pointer, and the plot dropped it), and the small profile plot's hover box was drawn
over the peak it was reading, or fell back to the distance alone when it did not fit.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", str(Path(os.environ.get("WINDIR", "")) / "Fonts"))

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QFont, QImage, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from stmbench.gui.widgets import PlotView  # noqa: E402
from stmbench.gui.widgets import plot_view as pv  # noqa: E402
from tests.test_gui_pins import app, click, make_plot, move  # noqa: E402, F401


def wheel(plot: PlotView, pos, delta: int = 120, mods=Qt.KeyboardModifier.NoModifier) -> QWheelEvent:
    pos = QPointF(pos)
    ev = QWheelEvent(pos, QPointF(plot.mapToGlobal(pos.toPoint())), QPoint(0, 0), QPoint(0, delta),
                     Qt.MouseButton.NoButton, mods, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(plot, ev)
    return ev


def x_at(plot: PlotView, pos) -> float:
    lay = plot._layout_now()
    return lay.x0 + (pos.x() - lay.plot.left()) / lay.plot.width() * (lay.x1 - lay.x0)


def at(plot: PlotView, frac: float) -> QPointF:
    r = plot.plot_rect()
    return QPointF(r.left() + frac * r.width(), r.center().y())


def plain(n: int = 101, size=(640, 340)) -> tuple[PlotView, np.ndarray, np.ndarray]:
    """y = x squared on 0..10: what is visible at the left end is small, at the right end large."""
    plot = PlotView()
    font = QFont("Arial")
    font.setPixelSize(16)
    plot.setFont(font)
    plot.resize(*size)
    x = np.linspace(0.0, 10.0, n)
    y = x ** 2
    plot.set_series(x, [("y", y)], "x", "y", "t")
    return plot, x, y


def rgb(plot) -> np.ndarray:
    img = plot.grab().toImage().convertToFormat(QImage.Format.Format_RGB32)
    buf = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)
    return buf[:, :img.width(), [2, 1, 0]].copy()


# ── zoom about the pointer ──
def test_a_notch_up_halves_the_x_span_about_the_pointer_and_a_notch_down_undoes_it(app):
    plot, x, _ = plain()
    changed = []
    plot.zoom_changed.connect(lambda: changed.append(1))
    full = plot.x_range()
    pos = at(plot, 0.7)
    under = x_at(plot, pos)
    ev = wheel(plot, pos)
    lo, hi = plot.x_range()
    assert ev.isAccepted() and plot.is_zoomed() and len(changed) == 1
    assert hi - lo == pytest.approx(0.5 * (full[1] - full[0]))
    assert x_at(plot, pos) == pytest.approx(under, abs=2 * (hi - lo) / plot.plot_rect().width())      # the point stayed put
    wheel(plot, pos)
    assert plot.x_range()[1] - plot.x_range()[0] == pytest.approx(0.25 * (full[1] - full[0]))
    wheel(plot, pos, -120)
    assert plot.x_range()[1] - plot.x_range()[0] == pytest.approx(0.5 * (full[1] - full[0]))
    wheel(plot, pos, -120)
    assert not plot.is_zoomed() and plot.x_range() == pytest.approx(full)                           # past whole: whole
    n = len(changed)
    wheel(plot, pos, -120)
    assert not plot.is_zoomed() and len(changed) == n                                              # nothing to zoom out of


def test_the_zoomed_range_stays_inside_the_data_and_goes_no_deeper_than_one_64th(app):
    plot, x, _ = plain()
    for frac in (0.0, 1.0):
        plot.fit()
        for _ in range(3):
            wheel(plot, at(plot, frac))
        lo, hi = plot.x_range()
        assert lo >= x[0] - 1e-9 and hi <= x[-1] + 1e-9 and hi - lo == pytest.approx(10.0 / 8)
        assert (lo == pytest.approx(0.0)) if frac == 0.0 else (hi == pytest.approx(10.0))
    for _ in range(12):
        wheel(plot, at(plot, 0.5))
    lo, hi = plot.x_range()
    assert hi - lo == pytest.approx(10.0 / pv.MAX_ZOOM)


def test_y_is_fitted_to_the_finite_data_in_the_visible_range_with_the_full_views_margin(app):
    plot, x, y = plain()
    full = plot._layout_now()
    assert (full.y0, full.y1) == pytest.approx((-5.0, 105.0))                  # 0..100 with 5 % each side
    for _ in range(3):
        wheel(plot, at(plot, 0.1))
    lo, hi = plot.x_range()
    seen = y[(x >= lo) & (x <= hi)]
    lay = plot._layout_now()
    margin = 0.05 * (seen.max() - seen.min())
    assert (lay.y0, lay.y1) == pytest.approx((seen.min() - margin, seen.max() + margin))
    assert lay.y1 < 20.0                                                       # the right end's large values are not in it
    plot.fit()
    for _ in range(3):
        wheel(plot, at(plot, 0.9))
    lay = plot._layout_now()
    assert lay.y0 > 40.0                                                       # and the left end's small ones are not


def test_a_gap_in_the_visible_data_is_skipped_not_taken_for_zero(app):
    plot, x, y = plain()
    y = y.copy()
    y[:40] = np.nan                                                            # nothing finite at the left end
    plot.set_series(x, [("y", y)], "x", "y", "t")
    for _ in range(3):
        wheel(plot, at(plot, 0.9))
    assert plot._layout_now().y0 > 40.0


# ── back to the whole range ──
def test_fit_and_new_data_go_back_to_the_whole_range_and_pins_stay_on_their_data(app):
    plot, x, y = plain()
    full = plot.x_range()
    click(plot, plot.point_position(80))
    wheel(plot, at(plot, 0.8))
    wheel(plot, at(plot, 0.8))
    assert plot.is_zoomed() and [p["index"] for p in plot.pinned_points()] == [80]
    plot.fit()
    assert not plot.is_zoomed() and plot.x_range() == pytest.approx(full)
    assert [p["index"] for p in plot.pinned_points()] == [80]                  # a pin is a data point, not a place
    wheel(plot, at(plot, 0.5))
    plot.set_series(x, [("y", y)], "x", "y", "t")                              # new data
    assert not plot.is_zoomed() and plot.x_range() == pytest.approx(full) and plot.pinned_points() == []
    wheel(plot, at(plot, 0.5))
    plot.clear("none")
    assert not plot.is_zoomed()


def test_a_pin_out_of_view_is_not_drawn_but_is_still_listed(app, monkeypatch):
    plot, x, _ = make_plot()
    for i in (20, 100, 180):
        click(plot, plot.point_position(i))
    for _ in range(3):
        wheel(plot, plot.point_position(100))
    lo, hi = plot.x_range()
    assert x[20] < lo and x[180] > hi and lo <= x[100] <= hi
    assert [p["index"] for p in plot.pinned_points()] == [20, 100, 180]
    drawn = []
    monkeypatch.setattr(plot, "_draw_reading", lambda p, lay, to_px, i, pinned: drawn.append(i))
    plot.grab()
    assert drawn == [100]
    listed = plot._list_lines([("1", 20), ("2", 100), ("3", 180)], pv.QFontMetrics(plot._mono_font()), plot.plot_rect(), 6)
    assert len(listed) == 3                                                    # the list is of every pin


def test_the_hover_stands_on_a_point_that_is_in_view(app):
    plot, x, _ = plain()
    for _ in range(3):
        wheel(plot, at(plot, 0.5))
    lo, hi = plot.x_range()
    for frac in (0.0, 0.5, 1.0):
        move(plot, at(plot, frac))
        assert plot._hover is not None and lo <= x[plot._hover] <= hi


# ── where the wheel does nothing ──
def test_the_wheel_is_taken_and_ignored_off_the_axes_with_ctrl_and_on_a_plot_that_does_not_zoom(app):
    plot, _, _ = plain()
    r = plot.plot_rect()
    for pos in (QPointF(r.left() - 5, r.center().y()), QPointF(r.center().x(), r.bottom() + 8), QPointF(2, 2)):
        ev = wheel(plot, pos)
        assert ev.isAccepted() and not plot.is_zoomed(), pos                    # off the axes: nothing, and no parent scrolls
    ev = wheel(plot, r.center(), mods=Qt.KeyboardModifier.ControlModifier)
    assert ev.isAccepted() and not plot.is_zoomed()
    ev = wheel(plot, r.center(), delta=0)
    assert ev.isAccepted() and not plot.is_zoomed()
    plot.set_wheel_zoom(False)                                                 # the small profile plot
    ev = wheel(plot, r.center())
    assert ev.isAccepted() and not plot.is_zoomed()
    plot.set_wheel_zoom(True)
    wheel(plot, r.center())
    assert plot.is_zoomed()


def test_an_empty_plot_takes_the_wheel_too(app):
    plot = PlotView()
    plot.resize(300, 200)
    plot.clear("nothing")
    assert wheel(plot, QPointF(100, 100)).isAccepted() and not plot.is_zoomed()


# ── the readout box ──
def test_the_box_goes_to_the_corner_with_the_fewest_curve_points_and_a_tie_to_the_one_farther_from_the_pointer(app):
    plot = PlotView()
    plot.resize(640, 340)
    x = np.linspace(0.0, 1.0, 101)
    plot.set_series(x, [("y", 1.0 - x)], "x", "y", "t")                         # from the top left to the bottom right
    lay = plot._layout_now()
    plot_r = lay.plot
    w, h = 150.0, 60.0
    left_mid = QPointF(plot_r.left() + 10, plot_r.center().y())
    right_mid = QPointF(plot_r.right() - 10, plot_r.center().y())
    top_right = (plot_r.right() - 6 - w, plot_r.top() + 6)
    bottom_left = (plot_r.left() + 6, plot_r.bottom() - 6 - h)
    for here, want in ((left_mid, top_right), (right_mid, bottom_left)):       # both are free of the curve: the farther one
        box = plot._pick_corner(lay, w, h, 6, here)
        assert (box.left(), box.top()) == pytest.approx(want)
    for corner in ((plot_r.left() + 6, plot_r.top() + 6), (plot_r.right() - 6 - w, plot_r.bottom() - 6 - h)):
        box = plot._pick_corner(lay, w, h, 6, left_mid)
        assert (box.left(), box.top()) != pytest.approx(corner)                 # the two the curve runs through


def dot_colour_at(plot: PlotView, i: int) -> tuple[int, int, int]:
    p = plot.point_position(i)
    return tuple(int(v) for v in rgb(plot)[int(round(p.y())), int(round(p.x()))])


def test_the_box_does_not_cover_the_point_being_read_when_both_top_corners_are_taken(app):
    plot = PlotView()
    font = QFont("Arial")
    font.setPixelSize(16)
    plot.setFont(font)
    plot.resize(640, 340)
    x = np.linspace(-1.0, 1.0, 201)
    y = 1.0 + np.exp(-((x + 0.9) / 0.1) ** 2) + np.exp(-((x - 0.9) / 0.1) ** 2) + 0.5 * np.exp(-((x + 0.2) / 0.08) ** 2)
    plot.set_series(x, [("dI/dV (nS)", y)], "Bias (V)", "dI/dV (nS)", "spectrum")
    for i in (10, 90, 190):                                                    # at each of the peaks, and between
        move(plot, plot.point_position(i))
        colour = dot_colour_at(plot, i)
        assert colour != tuple(pv.READOUT_BG.getRgb()[:3]), (i, colour)         # the dot, not the box, is under the pointer
        assert colour[2] > colour[0] + 40                                       # the curve's blue
    lay = plot._layout_now()
    for i in (10, 90, 190):                                                    # whichever corner, never over the point read
        here = plot.point_position(i)
        box = plot._pick_corner(lay, 200.0, 60.0, 6, here, here.x())
        assert not box.adjusted(-4, -4, 4, 4).contains(here), i


def test_a_plot_too_small_for_the_box_reads_distance_and_value_together_in_its_title_strip(app):
    plot = PlotView()
    font = QFont("Arial")
    font.setPixelSize(16)
    plot.setFont(font)
    plot.resize(190, 170)
    s = np.linspace(0.0, 2.5, 40)
    plot.set_series(s, [("profile (pm)", 20.0 * np.exp(-((s - 1.2) / 0.3) ** 2))], "distance (nm)", "pm", "A→B 2.50 nm")
    lay = plot._layout_now()
    assert lay.show_title                                                       # the strip is there
    move(plot, plot.point_position(20))
    i = plot._hover
    text = plot.strip_text(plot._fields(i), lay)
    assert text is not None
    for _col, _lab, number in plot._fields(i):
        assert number in text                                                   # the distance and the value, both
    img = rgb(plot)
    bg = tuple(pv.READOUT_BG.getRgb()[:3])
    assert tuple(int(v) for v in img[int(lay.vpad) + 2, int(lay.pad) + 3]) == bg   # the strip is the readout's now
    QApplication.sendEvent(plot, QEvent(QEvent.Type.Leave))
    assert tuple(int(v) for v in rgb(plot)[int(lay.vpad) + 2, int(lay.pad) + 3]) == (255, 255, 255)


def test_a_plot_with_room_keeps_the_box_and_leaves_the_title_alone(app):
    plot, x, ys = make_plot(size=(640, 340))
    lay = plot._layout_now()
    move(plot, plot.point_position(100))
    img = rgb(plot)
    assert tuple(int(v) for v in img[int(lay.vpad) + 2, int(lay.pad) + 3]) == (255, 255, 255)
    rows, _ = plot._readout_rows(plot._fields(100), pv.QFontMetrics(plot._mono_font()), plot.plot_rect(), 6)
    assert len(rows) == 2 and "Bias (V)" in rows[0]                             # the table: both fields, a row each


def test_a_title_strip_is_not_used_when_there_is_none_or_nothing_fits_it(app):
    plot = PlotView()
    plot.resize(150, 70)                                                        # no room for a title
    s = np.linspace(0.0, 1.0, 20)
    plot.set_series(s, [("v", s)], "distance (nm)", "pm", "title")
    lay = plot._layout_now()
    assert not lay.show_title and plot.strip_text(plot._fields(5), lay) is None
    move(plot, plot.point_position(5))
    plot.grab()                                                                 # paints without failing


def lone_dot_plot() -> PlotView:
    """Two curves: A is dense over the right 60 % of the axes (full height) and has a lone point at the top left; B is
    dense along the bottom. The top-left corner holds one curve point, the fewest, and it is the read dot."""
    plot = PlotView()
    font = QFont("Arial")
    font.setPixelSize(16)
    plot.setFont(font)
    plot.resize(640, 340)
    x = np.linspace(0.0, 1.0, 201)
    a = np.sin(2 * np.pi * 40 * x)
    a[:80] = np.nan
    a[5] = 1.4
    b = -1.3 + 0.05 * np.sin(2 * np.pi * 40 * x)
    plot.set_series(x, [("A", a), ("B", b)], "x", "y", "t")
    return plot


def test_the_box_never_covers_the_dots_being_read_even_when_that_corner_has_the_fewest_curve_points(app):
    plot = lone_dot_plot()
    lay = plot._layout_now()
    here = plot.point_position(5)
    w, h = 150.0, 60.0
    plain_box = plot._pick_corner(lay, w, h, 6, here, here.x())
    assert plain_box.adjusted(-4, -4, 4, 4).contains(here)                     # by curve points alone it is the top left
    dots = plot._dots_at(lay, [5])
    assert len(dots) == 2
    box = plot._pick_corner(lay, w, h, 6, here, here.x(), dots=dots)
    assert not any(box.adjusted(-4, -4, 4, 4).contains(d) for d in dots)
    move(plot, here)                                                           # and as painted
    assert plot._hover == 5
    colour = dot_colour_at(plot, 5)
    assert colour != tuple(pv.READOUT_BG.getRgb()[:3]) and colour[2] > colour[0] + 40


def test_the_pins_list_never_covers_a_pinned_dot(app):
    plot = lone_dot_plot()
    for i in (5, 150):
        click(plot, plot.point_position(i))
    lay = plot._layout_now()
    dots = plot._dots_at(lay, [5, 150])
    box = plot._pick_corner(lay, 300.0, 56.0, 6, None, dots=dots)
    assert not any(box.adjusted(-4, -4, 4, 4).contains(d) for d in dots)
    for i in (5, 150):
        colour = dot_colour_at(plot, i)                                        # series A's blue, not the list's background
        assert colour != tuple(pv.READOUT_BG.getRgb()[:3]) and colour[2] > colour[0] + 40, i
