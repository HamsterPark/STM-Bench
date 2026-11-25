"""The small profile plot, enlarged on a click, and the wheel on the viewer's plots, in the console window.

Written for the round after screen trial 23 (2026-11-25): the profile plot beside the marker table was about 65 px across
its axes, a click on it only pinned a point, the hover box fell back to the distance alone, and the agent spent eleven
steps reading a height; it was the fourth trial to meet this. The agent also tried the wheel on a spectrum.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint  # noqa: E402

from stmbench.gui import shots  # noqa: E402
from stmbench.gui.labels import one  # noqa: E402
from tests.gui_helpers import app, fonts, mark, open_window, run_button, scan  # noqa: E402, F401
from tests.test_gui_app import make_trace  # noqa: E402


def draw_profile(win) -> None:
    run_button(win, "view.tool.profile")
    fv = win.center.frame_view
    shots.drag(win, fv, fv.pixel_to_widget(5, 5).toPoint(), fv.pixel_to_widget(25, 20).toPoint())


def view_state(win) -> tuple:
    c = win.center
    fv = c.frame_view
    return (fv.view_rect(), fv.zoom(), [dict(m) for m in fv.markers()], dict(fv._profile), win.viewed, c.channel.value(),
            c.table.rowCount(), c.info.text())


def enlarge(win) -> None:
    shots.click(win, win.center.small_plot)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_a_click_on_the_small_plot_shows_it_large_and_the_way_back_leaves_the_frame_view_as_it_was(fonts, open_window, lang):
    win = open_window(lang)
    scan(win)
    c = win.center
    mark(win, 10, 12)
    draw_profile(win)
    run_button(win, "view.zoom.2")                                                  # a zoom, a marker, a profile line
    small = c.small_plot
    assert small.has_data() and c.small_hint.text() == one("plot_enlarge", lang)       # the small plot says it can be clicked
    assert c.stack.currentIndex() == 0 and c.toolbars.currentIndex() == 0
    before = view_state(win)
    enlarge(win)
    assert c.plot_enlarged() and c.stack.currentIndex() == 2 and c.toolbars.currentIndex() == 2
    big = c.profile_big
    assert big.isVisible() and not c.frame_view.isVisible() and big.width() > 4 * small.plot_rect().width()
    assert np.array_equal(big._x, small._x) and np.array_equal(big._values, small._values)    # the same curve
    assert small.pinned_points() == []                                               # the click pinned nothing
    back = shots.widget(win, "view.profile_back")
    assert back.isVisible() and back.text() == one("plot_back", lang)               # visible and labelled
    assert shots.widget(win, "view.profile_fit").text() == one("zoom_fit", lang)
    assert c.small_hint.text() == one("plot_return", lang)
    assert c.readout.full_text() == one("hint_plot", lang)
    shots.click(win, back)
    assert not c.plot_enlarged() and c.stack.currentIndex() == 0 and c.toolbars.currentIndex() == 0
    assert c.frame_view.isVisible() and view_state(win) == before                    # frame, zoom, markers, profile line
    assert c.small_hint.text() == one("plot_enlarge", lang)
    enlarge(win)                                                                     # a second click on the small plot returns
    assert c.plot_enlarged()
    enlarge(win)
    assert not c.plot_enlarged() and view_state(win) == before


def test_a_click_on_an_empty_small_plot_does_nothing_and_says_nothing(open_window):
    win = open_window("en")
    scan(win)
    c = win.center
    assert not c.small_plot.has_data() and c.small_hint.text() == ""
    enlarge(win)
    assert not c.plot_enlarged() and c.stack.currentIndex() == 0


def test_a_move_trace_in_the_small_plot_enlarges_too_and_returns_to_a_spectrum_that_was_on_show(open_window):
    win = open_window("en")
    c = win.center
    c.show_spectrum(make_trace("move_0001.dat", n=40))
    c.show_trace("move_tip", {"d": list(np.linspace(0, 1, 30)), "z": list(np.sin(np.linspace(0, 6, 30)))})
    assert c.small_plot.has_data() and c.small_hint.text() == one("plot_enlarge", "en")
    enlarge(win)
    assert c.plot_enlarged() and c.profile_big.has_data() and c.toolbars.currentIndex() == 2
    shots.click(win, shots.widget(win, "view.profile_back"))
    assert c.stack.currentIndex() == 1 and c.toolbars.currentIndex() == 1 and c.plot.has_data()      # the spectrum, as before


def test_a_new_frame_or_spectrum_leaves_the_big_view_as_it_does_any_other(open_window):
    win = open_window("en")
    scan(win)
    c = win.center
    draw_profile(win)
    enlarge(win)
    assert c.plot_enlarged()
    scan(win)                                                                         # a new frame
    assert not c.plot_enlarged() and c.stack.currentIndex() == 0 and c.toolbars.currentIndex() == 0
    assert c.small_hint.text() in ("", one("plot_enlarge", "en"))
    draw_profile(win)
    enlarge(win)
    c.show_spectrum(make_trace("move_0002.dat", n=40, index=2))                       # a new spectrum
    assert not c.plot_enlarged() and c.stack.currentIndex() == 1 and c.toolbars.currentIndex() == 1
    assert c.readout.full_text() == one("hint_plot", "en")


def test_new_profile_data_replaces_the_big_view_in_place_and_a_cleared_profile_returns(open_window):
    win = open_window("en")
    scan(win)
    c = win.center
    draw_profile(win)
    enlarge(win)
    s = list(np.linspace(0.0, 2.0, 25))
    c._profile({"frame": "f.sxm", "s_nm": s, "values": list(np.cos(s)), "unit": "pm"})
    assert c.plot_enlarged() and np.array_equal(c.profile_big._x, np.asarray(s)) and "A→B 2.00 nm" in c.profile_big._title
    c._profile({})
    assert not c.plot_enlarged() and c.stack.currentIndex() == 0 and not c.profile_big.has_data()


# ── the readout, in both sizes ──
def test_the_hover_reads_distance_and_value_together_in_the_big_plot_and_the_small_one(open_window):
    win = open_window("en")
    scan(win)
    c = win.center
    draw_profile(win)
    small, big = c.small_plot, c.profile_big
    i = len(small._x) // 2
    lay = small._layout_now()
    text = small.strip_text(small._fields(i), lay)                                   # small: one line over the title
    assert text is not None and all(n in text for _c, _l, n in small._fields(i))
    enlarge(win)
    shots.settle(win)
    rows, _ = big._readout_rows(big._fields(i), pv_metrics(big), big.plot_rect(), 6)
    assert len(rows) == 2 and "distance (nm)" in rows[0] and "profile (pm)" in rows[1]    # a row each, the value named with its unit


def pv_metrics(plot):
    from PySide6.QtGui import QFontMetrics
    return QFontMetrics(plot._mono_font())


# ── the wheel on the big profile, and the rest ──
def test_the_wheel_over_the_axes_of_the_big_profile_zooms_and_over_the_small_one_does_not(open_window):
    win = open_window("en")
    scan(win)
    c = win.center
    draw_profile(win)
    small = c.small_plot
    shots.wheel(win, small, small.plot_rect().center().toPoint())
    assert not small.is_zoomed()
    enlarge(win)
    shots.settle(win)
    big = c.profile_big
    where = big.plot_rect().center().toPoint()
    shots.wheel(win, big, where)
    assert big.is_zoomed() and not small.is_zoomed()
    shots.click(win, shots.widget(win, "view.profile_fit"))
    assert not big.is_zoomed()
    shots.wheel(win, big, QPoint(2, 2))
    assert not big.is_zoomed()                                                         # off the axes: nothing


def test_clear_plot_readings_clears_the_big_plots_pins_too(open_window):
    win = open_window("en")
    scan(win)
    c = win.center
    draw_profile(win)
    enlarge(win)
    shots.settle(win)
    big = c.profile_big
    shots.click(win, big, big.point_position(len(big._x) // 2).toPoint())
    assert len(big.pinned_points()) == 1
    run_button(win, "view.clear_readings")
    assert big.pinned_points() == []


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_new_controls_and_the_line_under_the_small_plot_fit_the_window(fonts, open_window, lang):
    win = open_window(lang)
    scan(win)
    c = win.center
    draw_profile(win)
    hint = c.small_hint
    for key in ("plot_enlarge", "plot_return"):
        assert hint.fontMetrics().horizontalAdvance(one(key, lang)) <= hint.width() - 4, key
    enlarge(win)
    shots.settle(win)
    tb = c.toolbars
    for name in ("view.profile_back", "view.profile_fit"):
        b = shots.widget(win, name)
        assert b.isVisible() and tb.rect().contains(b.mapTo(tb, b.rect().bottomRight()))
        assert b.fontMetrics().horizontalAdvance(b.text()) <= b.width()
