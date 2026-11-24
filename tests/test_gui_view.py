"""The viewer's zoom as a screen agent meets it in the console window: the zoom buttons zoom about the selected
marker, a repeat scan of the same window keeps the zoom, and the buttons always show the zoom in force.

Written for the round after screen trials 3-5 (trial 4 step 84: "Zoom x2/x4 always centres on the frame centre, not
on the marker"; trial 3 step 29: "selecting another frame resets Zoom to Fit").
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint  # noqa: E402

from stmbench.gui import shots  # noqa: E402
from stmbench.render import pixel_to_scan_nm  # noqa: E402
from tests.gui_helpers import app, fonts, mark, open_window, run_button, scan  # noqa: E402, F401

GEOM = {"cx_nm": 0.0, "cy_nm": 0.0, "w_nm": 8.0, "h_nm": 8.0, "angle_deg": 0.0, "nx": 32, "ny": 32}


def zoom_in_force(win) -> int:
    """The level the zoom buttons show, which must be the one the view has."""
    shown = [lvl for lvl in (1, 2, 4, 8) if shots.widget(win, f"view.zoom.{lvl}").isChecked()]
    assert len(shown) == 1, shown
    assert shown[0] == win.center.frame_view.zoom()
    return shown[0]


def view_centre(win) -> tuple[float, float]:
    c0, r0, wv, hv = win.center.frame_view.view_rect()
    return c0 + wv / 2, r0 + hv / 2


def select_row(win, row: int) -> None:
    t = win.center.table
    shots.click(win, t.viewport(), t.visualItemRect(t.item(row, 1)).center())


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_zoom_buttons_centre_the_view_on_the_selected_marker_else_keep_the_views_centre(fonts, open_window, lang):
    win = open_window(lang)
    scan(win)
    fv = win.center.frame_view
    mark(win, 12, 20)                                                             # marker 1: well inside the frame
    mark(win, 3, 4)                                                               # marker 2, the selected one: near a corner
    assert win.center.selected_row() == 1
    run_button(win, "view.zoom.4")
    assert zoom_in_force(win) == 4 and fv.view_rect()[:2] == (0.0, 0.5)           # held at the left edge, the marker in view
    assert fv.scan_to_widget(*pixel_to_scan_nm(GEOM, 3, 4)) is not None
    select_row(win, 0)                                                            # the table's selection decides
    run_button(win, "view.zoom.2")
    assert zoom_in_force(win) == 2
    c0, r0, wv, hv = fv.view_rect()
    assert (c0 + wv / 2, r0 + hv / 2) == pytest.approx((12.5, 20.5))             # centred on marker 1 (pixel 12, 20)
    run_button(win, "view.zoom.8")
    assert zoom_in_force(win) == 8 and view_centre(win) == pytest.approx((12.5, 20.5))
    pt = fv.scan_to_widget(*pixel_to_scan_nm(GEOM, 12, 20))
    r = fv.image_rect()
    assert (pt.x(), pt.y()) == pytest.approx((r.x() + r.width() / 2, r.y() + r.height() / 2), abs=1.0)
    run_button(win, "view.zoom.1")
    assert zoom_in_force(win) == 1 and fv.view_rect() == (0.0, 0.0, 32.0, 32.0)
    # no marker selected: about the centre of the view as it is, which a pan has moved
    run_button(win, "view.clear_markers")
    run_button(win, "view.zoom.4")
    run_button(win, "view.tool.pan")
    shots.drag(win, fv, fv.rect().center(), fv.rect().center() - QPoint(120, 70))
    centre = view_centre(win)
    assert centre != (16.0, 16.0)                                                 # the pan moved it
    run_button(win, "view.zoom.2")
    assert zoom_in_force(win) == 2 and view_centre(win) == pytest.approx(centre)  # the view's centre, not the frame's
    run_button(win, "view.zoom.4")
    assert zoom_in_force(win) == 4 and view_centre(win) == pytest.approx(centre)


def test_a_repeat_scan_of_the_window_keeps_the_zoom_and_the_place_and_another_window_is_shown_whole(open_window):
    win = open_window("en")
    first = scan(win)
    fv = win.center.frame_view
    mark(win, 20, 12)
    run_button(win, "view.zoom.4")
    assert zoom_in_force(win) == 4
    place = view_centre(win)
    assert place == pytest.approx((20.5, 12.5))
    second = scan(win)                                                            # the same window again, as after every drag
    assert second != first and win.viewed == ("frame", second)
    assert zoom_in_force(win) == 4 and view_centre(win) == pytest.approx(place)
    mark(win, 20, 12)                                                             # the next marker is at the scale the reader sees
    m1, m2 = fv.markers()
    assert (m1["x_nm"], m1["y_nm"]) == pytest.approx((m2["x_nm"], m2["y_nm"]))
    # an older frame of that window, by its thumbnail: the same
    shots.click(win, win.center.thumbs[1])
    shots.settle(win)
    assert win.viewed == ("frame", first) and zoom_in_force(win) == 4 and view_centre(win) == pytest.approx(place)
    # another channel of it keeps the view (as it did); another window, by a scan, is shown whole and says so
    run_button(win, "view.channel.current")
    shots.settle(win)
    assert zoom_in_force(win) == 4 and view_centre(win) == pytest.approx(place)
    third = scan(win, 0.0, 0.0, 6.0, 24)
    assert win.viewed == ("frame", third) and zoom_in_force(win) == 1 and fv.view_rect() == (0.0, 0.0, 24.0, 24.0)
    # and back, by the thumbnail of a frame of the first window: whole again (the view was the whole third frame)
    shots.click(win, win.center.thumbs[1])
    shots.settle(win)
    assert win.viewed == ("frame", second) and zoom_in_force(win) == 1


def test_a_scan_centred_elsewhere_keeps_where_the_reader_was_looking_when_it_reaches_there(open_window):
    win = open_window("en")
    scan(win, 0.0, 0.0, 8.0, 32)
    fv = win.center.frame_view
    run_button(win, "view.zoom.2")
    here = fv._view_centre_scan(fv._shown)
    assert here == pytest.approx((0.0, 0.0))
    scan(win, 1.0, 0.5, 8.0, 32)                                                  # the same window, a little aside
    assert zoom_in_force(win) == 2 and fv._view_centre_scan(fv._shown) == pytest.approx(here)
    scan(win, 30.0, 30.0, 8.0, 32)                                                # one that does not hold that place
    assert zoom_in_force(win) == 1 and fv.view_rect() == (0.0, 0.0, 32.0, 32.0)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_readings_pinned_on_a_spectrum_stay_in_the_picture_and_clear_readings_and_new_data_remove_them(fonts, open_window, lang):
    """Screen trial 3 step 44: the hover readout gives exact values, but only one per step. A click pins the point;
    the pins stay in the screenshot taken after the step's last event."""
    win = open_window(lang)
    scan(win)
    win.right.setCurrentWidget(win.spectroscopy)
    shots.fill(win, {"act.sts.v_start": "-0.6", "act.sts.v_end": "0.6", "act.sts.points": "64"})
    run_button(win, "run.sts")
    shots.settle(win)
    plot = win.center.plot
    assert win.viewed[0] == "spectrum" and plot.pinned_points() == []
    for i in (10, 30, 50):
        shots.click(win, plot, plot.point_position(i).toPoint())
    assert [p["index"] for p in plot.pinned_points()] == [10, 30, 50]
    clear = shots.widget(win, "view.clear_readings")
    assert clear.isVisible() and clear.width() >= 56
    shots.hit(win, clear)
    run_button(win, "view.clear_readings")
    assert plot.pinned_points() == []
    for i in (10, 30):
        shots.click(win, plot, plot.point_position(i).toPoint())
    run_button(win, "run.sts")                                                   # the next spectrum: new data
    shots.settle(win)
    assert win.center.plot.pinned_points() == []
    # the button is in the same place with a frame on show, and clears nothing there
    here = shots.widget(win, "view.clear_readings").mapTo(win, shots.widget(win, "view.clear_readings").rect().center())
    win.request_frame(win._data["frames"][-1]["frame"])
    shots.settle(win)
    assert shots.widget(win, "view.clear_readings").mapTo(win, shots.widget(win, "view.clear_readings").rect().center()) == here


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_two_clear_buttons_name_the_view_each_acts_on(fonts, open_window, lang):
    """Four blind screen trials clicked "Clear readings" to clear the image's markers (their values are readings too).
    Each label now names its view, in the screen's own words, and neither is a prefix or a word-for-word part of the
    other, so a reader cannot take one for the other."""
    win = open_window(lang)
    image = shots.widget(win, "view.clear_markers").text()
    plot = shots.widget(win, "view.clear_readings").text()
    words = {"en": (("image", "marker"), ("plot", "reading")), "zh": (("图像", "标记"), ("曲线", "读数"))}[lang]
    assert all(w in image for w in words[0]) and not any(w in image for w in words[1])
    assert all(w in plot for w in words[1]) and not any(w in plot for w in words[0])
    assert image != plot
