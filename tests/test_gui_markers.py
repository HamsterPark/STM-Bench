"""The viewer's markers as a screen agent meets them in the console window: what a marker is (the pixel it was placed
in, at that pixel's centre, with that pixel's value), the table that lists them, and the controls around them.

Written for the round after screen trials 3-5 (P4): the agents fitted Gaussians to marker values by hand and lost
steps to coordinates that were not the pixel's, a table that showed six rows, and a Clear markers button that was
not there when a trace was on show.
"""
from __future__ import annotations

import math

import pytest

from stmbench.gui import fmt, shots
from stmbench.gui.widgets import frame_view as fvm
from stmbench.render import pixel_to_scan_nm
from tests.gui_helpers import app, mark, open_window, pixel_point, run_button, scan  # noqa: F401

from PySide6.QtTest import QTest  # noqa: E402

GEOM_8NM_32PX = {"cx_nm": 0.0, "cy_nm": 0.0, "w_nm": 8.0, "h_nm": 8.0, "angle_deg": 0.0, "nx": 32, "ny": 32}


def table_row(win, row: int) -> list[str]:
    t = win.center.table
    return [t.item(row, c).text() for c in range(t.columnCount())]


# ── what a marker is ──
@pytest.mark.parametrize("lang", ["en", "zh"])
def test_the_readout_the_table_and_the_buttons_give_the_same_point_for_a_click_anywhere_in_a_pixel(open_window, lang):
    """Trial 5 steps 40-45: the table listed the clicked coordinates but the value of the pixel they fell in, so a
    Gaussian fitted to the table was 0.06 nm off until the agent worked out the pixel centres by hand. Now what the
    cursor reads, what the click marks, what the table lists and what the buttons copy are one point: the centre of
    the pixel, with its value."""
    win = open_window(lang)
    scan(win)
    fv = win.center.frame_view
    x, y = pixel_to_scan_nm(GEOM_8NM_32PX, 10, 12)
    value = fv._shown.values[12, 10]
    for dx, dy in ((0.3, -0.35), (-0.45, 0.45), (0.0, 0.0)):                 # three places inside pixel (10, 12)
        QTest.mouseMove(fv, pixel_point(win, 10, 12, dx, dy))
        want = f"pixel col 10, row 12   centre x {x:.3f} nm   y {y:.3f} nm   {value:.2f} pm"
        assert win.center.readout.full_text() == want
        mark(win, 10, 12, dx, dy)
        row = len(fv.markers()) - 1
        assert table_row(win, row)[1:4] == [f"{x:.3f}", f"{y:.3f}", f"{value:.2f} pm"]
    first, last = fv.markers()[0], fv.markers()[-1]
    assert first == last and (first["x_nm"], first["y_nm"]) == pytest.approx((x, y), abs=1e-9)
    assert (first["col"], first["row"], first["value"]) == (10, 12, value)
    # the buttons copy the coordinates as the table shows them
    run_button(win, "markers.tip")
    assert (shots.widget(win, "act.move_tip.x_nm").text(), shots.widget(win, "act.move_tip.y_nm").text()) == (
        f"{x:.3f}", f"{y:.3f}")
    run_button(win, "markers.scan")
    assert shots.widget(win, "act.scan.center_x_nm").text() == f"{x:.3f}"
    run_button(win, "markers.report")
    f = win.sidebar.report.fields
    assert (f["x_nm"].text(), f["y_nm"].text(), f["frame"].text()) == (f"{x:.3f}", f"{y:.3f}", win.viewed[1])


def test_a_marker_shown_on_another_frame_is_drawn_where_its_scan_position_lies_and_keeps_its_own_record(open_window):
    """The marker is a record of one pixel of one frame (its value, col, row, frame and start stay those); on another
    frame it is drawn at the same scan position, which there is generally not at a pixel's centre."""
    win = open_window("en")
    a = scan(win, 0.0, 0.0, 8.0, 32)
    fv = win.center.frame_view
    mark(win, 20, 9, 0.2, 0.2)
    (m,) = fv.markers()
    assert (m["col"], m["row"], m["frame"]) == (20, 9, a)
    before = table_row(win, 0)
    assert before[4] == a
    b = scan(win, 1.07, -0.93, 6.5, 26)
    assert b != a and win.viewed[1] == b
    assert fv.markers() == [m] and table_row(win, 0) == before          # the record is the frame it was placed on's
    geom_b = {"cx_nm": 1.07, "cy_nm": -0.93, "w_nm": 6.5, "h_nm": 6.5, "angle_deg": 0.0, "nx": 26, "ny": 26}
    col, row = fvm.scan_to_pixel(geom_b, m["x_nm"], m["y_nm"])
    assert not (col == int(col) and row == int(row))                    # not a pixel centre of the frame it is shown on
    pt = fv.scan_to_widget(m["x_nm"], m["y_nm"])
    want = fv.pixel_to_widget(col, row)
    assert (pt.x(), pt.y()) == pytest.approx((want.x(), want.y()), abs=1e-6)
    # the reader can still read frame b's own pixel there: the cursor says what b holds, not what a held
    QTest.mouseMove(fv, pt.toPoint())
    text = win.center.readout.full_text()
    assert text.startswith(f"pixel col {int(math.floor(col + 0.5))}, row {int(math.floor(row + 0.5))}   ")
    win.request_frame(a)
    shots.settle(win)
    back = fv.scan_to_widget(m["x_nm"], m["y_nm"])
    want = fv.pixel_to_widget(20, 9)
    assert (back.x(), back.y()) == pytest.approx((want.x(), want.y()), abs=1e-6)       # on its own frame: the pixel's centre
    # a second marker, placed on the second frame: each keeps its frame and its start time
    win.request_frame(b)
    shots.settle(win)
    mark(win, 5, 5)
    starts = {f["frame"]: f["t_start_s"] for f in win._data["frames"]}
    rows = [table_row(win, r) for r in range(2)]
    assert [(r[4], r[5]) for r in rows] == [(a, fmt.hms(starts[a])), (b, fmt.hms(starts[b]))]
