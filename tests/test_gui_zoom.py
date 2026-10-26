"""The viewer's zoom and pan (``FrameView.set_zoom``, the wheel, the ``"pan"`` tool): the same pixel / scan
mapping at every zoom, so that what is read off a zoomed image (cursor, markers, profile, axes) is what the
judge's mapping gives for the pixel under it, and nothing is found, fitted or snapped for the reader.

(The rest of the viewer's tests are in ``test_gui_widgets.py``; this file is the console's own, written when the
zoom was added.)"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from stmbench.gui.widgets import FrameView  # noqa: E402
from stmbench.gui.widgets import frame_view as fv  # noqa: E402
from stmbench.render import pixel_to_scan_nm  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def bump_frame(name="sim001.sxm", angle=0.0, center=(2.0, -1.0), size=10.0, n=64, channel="z", seed=3) -> dict:
    """A frame with a few Gaussian bumps at known pixels and a little noise (metres)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    z = rng.normal(0.0, 1.0, (n, n)) * 1e-12
    for cx, cy, amp in ((n * 0.25, n * 0.25, 90.0), (n * 0.75, n * 0.3, 60.0), (n * 0.4, n * 0.8, 120.0)):
        z += amp * 1e-12 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.0 ** 2))
    return {"frame": name, "array": z, "center_nm": center, "size_nm": size, "angle_deg": angle, "channel": channel,
            "direction": "forward", "unit": "m", "pixels": n}


def make_view(frame=None) -> FrameView:
    view = FrameView()
    view.resize(640, 560)
    if frame is not None:
        view.set_frame(frame)
    return view


def geom_of(frame: dict) -> dict:
    return fv.frame_geometry(frame)


def wheel(view: FrameView, pos: QPointF, delta: int = 120, mods=Qt.KeyboardModifier.NoModifier) -> bool:
    ev = QWheelEvent(QPointF(pos), QPointF(view.mapToGlobal(pos.toPoint())), QPoint(0, 0), QPoint(0, delta),
                     Qt.MouseButton.NoButton, mods, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(view, ev)
    return ev.isAccepted()


def inside(view: FrameView, fx: float, fy: float) -> QPointF:
    r = view.image_rect()
    return QPointF(r.x() + fx * r.width(), r.y() + fy * r.height())


def test_zoom_one_is_the_whole_frame_and_the_mapping_is_what_it_was(app):
    frame = bump_frame()
    view = make_view(frame)
    assert view.zoom() == 1 and view.view_rect() == (0.0, 0.0, 64.0, 64.0)
    r = view.image_rect()
    p = view._point_at(QPointF(r.x() + 0.25 * r.width(), r.y() + 0.75 * r.height()))
    assert p["col_f"] == pytest.approx(0.25 * 64 - 0.5) and p["row_f"] == pytest.approx(0.75 * 64 - 0.5)
    assert view._axis_spec()[0] == (2.0 - 5.0, 2.0 + 5.0) and view._axis_spec()[1] == (-1.0 - 5.0, -1.0 + 5.0)
    assert view.pixel_to_widget(10, 20) == QPointF(r.x() + 10.5 * r.width() / 64, r.y() + 20.5 * r.height() / 64)


def test_the_wheel_steps_the_zoom_about_the_pointer_and_what_is_under_it_stays_there(app):
    view = make_view(bump_frame())
    got: list[int] = []
    view.zoom_changed.connect(got.append)
    pos = inside(view, 0.3, 0.7)
    under = view._point_at(pos)
    for want in (2, 4, 8, 8):                                           # up notches: in, in, in, and the end of the range
        assert wheel(view, pos, +120)
        assert view.zoom() == want
        now = view._point_at(pos)
        assert now["x_nm"] == pytest.approx(under["x_nm"], abs=1e-6) and now["y_nm"] == pytest.approx(under["y_nm"], abs=1e-6)
    c0, r0, wv, hv = view.view_rect()
    assert (wv, hv) == (8.0, 8.0) and 0 <= c0 <= 56 and 0 <= r0 <= 56   # an eighth of the frame, inside it
    for want in (4, 2, 1, 1):                                           # down notches: back out
        assert wheel(view, pos, -120)
        assert view.zoom() == want
    assert view.view_rect() == (0.0, 0.0, 64.0, 64.0) and got[:3] == [2, 4, 8]
    # with Ctrl nothing happens, and the event is still taken
    assert wheel(view, pos, +120, Qt.KeyboardModifier.ControlModifier) and view.zoom() == 1
    # beside the image the view zooms about its centre
    wheel(view, QPointF(2, 2), +120)
    assert view.zoom() == 2 and view.view_rect() == (16.0, 16.0, 32.0, 32.0)
    view.fit()
    assert view.zoom() == 1 and view.view_rect() == (0.0, 0.0, 64.0, 64.0)


@pytest.mark.parametrize("angle", [0.0, 30.0])
def test_the_pixel_widget_and_scan_mappings_agree_at_every_zoom_and_pan(app, angle):
    frame = bump_frame(angle=angle)
    geom = geom_of(frame)
    view = make_view(frame)
    for level in (1, 2, 4, 8):
        view.set_zoom(level)
        for dx, dy in ((0, 0), (-90, 40), (500, -300)):               # the corners of the pan range, and between
            view.set_tool("pan")
            view.pan_by(dx, dy)
            for fx, fy in ((0.02, 0.03), (0.5, 0.5), (0.97, 0.9), (0.31, 0.68)):
                pos = inside(view, fx, fy)
                p = view._point_at(pos)
                x, y = pixel_to_scan_nm(geom, p["col_f"], p["row_f"])      # the judge's mapping, for the pixel under it
                assert (p["x_nm"], p["y_nm"]) == (pytest.approx(x), pytest.approx(y))
                back = view.pixel_to_widget(p["col_f"], p["row_f"])
                assert back.x() == pytest.approx(pos.x(), abs=1e-6) and back.y() == pytest.approx(pos.y(), abs=1e-6)
                again = view.scan_to_widget(p["x_nm"], p["y_nm"])
                assert again is not None and again.x() == pytest.approx(pos.x(), abs=1e-6)
                assert p["value"] == view._shown.values[p["row"], p["col"]]       # and its reading, untouched by the zoom
    # the whole frame is in view at zoom 1: every pixel of it has a position; zoomed, the ones out of view have none
    view.fit()
    corner = pixel_to_scan_nm(geom, 2, 2)
    assert view.scan_to_widget(*corner) is not None
    view.set_zoom(8)
    view.pan_by(1e4, 1e4)                                               # the view at the frame's top left
    assert view.view_rect()[:2] == (0.0, 0.0)
    assert view.scan_to_widget(*corner) is not None
    far = pixel_to_scan_nm(geom, 60, 60)
    assert view.scan_to_widget(*far) is None


def test_the_axes_show_the_part_of_the_frame_in_view(app):
    frame = bump_frame(center=(2.0, -1.0), size=10.0)
    view = make_view(frame)
    pos = inside(view, 0.75, 0.25)
    wheel(view, pos, +120)
    wheel(view, pos, +120)                                                # x4
    (x0, x1), (y0, y1), xt, yt = view._axis_spec()
    assert (xt, yt) == ("x (nm)", "y (nm)")
    assert x1 - x0 == pytest.approx(10.0 / 4) and y1 - y0 == pytest.approx(10.0 / 4)
    r = view.image_rect()
    left = view._point_at(QPointF(r.x() + 1e-6, r.center().y()))          # the scan x at the image's left and right edges
    right = view._point_at(QPointF(r.right() + 1 - 1e-6, r.center().y()))
    assert left["x_nm"] == pytest.approx(x0, abs=0.01) and right["x_nm"] == pytest.approx(x1, abs=0.01)
    top = view._point_at(QPointF(r.center().x(), r.y() + 1e-6))
    bottom = view._point_at(QPointF(r.center().x(), r.bottom() + 1 - 1e-6))
    assert top["y_nm"] == pytest.approx(y1, abs=0.01) and bottom["y_nm"] == pytest.approx(y0, abs=0.01)
    rot = make_view(bump_frame(angle=30.0))
    rot.set_zoom(2)
    (u0, u1), (v0, v1), ut, vt = rot._axis_spec()
    assert ut.startswith("u (nm), frame rotated 30") and u1 - u0 == pytest.approx(5.0) and v1 - v0 == pytest.approx(5.0)


def test_the_painted_image_is_the_nearest_pixels_of_the_part_in_view(app):
    view = make_view(bump_frame())
    view.set_zoom(4)
    view.set_tool("pan")
    view.pan_by(-60, -25)
    img = view.grab().toImage()
    sh = view._shown
    for fx, fy in ((0.5, 0.5), (0.1, 0.9), (0.93, 0.07), (0.62, 0.33)):
        pos = inside(view, fx, fy)
        p = view._point_at(pos)
        assert img.pixelColor(pos.toPoint()).red() == int(sh.grey[p["row"], p["col"]])
    # and at zoom 1 it is what it always was: pixel i shows source pixel floor((i + 0.5) * n / size)
    view.fit()
    r = view.image_rect()
    img = view.grab().toImage()
    for i, j in ((0, 0), (r.width() // 3, r.height() // 2), (r.width() - 1, r.height() - 1)):
        col, row = int((i + 0.5) * sh.nx / r.width()), int((j + 0.5) * sh.ny / r.height())
        assert img.pixelColor(r.x() + i, r.y() + j).red() == int(sh.grey[row, col])


def drag(view: FrameView, a: QPointF, b: QPointF, steps: int = 6) -> None:
    left, none = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    QTest.mousePress(view, left, none, a.toPoint())
    for i in range(1, steps + 1):
        QTest.mouseMove(view, (a + (b - a) * (i / steps)).toPoint())
    QTest.mouseRelease(view, left, none, b.toPoint())


def test_the_pan_tool_moves_the_zoomed_image_with_the_pointer_and_stops_at_the_frame(app):
    view = make_view(bump_frame())
    got = {"markers": [], "profiles": []}
    view.markers_changed.connect(got["markers"].append)
    view.profile_changed.connect(got["profiles"].append)
    view.set_tool("pan")
    drag(view, inside(view, 0.4, 0.4), inside(view, 0.7, 0.2))            # at zoom 1 there is nothing to move
    assert view.view_rect() == (0.0, 0.0, 64.0, 64.0)
    view.set_zoom(4)
    c0, r0, wv, hv = view.view_rect()
    a = inside(view, 0.5, 0.5)
    dx, dy = 20.0, -12.0
    drag(view, a, QPointF(a.x() + dx, a.y() + dy))
    r = view.image_rect()
    c1, r1, _, _ = view.view_rect()
    assert c1 == pytest.approx(c0 - dx * wv / r.width(), abs=0.05) and r1 == pytest.approx(r0 - dy * hv / r.height(), abs=0.05)
    for _ in range(3):
        drag(view, inside(view, 0.9, 0.5), inside(view, 0.0, 0.5), steps=10)  # far to the left: the right edge of the frame
        drag(view, inside(view, 0.5, 0.1), inside(view, 0.5, 0.95), steps=10) # and down: the top of the frame
    c, rr, wv, hv = view.view_rect()
    assert c == pytest.approx(64 - 16) and rr == pytest.approx(0.0, abs=1e-9)
    assert got["markers"] == [] and got["profiles"] == []                  # the pan tool placed no marker, drew no line
    # a marker placed afterwards is where the judge's mapping puts the pixel under the click
    frame = bump_frame()
    view.set_tool("marker")
    click = inside(view, 0.37, 0.62).toPoint()                              # a whole pixel, as a click is
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, click)
    (m,) = view.markers()
    p = view._point_at(QPointF(click))
    x, y = pixel_to_scan_nm(geom_of(frame), p["col_f"], p["row_f"])
    assert (m["x_nm"], m["y_nm"]) == (pytest.approx(x), pytest.approx(y))
    assert view.scan_to_widget(m["x_nm"], m["y_nm"]).x() == pytest.approx(click.x(), abs=1e-6)


def test_a_marker_out_of_view_is_kept_and_a_profile_reads_the_same_data_zoomed(app):
    frame = bump_frame()
    view = make_view(frame)
    far, near = inside(view, 0.1, 0.1), inside(view, 0.8, 0.8)
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, far.toPoint())
    QTest.mouseClick(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, near.toPoint())
    m_far, m_near = view.markers()
    wheel(view, near, +120)
    wheel(view, near, +120)                                               # x4 about the second marker's place
    assert view.scan_to_widget(m_far["x_nm"], m_far["y_nm"]) is None      # not in view: not drawn
    assert view.scan_to_widget(m_near["x_nm"], m_near["y_nm"]) is not None
    assert len(view.markers()) == 2                                       # but kept
    payloads: list[dict] = []
    view.profile_changed.connect(payloads.append)
    view.set_tool("profile")
    drag(view, inside(view, 0.2, 0.3), inside(view, 0.8, 0.7))
    (pl,) = payloads
    geom = geom_of(frame)
    c0, r0 = fv.scan_to_pixel(geom, pl["x0_nm"], pl["y0_nm"])
    c1, r1 = fv.scan_to_pixel(geom, pl["x1_nm"], pl["y1_nm"])
    t = np.linspace(0.0, 1.0, len(pl["s_nm"]))
    want = fv._bilinear(view._shown.values, c0 + t * (c1 - c0), r0 + t * (r1 - r0))
    assert np.allclose(pl["values"], want) and pl["unit"] == "pm"
    # a profile that ends beyond the view stops at the edge of the part in view, not of the frame
    drag(view, inside(view, 0.5, 0.5), QPointF(2, 2))
    ends = payloads[-1]
    ex, ey = view._point_at(inside(view, 0.0, 0.0), clamp=True)["x_nm"], view._point_at(inside(view, 0.0, 0.0), clamp=True)["y_nm"]
    assert (ends["x1_nm"], ends["y1_nm"]) == (pytest.approx(ex), pytest.approx(ey))
    view.fit()
    assert view.scan_to_widget(m_far["x_nm"], m_far["y_nm"]) is not None   # and out again, both are drawn


def test_a_new_frame_is_shown_whole_and_another_channel_of_the_same_frame_keeps_the_view(app):
    view = make_view(bump_frame("sim001.sxm"))
    got: list[int] = []
    view.zoom_changed.connect(got.append)
    view.set_zoom(4)
    view.set_tool("pan")
    view.pan_by(-30, -30)
    corner = view.view_rect()
    other = bump_frame("sim001.sxm", channel="current", seed=9)           # the same frame, its current channel
    view.set_frame(other)
    assert view.zoom() == 4 and view.view_rect() == corner
    view.set_frame(bump_frame("sim002.sxm", seed=5))                      # the next frame
    assert view.zoom() == 1 and view.view_rect() == (0.0, 0.0, 64.0, 64.0)
    assert got == [4, 1]
    view.set_frame(None)
    assert view.zoom() == 1


def test_a_level_that_is_not_one_is_refused_and_with_no_frame_the_control_is_answered_with_the_level_it_has(app):
    view = make_view()
    got: list[int] = []
    view.zoom_changed.connect(got.append)
    view.set_zoom(4)                                                       # nothing is shown: nothing to zoom
    assert view.zoom() == 1 and got == [1]
    view.set_frame(bump_frame())
    for bad in (0, 3, 16, -2):
        with pytest.raises(ValueError, match="zoom"):
            view.set_zoom(bad)
    view.set_tool("pan")
    assert view.cursor().shape() == Qt.CursorShape.OpenHandCursor
    with pytest.raises(ValueError):
        view.set_tool("lasso")                                             # not a tool
    assert math.isfinite(view.view_rect()[2])
