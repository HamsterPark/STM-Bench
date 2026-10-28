"""How the viewer draws a marker: an open ring, so that the pixel it marks (and its neighbours) stay visible at every
zoom and on every grey level, with its number beside it where no other number or ring is covered.

Written for the round after screen trial 4 (step 85: "three overlapping marker rings hid the atom in the Fit view").
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QFont, QImage, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from stmbench.gui.widgets import FrameView  # noqa: E402
from stmbench.gui.widgets import frame_view as fv  # noqa: E402
from stmbench.render import pixel_to_scan_nm  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def atom_frame(n: int = 96, name: str = "sim001.sxm", peak_pm: float = 70.0, center=(0.0, 0.0), size: float = 12.0) -> dict:
    """A terrace with one atom (sigma 4 px) in the middle and a little noise (metres)."""
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:n, 0:n]
    z = rng.normal(0.0, 0.5, (n, n)) * 1e-12 + peak_pm * 1e-12 * np.exp(-((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / (2 * 4.0 ** 2))
    return {"frame": name, "array": z, "center_nm": center, "size_nm": size, "angle_deg": 0.0, "channel": "z",
            "direction": "forward", "unit": "m", "pixels": n}


def make_view(frame=None, size=(760, 640)) -> FrameView:
    view = FrameView()
    font = QFont("Arial")
    font.setPixelSize(16)
    view.setFont(font)
    view.resize(*size)
    if frame is not None:
        view.set_frame(frame)
    return view


def click(view: FrameView, pos: QPointF) -> None:
    for kind, buttons in ((QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton), (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton)):
        ev = QMouseEvent(kind, pos, QPointF(view.mapToGlobal(pos.toPoint())), Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(view, ev)


def hued(shot: np.ndarray, color) -> np.ndarray:
    """The pixels that are visibly the marker colour (cyan or yellow), not grey, black or white, whatever they were
    blended with at the edge of a line."""
    r, g, b = (shot[..., i].astype(int) for i in range(3))
    if color.red() < 100:                                   # the cyan ring: green and blue well above red
        return (g - r > 70) & (b - r > 70)
    return (r - b > 110) & (g - b > 80)                     # the selected yellow: red and green well above blue


def rgb(view: FrameView) -> np.ndarray:
    img = view.grab().toImage().convertToFormat(QImage.Format.Format_RGB32)
    buf = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)
    return buf[:, :img.width(), [2, 1, 0]].copy()


def plus(view: FrameView, col: int, row: int, step: int = 3) -> None:
    """The five markers an agent put on an atom (trial 5): its pixel and four flanks."""
    for dc, dr in ((0, 0), (-step, 0), (step, 0), (0, -step), (0, step)):
        click(view, view.pixel_to_widget(col + dc, row + dr))


def discs_clear(view: FrameView, rects) -> None:
    """No number covers another number, and none covers a ring."""
    r = fv.marker_radius(view._fpx())
    pts = [view.scan_to_widget(m["x_nm"], m["y_nm"]) for m in view.markers()]
    for i, (a, _) in enumerate(rects):
        for j, (b, _) in enumerate(rects):
            if i < j:
                assert not a.intersects(b), (i + 1, j + 1)
        for j, c in enumerate(pts):
            nx, ny = min(max(c.x(), a.left()), a.right()), min(max(c.y(), a.top()), a.bottom())
            assert math.hypot(nx - c.x(), ny - c.y()) >= r + 1.0, (f"number {i + 1} covers the ring of marker {j + 1}")


# ── the ring is open ──
@pytest.mark.parametrize("zoom", [1, 2, 4, 8])
@pytest.mark.parametrize("peak", [70.0, -70.0])
def test_the_pixel_a_marker_is_on_and_its_neighbours_are_as_the_image_draws_them_at_every_zoom_and_grey_level(app, zoom, peak):
    """On a bright atom or a dark one, with the whole image or a part of it: inside the ring the picture is the picture."""
    frame = atom_frame(peak_pm=peak)
    view = make_view(frame)
    view.set_zoom(zoom, focus=pixel_to_scan_nm(fv.frame_geometry(frame), 48, 48))
    bare = rgb(view)
    click(view, view.pixel_to_widget(48, 48))
    (m,) = view.markers()
    pt = view.scan_to_widget(m["x_nm"], m["y_nm"])
    r = fv.marker_radius(view._fpx())
    shot = rgb(view)
    yy, xx = np.mgrid[0:shot.shape[0], 0:shot.shape[1]]
    inside = np.hypot(xx + 0.5 - pt.x(), yy + 0.5 - pt.y()) <= r - 3.0
    assert inside.sum() > 20
    assert np.array_equal(shot[inside], bare[inside])                           # nothing drawn over the pixel and its neighbours
    ring = (np.hypot(xx + 0.5 - pt.x(), yy + 0.5 - pt.y()) <= r + 2.5) & ~inside
    assert (shot[ring] != bare[ring]).any(axis=1).mean() > 0.4                  # and the ring itself is there, round it


def test_a_marker_is_the_rings_centre_and_there_is_no_cross_in_it(app):
    view = make_view(atom_frame())
    click(view, view.pixel_to_widget(30, 30))
    pt = view.scan_to_widget(view.markers()[0]["x_nm"], view.markers()[0]["y_nm"])
    shot = rgb(view)
    view.clear_markers()
    bare = rgb(view)
    r = fv.marker_radius(view._fpx())
    yy, xx = np.mgrid[0:shot.shape[0], 0:shot.shape[1]]
    centre = np.hypot(xx + 0.5 - pt.x(), yy + 0.5 - pt.y()) <= r - 3.0
    assert centre.sum() >= 30 and np.array_equal(shot[centre], bare[centre])    # a disc of the ring's own size: the picture


# ── the pixel's own outline, where it is large ──
def test_zoomed_in_the_pixel_a_marker_is_on_is_outlined_and_on_another_frame_it_is_not(app):
    a = atom_frame(n=32, name="sim001.sxm", size=8.0)
    view = make_view(a)
    view.set_zoom(8, focus=pixel_to_scan_nm(fv.frame_geometry(a), 16, 16))
    click(view, view.pixel_to_widget(16, 16))
    (m,) = view.markers()                                                       # the only marker is the selected one
    box_a, box_b = view.pixel_to_widget(15.5, 15.5), view.pixel_to_widget(16.5, 16.5)
    assert box_b.x() - box_a.x() >= 4.0 * fv.marker_radius(view._fpx())              # a pixel wider than four ring radii
    sel = np.array([fv.MARKER_SELECTED_COLOR.red(), fv.MARKER_SELECTED_COLOR.green(), fv.MARKER_SELECTED_COLOR.blue()])
    near = lambda shot: (np.abs(shot.astype(int) - sel).sum(axis=2) < 60)            # noqa: E731
    yc = int(round((box_a.y() + box_b.y()) / 2))
    left, right = int(round(box_a.x())), int(round(box_b.x()))
    colour = near(rgb(view))
    assert colour[yc - 2:yc + 3, left:left + 4].any() and colour[yc - 2:yc + 3, right - 4:right].any()    # its sides are drawn
    # the same scan position on a frame of another window: a ring only, nothing at the pixel's sides
    b = atom_frame(n=24, name="sim002.sxm", size=8.0, center=(0.13, -0.11))
    view.set_frame(b)
    view.set_zoom(8, focus=(m["x_nm"], m["y_nm"]))
    pt = view.scan_to_widget(m["x_nm"], m["y_nm"])
    r = fv.marker_radius(view._fpx())
    colour = near(rgb(view))
    yy, xx = np.mgrid[0:colour.shape[0], 0:colour.shape[1]]
    away = np.hypot(xx + 0.5 - pt.x(), yy + 0.5 - pt.y()) > r + 4.0
    away &= (yy > view.image_rect().top()) & (yy < view.image_rect().bottom()) & (xx > view.image_rect().left()) & (xx < view.image_rect().right())
    items = [(1, pt, True, m)]
    for rect, _far in view._marker_label_rects(view._layout(), items):          # its number's box has the colour's border
        away &= ~((xx >= rect.left() - 3) & (xx <= rect.right() + 3) & (yy >= rect.top() - 3) & (yy <= rect.bottom() + 3))
    assert colour[away].sum() == 0
    assert colour[(np.abs(np.hypot(xx + 0.5 - pt.x(), yy + 0.5 - pt.y()) - r) <= 1.2)].any()


# ── the numbers ──
@pytest.mark.parametrize("zoom", [1, 2, 4])
def test_the_five_markers_on_an_atom_are_all_seen_and_none_of_their_numbers_covers_another_or_a_ring(app, zoom):
    frame = atom_frame()
    view = make_view(frame)
    view.set_zoom(zoom, focus=pixel_to_scan_nm(fv.frame_geometry(frame), 48, 48))
    plus(view, 48, 48)
    assert len(view.markers()) == 5
    lay = view._layout()
    items = [(i + 1, view.scan_to_widget(m["x_nm"], m["y_nm"]), i == view._selected, m) for i, m in enumerate(view.markers())]
    assert all(pt is not None for _, pt, _, _ in items)
    rects = view._marker_label_rects(lay, items)
    discs_clear(view, rects)
    for rect, _ in rects:
        assert QRectF_contains(lay.img, rect)                                  # inside the image
    # the rings are all there: the colour of each is found round its own centre
    shot = rgb(view)
    yy, xx = np.mgrid[0:shot.shape[0], 0:shot.shape[1]]
    r = fv.marker_radius(view._fpx())
    for k, (_, pt, selected, _m) in enumerate(items):
        want = fv.MARKER_SELECTED_COLOR if selected else fv.MARKER_COLOR
        ring = (np.abs(np.hypot(xx + 0.5 - pt.x(), yy + 0.5 - pt.y()) - r) <= 1.0)
        assert hued(shot, want)[ring].mean() > 0.4, f"ring {k + 1} is covered"


def test_a_number_that_cannot_stand_beside_its_ring_stands_off_with_a_line_and_the_selected_marker_is_drawn_last(app):
    view = make_view(atom_frame())
    plus(view, 48, 48, step=2)                                                   # closer than a number is wide: the rings crowd
    lay = view._layout()
    items = [(i + 1, view.scan_to_widget(m["x_nm"], m["y_nm"]), i == view._selected, m) for i, m in enumerate(view.markers())]
    rects = view._marker_label_rects(lay, items)
    assert any(far for _, far in rects)                                          # some number stands off ...
    discs_clear(view, rects)                                                     # ... and still covers nothing
    view.select_marker(0)
    shot = rgb(view)
    sel = np.array([fv.MARKER_SELECTED_COLOR.red(), fv.MARKER_SELECTED_COLOR.green(), fv.MARKER_SELECTED_COLOR.blue()])
    assert (np.abs(shot.astype(int) - sel).sum(axis=2) < 60).sum() > 40          # the selected one is in its own colour


def test_markers_at_the_edge_keep_their_numbers_inside_the_image(app):
    view = make_view(atom_frame())
    for col, row in ((0, 0), (95, 0), (0, 95), (95, 95), (47, 0), (95, 47)):
        click(view, view.pixel_to_widget(col, row))
    lay = view._layout()
    items = [(i + 1, view.scan_to_widget(m["x_nm"], m["y_nm"]), False, m) for i, m in enumerate(view.markers())]
    rects = view._marker_label_rects(lay, items)
    for rect, _ in rects:
        assert QRectF_contains(lay.img, rect)
    discs_clear(view, rects)


def QRectF_contains(img, rect) -> bool:                                           # noqa: N802
    return img.x() <= rect.left() and rect.right() <= img.right() + 1 and img.y() <= rect.top() and rect.bottom() <= img.bottom() + 1
