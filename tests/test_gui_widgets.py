"""The desktop console's viewer widgets (stmbench.gui.widgets), run offscreen.

FrameView must show the image an MCP agent gets, and a point read off it must be a position the
judge finds covered by that frame: its mapping is pinned to ``stmbench.render.pixel_to_scan_nm``
(rotated frames included), the drawn grey levels to ``display_array``, and markers to the scan
coordinates they were placed at. Mouse input goes through real ``QMouseEvent``s (fractional
positions where a test needs exactness), the way a click reaches the widget in the console.
"""
from __future__ import annotations

import math
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
# Offscreen Qt on Windows has no fonts unless it is told where they are: every glyph becomes a box
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", str(Path(os.environ.get("WINDIR", "")) / "Fonts"))

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QFontDatabase, QFontInfo, QFontMetrics, QImage, QMouseEvent, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from stmbench.gui.widgets import FrameView, PlotView, frame_thumbnail  # noqa: E402
from stmbench.gui.widgets import frame_view as fv  # noqa: E402
from stmbench.gui.widgets import plot_view as pv  # noqa: E402
from stmbench.instrument.render import display_array  # noqa: E402
from stmbench.render import flatten as plane_flatten  # noqa: E402
from stmbench.render import pixel_to_scan_nm  # noqa: E402

WIDGET_SIZE = (860, 740)
BLUE = np.array([0x00, 0x72, 0xB2])                     # PlotView's first series colour
ORANGE = np.array([0xD5, 0x5E, 0x00])                   # its second
SERIES = [np.array(QColor(c).getRgb()[:3]) for c in pv.SERIES_COLORS]
READOUT_BG = pv.READOUT_BG


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


# ── helpers ────────────────────────────────────────────────────────────────────────────────
def make_view(app, size=WIDGET_SIZE, px=16) -> FrameView:
    view = FrameView()
    font = QFont("Arial")
    font.setPixelSize(px)
    view.setFont(font)
    view.resize(*size)
    return view


def make_frame(array, *, center=(0.0, 0.0), size=10.0, angle=0.0, channel="z", name="sim001.sxm") -> dict:
    """A dict shaped like ``Instrument.frame()``'s reply."""
    array = np.asarray(array, dtype=float)
    return {"frame": name, "center_nm": [center[0], center[1]], "size_nm": size, "pixels": array.shape[1],
            "angle_deg": angle, "nm_per_px": size / array.shape[1], "channel": channel, "direction": "forward",
            "unit": "m" if channel == "z" else "A", "t_start_s": 0.0, "bias_v": 0.05, "setpoint_pa": 200.0,
            "array": array}


def geometry(frame: dict) -> dict:
    """The geometry ``pixel_to_scan_nm`` takes, as the web console's /api/pixel builds it."""
    return {"cx_nm": frame["center_nm"][0], "cy_nm": frame["center_nm"][1], "w_nm": frame["size_nm"],
            "h_nm": frame["size_nm"], "angle_deg": frame["angle_deg"], "nx": frame["pixels"], "ny": frame["pixels"]}


def bumpy(n=64, seed=3, scale=1e-10) -> np.ndarray:
    """A z-like array in metres: a tilted plane, two bumps and noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    z = 0.4 * xx + 0.7 * yy
    for cx, cy in ((n * 0.3, n * 0.6), (n * 0.7, n * 0.3)):
        z += 25.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (n / 14) ** 2))
    return (z + rng.normal(0, 0.8, z.shape)) * scale / 100.0


def send(view, kind, pos, button=Qt.MouseButton.LeftButton, buttons=Qt.MouseButton.NoButton):
    pos = QPointF(pos)
    ev = QMouseEvent(kind, pos, QPointF(view.mapToGlobal(pos.toPoint())), button, buttons, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(view, ev)


def click(view, pos, button=Qt.MouseButton.LeftButton):
    send(view, QEvent.Type.MouseButtonPress, pos, button, button)
    send(view, QEvent.Type.MouseButtonRelease, pos, button, Qt.MouseButton.NoButton)


def drag(view, a, b):
    send(view, QEvent.Type.MouseButtonPress, a, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)
    send(view, QEvent.Type.MouseMove, b, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    send(view, QEvent.Type.MouseButtonRelease, b, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)


def move(view, pos):
    send(view, QEvent.Type.MouseMove, pos, Qt.MouseButton.NoButton, Qt.MouseButton.NoButton)


def to_array(img: QImage) -> np.ndarray:
    """An (h, w, 3) uint8 RGB array of a QImage."""
    img = img.convertToFormat(QImage.Format.Format_RGB32)
    buf = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)
    return buf[:, :img.width(), [2, 1, 0]].copy()           # little-endian 0xffRRGGBB is stored B, G, R, X


def rgb(widget) -> np.ndarray:
    """The widget as rendered."""
    return to_array(widget.grab().toImage())


def drawn_grey(view: FrameView, n: int) -> np.ndarray:
    """The grey level drawn at the centre of every pixel of the n x n frame on display."""
    shot = rgb(view)
    out = np.empty((n, n), np.uint8)
    for r in range(n):
        for c in range(n):
            pt = view.pixel_to_widget(c, r)
            px = shot[int(math.floor(pt.y())), int(math.floor(pt.x()))]
            assert px[0] == px[1] == px[2], f"pixel ({c}, {r}) is not grey: {px}"
            out[r, c] = px[0]
    return out


def near(shot: np.ndarray, color) -> np.ndarray:
    """Mask of the pixels within a small colour distance of ``color`` (an RGB triple or QColor)."""
    if isinstance(color, np.ndarray):
        target = color
    else:
        target = np.array([color.red(), color.green(), color.blue()])
    return np.abs(shot.astype(int) - target).sum(axis=2) < 60


def xy(pt) -> tuple[float, float]:
    """A QPointF as a tuple (``pytest.approx`` does not compare Qt points)."""
    return pt.x(), pt.y()


class Capture:
    """Everything a signal emitted, in order."""

    def __init__(self, signal) -> None:
        self.items: list = []
        signal.connect(self.items.append)


# ── ticks and number formatting ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("lo, hi, max_ticks, step", [(-15, 15, 8, 5.0), (-15, 15, 16, 2.0), (0, 1, 6, 0.2),
                                                     (12.3, 12.9, 5, 0.2), (0, 3.7e-9, 6, 1e-9), (2, 1000, 5, 200.0)])
def test_nice_ticks_are_the_finest_one_two_five_step_that_fits(lo, hi, max_ticks, step):
    ticks, got = pv.nice_ticks(lo, hi, max_ticks)
    assert got == pytest.approx(step)
    assert 1 <= len(ticks) <= max_ticks
    assert all(lo - 1e-9 * step <= t <= hi + 1e-9 * step for t in ticks)
    assert all(abs(t / step - round(t / step)) < 1e-6 for t in ticks)


def test_nice_ticks_degenerate_ranges_and_tick_text():
    assert pv.nice_ticks(1.0, 1.0, 5) == ([], 0.0)
    assert pv.nice_ticks(float("nan"), 2.0, 5) == ([], 0.0)
    ticks, step = pv.nice_ticks(-0.25, 0.25, 6)
    assert pv.format_ticks(ticks, step) == ["-0.2", "-0.1", "0.0", "0.1", "0.2"]
    assert pv.format_number(-0.0004, 2) == "0.00"             # never "-0.00"
    assert pv.format_number(1234.5, 0) in ("1234", "1235")
    assert pv.format_number(float("nan"), 2) == "nan"
    assert pv.format_number(2.5e9, 0) == "2.5e+09"


# ── coordinates ────────────────────────────────────────────────────────────────────────────
def test_scan_to_pixel_inverts_pixel_to_scan_nm():
    rng = np.random.default_rng(1)
    for angle in (0.0, 30.0, -77.0, 180.0):
        geom = {"cx_nm": 12.5, "cy_nm": -7.25, "w_nm": 20.0, "h_nm": 20.0, "angle_deg": angle, "nx": 128, "ny": 128}
        for col, row in rng.uniform(-3, 130, (20, 2)):
            x, y = pixel_to_scan_nm(geom, col, row)
            assert fv.scan_to_pixel(geom, x, y) == pytest.approx((col, row), abs=1e-9)


def test_frame_geometry_is_what_the_web_console_maps_through():
    frame = make_frame(np.zeros((48, 48)), center=(3.0, -4.5), size=7.5, angle=-20.0)
    assert fv.frame_geometry(frame) == pytest.approx(geometry(frame))


@pytest.mark.parametrize("angle", [0.0, 30.0])
def test_a_click_marks_the_centre_of_the_pixel_it_falls_in_and_reads_that_pixels_value(app, angle):
    """A position a fraction fx of the image width from its left edge is col = fx * nx - 0.5; the marker it
    adds is at the centre of the pixel floor(col + 0.5), in the judge's mapping, to 1e-6 nm, and its value is
    that pixel's: the coordinates and the reading belong to the same point."""
    n = 64
    frame = make_frame(bumpy(n), center=(3.5, -2.0), size=12.0, angle=angle)
    geom = geometry(frame)
    view = make_view(app)
    view.set_frame(frame)
    for col, row in [(-0.5, -0.5), (0.0, 0.0), (n - 1.0, 0.0), (0.0, n - 1.0), (n / 2 - 0.5, n / 2 - 0.5),
                     (12.3, 40.7), (n - 0.5001, n - 0.5001), (20.49, 7.51)]:
        pos = view.pixel_to_widget(col, row)
        got = Capture(view.marker_added)
        click(view, pos)
        (m,) = got.items
        ci, ri = m["col"], m["row"]
        assert (ci, ri) == tuple(min(max(int(math.floor(v + 0.5)), 0), n - 1) for v in (col, row)) or col % 1 == 0.5
        assert (m["x_nm"], m["y_nm"]) == pytest.approx(pixel_to_scan_nm(geom, ci, ri), abs=1e-6)
        assert fv.scan_to_pixel(geom, m["x_nm"], m["y_nm"]) == pytest.approx((ci, ri), abs=1e-9)
        assert m["value"] == view._shown.values[ri, ci] and m["unit"] == "pm" and m["frame"] == "sim001.sxm"
        assert set(m) == {"x_nm", "y_nm", "col", "row", "value", "unit", "frame"}
        # drawn on the pixel's centre, which is where the ring is centred
        assert xy(view.scan_to_widget(m["x_nm"], m["y_nm"])) == pytest.approx(xy(view.pixel_to_widget(ci, ri)), abs=1e-6)


@pytest.mark.parametrize("angle", [0.0, -30.0])
def test_a_marker_is_the_same_wherever_in_its_pixel_it_was_placed(app, angle):
    """Anywhere inside one pixel the click makes the same marker, and the cursor reads the same point: (x, y, value)
    is a measurement, not the place the pointer happened to be (trial 5 steps 40-45: coordinates 0.06 nm off the
    pixel the value came from)."""
    n = 40
    frame = make_frame(bumpy(n), center=(-1.0, 2.5), size=10.0, angle=angle)
    view = make_view(app)
    view.set_frame(frame)
    for zoom in (1, 4):
        view.set_zoom(zoom)
        view.set_tool("pan")
        view.pan_by(40, 25)                                              # the view is not at the frame's corner
        view.set_tool("marker")
        c0, r0, wv, hv = view.view_rect()
        ci, ri = int(c0 + wv / 2), int(r0 + hv / 2)
        marks, cursors = [], []
        for dc, dr in ((-0.45, -0.45), (0.0, 0.0), (0.45, -0.45), (-0.45, 0.45), (0.45, 0.45), (0.2, -0.1)):
            pos = view.pixel_to_widget(ci + dc, ri + dr)
            got, cur = Capture(view.marker_added), Capture(view.cursor_moved)
            move(view, pos)
            click(view, pos)
            marks.append(got.items[-1])
            cursors.append(cur.items[-1])
        assert all(m == marks[0] for m in marks) and (marks[0]["col"], marks[0]["row"]) == (ci, ri)
        assert all(c == cursors[0] for c in cursors)
        assert cursors[0] == {k: marks[0][k] for k in cursors[0]}      # what the cursor read is what the click marked
        view.clear_markers()


@pytest.mark.parametrize("angle", [0.0, 30.0, -135.0])
def test_widget_and_scan_positions_agree_with_the_geometry(app, angle):
    n = 40
    frame = make_frame(bumpy(n), center=(-6.0, 9.5), size=8.0, angle=angle)
    geom = geometry(frame)
    view = make_view(app)
    view.set_frame(frame)
    rect = view.image_rect()
    assert rect.width() == rect.height() and rect.width() > 300
    # pixel (col, row) sits where its fractional position in the image says
    for col, row in [(0, 0), (n - 1, n - 1), (7.25, 19.5)]:
        pt = view.pixel_to_widget(col, row)
        assert pt.x() == pytest.approx(rect.x() + (col + 0.5) * rect.width() / n, abs=1e-9)
        assert pt.y() == pytest.approx(rect.y() + (row + 0.5) * rect.height() / n, abs=1e-9)
        x, y = pixel_to_scan_nm(geom, col, row)
        assert xy(view.scan_to_widget(x, y)) == pytest.approx(xy(pt), abs=1e-6)
    assert xy(view.pixel_to_widget(-0.5, -0.5)) == pytest.approx((rect.x(), rect.y()), abs=1e-9)
    # the frame's own corners are covered; a point a pixel past the edge is not
    assert view.scan_to_widget(*pixel_to_scan_nm(geom, -0.5, -0.5)) is not None
    assert view.scan_to_widget(*pixel_to_scan_nm(geom, n - 0.5, n - 0.5)) is not None
    assert view.scan_to_widget(*pixel_to_scan_nm(geom, -1.5, 10)) is None
    assert view.scan_to_widget(*pixel_to_scan_nm(geom, 10, n + 0.5)) is None


def test_a_position_read_in_the_view_is_one_the_judge_finds_covered(app):
    from stmbench.trackB.claims import frame_covering

    n, angle = 64, 33.0
    frame = make_frame(bumpy(n), center=(12.0, -4.0), size=8.0, angle=angle)
    saved = {"kind": "scan_saved", "complete": True, "cx_m": 12e-9, "cy_m": -4e-9, "w_m": 8e-9, "h_m": 8e-9,
             "angle_deg": angle, "nx": n, "ny": n, "idx": 1}
    view = make_view(app)
    view.set_frame(frame)
    got = Capture(view.marker_added)
    for col, row in [(0, 0), (n - 1, 0), (0, n - 1), (n - 1, n - 1), (31.5, 31.5)]:
        click(view, view.pixel_to_widget(col, row))
        m = got.items[-1]
        assert frame_covering([saved], m["x_nm"], m["y_nm"], {"min_fov_nm": 5, "max_nm_per_px": 5}) is saved


def test_the_centre_pixel_of_an_unrotated_frame_is_the_scan_offset_and_row_0_is_high_y(app):
    n = 33                                                              # odd: one pixel is the middle of the frame
    view = make_view(app)
    view.set_frame(make_frame(bumpy(n), center=(5.0, 6.0), size=4.0))
    got = Capture(view.marker_added)
    click(view, view.pixel_to_widget(n // 2, n // 2))
    assert (got.items[0]["x_nm"], got.items[0]["y_nm"]) == pytest.approx((5.0, 6.0), abs=1e-6)
    click(view, view.pixel_to_widget(0, 0))
    click(view, view.pixel_to_widget(n - 1, n - 1))
    first, last = got.items[1], got.items[2]
    assert first["x_nm"] < last["x_nm"] and first["y_nm"] > last["y_nm"]
    # a pixel's centre is half a pixel inside the frame's edge: the corner pixels' centres are not at +-2 nm
    assert first["x_nm"] == pytest.approx(5.0 - 2.0 + 0.5 * 4.0 / n, abs=1e-9)
    assert first["y_nm"] == pytest.approx(6.0 + 2.0 - 0.5 * 4.0 / n, abs=1e-9)


# ── the picture ────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("channel, n", [("z", 48), ("current", 48), ("didv", 48), ("z", 256)])
def test_line_mode_draws_exactly_display_array(app, channel, n):
    """At every pixel centre, also where the enlargement is not a whole number (256 px in ~680)."""
    z = bumpy(n)
    array = z if channel == "z" else 2e-10 + 1e-12 * np.cumsum(z - z.mean(), axis=1) / n
    view = make_view(app)
    view.set_frame(make_frame(array, channel=channel))
    assert np.array_equal(drawn_grey(view, n), display_array(array, channel))


def test_plane_and_none_modes_use_the_same_percentile_mapping(app):
    n = 40
    z = bumpy(n)
    view = make_view(app)
    view.set_frame(make_frame(z))
    view.set_flatten("none")
    assert np.array_equal(drawn_grey(view, n), display_array(z, "current"))          # mapped as recorded
    view.set_flatten("plane")
    want = display_array(plane_flatten(z, "plane"), "current")
    assert np.abs(drawn_grey(view, n).astype(int) - want.astype(int)).max() <= 1
    view.set_flatten("line")
    assert np.array_equal(drawn_grey(view, n), display_array(z, "z"))


def test_unacquired_pixels_are_drawn_flat_blue_not_as_low(app):
    n = 32
    z = bumpy(n)
    z[10:14, :] = np.nan
    view = make_view(app)
    view.set_frame(make_frame(z))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                  # display_array: all-NaN lines have no median
        assert (display_array(z, "z")[10:14] == 0).all()                 # the agent's image shows them as the lowest level
    pt = view.pixel_to_widget(5, 11)
    assert tuple(rgb(view)[int(pt.y()), int(pt.x())]) == fv._NAN_RGB


@pytest.mark.parametrize("name", ["constant", "all nan", "nan row", "inf", "huge", "tiny", "2x2", "wide"])
@pytest.mark.parametrize("channel", ["z", "current"])
def test_degenerate_frames_draw_without_warnings_in_every_flatten_mode(app, name, channel):
    rng = np.random.default_rng(0)
    arrays = {
        "constant": np.full((16, 16), 1e-10), "all nan": np.full((16, 16), np.nan),
        "nan row": np.where(np.arange(16)[:, None] == 5, np.nan, rng.normal(0, 1e-11, (16, 16))),
        "inf": np.where((np.arange(16)[:, None] == 5) & (np.arange(16)[None, :] == 7), np.inf, rng.normal(0, 1e-11, (16, 16))),
        "huge": rng.normal(0, 1e-3, (16, 16)), "tiny": rng.normal(0, 1e-25, (16, 16)),
        "2x2": np.array([[0.0, 1e-11], [2e-11, 3e-11]]), "wide": rng.normal(0, 1e-11, (8, 40)),
    }
    frame = make_frame(arrays[name], channel=channel)
    view = make_view(app)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # a numpy RuntimeWarning from the data would show here
        for mode in ("line", "plane", "none"):
            view.set_flatten(mode)
            view.set_frame(frame)
            assert rgb(view).shape == (WIDGET_SIZE[1], WIDGET_SIZE[0], 3)
            assert view._layout().bar_ticks                      # the colour bar always has its two end labels
        thumb = frame_thumbnail(frame, 32)
    assert (thumb.width(), thumb.height()) == (32, 32)


def test_the_view_is_square_and_inside_the_widget_at_any_size(app):
    frame = make_frame(bumpy(64))
    for size, px in [(WIDGET_SIZE, 16), ((500, 420), 16), ((1000, 560), 16), ((300, 200), 16), (WIDGET_SIZE, 24)]:
        view = make_view(app, size, px)
        view.set_frame(frame)
        r = view.image_rect()
        assert r.width() == r.height() >= 32
        assert 0 <= r.x() and r.right() < size[0] and 0 <= r.y() and r.bottom() < size[1], (size, px, r)
        view.set_frame(None)
        r = view.image_rect()
        assert r.width() > 0 and 0 <= r.x() and r.right() < size[0] and 0 <= r.y() and r.bottom() < size[1]


def test_axes_are_scan_xy_when_unrotated_and_frame_uv_when_rotated(app):
    view = make_view(app)
    view.set_frame(make_frame(bumpy(32), center=(4.0, -6.0), size=10.0))
    (x0, x1), (y0, y1), xt, yt = view._axis_spec()
    assert (x0, x1, y0, y1) == pytest.approx((-1.0, 9.0, -11.0, -1.0)) and xt == "x (nm)" and yt == "y (nm)"
    view.set_frame(make_frame(bumpy(32), center=(4.0, -6.0), size=10.0, angle=30.0))
    (x0, x1), (y0, y1), xt, yt = view._axis_spec()
    assert (x0, x1, y0, y1) == pytest.approx((-5.0, 5.0, -5.0, 5.0)) and xt.startswith("u (nm)") and yt == "v (nm)"
    assert "30" in xt


def test_the_colour_bar_states_the_mapped_range_in_display_units(app):
    n = 64
    z = bumpy(n)
    view = make_view(app)
    view.set_frame(make_frame(z))
    sh = view._shown
    flat = z - np.median(z, axis=1, keepdims=True)
    lo, hi = np.percentile(flat, [0.5, 99.5]) * 1e12
    assert sh.unit == "pm" and (sh.lo, sh.hi) == pytest.approx((lo, hi))
    grey = display_array(z, "z")
    assert grey[np.unravel_index(flat.argmax(), flat.shape)] == 255 and grey[np.unravel_index(flat.argmin(), flat.shape)] == 0
    labels = dict(view._layout().bar_ticks)                            # fraction from the top -> text
    for frac, want in ((0.0, hi), (1.0, lo)):
        decimals = len(labels[frac].partition(".")[2])
        assert abs(float(labels[frac]) - want) <= 0.5 * 10.0 ** -decimals + 1e-9
    view.set_frame(make_frame(2e-10 + 1e-12 * z / 1e-10, channel="current"))
    assert view._shown.unit == "pA"
    view.set_frame(make_frame(2e-10 + 1e-12 * z / 1e-10, channel="didv"))
    assert view._shown.unit == "pA"


def test_set_frame_refuses_a_frame_it_cannot_draw_and_keeps_the_old_one(app):
    view = make_view(app)
    good = make_frame(bumpy(16))
    view.set_frame(good)
    for bad in (make_frame(np.zeros((1, 8))), {**good, "size_nm": 0.0}, {**good, "array": np.zeros(5)}):
        with pytest.raises(ValueError):
            view.set_frame(bad)
    assert view._shown is not None and view._shown.nx == 16


# ── readouts ───────────────────────────────────────────────────────────────────────────────
def test_cursor_readout_and_leaving_the_image(app):
    n = 32
    z = bumpy(n)
    frame = make_frame(z, center=(1.0, 2.0), size=6.0)
    view = make_view(app)
    view.set_frame(frame)
    got = Capture(view.cursor_moved)
    pos = view.pixel_to_widget(9.0, 20.0)
    move(view, pos)
    c = got.items[0]
    assert set(c) == {"x_nm", "y_nm", "col", "row", "value", "unit"}
    assert (c["col"], c["row"], c["unit"]) == (9, 20, "pm")
    assert (c["x_nm"], c["y_nm"]) == pytest.approx(pixel_to_scan_nm(geometry(frame), 9.0, 20.0), abs=1e-6)
    flat = z - np.median(z, axis=1, keepdims=True)
    assert c["value"] == pytest.approx(flat[20, 9] * 1e12)
    rect = view.image_rect()
    move(view, QPointF(rect.x() - 5, rect.y() + 30))                  # into the margin: {} once, not on every move
    move(view, QPointF(rect.x() - 9, rect.y() + 31))
    assert got.items[1:] == [{}]
    move(view, pos)
    QApplication.sendEvent(view, QEvent(QEvent.Type.Leave))
    assert got.items[2]["col"] == 9 and got.items[3:] == [{}]


def test_values_follow_the_flatten_mode_in_display_units(app):
    n = 32
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    z = (3.0 * xx + 2.0 * yy + 5.0) * 1e-12
    z[8, 8] += 40e-12                                                # one 40 pm bump on a tilted plane
    view = make_view(app)
    view.set_frame(make_frame(z))
    got = Capture(view.marker_added)
    seen = {}
    for mode in ("line", "plane", "none"):
        view.set_flatten(mode)
        n0 = len(got.items)
        click(view, view.pixel_to_widget(8, 8))
        click(view, view.pixel_to_widget(24, 24))
        seen[mode] = [m["value"] for m in got.items[n0:]]
    assert {m["unit"] for m in got.items} == {"pm"}
    assert seen["none"] == pytest.approx([z[8, 8] * 1e12, z[24, 24] * 1e12])             # raw, in pm
    assert seen["line"] == pytest.approx([(z[8, 8] - np.median(z[8])) * 1e12, (z[24, 24] - np.median(z[24])) * 1e12])
    assert seen["plane"] == pytest.approx([40.0, 0.0], abs=1.0)                           # the plane is gone, the bump stays
    view.set_frame(make_frame(2e-10 + 1e-12 * np.ones((n, n)), channel="current"))
    click(view, view.pixel_to_widget(3, 3))
    assert (got.items[-1]["unit"], got.items[-1]["value"]) == ("pA", pytest.approx(201.0))


def test_a_pixel_without_data_reads_nan(app):
    z = bumpy(32)
    z[4, 5] = np.nan
    view = make_view(app)
    view.set_frame(make_frame(z))
    got = Capture(view.marker_added)
    click(view, view.pixel_to_widget(5, 4))
    assert math.isnan(got.items[0]["value"])


# ── markers ────────────────────────────────────────────────────────────────────────────────
def test_marker_api(app):
    view = make_view(app)
    view.set_frame(make_frame(bumpy(32)))
    added, changed = Capture(view.marker_added), Capture(view.markers_changed)
    assert view.markers() == [] and view.selected_marker() is None
    for pt in ((4, 5), (10, 11), (20, 6)):
        click(view, view.pixel_to_widget(*pt))
    assert [(m["col"], m["row"]) for m in view.markers()] == [(4, 5), (10, 11), (20, 6)]
    assert [len(c) for c in changed.items] == [1, 2, 3]
    assert changed.items[-1] == view.markers() and all(m["frame"] == "sim001.sxm" for m in view.markers())
    assert view.selected_marker() == view.markers()[-1]                # a new marker is the selected one
    n_changed = len(changed.items)
    view.select_marker(0)
    assert view.selected_marker()["col"] == 4 and len(changed.items) == n_changed        # selecting changes no list
    view.select_marker(7)
    assert view.selected_marker() is None
    view.select_marker(2)
    view.select_marker(None)
    assert view.selected_marker() is None
    view.select_marker(1)
    copy = view.markers()
    copy[0]["x_nm"] = 1e9                                             # the caller's copy is its own
    assert view.markers()[0]["x_nm"] != 1e9
    view.remove_marker(0)
    assert [(m["col"], m["row"]) for m in view.markers()] == [(10, 11), (20, 6)] and changed.items[-1] == view.markers()
    view.clear_markers()
    assert view.markers() == [] and changed.items[-1] == [] and len(added.items) == 3
    n_changed = len(changed.items)
    view.clear_markers()
    assert len(changed.items) == n_changed                             # nothing to clear: nothing emitted


def test_a_double_click_places_one_marker(app):
    """What an agent's pyautogui.doubleClick delivers: press, release, double-click, release."""
    view = make_view(app)
    view.set_frame(make_frame(bumpy(32)))
    pos = view.pixel_to_widget(12, 9)
    L, N = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
    send(view, QEvent.Type.MouseButtonPress, pos, L, L)
    send(view, QEvent.Type.MouseButtonRelease, pos, L, N)
    send(view, QEvent.Type.MouseButtonDblClick, pos, L, L)
    send(view, QEvent.Type.MouseButtonRelease, pos, L, N)
    assert len(view.markers()) == 1
    view.set_tool("profile")
    got = Capture(view.profile_changed)
    for kind, buttons in ((QEvent.Type.MouseButtonPress, L), (QEvent.Type.MouseButtonRelease, N),
                          (QEvent.Type.MouseButtonDblClick, L), (QEvent.Type.MouseButtonRelease, N)):
        send(view, kind, pos, L, buttons)
    assert got.items == [] and len(view.markers()) == 1


def test_the_marker_list_is_announced_before_the_marker_event(app):
    """Both already hold the new marker (and its selection) by the time any slot runs."""
    view = make_view(app)
    view.set_frame(make_frame(bumpy(32)))
    order: list = []
    view.markers_changed.connect(lambda ms: order.append(("changed", len(ms), view.selected_marker()["col"])))
    view.marker_added.connect(lambda m: order.append(("added", len(view.markers()), view.selected_marker()["col"])))
    click(view, view.pixel_to_widget(6, 6))
    click(view, view.pixel_to_widget(25, 25))
    assert order == [("changed", 1, 6), ("added", 1, 6), ("changed", 2, 25), ("added", 2, 25)]


def test_only_a_left_click_with_the_marker_tool_on_the_image_adds_a_marker(app):
    view = make_view(app)
    got = Capture(view.marker_added)
    click(view, QPointF(300, 300))                                     # no frame yet
    view.set_frame(make_frame(bumpy(32)))
    rect = view.image_rect()
    centre = QPointF(rect.center())
    click(view, centre, Qt.MouseButton.RightButton)
    click(view, centre, Qt.MouseButton.MiddleButton)
    click(view, QPointF(rect.x() - 3, rect.y() + 40))                  # beside the image
    click(view, QPointF(rect.right() + 1, rect.center().y()))
    view.set_tool("none")
    click(view, centre)
    view.set_tool("profile")
    click(view, centre)                                                # a click is not a line either
    assert got.items == [] and view.markers() == []
    view.set_tool("marker")
    click(view, centre)
    assert len(got.items) == 1
    with pytest.raises(ValueError):
        view.set_tool("lasso")
    with pytest.raises(ValueError):
        view.set_flatten("cubic")


def test_a_marker_placed_on_one_frame_is_drawn_where_the_same_scan_coordinates_lie_on_the_next(app):
    view = make_view(app)
    n = 64
    a = make_frame(bumpy(n, seed=1), center=(0.0, 0.0), size=10.0, name="sim001.sxm")
    b = make_frame(bumpy(n, seed=2), center=(2.5, -1.0), size=10.0, angle=30.0, name="sim002.sxm")
    view.set_frame(a)
    click(view, view.pixel_to_widget(40, 25))                          # marker 1: inside both frames
    click(view, view.pixel_to_widget(2, 60))                           # marker 2 (the selected one): outside frame b
    m1, m2 = view.markers()
    assert m1["frame"] == m2["frame"] == "sim001.sxm"
    r = fv.marker_radius(16)
    for frame in (a, b):
        view.set_frame(frame)
        col, row = fv.scan_to_pixel(geometry(frame), m1["x_nm"], m1["y_nm"])
        want = view.pixel_to_widget(col, row)
        assert xy(view.scan_to_widget(m1["x_nm"], m1["y_nm"])) == pytest.approx(xy(want), abs=1e-6)
        # ... and the ring is really there: its colour, centred on that point, in the rendered widget
        shot = rgb(view)
        yy, xx = np.mgrid[0:shot.shape[0], 0:shot.shape[1]]
        ring = near(shot, fv.MARKER_COLOR) & (np.hypot(xx + 0.5 - want.x(), yy + 0.5 - want.y()) <= r + 2.5)
        assert ring.sum() > 20
        assert xx[ring].mean() + 0.5 == pytest.approx(want.x(), abs=1.0)
        assert yy[ring].mean() + 0.5 == pytest.approx(want.y(), abs=1.0)
    # marker 2 lies outside frame b: it is not drawn there (its colour, the selected one, is absent)
    assert view.scan_to_widget(m2["x_nm"], m2["y_nm"]) is None
    rect = view.image_rect()
    picture = rgb(view)[rect.top():rect.bottom() + 1, rect.left():rect.right() + 1]
    assert near(picture, fv.MARKER_SELECTED_COLOR).sum() == 0
    view.set_frame(a)
    picture = rgb(view)[rect.top():rect.bottom() + 1, rect.left():rect.right() + 1]
    assert near(picture, fv.MARKER_SELECTED_COLOR).sum() > 20          # ... and it is drawn on frame a


def test_markers_survive_an_empty_view_and_come_back(app):
    view = make_view(app)
    a = make_frame(bumpy(32))
    view.set_frame(a)
    click(view, view.pixel_to_widget(5, 5))
    view.set_frame(None)
    assert len(view.markers()) == 1 and view.scan_to_widget(0.0, 0.0) is None
    view.set_frame(a)
    assert view.scan_to_widget(view.markers()[0]["x_nm"], view.markers()[0]["y_nm"]) is not None


# ── profile ────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("angle", [0.0, 30.0])
def test_a_profile_across_a_ramp_has_the_expected_slope_and_length(app, angle):
    n = 64
    geom = geometry(make_frame(np.zeros((n, n)), center=(1.0, 2.0), size=8.0, angle=angle))
    x_nm = np.array([[pixel_to_scan_nm(geom, c, r)[0] for c in range(n)] for r in range(n)])
    frame = make_frame(x_nm * 1e-12, center=(1.0, 2.0), size=8.0, angle=angle, channel="current")   # I = x, in pA
    view = make_view(app)
    view.set_frame(frame)
    view.set_tool("profile")
    got = Capture(view.profile_changed)
    (ca, ra), (cb, rb) = (9.0, 12.0), (50.0, 41.0)
    drag(view, view.pixel_to_widget(ca, ra), view.pixel_to_widget(cb, rb))
    (p,) = got.items
    assert set(p) == {"frame", "x0_nm", "y0_nm", "x1_nm", "y1_nm", "s_nm", "values", "unit"}
    assert p["frame"] == "sim001.sxm" and p["unit"] == "pA"
    (x0, y0), (x1, y1) = pixel_to_scan_nm(geom, ca, ra), pixel_to_scan_nm(geom, cb, rb)
    assert (p["x0_nm"], p["y0_nm"], p["x1_nm"], p["y1_nm"]) == pytest.approx((x0, y0, x1, y1), abs=1e-6)
    length = math.hypot(x1 - x0, y1 - y0)
    s, v = np.array(p["s_nm"]), np.array(p["values"])
    assert s[0] == 0.0 and s[-1] == pytest.approx(length, abs=1e-9) and np.all(np.diff(s) > 0)
    assert len(s) == len(v) == round(length / frame["nm_per_px"]) + 1          # about one sample per image pixel
    assert np.polyfit(s, v, 1)[0] == pytest.approx((x1 - x0) / length, abs=1e-9)   # pA per nm along the line
    assert v == pytest.approx(x0 + s * (x1 - x0) / length, abs=1e-6)


def test_a_profile_is_measured_on_the_flattened_data_and_follows_the_view(app):
    n = 48
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    z = (2.0 * xx + 1.0 * yy) * 1e-12
    z[20:24, 20:24] += 30e-12
    view = make_view(app)
    view.set_frame(make_frame(z, name="sim001.sxm"))
    view.set_tool("profile")
    got = Capture(view.profile_changed)
    drag(view, view.pixel_to_widget(4, 22), view.pixel_to_widget(43, 22))
    line = np.array(got.items[-1]["values"])
    view.set_flatten("none")
    raw = np.array(got.items[-1]["values"])
    assert len(got.items) == 2 and got.items[-1]["unit"] == "pm"
    assert raw[0] == pytest.approx(z[22, 4] * 1e12) and raw[-1] == pytest.approx(z[22, 43] * 1e12)
    assert line[0] == pytest.approx((z[22, 4] - np.median(z[22])) * 1e12)
    view.set_flatten("plane")
    assert abs(got.items[-1]["values"][0]) < 3.0
    # the same frame name in another channel: the line stays, measured again, in that channel's unit
    view.set_flatten("line")                                            # current is shown as recorded in this mode
    view.set_frame(make_frame(2e-10 + 0 * z, name="sim001.sxm", channel="current"))
    assert got.items[-1]["unit"] == "pA" and got.items[-1]["values"][0] == pytest.approx(200.0)
    n_before = len(got.items)
    view.set_frame(make_frame(z, name="sim002.sxm"))                    # a different frame: the line is dropped
    assert got.items[n_before:] == [{}]
    view.set_frame(make_frame(z, name="sim001.sxm"))
    assert len(got.items) == n_before + 1                              # and does not come back


def test_a_drag_that_ends_outside_the_image_stops_at_its_edge_and_a_click_makes_no_line(app):
    frame = make_frame(bumpy(32))
    view = make_view(app)
    view.set_frame(frame)
    view.set_tool("profile")
    got = Capture(view.profile_changed)
    rect = view.image_rect()
    drag(view, view.pixel_to_widget(5, 16), QPointF(rect.right() + 80, rect.center().y()))
    (p,) = got.items
    assert fv.scan_to_pixel(fv.frame_geometry(frame), p["x1_nm"], p["y1_nm"])[0] == pytest.approx(31.5, abs=1e-6)
    click(view, view.pixel_to_widget(9, 9))
    assert len(got.items) == 1
    view.clear_profile()
    assert got.items[-1] == {}
    view.clear_profile()
    assert len(got.items) == 2


def test_the_profile_line_and_its_ends_are_drawn(app):
    view = make_view(app)
    view.set_frame(make_frame(bumpy(48)))
    before = rgb(view)
    view.set_tool("profile")
    drag(view, view.pixel_to_widget(6, 30), view.pixel_to_widget(40, 12))
    after = rgb(view)
    mid = view.pixel_to_widget(23, 21)
    assert near(after, fv.PROFILE_COLOR)[int(mid.y()) - 2:int(mid.y()) + 3, int(mid.x()) - 2:int(mid.x()) + 3].any()
    assert near(before, fv.PROFILE_COLOR).sum() == 0


# ── a wheel with Ctrl, menus and focus: nothing (the plain wheel zooms: tests/test_gui_zoom.py) ───
def test_a_wheel_with_ctrl_changes_nothing_and_neither_widget_has_a_menu_or_takes_focus(app):
    """Ctrl + wheel is taken and dropped by both widgets: the picture, the layout, the markers and the signals
    stay as they were, and no parent scrolls. (The plain wheel is the viewer's zoom, tested in test_gui_zoom.py;
    in the plot it does nothing: test_a_wheel_over_the_plot_leaves_the_cursor_where_it_is.) Neither widget
    has a context menu or takes the keyboard focus."""
    view = make_view(app)
    view.set_frame(make_frame(bumpy(32)))
    click(view, view.pixel_to_widget(7, 7))
    plot = PlotView()
    plot.resize(400, 250)
    plot.set_series([0, 1, 2], [("a", [1, 3, 2])], "x", "y")
    emitted = [Capture(s) for s in (view.cursor_moved, view.marker_added, view.markers_changed, view.profile_changed)]
    rect = view.image_rect()
    for widget in (view, plot):
        before = rgb(widget)
        for delta in (120, -120, 480):
            ev = QWheelEvent(QPointF(100, 100), QPointF(widget.mapToGlobal(QPoint(100, 100))), QPoint(0, 0), QPoint(0, delta),
                             Qt.MouseButton.NoButton, Qt.KeyboardModifier.ControlModifier, Qt.ScrollPhase.NoScrollPhase, False)
            QApplication.sendEvent(widget, ev)
            assert ev.isAccepted()                                     # swallowed: a parent does not scroll either
        assert np.array_equal(rgb(widget), before)
    assert view.image_rect() == rect and len(view.markers()) == 1
    assert [c.items for c in emitted][0:1] == [[]] and emitted[1].items == [] and emitted[3].items == []
    for widget in (view, plot):
        assert widget.contextMenuPolicy() == Qt.ContextMenuPolicy.NoContextMenu
        assert widget.focusPolicy() == Qt.FocusPolicy.NoFocus


# ── thumbnails ─────────────────────────────────────────────────────────────────────────────
def test_thumbnail_of_a_small_frame_enlarges_by_nearest_neighbour(app):
    z = bumpy(16)
    img = frame_thumbnail(make_frame(z), 96)
    assert (img.width(), img.height()) == (96, 96)
    assert np.array_equal(to_array(img)[..., 0], np.kron(display_array(z, "z"), np.ones((6, 6), np.uint8)))


def test_thumbnail_of_a_large_frame_averages_and_keeps_the_mean(app):
    z = bumpy(256)
    img = frame_thumbnail(make_frame(z), 96)
    assert (img.width(), img.height()) == (96, 96)
    assert abs(float(to_array(img)[..., 0].mean()) - float(display_array(z, "z").mean())) < 0.6
    # a 3 x 3 spot keeps its weight when it is shrunk: area averaging, not point sampling
    spot = np.zeros((256, 256))
    spot[100:103, 100:103] = 1.0
    t = to_array(frame_thumbnail(make_frame(spot, channel="current"), 96))[..., 0]
    assert t.sum() == pytest.approx(display_array(spot, "current").sum() * (96 / 256) ** 2, rel=0.05)


# ── PlotView ───────────────────────────────────────────────────────────────────────────────
def test_plotview_survives_nan_inf_empty_and_ragged_input(app):
    plot = PlotView()
    plot.resize(520, 330)
    x = np.linspace(-1, 1, 101)
    y = np.sin(3 * x)
    cases = [
        ([], []), (x, []), ([], [("a", [])]), (x, [("a", [np.nan] * 101)]), ([np.nan] * 5, [("a", [1, 2, 3, 4, 5])]),
        (x, [("a", y), ("b", y[:50])]), (x, [("a", np.r_[y[:50], np.inf, y[51:]])]), ([0.0], [("a", [1.0])]),
        ([0, 1, 2], [("flat", [4, 4, 4])]), (x, [("a", y)] * 9), (x, [("a", ["x"] * 101)]), (x, [("a", y * 1e-18)]),
        (x, [("a", y * 1e15)]), ([1e-12, 2e-12], [("a", [1e-12, 5e-12])]), (x, [("a", y[:3]), ("b", np.zeros(500))]),
    ]
    for xs, series in cases:
        plot.set_series(xs, series, "Bias (V)", "dI/dV (nS)", "a title")
        assert rgb(plot).shape == (330, 520, 3)
    plot.clear()
    assert (rgb(plot) == 255).all()                                    # nothing but white
    plot.clear("No spectrum yet")
    assert (rgb(plot) != 255).any()                                    # the message
    plot.set_series([], [], "x", "y")
    assert (rgb(plot) != 255).any()                                    # the "No data" message
    tiny = PlotView()
    tiny.resize(40, 30)
    tiny.set_series(x, [("a", y), ("b", -y)], "a long x label", "a long y label", "a title that cannot fit")
    assert rgb(tiny).shape == (30, 40, 3)


def test_plotview_draws_each_series_in_its_colour_and_leaves_gaps_for_nan(app):
    plot = PlotView()
    plot.resize(520, 330)
    x = np.linspace(0, 1, 201)
    y = x.copy()
    plot.set_series(x, [("a", y)], "x", "y")
    full = near(rgb(plot), BLUE)
    assert full.sum() > 200
    cols = np.where(full.any(axis=0))[0]
    gap = y.copy()
    gap[80:120] = np.nan                                               # x in [0.40, 0.60)
    plot.set_series(x, [("a", gap)], "x", "y")
    holed = near(rgb(plot), BLUE)
    removed = np.where((full & ~holed).any(axis=0))[0]
    span = cols.max() - cols.min()
    assert len(removed) > 0.1 * span
    assert removed.min() > cols.min() + 0.35 * span and removed.max() < cols.min() + 0.65 * span
    inf = y.copy()
    inf[80:120] = np.inf                                               # skipped exactly like NaN
    plot.set_series(x, [("a", inf)], "x", "y")
    assert np.array_equal(near(rgb(plot), BLUE), holed)
    plot.set_series(x, [("a", y), ("b", 1 - y)], "x", "y")
    assert near(rgb(plot), BLUE).sum() > 200 and near(rgb(plot), ORANGE).sum() > 200


def test_a_short_plot_gives_up_its_captions_before_its_curve(app):
    """The line profile sits in a strip beside the marker table, a few hundred by ~140 px: the x
    label goes first, then the title, so the curve keeps room to be read."""
    s = np.linspace(0, 4.4, 41)
    z = 120 + 25 * np.exp(-((s - 2.2) / 0.7) ** 2)
    plot = PlotView()
    font = QFont("Arial")
    font.setPixelSize(16)
    plot.setFont(font)
    regular, bold = plot._fonts()
    fh, bold_h, tpx = QFontMetrics(regular).height(), QFontMetrics(bold).height(), plot._tpx()
    vpad, tick_len, gap = max(3, tpx // 4), max(4, tpx // 4), 4
    with_captions = vpad + (bold_h + 4) + vpad + tick_len + gap + fh + gap + fh      # what a tall plot spends round its axes
    heights = {}
    for h in (400, 190, 140, 100):
        plot.resize(330, h)
        plot.set_series(s, [("z", z)], "distance (nm)", "pm", "A to B  4.4 nm  (sim001.sxm)")
        heights[h] = plot.plot_rect().height()
        assert plot.plot_rect().bottom() < h and plot.plot_rect().top() >= 0
        full = h - with_captions
        if full >= 4.3 * fh:
            assert heights[h] == pytest.approx(full)                                  # room enough: every caption stays
        else:
            assert heights[h] >= full + fh                                            # else at least the x label row went
    assert heights[400] == pytest.approx(400 - with_captions)
    assert heights[140] >= 140 - with_captions + fh and heights[100] >= 20
    plot.resize(330, 400)
    full = rgb(plot)
    plot.set_series(s, [("z", z)], "", "pm", "")                    # no captions at all: the same axes, taller
    assert plot.plot_rect().height() > heights[400]
    assert not (rgb(plot) == full).all()
    assert PlotView().plot_rect().isNull()


def test_plotview_text_is_at_least_14_px_whatever_the_widget_font(app):
    plot = PlotView()
    small = QFont("Arial")
    small.setPixelSize(9)
    plot.setFont(small)
    assert plot._fpx() == 14
    big = QFont("Arial")
    big.setPixelSize(22)
    plot.setFont(big)
    assert plot._fpx() == 22
    assert plot.sizeHint().height() > plot.minimumSizeHint().height() > 0


# ── PlotView: the cursor readout ───────────────────────────────────────────────────────────
NAMES = ["dI/dV (nS)", "dI/dV back (nS)", "I (pA)"]


def sts_plot(n=201, size=(600, 330), series=1, px=16):
    """A PlotView holding STS-like curves on a +-1 V sweep: what an agent reads an energy off."""
    plot = PlotView()
    font = QFont("Arial")
    font.setPixelSize(px)
    plot.setFont(font)
    plot.resize(*size)
    x = np.linspace(-1.0, 1.0, n)
    ys = [5.0 + 2.0 * np.exp(-((x + 0.43) / 0.08) ** 2) + 0.7 * k * x for k in range(series)]
    plot.set_series(x, list(zip(NAMES, ys)), "Bias (V)", "dI/dV (nS)", "spectrum")
    return plot, x, ys


def reading(i, x, ys, names=NAMES):
    """What the cursor must report at point i."""
    return {"index": i, "x": float(x[i]), "values": [[nm, float(y[i])] for nm, y in zip(names, ys)]}


def leave(widget):
    QApplication.sendEvent(widget, QEvent(QEvent.Type.Leave))


def box_of(plot: PlotView):
    """(left, top, right, bottom) of the readout box in the rendered axes, found as the pixels of
    exactly its background colour; None when none is drawn."""
    shot, r = rgb(plot), plot.plot_rect()
    bg = np.array([READOUT_BG.red(), READOUT_BG.green(), READOUT_BG.blue()])
    x0, y0 = int(r.left()), int(r.top())
    mask = (shot[y0:int(r.bottom()), x0:int(r.right())] == bg).all(axis=2)
    ys_, xs_ = np.nonzero(mask)
    return None if xs_.size == 0 else (x0 + xs_.min(), y0 + ys_.min(), x0 + xs_.max(), y0 + ys_.max())


def test_a_wheel_over_the_plot_zooms_it_and_leaves_the_pin_and_off_the_axes_changes_nothing(app):
    """The wheel zooms (see test_gui_plot_zoom.py); it never picks a point or moves a pin, and off the axes it is taken
    and does nothing at all."""
    plot, x, ys = sts_plot()
    got = Capture(plot.point_picked)
    click(plot, plot.point_position(46))

    def notch(pos, delta):
        ev = QWheelEvent(QPointF(pos), QPointF(plot.mapToGlobal(QPoint(int(pos.x()), int(pos.y())))), QPoint(0, 0),
                         QPoint(0, delta), Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                         Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(plot, ev)
        assert ev.isAccepted()

    before = rgb(plot)
    for delta in (120, -120, 480):
        notch(QPointF(2, 2), delta)                                            # off the axes
    assert np.array_equal(rgb(plot), before)
    for delta in (120, -120, 480):
        notch(plot.plot_rect().center(), delta)                                # over them
    assert plot.pinned_point() == reading(46, x, ys) and len(got.items) == 1


def test_the_readout_digits_follow_each_quantitys_range():
    assert [pv.reading_decimals(s) for s in (2.0, 0.2, 20.0, 0.0, float("nan"))] == [3, 4, 2, 3, 3]
    assert pv.format_number(-0.432161, 3, sig=4) == "-0.432"                  # a peak to a millivolt on a +-1 V sweep
    assert pv.format_number(5.9123, 3, sig=4) == "5.912"
    assert pv.format_number(1.23456e-7, 3, sig=4) == "1.235e-07"
    assert pv.format_number(1.23456e-7, 3) == "1.23e-07"                       # the axes keep their three
    plot, x, _ = sts_plot(201)
    assert plot._x_decimals == 3 and plot._y_decimals == [3]
    plot, x, _ = sts_plot(4097)                                                 # 0.5 mV apart: a tenth of a millivolt shows
    assert plot._x_decimals == 4
    plot.set_series([0.0, 100.0, 200.0], [("a", [1.0, 2.0, 3.0]), ("b", [1000.0, 2000.0, 3000.0])], "x", "y")
    assert plot._x_decimals == 1 and plot._y_decimals == [3, 0]                 # four digits of each range, curve by curve


@pytest.mark.parametrize("i", [0, 17, 100, 183, 200])
def test_hovering_a_data_point_reads_that_points_values(app, i):
    plot, x, ys = sts_plot()
    got = Capture(plot.point_picked)
    move(plot, plot.point_position(i))
    assert plot.cursor_point() == reading(i, x, ys)                            # exactly the point's own numbers
    assert plot.pinned_point() is None and got.items == []                     # a hover is not a pick
    leave(plot)
    assert plot.cursor_point() is None


@pytest.mark.parametrize("i", [17, 100, 183])
def test_the_cursor_is_drawn_at_the_point_with_a_dot_on_the_curve(app, i):
    plot, x, ys = sts_plot()
    bare = rgb(plot)
    pos = plot.point_position(i)
    move(plot, pos)
    shot = rgb(plot)
    r = plot.plot_rect()
    diff = np.abs(shot.astype(int) - bare.astype(int)).sum(axis=2)[int(r.top()):int(r.bottom())]
    column = int(diff.sum(axis=0).argmax())                                    # the cursor line: the column changed most, top to bottom
    assert column == round(pos.x()) and (diff[:, column] > 0).sum() > 0.9 * diff.shape[0]
    yy, xx = np.mgrid[0:shot.shape[0], 0:shot.shape[1]]
    disc = np.hypot(xx + 0.5 - pos.x(), yy + 0.5 - pos.y()) <= 3.5            # a dot fills it; the curve alone covers about half
    assert near(shot, BLUE)[disc].sum() >= 0.9 * disc.sum()


def test_the_pointer_between_points_picks_the_nearest_in_x_whatever_its_height(app):
    plot, x, ys = sts_plot()
    r = plot.plot_rect()
    step = r.width() / 200
    for i, off, want in ((60, 0.3, 60), (60, 0.7, 61), (60, -0.4, 60), (60, -0.6, 59)):
        move(plot, QPointF(plot.point_position(i).x() + off * step, r.bottom() - 3))      # near the bottom: y plays no part
        assert plot.cursor_point()["index"] == want


def test_the_pointer_off_the_axes_shows_no_hover(app):
    plot, x, ys = sts_plot()
    r = plot.plot_rect()
    mid = plot.point_position(100)
    for pos in (QPointF(r.left() - 3, mid.y()), QPointF(mid.x(), r.top() - 3), QPointF(mid.x(), r.bottom() + 3),
                QPointF(r.right() + 3, mid.y())):
        move(plot, mid)
        assert plot.cursor_point() is not None
        move(plot, pos)
        assert plot.cursor_point() is None


def test_a_click_pins_the_point_and_reports_it(app):
    plot, x, ys = sts_plot(series=2)
    names = NAMES[:2]
    got = Capture(plot.point_picked)
    click(plot, plot.point_position(46))
    assert got.items == [reading(46, x, ys, names)]
    (p,) = got.items
    assert set(p) == {"index", "x", "values"} and type(p["index"]) is int and type(p["x"]) is float
    assert all(type(lab) is str and type(v) is float for lab, v in p["values"])
    assert plot.pinned_point() == reading(46, x, ys, names) == plot.cursor_point()
    # the line and box stay when the pointer leaves, drawn as they were
    with_pointer = rgb(plot)
    leave(plot)
    assert plot.cursor_point() == plot.pinned_point() == reading(46, x, ys, names)
    assert np.array_equal(rgb(plot), with_pointer)
    assert len(got.items) == 1
    # a later click moves the pin
    click(plot, plot.point_position(150, 1))
    assert got.items[-1] == reading(150, x, ys, names) and plot.pinned_point() == reading(150, x, ys, names)
    assert len(got.items) == 2


def test_hovering_shows_the_point_under_the_pointer_and_the_pin_returns_when_it_leaves(app):
    plot, x, ys = sts_plot()
    got = Capture(plot.point_picked)
    click(plot, plot.point_position(46))
    move(plot, plot.point_position(120))
    assert plot.cursor_point() == reading(120, x, ys) and plot.pinned_point() == reading(46, x, ys)
    assert len(got.items) == 1                                                 # moving picks nothing
    leave(plot)
    assert plot.cursor_point() == reading(46, x, ys)
    pinned_picture = rgb(plot)
    move(plot, plot.point_position(46))                                        # over the pin itself: drawn the same
    assert np.array_equal(rgb(plot), pinned_picture)


def test_new_data_or_clear_removes_the_pin_and_the_hover_for_good(app):
    plot, x, ys = sts_plot()
    fresh = rgb(sts_plot()[0])
    for how in ("set_series", "clear"):
        click(plot, plot.point_position(46))
        move(plot, plot.point_position(120))
        assert rgb(plot).tobytes() != fresh.tobytes()
        if how == "set_series":
            plot.set_series(x, [(NAMES[0], ys[0])], "Bias (V)", "dI/dV (nS)", "spectrum")
        else:
            plot.clear("nothing")
        assert plot.cursor_point() is None and plot.pinned_point() is None
        shot = rgb(plot)                                                      # (a grab of a hidden widget resizes it: still none)
        assert plot.cursor_point() is None
        if how == "set_series":
            assert np.array_equal(shot, fresh)                                # no line, dots or box left over
        leave(plot)
        assert plot.cursor_point() is None


def test_points_with_nothing_to_read_are_skipped_when_finding_the_nearest(app):
    nan, inf = float("nan"), float("inf")
    plot = PlotView()
    plot.resize(600, 330)
    x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, nan, 7.0, inf, 9.0]
    a = [1.0, nan, 3.0, 4.0, nan, 6.0, 7.0, 8.0, 9.0, 10.0]
    b = [nan, nan, 3.0, nan, nan, 6.0, 7.0, 8.0, 9.0, nan]
    plot.set_series(x, [("a", a), ("b", b)], "x", "y")
    r = plot.plot_rect()

    def px(v: float) -> float:                                                  # the finite points span x = 0 .. 9
        return r.left() + v / 9.0 * r.width()

    got = Capture(plot.point_picked)
    # readable points sit at x = 0, 2, 3, 5, 7, 9: 1 (nothing to read), 4 (likewise), 6 (no x) and 8 (infinite x) are skipped
    for xv, want in ((0.8, 0), (1.2, 2), (3.8, 3), (4.2, 5), (5.7, 5), (6.3, 7), (7.8, 7), (8.2, 9)):
        move(plot, QPointF(px(xv), r.center().y()))
        assert plot.cursor_point()["index"] == want, xv
    assert plot.point_position(1) is None and plot.point_position(6) is None and plot.point_position(8) is None
    assert plot.point_position(3, 1) is None and plot.point_position(3, 0) is not None
    click(plot, QPointF(px(3.2), r.center().y()))                              # a point with one series missing
    (p,) = got.items
    assert p["index"] == 3 and p["x"] == 3.0 and p["values"][0] == ["a", 4.0]
    assert p["values"][1][0] == "b" and math.isnan(p["values"][1][1])
    for xv in np.linspace(0, 9, 90):                                           # sweeping the pointer never lands on a skipped point
        move(plot, QPointF(px(xv), r.center().y()))
        assert plot.cursor_point()["index"] in (0, 2, 3, 5, 7, 9)


@pytest.mark.parametrize("build", ["fresh", "cleared", "empty", "all nan x", "all nan y"])
def test_a_plot_with_no_data_has_no_cursor(app, build):
    plot = PlotView()
    plot.resize(600, 330)
    if build == "cleared":
        plot.clear("No spectrum yet")
    elif build == "empty":
        plot.set_series([], [("a", [])], "x", "y")
    elif build == "all nan x":
        plot.set_series([float("nan")] * 4, [("a", [1.0, 2.0, 3.0, 4.0])], "x", "y")
    elif build == "all nan y":
        plot.set_series([1.0, 2.0, 3.0], [("a", [float("nan")] * 3)], "x", "y")
    got = Capture(plot.point_picked)
    before = rgb(plot)
    for pos in (QPointF(300, 150), QPointF(20, 20), QPointF(590, 320)):
        move(plot, pos)
        click(plot, pos)
    assert plot.cursor_point() is None and plot.pinned_point() is None and got.items == []
    assert plot.point_position(0) is None and plot.plot_rect().isNull()
    assert np.array_equal(rgb(plot), before)


def test_only_a_left_click_on_the_axes_picks_and_a_double_click_picks_once(app):
    plot, x, ys = sts_plot()
    got = Capture(plot.point_picked)
    r = plot.plot_rect()
    pos = plot.point_position(60)
    click(plot, pos, Qt.MouseButton.RightButton)
    click(plot, pos, Qt.MouseButton.MiddleButton)
    click(plot, QPointF(r.left() - 5, pos.y()))                                # the margin
    click(plot, QPointF(pos.x(), r.bottom() + 5))
    assert got.items == [] and plot.pinned_point() is None
    L, N = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
    send(plot, QEvent.Type.MouseButtonPress, pos, L, L)
    send(plot, QEvent.Type.MouseButtonRelease, pos, L, N)
    send(plot, QEvent.Type.MouseButtonDblClick, pos, L, L)
    send(plot, QEvent.Type.MouseButtonRelease, pos, L, N)
    assert len(got.items) == 1 and plot.pinned_point()["index"] == 60


def test_the_readout_box_sits_in_a_corner_clear_of_the_curve_and_of_the_point_read(app):
    """Screen trial 23: the box was drawn over the peak being read. It now takes the corner with the fewest curve points
    (here the peak's flank is in the top-left corner, the baseline along the bottom), and never covers the point."""
    plot, x, ys = sts_plot()
    r = plot.plot_rect()
    samples = plot._curve_samples(plot._layout_now())
    for i in (180, 10, 60):
        move(plot, plot.point_position(i))
        left, top, right, bottom = box_of(plot)
        assert right >= r.right() - 12 and top <= r.top() + 12 and left > r.center().x(), i      # the top right
        inside = (samples[:, 0] >= left) & (samples[:, 0] <= right) & (samples[:, 1] >= top) & (samples[:, 1] <= bottom)
        assert not inside.any(), i                                              # no point of the curve under the box
        here = plot.point_position(i)
        assert not (left <= here.x() <= right and top <= here.y() <= bottom)         # and the point itself is in the clear


def test_a_big_plot_reads_a_table_and_a_small_one_a_single_line(app):
    lh = QFontMetrics(sts_plot()[0]._mono_font()).height()
    plot, x, ys = sts_plot(series=3)
    move(plot, plot.point_position(180))
    left, top, right, bottom = box_of(plot)
    assert bottom - top + 1 >= 4 * lh                                          # the x row and one row per curve
    swatches = [near(rgb(plot), SERIES[k])[top:bottom, left:right].sum() for k in range(3)]
    assert all(n > 20 for n in swatches)                                       # each row carries its curve's colour
    small, x, ys = sts_plot(size=(310, 140))
    small.set_series(x, [(NAMES[0], ys[0])], "distance (nm)", "pm", "A to B  4.400 nm")
    move(small, small.point_position(150))
    left, top, right, bottom = box_of(small)
    assert bottom - top + 1 <= lh + 12                                         # one line, not the two-row table
    tiny = PlotView()
    tiny.resize(120, 60)
    tiny.set_series([0, 1, 2], [("a", [1, 3, 2])], "x", "y")
    move(tiny, tiny.point_position(1))                                         # no room for a box: the line and dot remain
    assert tiny.cursor_point()["index"] == 1 and box_of(tiny) is None


def test_a_readout_too_wide_for_the_plot_gives_up_words_before_digits(app):
    """The cascade: labelled table, one labelled line, one line with x= / y= tags, the x alone, the number alone."""
    holder = QWidget()                               # a child: a top-level window cannot be made as narrow as this
    plot = PlotView(holder)
    plot.resize(1000, 200)
    plot.set_series([0.0, 0.5, 1.0], [("a very long series label indeed", [1.0, 2.5, 3.0])],
                    "bias voltage of the sample against the tip (V)", "pA")
    mfm = QFontMetrics(plot._mono_font())

    def need(text):                                  # the axes width a line of this text asks for (box padding + margins)
        return 12 + mfm.horizontalAdvance(text) + 12

    def rows_at(axes_width):
        for _ in range(8):                           # the margins depend a little on the width (tick labels): home in
            delta = axes_width - plot.plot_rect().width()
            if abs(delta) <= 1:
                break
            plot.resize(plot.width() + round(delta), 200)
        assert abs(plot.plot_rect().width() - axes_width) <= 1
        rows, _ = plot._readout_rows(plot._fields(1), mfm, plot.plot_rect(), 6)
        move(plot, plot.point_position(1))
        assert plot.cursor_point()["index"] == 1 and rgb(plot).shape == (200, plot.width(), 3)   # it draws, whatever the width
        return rows

    table = rows_at(1000)                                                                   # room: a row per field, equals aligned
    assert len(table) == 2 and table[0].index("=") == table[1].index("=") and table[0].endswith("= 0.500")
    assert rows_at(need("x=0.500  y=2.500") + 5) == ["x=0.500  y=2.500"]                    # labels too long: x= and y= tags
    assert rows_at(need("x=0.500  y=2.500") - 5) == ["x=0.500"]                             # then x alone
    assert rows_at(need("x=0.500") - 5) == ["0.500"]                                        # then just the number


def test_the_readout_text_is_fixed_pitch_and_at_least_14_px(app):
    plot, x, ys = sts_plot()
    font = plot._mono_font()
    assert font.pixelSize() >= 14
    if any(f in QFontDatabase.families() for f in pv.MONO_FAMILIES):
        assert QFontInfo(font).fixedPitch()
    small = PlotView()
    tiny_font = QFont("Arial")
    tiny_font.setPixelSize(9)
    small.setFont(tiny_font)
    assert small._mono_font().pixelSize() >= 14


def test_a_series_without_a_name_reads_with_the_y_label_and_an_unlabelled_x_as_x(app):
    plot = PlotView()
    plot.resize(600, 330)
    plot.set_series([0.0, 1.0, 2.0], [("", [1.0, 2.0, 3.0])], "", "pm")
    move(plot, plot.point_position(1))
    assert plot.cursor_point()["values"] == [["pm", 2.0]]
    assert plot._fields(1)[0][1] == "x"
    plot.set_series([0.0, 1.0, 2.0], [("", [1.0, 2.0, 3.0])], "t", "")
    assert plot._fields(1)[1][1] == "y"


# ── a real frame from the instrument ────────────────────────────────────────────────────────
def test_a_real_instrument_frame_round_trips_through_the_view(app):
    from stmbench.instrument import Instrument

    inst = Instrument("P4_atom_positioning_cu111", 0)
    name = inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=8.0, pixels=32)["frame"]
    frame = inst.frame(name, "z", "forward")
    view = make_view(app)
    view.set_frame(frame)
    assert np.array_equal(drawn_grey(view, 32), display_array(frame["array"], "z"))
    got = Capture(view.marker_added)
    click(view, view.pixel_to_widget(15.5, 15.5))                       # the middle of the frame: the corner of four pixels
    (m,) = got.items
    geom = fv.frame_geometry(frame)
    assert (m["x_nm"], m["y_nm"]) == pytest.approx(pixel_to_scan_nm(geom, m["col"], m["row"]), abs=1e-9)
    assert abs(m["x_nm"]) <= 0.5 * 8.0 / 32 + 1e-9 and abs(m["y_nm"]) <= 0.5 * 8.0 / 32 + 1e-9    # within half a pixel of it
    thumb = frame_thumbnail(frame, 64)
    assert (thumb.width(), thumb.height()) == (64, 64)


# ── numerics behind the picture ────────────────────────────────────────────────────────────
def test_plane_fit_removes_a_plane_with_holes():
    ny, nx = 30, 50
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    z = 3e-12 * xx - 5e-12 * yy + 7e-11
    z[4:9, 10:20] = np.nan
    z[:, 0] = np.nan
    plane = fv._fit_plane(z)
    assert np.nanmax(np.abs(z - plane)) < 1e-22
    assert np.isfinite(plane).all()
    assert np.array_equal(fv._fit_plane(np.full((5, 5), np.nan)), np.zeros((5, 5)))


def test_bilinear_sampling():
    a = np.arange(12, dtype=float).reshape(3, 4)                       # a[r, c] = 4 r + c
    col, row = np.array([0.0, 1.5, 3.0, 2.25, -2.0, 9.0]), np.array([0.0, 0.5, 2.0, 1.75, 0.0, 1.0])
    got = fv._bilinear(a, col, row)
    assert got == pytest.approx([0.0, 2 + 1.5, 11.0, 4 * 1.75 + 2.25, 0.0, 7.0])           # beyond the centres: the edge
    a[1, 1] = np.nan
    assert fv._bilinear(a, np.array([0.0]), np.array([1.0]))[0] == 4.0       # a NaN neighbour with no weight is ignored
    assert np.isnan(fv._bilinear(a, np.array([1.0, 1.5]), np.array([1.0, 1.0]))).all()
    a[1, 1] = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                       # ... and so is an infinite one, silently
        assert fv._bilinear(a, np.array([0.0]), np.array([1.0]))[0] == 4.0


def test_resample_axis_averages_by_area():
    a = np.arange(10, dtype=float)
    # cell k covers [2.5 k, 2.5 k + 2.5) of the source: pixels are weighted by the part inside
    got = fv._resample_axis(a, 4, 0, mean=True)
    assert got == pytest.approx([(0 + 1 + 0.5 * 2) / 2.5, (0.5 * 2 + 3 + 4) / 2.5, (5 + 6 + 0.5 * 7) / 2.5, (0.5 * 7 + 8 + 9) / 2.5])
    assert got.mean() == pytest.approx(a.mean())
    assert np.array_equal(fv._resample_axis(a, 20, 0, mean=False), np.repeat(a, 2))


# ── the widgets stay a screen-only affair ──────────────────────────────────────────────────
def test_the_widget_modules_import_no_network_module_and_read_no_hidden_state():
    root = Path(fv.__file__).parent
    banned = re.compile(r"\b(socket|http|urllib|QLocalServer|QTcpServer|QUdpSocket|QWebSocket|QNetwork\w*)\b|\.world\b|truth|verdict",
                        re.IGNORECASE)
    for path in sorted(root.glob("*.py")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            assert not banned.search(line), f"{path.name}:{n}: {line.strip()}"
