"""FrameView: a saved frame as the desktop console shows it, with nm axes, a colour bar, numbered
markers and a line profile; and :func:`frame_thumbnail` for the strip of recent frames.

The image is the one an MCP agent gets: with the default ``"line"`` flatten it is exactly
``stmbench.instrument.render.display_array(array, channel)``, greyscale, drawn by nearest neighbour
(each screen pixel takes one frame pixel; no smoothing that would invent detail). Everything read
off it (cursor value, marker value, profile) comes from the same flattened array, in the unit
the colour bar states.

Coordinates. A point at fractional pixel ``(col, row)`` has pixel centres at integers, row 0 at
the top, and maps to scan nm through :func:`stmbench.render.pixel_to_scan_nm`, the inverse of the
judge's footprint mapping; a position a fraction ``fx`` of the image width from its left edge has
``col = fx * nx - 0.5`` (the web console's rule). Markers live in scan coordinates and are drawn
on every frame whose footprint holds them, so a marker placed in one frame shows where the same
scan coordinates lie in the next.

There is no zoom, pan, wheel handling or context menu: the agent that works this widget has a
mouse and a screenshot, and every extra gesture is one it can trigger by accident.
"""
from __future__ import annotations

import math
import warnings
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from stmbench.instrument.render import display_array
from stmbench.render import pixel_to_scan_nm

from .plot_view import MIN_FONT_PX, font_px, format_number, format_ticks, nice_ticks, tick_decimals

FLATTEN_MODES = ("line", "plane", "none")
TOOLS = ("marker", "profile", "none")

#: channel -> (display unit, factor from the frame's SI unit); z is in m, current and lock-in X in A
_DISPLAY = {"z": ("pm", 1e12), "current": ("pA", 1e12), "didv": ("pA", 1e12)}
#: ``display_array`` removes line medians from z only; any other name maps an array as it stands
_AS_RECORDED = "current"
#: the percentile range ``display_array`` maps onto 0-255
_PERCENTILES = (0.5, 99.5)

# Overlay colours are fixed (not taken from the palette): they must read on every grey level
MARKER_COLOR = QColor("#00e5ff")
MARKER_SELECTED_COLOR = QColor("#ffd400")
PROFILE_COLOR = QColor("#ff4dff")
_BACKGROUND = QColor("#ffffff")
_TEXT = QColor("#111111")
_MUTED = QColor("#555b63")
_AXIS = QColor("#2b3036")
_LABEL_BG = QColor(10, 12, 24, 232)
_PLACEHOLDER_BG = QColor("#eceff3")
#: rows never acquired: a flat dark blue, not "black = low" (as in ``stmbench.render``)
_NAN_RGB = (40, 40, 60)


# ── geometry ───────────────────────────────────────────────────────────────────────────────
def frame_geometry(frame: dict) -> dict:
    """``pixel_to_scan_nm``'s geometry (``cx_nm cy_nm w_nm h_nm angle_deg nx ny``) of an
    ``Instrument.frame()`` dict. Pixels are square: the height follows from the pixel count."""
    ny, nx = np.shape(frame["array"])
    cx, cy = frame["center_nm"]
    w = float(frame["size_nm"])
    return {"cx_nm": float(cx), "cy_nm": float(cy), "w_nm": w, "h_nm": w * ny / nx,
            "angle_deg": float(frame.get("angle_deg") or 0.0), "nx": int(nx), "ny": int(ny)}


def scan_to_pixel(geom: dict, x_nm: float, y_nm: float) -> tuple[float, float]:
    """Inverse of :func:`stmbench.render.pixel_to_scan_nm`: scan nm -> fractional ``(col, row)``."""
    w, h = float(geom["w_nm"]), float(geom["h_nm"])
    a = math.radians(float(geom.get("angle_deg", 0.0) or 0.0))
    c, s = math.cos(a), math.sin(a)
    dx, dy = x_nm - float(geom["cx_nm"]), y_nm - float(geom["cy_nm"])
    u, v = c * dx + s * dy, -s * dx + c * dy
    return (u + w / 2) * int(geom["nx"]) / w - 0.5, (h / 2 - v) * int(geom["ny"]) / h - 0.5


# ── numerics ───────────────────────────────────────────────────────────────────────────────
def _fit_plane(z: np.ndarray) -> np.ndarray:
    """The least-squares plane through the finite pixels, on the whole grid (normal equations in
    centred coordinates: no design matrix of pixel-count rows for a 1024 x 1024 frame)."""
    ny, nx = z.shape
    ok = np.isfinite(z)
    if int(ok.sum()) < 3:
        return np.zeros_like(z)
    X = np.broadcast_to((np.arange(nx) - (nx - 1) / 2.0)[None, :], z.shape)
    Y = np.broadcast_to((np.arange(ny) - (ny - 1) / 2.0)[:, None], z.shape)
    w = ok.astype(np.float64)
    wx, wy, z0 = w * X, w * Y, np.where(ok, z, 0.0)
    a = np.array([[(wx * X).sum(), (wx * Y).sum(), wx.sum()],
                  [(wx * Y).sum(), (wy * Y).sum(), wy.sum()],
                  [wx.sum(), wy.sum(), w.sum()]])
    b = np.array([(z0 * X).sum(), (z0 * Y).sum(), z0.sum()])
    try:
        cx, cy, c0 = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:                    # all finite pixels on one line
        cx, cy, c0 = np.linalg.lstsq(a, b, rcond=None)[0]
    return cx * X + cy * Y + c0


def _bilinear(a: np.ndarray, col: np.ndarray, row: np.ndarray) -> np.ndarray:
    """Samples of ``a`` at fractional ``(col, row)`` (pixel centres at integers); beyond the
    outermost centres the edge value holds. A NaN neighbour only matters where it has weight."""
    ny, nx = a.shape
    c = np.clip(col, 0.0, nx - 1.0)
    r = np.clip(row, 0.0, ny - 1.0)
    c0 = np.minimum(np.floor(c).astype(np.intp), nx - 2)
    r0 = np.minimum(np.floor(r).astype(np.intp), ny - 2)
    fc, fr = c - c0, r - r0
    out = np.zeros(c.shape)
    for dr, wr in ((0, 1.0 - fr), (1, fr)):
        for dc, wc in ((0, 1.0 - fc), (1, fc)):
            w = wr * wc
            out += np.where(w > 0.0, w * a[r0 + dr, c0 + dc], 0.0)
    return out


def _resample_axis(a: np.ndarray, m: int, axis: int, mean: bool) -> np.ndarray:
    """Resize ``a`` to ``m`` samples along ``axis``: area-weighted means of the source cells that
    each output cell covers (shrinking), or the nearest source cell (enlarging)."""
    n = a.shape[axis]
    if not mean:
        idx = np.minimum(((np.arange(m) + 0.5) * n / m).astype(np.intp), n - 1)
        return np.take(a, idx, axis=axis)
    edges = np.linspace(0.0, float(n), m + 1)
    cum = np.concatenate([np.zeros_like(np.take(a, [0], axis=axis)), np.cumsum(a, axis=axis)], axis=axis)
    i = np.minimum(np.floor(edges).astype(np.intp), n - 1)
    frac = edges - i
    shape = [1] * a.ndim
    shape[axis] = -1
    integral = np.take(cum, i, axis=axis) + np.take(a, i, axis=axis) * frac.reshape(shape)
    return np.diff(integral, axis=axis) / np.diff(edges).reshape(shape)


def _display_values(array: np.ndarray, channel: str, mode: str):
    """``(flattened array in SI, 8-bit image, lo, hi)`` for a flatten mode; ``lo``/``hi`` (SI) are
    the range the image maps onto 0-255. In ``"line"`` mode the image is ``display_array``'s own."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # an all-NaN line: NaN, as in display_array
        if mode == "line":
            flat = array - np.nanmedian(array, axis=1, keepdims=True) if channel == "z" else array
            grey = display_array(array, channel)
        else:
            flat = array - _fit_plane(array) if mode == "plane" else array
            grey = display_array(flat, _AS_RECORDED)
        finite = flat[np.isfinite(flat)]
        if finite.size:
            lo, hi = (float(v) for v in np.percentile(finite, _PERCENTILES))
            hi = hi if hi > lo else lo + 1e-30
        else:
            lo = hi = 0.0
    return flat, grey, lo, hi


def _gray_image(grey: np.ndarray, nan_mask: np.ndarray | None = None) -> QImage:
    """An owned RGBX image from an 8-bit grey array (``nan_mask`` pixels in the flat NaN colour)."""
    h, w = grey.shape
    rgb = np.empty((h, w, 4), np.uint8)
    rgb[..., 0] = rgb[..., 1] = rgb[..., 2] = grey
    rgb[..., 3] = 255
    if nan_mask is not None and nan_mask.any():
        rgb[nan_mask, 0], rgb[nan_mask, 1], rgb[nan_mask, 2] = _NAN_RGB
    buf = rgb.tobytes()
    return QImage(buf, w, h, w * 4, QImage.Format.Format_RGBX8888).copy()


def frame_thumbnail(frame: dict, px: int = 96) -> QImage:
    """A ``px`` x ``px`` greyscale thumbnail of a frame dict, as an MCP agent's image would look
    (``display_array``; line-flattened z). Shrinking averages the pixels each thumbnail pixel
    covers, so atoms do not alias; no axes, markers or text are drawn."""
    px = max(8, int(px))
    array = np.array(frame["array"], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        grey = display_array(array, str(frame.get("channel", "z")))
    ny, nx = grey.shape
    g = grey.astype(np.float64)
    g = _resample_axis(g, px, 0, mean=ny > px)
    g = _resample_axis(g, px, 1, mean=nx > px)
    return _gray_image(np.clip(np.round(g), 0, 255).astype(np.uint8))


def marker_radius(fpx: int) -> float:
    """Radius in px of a marker's ring for a base font of ``fpx`` px."""
    return max(8.0, 0.6 * fpx)


class _Shown:
    """The frame on display and everything derived from it for the current flatten mode."""

    def __init__(self, frame: dict, mode: str) -> None:
        array = np.array(frame["array"], dtype=np.float64)
        if array.ndim != 2 or min(array.shape) < 2:
            raise ValueError("a frame needs a 2-D array of at least 2 x 2 pixels")
        size = float(frame["size_nm"])
        if not size > 0.0:
            raise ValueError("a frame needs a positive size_nm")
        self.name = str(frame.get("frame", ""))
        self.channel = str(frame.get("channel", "z"))
        self.geom = frame_geometry({**frame, "array": array})
        self.ny, self.nx = array.shape
        self.nm_per_px = self.geom["w_nm"] / self.nx
        self.unit, self.scale = _DISPLAY.get(self.channel, (str(frame.get("unit", "")), 1.0))
        self.array = array
        self.set_mode(mode)

    def set_mode(self, mode: str) -> None:
        flat, self.grey, lo, hi = _display_values(self.array, self.channel, mode)
        self.values = flat * self.scale                           # what every readout shows
        self.lo, self.hi = lo * self.scale, hi * self.scale        # the colour bar's range
        nan = ~np.isfinite(flat)
        self.nan_mask = nan if nan.any() else None


class FrameView(QWidget):
    """The image display of the console. See the module docstring for the conventions.

    Signals: ``cursor_moved`` (``x_nm y_nm col row value unit``; ``{}`` once the pointer leaves the
    image), ``marker_added`` (those plus ``frame``, ``col_f``/``row_f`` the fractional pixel),
    ``markers_changed`` (every marker, in order), ``profile_changed`` (``frame x0_nm y0_nm x1_nm
    y1_nm s_nm values unit``; ``{}`` when the profile line is dropped). ``col``/``row`` are the
    integer pixel the point falls in; ``value`` is NaN where the pixel holds no data.
    """

    cursor_moved = Signal(dict)
    marker_added = Signal(dict)
    markers_changed = Signal(list)
    profile_changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._shown: _Shown | None = None
        self._flatten = "line"
        self._tool = "marker"
        self._markers: list[dict] = []
        self._selected = -1
        self._profile: dict | None = None            # its ends in scan nm, and the frame it was drawn on
        self._drag: tuple[QPointF, QPointF] | None = None
        self._hover = False
        self._placeholder = "No frame"
        self._layout_cache: SimpleNamespace | None = None
        self._image_cache: tuple | None = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.CrossCursor)

    # ── what is shown ──
    def set_frame(self, frame: dict | None) -> None:
        """Show an ``Instrument.frame()`` dict (None: the empty view). Markers stay; a profile line
        stays only for the frame it was drawn on, and is measured again on this data."""
        self._shown = None if frame is None else _Shown(frame, self._flatten)
        self._invalidate()
        self._reconcile_profile()
        self.update()

    def set_flatten(self, mode: str) -> None:
        if mode not in FLATTEN_MODES:
            raise ValueError(f"flatten mode {mode!r}: one of {FLATTEN_MODES}")
        if mode == self._flatten:
            return
        self._flatten = mode
        if self._shown is not None:
            self._shown.set_mode(mode)
        self._invalidate()
        self._reconcile_profile()
        self.update()

    def set_tool(self, tool: str) -> None:
        if tool not in TOOLS:
            raise ValueError(f"tool {tool!r}: one of {TOOLS}")
        self._tool = tool
        self._drag = None
        self.setCursor(Qt.CursorShape.ArrowCursor if tool == "none" else Qt.CursorShape.CrossCursor)
        self.update()

    def set_placeholder(self, text: str) -> None:
        """The text of the empty view (the widget has no language of its own)."""
        self._placeholder = str(text)
        self.update()

    # ── markers ──
    def markers(self) -> list[dict]:
        return [dict(m) for m in self._markers]

    def clear_markers(self) -> None:
        if not self._markers:
            return
        self._markers.clear()
        self._selected = -1
        self.update()
        self.markers_changed.emit([])

    def remove_marker(self, index: int) -> None:
        """Drop one marker; the later ones are renumbered. The selection moves to the last one."""
        if not 0 <= index < len(self._markers):
            return
        del self._markers[index]
        self._selected = len(self._markers) - 1
        self.update()
        self.markers_changed.emit(self.markers())

    def select_marker(self, index: int | None) -> None:
        """Highlight marker ``index`` (0-based); None or an index outside the list clears the selection."""
        self._selected = index if index is not None and 0 <= index < len(self._markers) else -1
        self.update()

    def selected_marker(self) -> dict | None:
        return dict(self._markers[self._selected]) if 0 <= self._selected < len(self._markers) else None

    # ── profile ──
    def clear_profile(self) -> None:
        if self._profile is not None:
            self._profile = None
            self.update()
            self.profile_changed.emit({})

    def _reconcile_profile(self) -> None:
        """After new data: re-measure the line on it, or drop it when it was drawn on another frame."""
        if self._profile is None:
            return
        if self._shown is not None and self._shown.name == self._profile["frame"]:
            self.profile_changed.emit(self._profile_payload())
        else:
            self._profile = None
            self.profile_changed.emit({})

    def _profile_payload(self) -> dict:
        sh, pr = self._shown, self._profile
        c0, r0 = scan_to_pixel(sh.geom, pr["x0_nm"], pr["y0_nm"])
        c1, r1 = scan_to_pixel(sh.geom, pr["x1_nm"], pr["y1_nm"])
        length = math.hypot(pr["x1_nm"] - pr["x0_nm"], pr["y1_nm"] - pr["y0_nm"])
        n = max(2, int(round(length / sh.nm_per_px)) + 1)          # about one sample per image pixel
        t = np.linspace(0.0, 1.0, n)
        values = _bilinear(sh.values, c0 + t * (c1 - c0), r0 + t * (r1 - r0))
        return {"frame": sh.name, "x0_nm": pr["x0_nm"], "y0_nm": pr["y0_nm"], "x1_nm": pr["x1_nm"],
                "y1_nm": pr["y1_nm"], "s_nm": (t * length).tolist(), "values": values.tolist(), "unit": sh.unit}

    # ── layout and coordinates ──
    def _fpx(self) -> int:
        return max(MIN_FONT_PX, font_px(self.font()))

    def _fonts(self, fpx: int) -> tuple[QFont, QFont]:
        regular = QFont(self.font())
        regular.setPixelSize(fpx)
        regular.setBold(False)
        bold = QFont(regular)
        bold.setBold(True)
        return regular, bold

    def _invalidate(self) -> None:
        self._layout_cache = None
        self._image_cache = None

    def resizeEvent(self, ev) -> None:
        self._invalidate()
        super().resizeEvent(ev)

    def changeEvent(self, ev) -> None:
        super().changeEvent(ev)
        if ev.type() == QEvent.Type.FontChange:
            self._invalidate()
            self.update()

    def sizeHint(self) -> QSize:
        k = self._fpx() / 16.0
        return QSize(int(830 * k), int(730 * k))

    def minimumSizeHint(self) -> QSize:
        k = self._fpx() / 16.0
        return QSize(int(430 * k), int(390 * k))

    def _axis_spec(self) -> tuple[tuple[float, float], tuple[float, float], str, str]:
        """Axis ranges and titles: scan x/y of an unrotated frame; u/v, the frame's own axes
        measured from its centre, of a rotated one (the scan axes are no longer its edges)."""
        g = self._shown.geom
        w, h, ang = g["w_nm"], g["h_nm"], g["angle_deg"]
        if abs(ang) < 1e-9:
            return ((g["cx_nm"] - w / 2, g["cx_nm"] + w / 2), (g["cy_nm"] - h / 2, g["cy_nm"] + h / 2),
                    "x (nm)", "y (nm)")
        return (-w / 2, w / 2), (-h / 2, h / 2), f"u (nm), frame rotated {ang:g}°", "v (nm)"

    def _bar_ticks(self, fm: QFontMetrics, bar_h: int) -> tuple[list[tuple[float, str]], int]:
        """Colour-bar labels as ``(fraction from the top, text)`` and the decimals they use: the two
        ends of the mapped range, and nice values between them where there is room."""
        sh = self._shown
        span = sh.hi - sh.lo
        d = 2 if span <= 0 else min(6, max(0, 2 - math.floor(math.log10(span))))
        ticks, step = nice_ticks(sh.lo, sh.hi, max(2, int(bar_h / (fm.height() * 2.4))))
        d = max(d, tick_decimals(step)) if ticks else d
        out = [(0.0, format_number(sh.hi, d)), (1.0, format_number(sh.lo, d))]
        room = 1.15 * fm.height() / max(1, bar_h)
        for t in ticks:
            f = 1.0 - (t - sh.lo) / span if span > 0 else 0.5
            if room < f < 1.0 - room:
                out.append((f, format_number(t, d)))
        return out, d

    def _layout(self) -> SimpleNamespace:
        if self._layout_cache is None:
            self._layout_cache = self._compute_layout()
        return self._layout_cache

    def _compute_layout(self) -> SimpleNamespace:
        W, H = max(1, self.width()), max(1, self.height())
        fpx = self._fpx()
        tpx = fpx + 2          # ticks, titles and labels are read off a screenshot that may be shrunk: above the base text
        regular, bold = self._fonts(tpx)
        fm = QFontMetrics(regular)
        fh = fm.height()
        pad = max(6, fpx // 2)
        tick_len = max(5, fpx // 3)
        gap = 4
        top = pad + fh // 2 + 2                              # half of the top y label may stick out
        bottom = pad + tick_len + gap + fh + gap + fh        # tick labels and the x title
        sh = self._shown
        lay = SimpleNamespace(fpx=fpx, fh=fh, pad=pad, tick_len=tick_len, gap=gap, regular=regular, bold=bold,
                              x_ticks=[], y_ticks=[], x_title="", y_title="", bar=None, bar_ticks=[], bar_w=0)
        if sh is None:
            side = max(32, min(W, H) - 2 * pad)
            lay.img = QRect((W - side) // 2, (H - side) // 2, side, side)
            return lay
        asp = sh.geom["w_nm"] / sh.geom["h_nm"]
        (x0, x1), (y0, y1), lay.x_title, lay.y_title = self._axis_spec()
        avail_h = max(32, H - top - bottom)
        yt, ystep = nice_ticks(y0, y1, max(3, int(avail_h / (tpx * 3.2))))
        yw = max((fm.horizontalAdvance(s) for s in format_ticks(yt, ystep)), default=0)
        left = pad + fh + gap + yw + gap + tick_len
        lay.bar_w = max(14, int(tpx))
        bar_labels, _ = self._bar_ticks(fm, int(avail_h * 0.9))
        right = 12 + lay.bar_w + 6 + max(fm.horizontalAdvance(s) for _, s in bar_labels) + pad
        avail_w = max(32, W - left - right)
        w_px = max(32, int(min(avail_w, avail_h * asp)))
        h_px = max(32, int(round(w_px / asp)))
        x = (W - (left + w_px + right)) // 2 + left
        y = (H - (top + h_px + bottom)) // 2 + top
        lay.img = QRect(x, y, w_px, h_px)
        xt, xstep = nice_ticks(x0, x1, max(3, int(w_px / (tpx * 4.0))))
        yt, ystep = nice_ticks(y0, y1, max(3, int(h_px / (tpx * 3.2))))
        lay.x_ticks = [((t - x0) / (x1 - x0) * w_px, s) for t, s in zip(xt, format_ticks(xt, xstep))]
        lay.y_ticks = [((y1 - t) / (y1 - y0) * h_px, s) for t, s in zip(yt, format_ticks(yt, ystep))]
        bar_top = lay.img.top() + int(fh * 1.8)               # the unit sits in the row above the bar
        lay.bar = QRect(lay.img.right() + 1 + 12, bar_top, lay.bar_w, lay.img.bottom() + 1 - bar_top)
        lay.bar_ticks, _ = self._bar_ticks(fm, lay.bar.height())
        return lay

    def image_rect(self) -> QRect:
        """Where the image is drawn, in widget coordinates (the empty view's box without a frame)."""
        return QRect(self._layout().img)

    def pixel_to_widget(self, col: float, row: float) -> QPointF:
        """Widget position of fractional pixel ``(col, row)`` (pixel centres at integers)."""
        r = self._layout().img
        sh = self._shown
        nx, ny = (sh.nx, sh.ny) if sh is not None else (1, 1)
        return QPointF(r.x() + (col + 0.5) * r.width() / nx, r.y() + (row + 0.5) * r.height() / ny)

    def scan_to_widget(self, x_nm: float, y_nm: float) -> QPointF | None:
        """Widget position of a scan-frame point; None when this frame does not cover it."""
        sh = self._shown
        if sh is None:
            return None
        col, row = scan_to_pixel(sh.geom, x_nm, y_nm)
        eps = 1e-9
        if not (-0.5 - eps <= col <= sh.nx - 0.5 + eps and -0.5 - eps <= row <= sh.ny - 0.5 + eps):
            return None
        return self.pixel_to_widget(col, row)

    def _point_at(self, pos: QPointF, clamp: bool = False) -> dict | None:
        """What lies under a widget position; None outside the image (``clamp``: the nearest edge point)."""
        sh = self._shown
        if sh is None:
            return None
        r = self._layout().img
        fx, fy = (pos.x() - r.x()) / r.width(), (pos.y() - r.y()) / r.height()
        if clamp:
            fx, fy = min(max(fx, 0.0), 1.0), min(max(fy, 0.0), 1.0)
        elif not (0.0 <= fx < 1.0 and 0.0 <= fy < 1.0):
            return None
        col, row = fx * sh.nx - 0.5, fy * sh.ny - 0.5
        x, y = pixel_to_scan_nm(sh.geom, col, row)
        ci = min(max(int(math.floor(col + 0.5)), 0), sh.nx - 1)
        ri = min(max(int(math.floor(row + 0.5)), 0), sh.ny - 1)
        return {"x_nm": x, "y_nm": y, "col": ci, "row": ri, "value": float(sh.values[ri, ci]), "unit": sh.unit,
                "col_f": col, "row_f": row}

    # ── mouse ──
    def mouseMoveEvent(self, ev) -> None:
        pos = ev.position()
        if self._drag is not None:
            self._drag = (self._drag[0], self._clamped(pos))
            self.update()
        p = self._point_at(pos)
        if p is not None:
            self._hover = True
            self.cursor_moved.emit({k: p[k] for k in ("x_nm", "y_nm", "col", "row", "value", "unit")})
        elif self._hover:
            self._hover = False
            self.cursor_moved.emit({})

    def leaveEvent(self, ev) -> None:
        if self._hover:
            self._hover = False
            self.cursor_moved.emit({})
        super().leaveEvent(ev)

    def _clamped(self, pos: QPointF) -> QPointF:
        r = self._layout().img
        return QPointF(min(max(pos.x(), r.x()), r.x() + r.width()), min(max(pos.y(), r.y()), r.y() + r.height()))

    def mousePressEvent(self, ev) -> None:
        if ev.button() != Qt.MouseButton.LeftButton or self._shown is None or self._tool == "none":
            ev.accept()
            return
        p = self._point_at(ev.position())
        if p is None:
            ev.accept()
            return
        if self._tool == "marker":
            m = {**p, "frame": self._shown.name}
            self._markers.append(m)
            self._selected = len(self._markers) - 1
            self.update()
            self.markers_changed.emit(self.markers())      # the list first, then the event: a slot of the
            self.marker_added.emit(dict(m))                # latter finds the former already refreshed
        else:
            self._drag = (QPointF(ev.position()), QPointF(ev.position()))
            self.update()
        ev.accept()

    def mouseDoubleClickEvent(self, ev) -> None:
        ev.accept()                                  # a double click is one click: the press before it did the work

    def mouseReleaseEvent(self, ev) -> None:
        if ev.button() != Qt.MouseButton.LeftButton or self._drag is None:
            ev.accept()
            return
        a, b = self._drag[0], self._clamped(ev.position())
        self._drag = None
        if math.hypot(b.x() - a.x(), b.y() - a.y()) >= 3.0:         # shorter is a click, not a line
            pa, pb = self._point_at(a, clamp=True), self._point_at(b, clamp=True)
            self._profile = {"frame": self._shown.name, "x0_nm": pa["x_nm"], "y0_nm": pa["y_nm"],
                             "x1_nm": pb["x_nm"], "y1_nm": pb["y_nm"]}
            self.profile_changed.emit(self._profile_payload())
        self.update()
        ev.accept()

    def wheelEvent(self, ev) -> None:                # accepted and dropped: no zoom, no pan, no scrolling of a parent
        ev.accept()

    def contextMenuEvent(self, ev) -> None:
        ev.accept()

    # ── painting ──
    def _scaled_image(self, w: int, h: int) -> QImage:
        """The grey image enlarged to ``w`` x ``h`` device pixels, nearest neighbour: widget pixel
        ``i`` shows source pixel ``floor((i + 0.5) * n / size)``, whatever the platform's scaler."""
        sh = self._shown
        dpr = self.devicePixelRatioF()
        pw, ph = max(1, round(w * dpr)), max(1, round(h * dpr))
        key = (id(sh), pw, ph, self._flatten)
        if self._image_cache is not None and self._image_cache[0] == key:
            return self._image_cache[1]
        ri = np.minimum(((np.arange(ph) + 0.5) * sh.ny / ph).astype(np.intp), sh.ny - 1)
        ci = np.minimum(((np.arange(pw) + 0.5) * sh.nx / pw).astype(np.intp), sh.nx - 1)
        nan = None if sh.nan_mask is None else sh.nan_mask[ri[:, None], ci[None, :]]
        img = _gray_image(sh.grey[ri[:, None], ci[None, :]], nan)
        img.setDevicePixelRatio(dpr)
        self._image_cache = (key, img)
        return img

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), _BACKGROUND)
        lay = self._layout()
        img = lay.img
        outline = QRect(img.x() - 1, img.y() - 1, img.width() + 1, img.height() + 1)
        if self._shown is None:
            p.fillRect(img, _PLACEHOLDER_BG)
            p.setPen(QPen(_AXIS, 1))
            p.drawRect(outline)
            p.setFont(lay.regular)
            p.setPen(_MUTED)
            p.drawText(QRectF(img).adjusted(12, 12, -12, -12),
                       int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap), self._placeholder)
            p.end()
            return
        p.drawImage(img.topLeft(), self._scaled_image(img.width(), img.height()))
        p.setPen(QPen(_AXIS, 1))
        p.drawRect(outline)
        self._paint_axes(p, lay)
        self._paint_bar(p, lay)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setClipRect(img)
        self._paint_profile(p, lay)
        for i, m in enumerate(self._markers):
            pt = self.scan_to_widget(m["x_nm"], m["y_nm"])
            if pt is not None:
                self._paint_marker(p, lay, pt, i + 1, i == self._selected)
        p.end()

    def _paint_axes(self, p: QPainter, lay: SimpleNamespace) -> None:
        img, fh = lay.img, lay.fh
        p.setFont(lay.regular)
        for x, s in lay.x_ticks:
            gx = img.x() + x
            tx = math.floor(gx + 0.5) if gx < img.x() + img.width() else img.right()
            p.setPen(QPen(_AXIS, 1))
            p.drawLine(QPointF(tx + 0.5, img.bottom() + 1), QPointF(tx + 0.5, img.bottom() + 1 + lay.tick_len))
            p.setPen(_TEXT)
            p.drawText(QRectF(gx - 120, img.bottom() + 1 + lay.tick_len + lay.gap, 240, fh),
                       int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop), s)
        yw = 0
        for y, s in lay.y_ticks:
            gy = img.y() + y
            ty = math.floor(gy + 0.5) if gy < img.y() + img.height() else img.bottom()
            p.setPen(QPen(_AXIS, 1))
            p.drawLine(QPointF(img.x() - 1 - lay.tick_len, ty + 0.5), QPointF(img.x() - 1, ty + 0.5))
            p.setPen(_TEXT)
            p.drawText(QRectF(img.x() - 1 - lay.tick_len - lay.gap - 200, gy - fh / 2, 200, fh),
                       int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), s)
            yw = max(yw, QFontMetrics(lay.regular).horizontalAdvance(s))
        p.setFont(lay.bold)
        p.setPen(_TEXT)
        fm = QFontMetrics(lay.bold)
        p.drawText(QRectF(img.x(), img.bottom() + 1 + lay.tick_len + lay.gap + fh + lay.gap, img.width(), fh),
                   int(Qt.AlignmentFlag.AlignCenter),
                   fm.elidedText(lay.x_title, Qt.TextElideMode.ElideMiddle, img.width()))
        p.save()
        p.translate(img.x() - 1 - lay.tick_len - lay.gap - yw - lay.gap - fh / 2, img.center().y())
        p.rotate(-90)
        p.drawText(QRectF(-img.height() / 2, -fh / 2, img.height(), fh), int(Qt.AlignmentFlag.AlignCenter),
                   fm.elidedText(lay.y_title, Qt.TextElideMode.ElideMiddle, img.height()))
        p.restore()

    def _paint_bar(self, p: QPainter, lay: SimpleNamespace) -> None:
        bar, fh = lay.bar, lay.fh
        for i in range(bar.height()):                       # 255 at the top (the range's high end), 0 at the bottom
            g = int(round(255.0 * (1.0 - (i + 0.5) / bar.height())))
            p.fillRect(bar.x(), bar.y() + i, bar.width(), 1, QColor(g, g, g))
        p.setPen(QPen(_AXIS, 1))
        p.drawRect(QRect(bar.x() - 1, bar.y() - 1, bar.width() + 1, bar.height() + 1))
        p.setFont(lay.regular)
        p.setPen(_TEXT)
        for f, s in lay.bar_ticks:
            y = bar.y() + f * bar.height()
            p.setPen(QPen(_AXIS, 1))
            p.drawLine(QPointF(bar.right() + 1, round(y) + 0.5), QPointF(bar.right() + 5, round(y) + 0.5))
            p.setPen(_TEXT)
            p.drawText(QRectF(bar.right() + 8, y - fh / 2, 200, fh),
                       int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), s)
        p.setFont(lay.bold)
        p.drawText(QRectF(bar.x(), lay.img.y(), 200, fh), int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                   self._shown.unit)

    def _label(self, p: QPainter, lay: SimpleNamespace, text: str, rect: QRectF, color: QColor) -> None:
        """A dark label with a bold white number or letter, kept inside the widget."""
        rect = QRectF(rect)
        if rect.right() > self.width() - 2:
            rect.moveRight(self.width() - 2)
        if rect.left() < 2:
            rect.moveLeft(2)
        if rect.top() < 2:
            rect.moveTop(2)
        if rect.bottom() > self.height() - 2:
            rect.moveBottom(self.height() - 2)
        p.save()
        p.setClipping(False)
        p.setPen(QPen(color, 1.5))
        p.setBrush(_LABEL_BG)
        p.drawRoundedRect(rect, 3, 3)
        p.setFont(lay.bold)
        p.setPen(QColor("#ffffff"))
        p.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), text)
        p.restore()

    def _label_rect(self, lay: SimpleNamespace, text: str, x: float, y: float) -> QRectF:
        w = QFontMetrics(lay.bold).horizontalAdvance(text) + 10
        return QRectF(x, y, max(w, lay.fh), lay.fh + 2)

    def _paint_marker(self, p: QPainter, lay: SimpleNamespace, pt: QPointF, number: int, selected: bool) -> None:
        r = marker_radius(lay.fpx)
        color = MARKER_SELECTED_COLOR if selected else MARKER_COLOR
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(0, 0, 0, 235), 5.0))             # a dark rim under the colour: reads on white atoms too
        p.drawEllipse(pt, r, r)
        p.setPen(QPen(color, 2.5 if selected else 2.0))
        p.drawEllipse(pt, r, r)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):      # a small cross: the exact point
            a, b = QPointF(pt.x() + dx * 2, pt.y() + dy * 2), QPointF(pt.x() + dx * 5, pt.y() + dy * 5)
            p.setPen(QPen(QColor(0, 0, 0, 235), 3.5))
            p.drawLine(a, b)
            p.setPen(QPen(color, 1.5))
            p.drawLine(a, b)
        text = str(number)
        rect = self._label_rect(lay, text, pt.x() + r + 3, pt.y() - r - 3 - (lay.fh + 2))
        if rect.right() > lay.img.right() - 2:
            rect.moveLeft(pt.x() - r - 3 - rect.width())
        if rect.top() < lay.img.top() + 2:
            rect.moveTop(pt.y() + r + 3)
        self._label(p, lay, text, rect, color)

    def _paint_profile(self, p: QPainter, lay: SimpleNamespace) -> None:
        ends = None
        if self._drag is not None:
            ends = self._drag
        elif self._profile is not None and self._shown is not None and self._shown.name == self._profile["frame"]:
            sh, pr = self._shown, self._profile
            ends = tuple(self.pixel_to_widget(*scan_to_pixel(sh.geom, pr[kx], pr[ky]))
                         for kx, ky in (("x0_nm", "y0_nm"), ("x1_nm", "y1_nm")))
        if ends is None:
            return
        a, b = ends
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(0, 0, 0, 235), 5.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(a, b)
        p.setPen(QPen(PROFILE_COLOR, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(a, b)
        d = math.hypot(b.x() - a.x(), b.y() - a.y())
        if d < 1.0:
            return
        ux, uy = (b.x() - a.x()) / d, (b.y() - a.y()) / d
        for text, pt, sign in (("A", a, -1.0), ("B", b, 1.0)):    # each letter sits beyond its own end
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(PROFILE_COLOR)
            p.drawEllipse(pt, 3.5, 3.5)
            rect = self._label_rect(lay, text, 0, 0)
            reach = 10.0 + 0.5 * math.hypot(rect.width(), rect.height())
            rect.moveCenter(QPointF(pt.x() + sign * ux * reach, pt.y() + sign * uy * reach))
            self._label(p, lay, text, rect, PROFILE_COLOR)
