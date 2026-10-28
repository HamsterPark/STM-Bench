"""FrameView: a saved frame as the desktop console shows it, with nm axes, a colour bar, numbered
markers and a line profile; and :func:`frame_thumbnail` for the strip of recent frames.

The image is the one an MCP agent gets: with the default ``"line"`` flatten it is exactly
``stmbench.instrument.render.display_array(array, channel)``, greyscale, drawn by nearest neighbour
(each screen pixel takes one frame pixel; no smoothing that would invent detail). Everything read
off it (cursor value, marker value, profile) comes from the same flattened array, in the unit
the colour bar states. ``set_contrast("full")`` spreads the grey scale over the frame's whole range
instead of its 0.5-99.5 percentiles (which saturate the top of an atom); the readings do not change.

Coordinates. A point at fractional pixel ``(col, row)`` has pixel centres at integers, row 0 at
the top, and maps to scan nm through :func:`stmbench.render.pixel_to_scan_nm`, the inverse of the
judge's footprint mapping; a position a fraction ``fx`` of the image width from its left edge has
``col = fx * nx - 0.5`` (the web console's rule).

What a reading is. The image is drawn pixel by pixel, so a position over it is read as the pixel it falls
in: the cursor and a marker report that pixel's ``col`` and ``row``, its value, and the scan coordinates of
its *centre* (``pixel_to_scan_nm`` at integer ``col``, ``row``), never the pointer's own position, which no
value belongs to. A marker placed anywhere in a pixel is therefore the same marker, and its (x, y, value) is
a point the instrument measured. Markers live in scan coordinates and are drawn on every frame whose
footprint holds them, so a marker placed in one frame shows where the same scan coordinates lie in the next;
its record (value, pixel, frame) stays that of the frame it was placed in. A profile line's ends are the
exception: they keep the exact positions the drag gave.

Zoom. The wheel (without Ctrl) steps the view through x1, x2, x4 and x8 about the pointer, ``set_zoom`` and
``fit`` do the same from a control (about the centre of the view, or centred on a point: the selected marker),
and with the ``"pan"`` tool a drag moves the zoomed image. The axes, the cursor, the markers and the profile use
the same pixel / scan mapping at every zoom (the view is a window of the frame in pixel edges, ``view_rect``), and
the grey scale and the colour bar do not change with it unless the contrast is ``"visible"``. A new frame is shown
whole, except when it is another scan of the same window (the same size, pixels and angle: the next frame after a
drag, a repeat for drift): then the zoom stays, and the view keeps its centre in scan coordinates, where the
reader was looking, as far as the new frame reaches. Nothing is found, fitted or snapped to a peak.

There is no context menu and no keyboard focus, and a wheel with Ctrl does nothing: the agent that works this
widget has a mouse and a screenshot, and every extra gesture is one it can trigger by accident.
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
#: how the grey scale is set: ``auto`` maps the 0.5-99.5 percentile range (the image an MCP agent gets), ``full``
#: the frame's own minimum to maximum, so that the top of an atom is not saturated white, ``visible`` the minimum to
#: maximum of the part of the frame in view (a step far outside a zoomed view no longer flattens what is in it)
CONTRAST_MODES = ("auto", "full", "visible")
TOOLS = ("marker", "profile", "pan", "none")
#: the zoom levels, steps of the wheel: the whole frame, then a half, a quarter, an eighth of its width
ZOOM_LEVELS = (1, 2, 4, 8)

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


def _same_window(a: "_Shown", b: "_Shown") -> bool:
    """Are two frames scans of the same window: the same size, pixels and angle (their centres may differ)?"""
    ga, gb = a.geom, b.geom
    return ((a.nx, a.ny) == (b.nx, b.ny) and math.isclose(ga["w_nm"], gb["w_nm"], rel_tol=1e-9, abs_tol=1e-12)
            and math.isclose(ga["angle_deg"], gb["angle_deg"], rel_tol=0.0, abs_tol=1e-9))


def _same_place(a: "_Shown", b: "_Shown") -> bool:
    """Do two frames of the same window have the same centre?"""
    return (math.isclose(a.geom["cx_nm"], b.geom["cx_nm"], rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(a.geom["cy_nm"], b.geom["cy_nm"], rel_tol=0.0, abs_tol=1e-9))


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
    with np.errstate(invalid="ignore"):                       # 0 * inf: a hot pixel with no weight
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


def _display_values(array: np.ndarray, channel: str, mode: str, contrast: str = "auto",
                    region: tuple[int, int, int, int] | None = None):
    """``(flattened array in SI, 8-bit image, lo, hi)`` for a flatten mode; ``lo``/``hi`` (SI) are
    the range the image maps onto 0-255. In ``"line"`` mode with ``"auto"`` contrast the image is
    ``display_array``'s own; ``"full"`` maps the flattened frame's minimum and maximum instead, and
    ``"visible"`` those of ``region`` (``row0, row1, col0, col1``: the part in view; None: the frame)."""
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
            if contrast in ("full", "visible"):
                span = finite
                if contrast == "visible" and region is not None:
                    r0, r1, c0, c1 = region
                    part = flat[r0:r1, c0:c1]
                    span = part[np.isfinite(part)] if np.isfinite(part).any() else finite
                lo, hi = float(span.min()), float(span.max())
            else:
                lo, hi = (float(v) for v in np.percentile(finite, _PERCENTILES))
            hi = hi if hi > lo else lo + 1e-30
            if contrast in ("full", "visible"):                   # the same mapping as display_array's, over that range
                y = np.clip((np.nan_to_num(flat, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
                grey = np.round(y * 255.0).astype(np.uint8)
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
    """Radius in px of a marker's ring for a base font of ``fpx`` px. Small, and open: the markers of one atom
    stand a few pixels apart (five of them, at 4.7 screen px a pixel, 14 px from one another), and a ring wider
    than that, or one with a cross inside it, hid the pixel it marks and its neighbours (screen trial 4, step 85)."""
    return max(6.0, 0.42 * fpx)


class _Shown:
    """The frame on display and everything derived from it for the current flatten mode."""

    def __init__(self, frame: dict, mode: str, contrast: str = "auto") -> None:
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
        self.set_mode(mode, contrast)

    def set_mode(self, mode: str, contrast: str = "auto", region=None) -> None:
        flat, self.grey, lo, hi = _display_values(self.array, self.channel, mode, contrast, region)
        self.values = flat * self.scale                           # what every readout shows
        self.lo, self.hi = lo * self.scale, hi * self.scale        # the colour bar's range
        nan = ~np.isfinite(flat)
        self.nan_mask = nan if nan.any() else None


class FrameView(QWidget):
    """The image display of the console. See the module docstring for the conventions.

    Signals: ``cursor_moved`` (``x_nm y_nm col row value unit``; ``{}`` once the pointer leaves the
    image), ``marker_added`` (those plus ``frame``), ``markers_changed`` (every marker, in order),
    ``profile_changed`` (``frame x0_nm y0_nm x1_nm y1_nm s_nm values unit``; ``{}`` when the profile
    line is dropped). ``col``/``row`` are the pixel the position falls in, ``x_nm``/``y_nm`` the scan
    position of that pixel's centre and ``value`` its value (NaN where the pixel holds no data).
    """

    cursor_moved = Signal(dict)
    marker_added = Signal(dict)
    markers_changed = Signal(list)
    profile_changed = Signal(dict)
    zoom_changed = Signal(int)                       # the zoom level, when it changes (or a control asked for another)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._shown: _Shown | None = None
        self._flatten = "line"
        self._contrast = "auto"
        self._tool = "marker"
        self._zoom = 1
        self._c0 = self._r0 = 0.0                    # the view's top-left corner, in pixel edges of the frame
        self._pan_last: QPointF | None = None        # where the pointer was at the last move of a pan drag
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
        stays only for the frame it was drawn on, and is measured again on this data. The view is the
        whole frame again, unless this is the same frame in another channel or direction, or another scan of
        the same window (:func:`_same_window`), which keeps the zoom and the place the reader was looking at."""
        new = None if frame is None else _Shown(frame, self._flatten, self._contrast)
        old = self._shown
        same = (new is not None and old is not None and new.name == old.name and (new.nx, new.ny) == (old.nx, old.ny))
        origin = (self._c0, self._r0) if same else None
        if origin is None and new is not None and old is not None and self._zoom > 1 and _same_window(old, new):
            if _same_place(old, new):                      # a repeat of the very scan: the view is exactly where it was
                origin = (self._c0, self._r0)
            else:
                origin = self._origin_for_centre(new, *self._view_centre_scan(old))
        was = self._zoom
        self._shown = new
        if origin is None:
            self._zoom, self._c0, self._r0 = 1, 0.0, 0.0
        else:
            self._c0, self._r0 = origin
            if self._contrast == "visible":                # the part in view sets the scale
                new.set_mode(self._flatten, "visible", self._region())
        self._pan_last = None
        self._invalidate()
        self._reconcile_profile()
        self.update()
        if self._zoom != was:
            self.zoom_changed.emit(self._zoom)

    def set_flatten(self, mode: str) -> None:
        if mode not in FLATTEN_MODES:
            raise ValueError(f"flatten mode {mode!r}: one of {FLATTEN_MODES}")
        if mode == self._flatten:
            return
        self._flatten = mode
        if self._shown is not None:
            self._shown.set_mode(mode, self._contrast, self._region())
        self._invalidate()
        self._reconcile_profile()
        self.update()

    def set_contrast(self, mode: str) -> None:
        """``"auto"`` (the default): the grey scale spans the 0.5-99.5 percentile range, as the image an MCP
        agent gets. ``"full"``: it spans the frame's minimum to maximum, which keeps the top of an atom
        from saturating; ``"visible"``: the minimum to maximum of the part in view, so that zoomed onto a terrace
        its atoms are not flattened by a step elsewhere in the frame (with the whole frame in view it is
        ``"full"``). The colour bar follows. Readouts, markers and profiles are the same in every mode (they
        read the flattened data, not the grey level), and the mode holds for the frames shown next."""
        if mode not in CONTRAST_MODES:
            raise ValueError(f"contrast mode {mode!r}: one of {CONTRAST_MODES}")
        if mode == self._contrast:
            return
        self._contrast = mode
        if self._shown is not None:
            self._shown.set_mode(self._flatten, mode, self._region())
        self._invalidate()
        self.update()

    def contrast(self) -> str:
        return self._contrast

    # ── zoom and pan ──
    def zoom(self) -> int:
        return self._zoom

    def view_rect(self) -> tuple[float, float, float, float]:
        """The part of the frame the image shows, in pixel edges: ``(col0, row0, width, height)``; the
        whole frame (``0, 0, nx, ny``) at zoom 1."""
        return self._view_rect()

    def _view_rect(self) -> tuple[float, float, float, float]:
        sh = self._shown
        nx, ny = (sh.nx, sh.ny) if sh is not None else (1, 1)
        z = float(self._zoom)
        return self._c0, self._r0, nx / z, ny / z

    def _clamp_origin(self, c0: float, r0: float) -> tuple[float, float]:
        """The corner kept inside the frame: the view never shows what is not there."""
        _, _, wv, hv = self._view_rect()
        sh = self._shown
        return min(max(c0, 0.0), sh.nx - wv), min(max(r0, 0.0), sh.ny - hv)

    def set_zoom(self, level: int, about: QPointF | None = None, focus: tuple[float, float] | None = None) -> None:
        """Show ``1/level`` of the frame's width and height. ``about`` is a widget position that keeps what is
        under it (the wheel's pointer); ``focus`` a scan position the view is centred on (as far as the frame
        goes: the selected marker, when a control zooms); without either the view's centre stays. A ``focus`` the
        frame does not cover is not used. A level that is not one of :data:`ZOOM_LEVELS` is refused; with no frame
        shown only :attr:`zoom_changed` answers, so that a control that asked for another level goes back."""
        if level not in ZOOM_LEVELS:
            raise ValueError(f"zoom {level!r}: one of {ZOOM_LEVELS}")
        sh = self._shown
        centre = None if sh is None or focus is None else self._edge_of(sh, *focus)
        if sh is None or (level == self._zoom and centre is None):
            self.zoom_changed.emit(self._zoom)
            return
        if centre is not None:                              # centred on the point; the same level re-centres on it
            self._zoom = level
            self._c0, self._r0 = self._clamp_origin(centre[0] - sh.nx / level / 2, centre[1] - sh.ny / level / 2)
            self._view_changed()
            self.zoom_changed.emit(self._zoom)
            return
        img = self._layout().img
        anchored = about is not None and img.contains(about.toPoint())
        fx, fy = ((about.x() - img.x()) / img.width(), (about.y() - img.y()) / img.height()) if anchored else (0.5, 0.5)
        c0, r0, wv, hv = self._view_rect()
        cx, cy = c0 + fx * wv, r0 + fy * hv                 # the pixel edge under the pointer: it stays there
        self._zoom = level
        wv, hv = sh.nx / level, sh.ny / level
        self._c0, self._r0 = self._clamp_origin(cx - fx * wv, cy - fy * hv)
        for _ in range(4):
            # the axes' labels have other widths now, so the image may sit a few pixels elsewhere: place the
            # view again for where it is, until the pixel under the pointer is the one that was
            self._invalidate()
            if not anchored:
                break
            img = self._layout().img
            fx, fy = min(max((about.x() - img.x()) / img.width(), 0.0), 1.0), min(max((about.y() - img.y()) / img.height(), 0.0), 1.0)
            c, r = self._clamp_origin(cx - fx * wv, cy - fy * hv)
            if abs(c - self._c0) < 1e-9 and abs(r - self._r0) < 1e-9:
                break
            self._c0, self._r0 = c, r
        self._view_changed()
        self.zoom_changed.emit(self._zoom)

    @staticmethod
    def _edge_of(sh: "_Shown", x_nm: float, y_nm: float) -> tuple[float, float] | None:
        """A scan position as a point in the frame's pixel edges (the unit of the view's origin), or None where the
        frame does not cover it."""
        col, row = scan_to_pixel(sh.geom, x_nm, y_nm)
        eps = 1e-9
        if -0.5 - eps <= col <= sh.nx - 0.5 + eps and -0.5 - eps <= row <= sh.ny - 0.5 + eps:
            return col + 0.5, row + 0.5
        return None

    def _view_centre_scan(self, sh: "_Shown") -> tuple[float, float]:
        """The scan position at the centre of the view, on ``sh``."""
        c0, r0, wv, hv = self._view_rect()
        return pixel_to_scan_nm(sh.geom, c0 + wv / 2 - 0.5, r0 + hv / 2 - 0.5)

    def _origin_for_centre(self, sh: "_Shown", x_nm: float, y_nm: float) -> tuple[float, float] | None:
        """The origin of a view at the current zoom that is centred on a scan position of ``sh`` (kept inside the
        frame), or None where ``sh`` does not cover it."""
        edge = self._edge_of(sh, x_nm, y_nm)
        if edge is None:
            return None
        z = float(self._zoom)
        wv, hv = sh.nx / z, sh.ny / z
        return min(max(edge[0] - wv / 2, 0.0), sh.nx - wv), min(max(edge[1] - hv / 2, 0.0), sh.ny - hv)

    def zoom_step(self, direction: int, about: QPointF | None = None) -> None:
        """One level in (``direction`` > 0) or out, as one notch of the wheel does."""
        i = ZOOM_LEVELS.index(self._zoom) + (1 if direction > 0 else -1)
        self.set_zoom(ZOOM_LEVELS[min(max(i, 0), len(ZOOM_LEVELS) - 1)], about)

    def fit(self) -> None:
        """The whole frame."""
        self.set_zoom(1)

    def pan_by(self, dx_px: float, dy_px: float) -> None:
        """Move the zoomed image by ``dx_px``, ``dy_px`` widget pixels (a drag: the image follows the pointer),
        as far as the frame goes."""
        if self._shown is None or self._zoom == 1:
            return
        img = self._layout().img
        c0, r0, wv, hv = self._view_rect()
        c, r = self._clamp_origin(c0 - dx_px * wv / img.width(), r0 - dy_px * hv / img.height())
        if (c, r) != (self._c0, self._r0):
            self._c0, self._r0 = c, r
            self._view_changed()

    def _region(self) -> tuple[int, int, int, int] | None:
        """The rows and columns the view covers (``row0, row1, col0, col1``), or None when it is the whole frame."""
        if self._shown is None or self._zoom == 1:
            return None
        c0, r0, wv, hv = self._view_rect()
        return int(math.floor(r0)), int(math.ceil(r0 + hv)), int(math.floor(c0)), int(math.ceil(c0 + wv))

    def _view_changed(self) -> None:
        if self._contrast == "visible" and self._shown is not None:      # the grey scale follows what is in view
            self._shown.set_mode(self._flatten, "visible", self._region())
        self._invalidate()
        self.update()

    def set_tool(self, tool: str) -> None:
        if tool not in TOOLS:
            raise ValueError(f"tool {tool!r}: one of {TOOLS}")
        self._tool = tool
        self._drag = None
        self._pan_last = None
        self.setCursor({"none": Qt.CursorShape.ArrowCursor, "pan": Qt.CursorShape.OpenHandCursor}.get(
            tool, Qt.CursorShape.CrossCursor))
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
        rotated = abs(ang) >= 1e-9
        if self._zoom == 1:
            if not rotated:
                return ((g["cx_nm"] - w / 2, g["cx_nm"] + w / 2), (g["cy_nm"] - h / 2, g["cy_nm"] + h / 2),
                        "x (nm)", "y (nm)")
            return (-w / 2, w / 2), (-h / 2, h / 2), f"u (nm), frame rotated {ang:g}°", "v (nm)"
        # zoomed: the part of the frame in view, as fractions of its width and height (rows go down)
        c0, r0, wv, hv = self._view_rect()
        nx, ny = self._shown.nx, self._shown.ny
        fx0, fx1, fy0, fy1 = c0 / nx, (c0 + wv) / nx, r0 / ny, (r0 + hv) / ny
        x_lo, y_top = (-w / 2, h / 2) if rotated else (g["cx_nm"] - w / 2, g["cy_nm"] + h / 2)
        xs, ys = (x_lo + fx0 * w, x_lo + fx1 * w), (y_top - fy1 * h, y_top - fy0 * h)
        if rotated:
            return xs, ys, f"u (nm), frame rotated {ang:g}°", "v (nm)"
        return xs, ys, "x (nm)", "y (nm)"

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
        c0, r0, wv, hv = self._view_rect()
        return QPointF(r.x() + (col + 0.5 - c0) * r.width() / wv, r.y() + (row + 0.5 - r0) * r.height() / hv)

    def scan_to_widget(self, x_nm: float, y_nm: float) -> QPointF | None:
        """Widget position of a scan-frame point; None when this frame does not cover it, or the view (zoomed)
        does not show it."""
        sh = self._shown
        if sh is None:
            return None
        col, row = scan_to_pixel(sh.geom, x_nm, y_nm)
        c0, r0, wv, hv = self._view_rect()
        eps = 1e-9
        if not (c0 - 0.5 - eps <= col <= c0 + wv - 0.5 + eps and r0 - 0.5 - eps <= row <= r0 + hv - 0.5 + eps):
            return None
        return self.pixel_to_widget(col, row)

    def _point_at(self, pos: QPointF, clamp: bool = False) -> dict | None:
        """What lies under a widget position; None outside the image (``clamp``: the nearest edge point).

        The point is the pixel the position falls in: ``col``/``row``, its ``value`` and the scan position of
        its centre (``x_nm``/``y_nm``), so that a reading and the coordinates it is given at belong together.
        ``at_x_nm``/``at_y_nm`` and ``col_f``/``row_f`` are the position itself (a profile's ends use them)."""
        sh = self._shown
        if sh is None:
            return None
        r = self._layout().img
        fx, fy = (pos.x() - r.x()) / r.width(), (pos.y() - r.y()) / r.height()
        if clamp:
            fx, fy = min(max(fx, 0.0), 1.0), min(max(fy, 0.0), 1.0)
        elif not (0.0 <= fx < 1.0 and 0.0 <= fy < 1.0):
            return None
        c0, r0, wv, hv = self._view_rect()
        col, row = c0 + fx * wv - 0.5, r0 + fy * hv - 0.5
        ci = min(max(int(math.floor(col + 0.5)), 0), sh.nx - 1)
        ri = min(max(int(math.floor(row + 0.5)), 0), sh.ny - 1)
        x, y = pixel_to_scan_nm(sh.geom, ci, ri)
        ax, ay = pixel_to_scan_nm(sh.geom, col, row)
        return {"x_nm": x, "y_nm": y, "col": ci, "row": ri, "value": float(sh.values[ri, ci]), "unit": sh.unit,
                "at_x_nm": ax, "at_y_nm": ay, "col_f": col, "row_f": row}

    # ── mouse ──
    def mouseMoveEvent(self, ev) -> None:
        pos = ev.position()
        if self._pan_last is not None:                       # a pan drag carries the image with the pointer
            self.pan_by(pos.x() - self._pan_last.x(), pos.y() - self._pan_last.y())
            self._pan_last = QPointF(pos)
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
        if self._tool == "pan":
            self._pan_last = QPointF(ev.position())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            ev.accept()
            return
        if self._tool == "marker":
            m = {**{k: p[k] for k in ("x_nm", "y_nm", "col", "row", "value", "unit")}, "frame": self._shown.name}
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
        if ev.button() == Qt.MouseButton.LeftButton and self._pan_last is not None:
            self._pan_last = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            ev.accept()
            return
        if ev.button() != Qt.MouseButton.LeftButton or self._drag is None:
            ev.accept()
            return
        a, b = self._drag[0], self._clamped(ev.position())
        self._drag = None
        if math.hypot(b.x() - a.x(), b.y() - a.y()) >= 3.0:         # shorter is a click, not a line
            pa, pb = self._point_at(a, clamp=True), self._point_at(b, clamp=True)
            self._profile = {"frame": self._shown.name, "x0_nm": pa["at_x_nm"], "y0_nm": pa["at_y_nm"],
                             "x1_nm": pb["at_x_nm"], "y1_nm": pb["at_y_nm"]}
            self.profile_changed.emit(self._profile_payload())
        self.update()
        ev.accept()

    def wheelEvent(self, ev) -> None:
        """A notch of the wheel is one zoom level about the pointer (up: in). Always accepted: a parent never
        scrolls. With Ctrl it does nothing."""
        ev.accept()
        if self._shown is None or ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            return
        dy = ev.angleDelta().y()
        if dy:
            self.zoom_step(1 if dy > 0 else -1, ev.position())

    def contextMenuEvent(self, ev) -> None:
        ev.accept()

    # ── painting ──
    def _scaled_image(self, w: int, h: int) -> QImage:
        """The grey image enlarged to ``w`` x ``h`` device pixels, nearest neighbour: widget pixel
        ``i`` shows source pixel ``floor((i + 0.5) * n / size)``, whatever the platform's scaler."""
        sh = self._shown
        dpr = self.devicePixelRatioF()
        pw, ph = max(1, round(w * dpr)), max(1, round(h * dpr))
        c0, r0, wv, hv = self._view_rect()
        key = (id(sh), pw, ph, self._flatten, self._contrast, self._zoom, c0, r0)
        if self._image_cache is not None and self._image_cache[0] == key:
            return self._image_cache[1]
        ri = np.minimum(np.floor(r0 + (np.arange(ph) + 0.5) * hv / ph).astype(np.intp), sh.ny - 1)
        ci = np.minimum(np.floor(c0 + (np.arange(pw) + 0.5) * wv / pw).astype(np.intp), sh.nx - 1)
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
        items = []
        for i, m in enumerate(self._markers):
            pt = self.scan_to_widget(m["x_nm"], m["y_nm"])
            if pt is not None:
                items.append((i + 1, pt, i == self._selected, m))
        items.sort(key=lambda it: it[2])                          # the selected marker last: it is on top of the others
        rects = self._marker_label_rects(lay, items)
        for (_number, pt, selected, m), (rect, leader) in zip(items, rects):
            self._paint_marker(p, lay, pt, m, selected)
            if leader:                                            # a number that had to stand off: a line to its ring
                self._paint_leader(p, lay, pt, rect, selected)
        for (number, _pt, selected, _m), (rect, _leader) in zip(items, rects):
            self._label(p, lay, str(number), rect, MARKER_SELECTED_COLOR if selected else MARKER_COLOR)
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

    def _paint_marker(self, p: QPainter, lay: SimpleNamespace, pt: QPointF, m: dict, selected: bool) -> None:
        """An open ring on the marker's position, in a dark rim that reads on a white atom too; the pixel inside is
        left as it is. Where one pixel is wider than two rings (zoomed in), the pixel the marker is on is outlined,
        so that the pixel its row of the table speaks of is told from its neighbours."""
        r = marker_radius(lay.fpx)
        color = MARKER_SELECTED_COLOR if selected else MARKER_COLOR
        p.setBrush(Qt.BrushStyle.NoBrush)
        sh = self._shown
        if m.get("frame") == sh.name and "col" in m:           # placed on this frame: it is at a pixel's centre
            a = self.pixel_to_widget(m["col"] - 0.5, m["row"] - 0.5)
            b = self.pixel_to_widget(m["col"] + 0.5, m["row"] + 0.5)
            box = QRectF(a, b).adjusted(1.5, 1.5, -1.5, -1.5)
            if box.width() >= 4.0 * r:
                p.setPen(QPen(QColor(0, 0, 0, 235), 4.0))
                p.drawRect(box)
                p.setPen(QPen(color, 2.0))
                p.drawRect(box)
        p.setPen(QPen(QColor(0, 0, 0, 235), 4.5))
        p.drawEllipse(pt, r, r)
        p.setPen(QPen(color, 2.6 if selected else 2.0))
        p.drawEllipse(pt, r, r)

    def _marker_label_rects(self, lay: SimpleNamespace, items: list) -> list[tuple[QRectF, bool]]:
        """Where each marker's number goes, and whether it stands off from its ring (then a line joins them). The
        markers of one atom stand a few pixels apart, and five numbers in boxes on top of their rings and of one
        another cannot be read: a number goes beside its ring, in the first of eight places (upper right first, then
        the other corners, the sides, above and below) that covers no other number and no other marker's ring; where
        all eight are taken it stands one label further out, then two."""
        r = marker_radius(lay.fpx)
        bound = QRectF(lay.img).adjusted(2, 2, -2, -2)
        discs = [(pt, r + 3.0) for _n, pt, _s, _m in items]
        placed: list[QRectF] = []
        out: list[tuple[QRectF, bool]] = []
        k45 = math.sqrt(0.5)
        dirs = ((k45, -k45), (-k45, -k45), (k45, k45), (-k45, k45), (1.0, 0.0), (-1.0, 0.0), (0.0, -1.0), (0.0, 1.0))
        for k, (number, pt, _selected, _m) in enumerate(items):
            size = self._label_rect(lay, str(number), 0, 0)
            w, h = size.width(), size.height()
            g = 3.0
            best, best_cost, far = None, None, False
            for step in range(3):
                extra = step * (h + 4.0)
                for ux, uy in dirs:
                    reach = r + g + extra
                    cx = pt.x() + ux * reach + (w / 2 if ux > 0.1 else -w / 2 if ux < -0.1 else 0.0)
                    cy = pt.y() + uy * reach + (h / 2 if uy > 0.1 else -h / 2 if uy < -0.1 else 0.0)
                    rect = QRectF(cx - w / 2, cy - h / 2, w, h)
                    cost = 0.0 if bound.contains(rect) else 500.0
                    cost += 100.0 * sum(rect.intersects(q) for q in placed)
                    for j, (c, rad) in enumerate(discs):
                        if j != k:
                            nx, ny = min(max(c.x(), rect.left()), rect.right()), min(max(c.y(), rect.top()), rect.bottom())
                            cost += 60.0 if (nx - c.x()) ** 2 + (ny - c.y()) ** 2 < rad * rad else 0.0
                    cost += 5.0 * step                              # the nearer place, unless it covers something
                    if best_cost is None or cost < best_cost:
                        best, best_cost, far = rect, cost, step > 0
                    if cost == 0.0:
                        break
                if best_cost is not None and best_cost < 5.0:
                    break
            placed.append(best)
            out.append((best, far))
        return out

    def _paint_leader(self, p: QPainter, lay: SimpleNamespace, pt: QPointF, rect: QRectF, selected: bool) -> None:
        """A thin line from a ring to its number's box, which stands a little off."""
        r = marker_radius(lay.fpx)
        nx, ny = min(max(pt.x(), rect.left()), rect.right()), min(max(pt.y(), rect.top()), rect.bottom())
        d = math.hypot(nx - pt.x(), ny - pt.y())
        if d <= r + 2.0:
            return
        a = QPointF(pt.x() + (nx - pt.x()) / d * r, pt.y() + (ny - pt.y()) / d * r)
        b = QPointF(nx, ny)
        p.setPen(QPen(QColor(0, 0, 0, 235), 3.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(a, b)
        p.setPen(QPen(MARKER_SELECTED_COLOR if selected else MARKER_COLOR, 1.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(a, b)

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
            if not lay.img.adjusted(-6, -6, 6, 6).contains(pt.toPoint()):
                continue                                         # an end beyond the view (zoomed): only the line shows
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(PROFILE_COLOR)
            p.drawEllipse(pt, 3.5, 3.5)
            rect = self._label_rect(lay, text, 0, 0)
            reach = 10.0 + 0.5 * math.hypot(rect.width(), rect.height())
            rect.moveCenter(QPointF(pt.x() + sign * ux * reach, pt.y() + sign * uy * reach))
            self._label(p, lay, text, rect, PROFILE_COLOR)
