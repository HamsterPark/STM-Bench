"""PlotView: a plain line plot for the desktop console (spectra, line profiles, any column pair).

White background, 1-2-5 axis ticks with light grid lines, several series in colours that stay
distinct for the common colour-vision deficiencies, a legend strip when there is more than one
series. The picture is read from a screenshot, so it favours large text (never below
:data:`MIN_FONT_PX`) and thick lines over detail. Non-finite points are skipped, never drawn as
zero; a series with nothing finite, or no series at all, shows a message.

The one interaction is a cursor, as on an instrument's graph: over the axes a vertical line marks
the data point nearest the pointer in x, with a dot on every curve there and a box that reads the
point's x and each series' y (fixed-pitch numbers; the digits come from each quantity's own range,
a millivolt on a +-1 V sweep), so an energy is read off the spectrum, not estimated from the ticks.
A left click pins the point (and emits ``point_picked``): off the axes the readout shows the pin,
over them it follows the pointer. A later click on another point pins that one too, up to :data:`MAX_PINS`
(a screenshot is taken after the last event of a step, so several readings are read in one step only if they
stay): with more than one pin each is numbered, and the readout lists them, one aligned line each, in the
numbers the hover readout gives; ``clear_pins`` (the viewer's Clear readings button) and new data remove them.
Nothing is found: the pins are where the reader clicked. There is no zoom, pan, wheel handling or context menu.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

MIN_FONT_PX = 14
#: most readings pinned at once; one more clicked drops the oldest (the list must fit the axes)
MAX_PINS = 6
#: Okabe-Ito colours (no yellow: it vanishes on white); more series than colours cycle
SERIES_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
#: the readout's numbers are fixed-pitch so that digits line up; the first family the machine has wins
MONO_FAMILIES = ("Cascadia Mono", "Consolas", "DejaVu Sans Mono", "Liberation Mono", "Noto Sans Mono", "Courier New")
#: the readout box: opaque, so grid lines and curves under it do not run through the numbers
READOUT_BG = QColor("#eef3fb")
_BACKGROUND = QColor("#ffffff")
_TEXT = QColor("#111111")
_MUTED = QColor("#555b63")
_GRID = QColor("#e1e5ea")
_AXIS = QColor("#3d434b")
_CURSOR_LIVE = QColor(25, 25, 25, 215)
_CURSOR_PIN = QColor("#111111")
_NO_DATA = "No data"


# ── shared helpers (FrameView draws its axes with the same ticks) ──────────────────────────
def font_px(font: QFont) -> int:
    """Pixel size of ``font``; a point-sized font is converted at the usual 96 dpi."""
    px = font.pixelSize()
    if px > 0:
        return px
    pt = font.pointSizeF()
    return max(1, round(pt * 96.0 / 72.0)) if pt > 0 else 12


def nice_ticks(lo: float, hi: float, max_ticks: int) -> tuple[list[float], float]:
    """Multiples of the smallest 1-2-5 step inside ``[lo, hi]`` that give at most ``max_ticks``
    ticks; also the step. ``([], 0.0)`` for an empty or non-finite range."""
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [], 0.0
    max_ticks = max(2, int(max_ticks))
    e0 = math.floor(math.log10(hi - lo))
    for exp in range(e0 - 3, e0 + 2):
        for m in (1.0, 2.0, 5.0):
            step = m * 10.0 ** exp
            first, last = math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9)
            if last - first + 1 <= max_ticks:
                return [round(k * step, 12) for k in range(first, last + 1)], step
    step = 10.0 ** (e0 + 1)                       # not reached for max_ticks >= 2; one tick at most
    return [round(k * step, 12) for k in range(math.ceil(lo / step), math.floor(hi / step) + 1)], step


def tick_decimals(step: float) -> int:
    """Digits after the decimal point that show multiples of ``step`` exactly."""
    if step <= 0:
        return 0
    return min(9, max(0, -math.floor(math.log10(step) + 1e-9)))


def format_number(v: float, decimals: int, sig: int = 3) -> str:
    """Fixed-point text without a negative zero; scientific notation (``sig`` significant digits)
    beyond 1e7 or below 1e-5."""
    if not math.isfinite(v):
        return "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")
    if abs(v) >= 1e7 or (v != 0 and abs(v) < 1e-5):
        return f"{v:.{sig}g}"
    s = f"{v:.{decimals}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def reading_decimals(span: float, sig: int = 4) -> int:
    """Digits after the point that give ``sig`` significant digits of a quantity that ranges over
    ``span``: with four, an axis of +-1 V (span 2) reads to a millivolt."""
    if not math.isfinite(span) or span <= 0:
        return sig - 1
    return min(9, max(0, sig - 1 - math.floor(math.log10(span))))


def format_ticks(ticks: list[float], step: float) -> list[str]:
    d = tick_decimals(step)
    return [format_number(t, d) for t in ticks]


def _as_floats(a) -> np.ndarray:
    try:
        return np.asarray(a, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return np.empty(0)


def _finite_span(a: np.ndarray) -> float:
    """Largest minus smallest finite value (0 with none)."""
    f = a[np.isfinite(a)]
    return float(f.max() - f.min()) if f.size else 0.0


class PlotView(QWidget):
    """Line plot of one shared x and any number of named y series, with a cursor readout. See the
    module docstring.

    ``point_picked`` (a left click on the axes, never a hover): ``{"index": i, "x": x, "values":
    [[series label, y], ...]}``, the point's index in the arrays given to :meth:`set_series`, its x,
    and every series' y there (NaN where a series has none)."""

    point_picked = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._x = np.empty(0)
        self._series: list[tuple[str, np.ndarray]] = []
        self._x_label = ""
        self._y_label = ""
        self._title = ""
        self._message = ""
        self._ext: tuple[float, float, float, float] | None = None
        self._layout_cache: tuple | None = None
        self._values = np.empty((0, 0))                 # y of every series by point; NaN beyond a series' end
        self._valid = np.empty(0, dtype=np.intp)        # the points the cursor can stand on
        self._x_decimals = 3
        self._y_decimals: list[int] = []
        self._hover: int | None = None
        self._pin: int | None = None                    # the last point pinned
        self._pins: list[int] = []                      # every point pinned, in the order they were clicked
        self._pointer: QPointF | None = None            # where the pointer last was over the widget
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)

    # ── data ──
    def set_series(self, x, series: list[tuple[str, "array-like"]], x_label: str, y_label: str,
                   title: str = "") -> None:
        """Plot each ``(name, y)`` against ``x``. A series longer than ``x`` is cut to it (and the
        reverse); NaN and infinities leave gaps; nothing finite shows ``No data``. Removes the
        cursor's pin."""
        xs = _as_floats(x)
        self._x = xs
        self._series = [(str(name), _as_floats(y)[:len(xs)]) for name, y in series]
        self._x_label, self._y_label, self._title = str(x_label), str(y_label), str(title)
        self._ext = self._extent()
        self._message = "" if self._ext is not None else _NO_DATA
        self._index_points()
        self.update()

    def clear(self, message: str = "") -> None:
        """Empty the plot (and its cursor); ``message`` (default none) is shown in its place."""
        self._x = np.empty(0)
        self._series = []
        self._x_label = self._y_label = self._title = ""
        self._message = str(message)
        self._ext = None
        self._index_points()
        self.update()

    def _index_points(self) -> None:
        """What the cursor needs from new data: y by series and point, the points it may stand on
        (finite x and some finite y: a point nothing is drawn at is skipped), and the digits its
        readout shows. Drops the hover and the pin."""
        n = len(self._x)
        vals = np.full((len(self._series), n), np.nan)
        for k, (_, y) in enumerate(self._series):
            vals[k, :len(y)] = y
        self._values = vals
        self._valid = np.flatnonzero(np.isfinite(self._x) & np.isfinite(vals).any(axis=0)) if len(vals) \
            else np.empty(0, dtype=np.intp)
        xs = self._x[self._valid]
        digits = reading_decimals(_finite_span(xs))
        steps = np.diff(np.unique(xs)) if xs.size > 1 else np.empty(0)
        if steps.size:                                  # and at least half the point spacing: neighbours never print alike
            digits = max(digits, min(9, math.ceil(-math.log10(float(np.median(steps)) / 2) - 1e-9)))
        self._x_decimals = digits
        self._y_decimals = [reading_decimals(_finite_span(row)) for row in vals]
        self._hover = self._pin = self._pointer = None   # until the pointer moves again
        self._pins = []
        self._layout_cache = None

    def _extent(self) -> tuple[float, float, float, float] | None:
        """Finite data bounds ``(x0, x1, y0, y1)``, or None when no point has both coordinates."""
        xs, ys = [], []
        for _, y in self._series:
            x = self._x[:len(y)]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.any():
                xs.append(x[ok])
                ys.append(y[ok])
        if not xs:
            return None
        x, y = np.concatenate(xs), np.concatenate(ys)
        return float(x.min()), float(x.max()), float(y.min()), float(y.max())

    # ── size and events ──
    def _fpx(self) -> int:
        return max(MIN_FONT_PX, font_px(self.font()))

    def sizeHint(self) -> QSize:
        k = self._fpx() / 16.0
        return QSize(int(520 * k), int(300 * k))

    def minimumSizeHint(self) -> QSize:
        k = self._fpx() / 16.0
        return QSize(int(220 * k), int(140 * k))

    def wheelEvent(self, ev) -> None:        # accepted and dropped: no zoom, and no scrolling of a parent
        ev.accept()

    def contextMenuEvent(self, ev) -> None:
        ev.accept()

    def changeEvent(self, ev) -> None:
        super().changeEvent(ev)
        if ev.type() == QEvent.Type.FontChange:
            self.update()

    def resizeEvent(self, ev) -> None:
        # the pointer stayed where it was, the data moved under it (a hidden widget that is grabbed
        # gets a resize event of its own size each time, which must change nothing)
        self._hover = None if self._pointer is None else self._index_at(self._pointer)
        super().resizeEvent(ev)

    # ── the cursor ──
    def _index_at(self, pos: QPointF) -> int | None:
        """The point the cursor stands on for a pointer at ``pos``: the one nearest in x among the
        points that can be read, when the pointer is over the axes."""
        lay = self._layout_now()
        if lay is None or self._valid.size == 0:
            return None
        plot = lay.plot
        if not (plot.left() <= pos.x() <= plot.right() and plot.top() <= pos.y() <= plot.bottom()):
            return None                           # closed: the last point's own position is on the edge
        target = lay.x0 + (pos.x() - plot.left()) / plot.width() * (lay.x1 - lay.x0)
        return int(self._valid[int(np.argmin(np.abs(self._x[self._valid] - target)))])

    def _series_label(self, k: int) -> str:
        return self._series[k][0] or self._y_label or "y"

    def _payload(self, i: int) -> dict:
        return {"index": int(i), "x": float(self._x[i]),
                "values": [[self._series_label(k), float(self._values[k, i])] for k in range(len(self._series))]}

    def cursor_point(self) -> dict | None:
        """The point the readout shows now (the hovered one, else the pinned one), as
        ``point_picked`` carries it; None when there is no cursor."""
        i = self._hover if self._hover is not None else self._pin
        return None if i is None else self._payload(i)

    def pinned_point(self) -> dict | None:
        """The last point pinned (None: none)."""
        return None if self._pin is None else self._payload(self._pin)

    def pinned_points(self) -> list[dict]:
        """Every point pinned, in the order they were clicked."""
        return [self._payload(i) for i in self._pins]

    def clear_pins(self) -> None:
        """Remove every pinned reading."""
        if self._pins:
            self._pins, self._pin = [], None
            self.update()

    def point_position(self, index: int, series: int = 0) -> QPointF | None:
        """Widget position of data point ``index`` of series ``series``; None when it is not drawn."""
        lay = self._layout_now()
        if lay is None or not (0 <= index < len(self._x) and 0 <= series < len(self._series)):
            return None
        xv, yv = float(self._x[index]), float(self._values[series, index])
        return self._to_px(lay, xv, yv) if math.isfinite(xv) and math.isfinite(yv) else None

    def mouseMoveEvent(self, ev) -> None:
        self._pointer = QPointF(ev.position())
        i = self._index_at(self._pointer)
        if i != self._hover:
            self._hover = i
            self.update()

    def leaveEvent(self, ev) -> None:
        self._pointer = None
        if self._hover is not None:
            self._hover = None
            self.update()
        super().leaveEvent(ev)

    def mousePressEvent(self, ev) -> None:
        self._pointer = QPointF(ev.position())
        i = self._index_at(self._pointer) if ev.button() == Qt.MouseButton.LeftButton else None
        if i is not None:
            if i not in self._pins:                      # a click on a pinned point changes nothing but the last one
                self._pins = [*self._pins, i][-MAX_PINS:]
            self._pin = self._hover = i
            self.update()
            self.point_picked.emit(self._payload(i))
        ev.accept()

    def mouseDoubleClickEvent(self, ev) -> None:
        ev.accept()                              # a double click is one click: the press before it did the work

    # ── painting ──
    def _tpx(self) -> int:
        """Pixel size of tick numbers, labels and legend: a little above the widget's base text,
        which is itself never below :data:`MIN_FONT_PX`."""
        return self._fpx() + 1

    def _fonts(self) -> tuple[QFont, QFont]:
        regular = QFont(self.font())
        regular.setPixelSize(self._tpx())
        regular.setBold(False)
        bold = QFont(regular)
        bold.setBold(True)
        return regular, bold

    def _legend_rows(self, fm: QFontMetrics, width: float) -> list[list[tuple[str, QColor, int]]]:
        """Legend entries (name, colour, text width) wrapped into rows no wider than ``width``;
        none for a single series, whose y label already names it."""
        if len(self._series) < 2:
            return []
        rows: list[list[tuple[str, QColor, int]]] = [[]]
        used = 0
        for i, (name, _) in enumerate(self._series):
            if not name:
                continue
            w = fm.horizontalAdvance(name)
            item = 26 + 6 + w + 18
            if rows[-1] and used + item > width:
                rows.append([])
                used = 0
            rows[-1].append((name, QColor(SERIES_COLORS[i % len(SERIES_COLORS)]), w))
            used += item
        return rows if rows[0] else []

    def _layout_now(self) -> SimpleNamespace | None:
        """The layout for the data, size and font as they are (None with nothing to draw), kept
        until one of them changes: the cursor asks for it on every pointer move."""
        if self._ext is None:
            return None
        regular, bold = self._fonts()
        key = (self.width(), self.height(), self._tpx(), regular.key())
        if self._layout_cache is None or self._layout_cache[0] != key:
            lay = self._layout(self._ext, QFontMetrics(regular), QFontMetrics(bold), self._tpx())
            self._layout_cache = (key, lay)
        return self._layout_cache[1]

    @staticmethod
    def _to_px(lay: SimpleNamespace, xv: float, yv: float) -> QPointF:
        plot = lay.plot
        return QPointF(plot.left() + (xv - lay.x0) / (lay.x1 - lay.x0) * plot.width(),
                       plot.bottom() - (yv - lay.y0) / (lay.y1 - lay.y0) * plot.height())

    def plot_rect(self) -> QRectF:
        """The axes rectangle in widget coordinates (a null rectangle when there is nothing to draw)."""
        lay = self._layout_now()
        return QRectF() if lay is None else QRectF(lay.plot)

    def _layout(self, ext, fm: QFontMetrics, bold_fm: QFontMetrics, fpx: int) -> SimpleNamespace:
        W, H = self.width(), self.height()
        pad = max(6, fpx // 2)
        vpad = max(3, fpx // 4)                      # height is what a small plot lacks
        fh = fm.height()
        tick_len = max(4, fpx // 4)
        gap = 4
        x0, x1, y0, y1 = ext
        # a flat series still needs a span to be drawn in; y gets headroom so a peak is not clipped
        if x1 <= x0:
            half = 0.5 * max(1.0, abs(x0) * 0.1)
            x0, x1 = x0 - half, x1 + half
        if y1 <= y0:
            half = 0.5 * max(1.0, abs(y0) * 0.1)
            y0, y1 = y0 - half, y1 + half
        else:
            margin = 0.05 * (y1 - y0)
            y0, y1 = y0 - margin, y1 + margin
        legend = self._legend_rows(fm, W - 2 * pad)
        legend_h = len(legend) * (fh + 3) + (3 if legend else 0)
        show_x_label, show_title = bool(self._x_label), bool(self._title)

        def vertical() -> tuple[int, int, int]:
            top = vpad + ((bold_fm.height() + 4) if show_title else 0) + legend_h
            bottom = vpad + tick_len + gap + fh + ((gap + fh) if show_x_label else 0)
            return top, bottom, H - top - bottom

        top, bottom, plot_h = vertical()
        # a short plot (the line profile beside the marker table) keeps its curve rather than its
        # captions: the x label goes first (the title usually names the unit too), then the title
        if plot_h < 4.3 * fh and show_x_label:
            show_x_label = False
            top, bottom, plot_h = vertical()
        if plot_h < 2.8 * fh and show_title:
            show_title = False
            top, bottom, plot_h = vertical()
        title_h = bold_fm.height() + 4 if show_title else 0
        plot_h = max(20, plot_h)
        # y ticks first: their label width sets the left margin, which sets the width x gets
        yt, ystep = nice_ticks(y0, y1, max(2, min(8, int(plot_h / (fh * 2.3)))))
        ylabels = format_ticks(yt, ystep)
        yw = max((fm.horizontalAdvance(s) for s in ylabels), default=0)
        left = pad + ((fh + gap) if self._y_label else 0) + yw + gap + tick_len
        plot_w0 = max(20, W - left - pad - 2 * fpx)
        for max_ticks in range(9, 1, -1):          # thin the x ticks out until their labels do not touch
            xt, xstep = nice_ticks(x0, x1, max_ticks)
            xlabels = format_ticks(xt, xstep)
            pos = [(t - x0) / (x1 - x0) * plot_w0 for t in xt]
            widths = [fm.horizontalAdvance(s) for s in xlabels]
            if all(pos[i + 1] - pos[i] >= 0.5 * (widths[i] + widths[i + 1]) + 10 for i in range(len(xt) - 1)):
                break
        right = max(pad, (widths[-1] // 2 + 3) if widths else pad)
        plot = QRectF(left, top, max(20.0, W - left - right), plot_h)
        return SimpleNamespace(W=W, H=H, pad=pad, vpad=vpad, fh=fh, tick_len=tick_len, gap=gap, plot=plot,
                               legend=legend, title_h=title_h, show_x_label=show_x_label, show_title=show_title,
                               x0=x0, x1=x1, y0=y0, y1=y1, xt=xt, xlabels=xlabels, yt=yt, ylabels=ylabels, top=top)

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), _BACKGROUND)
        lay = self._layout_now()
        regular, bold = self._fonts()
        if lay is None:
            if self._message:
                p.setFont(regular)
                p.setPen(_MUTED)
                p.drawText(QRectF(self.rect()).adjusted(12, 12, -12, -12),
                           int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap), self._message)
            p.end()
            return
        bold_fm = QFontMetrics(bold)
        plot = lay.plot
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setFont(regular)

        def to_px(xv: float, yv: float) -> QPointF:
            return self._to_px(lay, xv, yv)

        # grid, then the data under the frame
        p.setPen(QPen(_GRID, 1))
        for t in lay.xt:
            gx = round(to_px(t, lay.y0).x()) + 0.5
            p.drawLine(QPointF(gx, plot.top()), QPointF(gx, plot.bottom()))
        for t in lay.yt:
            gy = round(to_px(lay.x0, t).y()) + 0.5
            p.drawLine(QPointF(plot.left(), gy), QPointF(plot.right(), gy))
        p.save()
        p.setClipRect(plot.adjusted(-2, -2, 2, 2))
        for i, (_, y) in enumerate(self._series):
            self._draw_series(p, self._x[:len(y)], y, QColor(SERIES_COLORS[i % len(SERIES_COLORS)]), to_px)
        p.restore()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.setPen(QPen(_AXIS, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(plot.adjusted(0, 0, -1, -1))

        # tick marks and labels
        p.setPen(_TEXT)
        for t, s in zip(lay.xt, lay.xlabels):
            gx = to_px(t, lay.y0).x()
            p.setPen(QPen(_AXIS, 1))
            p.drawLine(QPointF(round(gx) + 0.5, plot.bottom() - 1), QPointF(round(gx) + 0.5, plot.bottom() - 1 + lay.tick_len))
            p.setPen(_TEXT)
            p.drawText(QRectF(gx - 100, plot.bottom() + lay.tick_len + 1, 200, lay.fh),
                       int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop), s)
        for t, s in zip(lay.yt, lay.ylabels):
            gy = to_px(lay.x0, t).y()
            p.setPen(QPen(_AXIS, 1))
            p.drawLine(QPointF(plot.left() - lay.tick_len, round(gy) + 0.5), QPointF(plot.left(), round(gy) + 0.5))
            p.setPen(_TEXT)
            p.drawText(QRectF(plot.left() - lay.tick_len - lay.gap - 200, gy - lay.fh / 2, 200, lay.fh),
                       int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), s)

        # axis labels and title
        p.setFont(bold)
        p.setPen(_TEXT)
        if lay.show_x_label:
            r = QRectF(plot.left(), plot.bottom() + lay.tick_len + lay.gap + lay.fh, plot.width(), lay.fh)
            p.drawText(r, int(Qt.AlignmentFlag.AlignCenter),
                       bold_fm.elidedText(self._x_label, Qt.TextElideMode.ElideMiddle, int(plot.width())))
        if self._y_label:
            p.save()
            p.translate(lay.pad + lay.fh / 2, plot.center().y())
            p.rotate(-90)
            p.drawText(QRectF(-plot.height() / 2, -lay.fh / 2, plot.height(), lay.fh),
                       int(Qt.AlignmentFlag.AlignCenter),
                       bold_fm.elidedText(self._y_label, Qt.TextElideMode.ElideMiddle, int(plot.height())))
            p.restore()
        if lay.show_title:
            r = QRectF(lay.pad, lay.vpad, lay.W - 2 * lay.pad, lay.title_h)
            p.drawText(r, int(Qt.AlignmentFlag.AlignCenter),
                       bold_fm.elidedText(self._title, Qt.TextElideMode.ElideRight, int(r.width())))

        # legend strip between the title and the axes
        p.setFont(regular)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        y = lay.vpad + lay.title_h
        for row in lay.legend:
            x = lay.pad
            for name, color, w in row:
                p.setPen(QPen(color, 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.drawLine(QPointF(x + 2, y + lay.fh / 2 + 1), QPointF(x + 24, y + lay.fh / 2 + 1))
                p.setPen(_TEXT)
                p.drawText(QRectF(x + 32, y, w + 4, lay.fh), int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), name)
                x += 26 + 6 + w + 18
            y += lay.fh + 3
        self._paint_cursor(p, lay, to_px)
        p.end()

    # the cursor: a line at the point, a dot on each curve, a box with the numbers
    def _mono_font(self) -> QFont:
        have = set(QFontDatabase.families())            # ask only for families the machine has: a missing one is looked up (slowly) each time
        f = QFont()
        f.setFamilies([n for n in MONO_FAMILIES if n in have] or ["monospace"])
        f.setStyleHint(QFont.StyleHint.Monospace)
        f.setFixedPitch(True)
        f.setPixelSize(self._tpx() + 2)                 # the numbers the picture is read for: larger than the ticks
        return f

    def _fields(self, i: int) -> list[tuple[QColor | None, str, str]]:
        """What the readout says at point ``i``: (curve colour, or None for x; label; number), x first."""
        fields: list[tuple[QColor | None, str, str]] = [
            (None, self._x_label or "x", format_number(float(self._x[i]), self._x_decimals, sig=4))]
        for k in range(len(self._series)):
            fields.append((QColor(SERIES_COLORS[k % len(SERIES_COLORS)]), self._series_label(k),
                           format_number(float(self._values[k, i]), self._y_decimals[k], sig=4)))
        return fields

    @staticmethod
    def _readout_rows(fields, mfm: QFontMetrics, plot: QRectF, pad: int) -> tuple[list[str], int]:
        """The readout's text rows and the width of the swatch column in front of them. The table, one
        ``label = number`` row per field with the equal signs and the digits lined up, when it takes
        up little of the axes; else a single line, the most of it that fits, so that a small plot
        (the line profile beside the marker table) keeps its curve in view."""
        wl, wv = max(len(f[1]) for f in fields), max(len(f[2]) for f in fields)
        swatch = 16 if len(fields) > 2 else 0               # a colour key where several curves are read
        rows = [f"{lab.ljust(wl)} = {val.rjust(wv)}" for _, lab, val in fields]
        w = 2 * pad + swatch + max(mfm.horizontalAdvance(r) for r in rows)
        if w <= 0.8 * plot.width() and 2 * pad + len(rows) * mfm.height() <= 0.5 * plot.height():
            return rows, swatch
        labelled = [f"{lab} = {val}" for _, lab, val in fields]
        tags = [f"x={fields[0][2]}"] + ([f"y={fields[1][2]}"] if len(fields) == 2
                                        else [f"y{k}={f[2]}" for k, f in enumerate(fields[1:], 1)])
        for text in ("  ".join(labelled), "  ".join(labelled[:2]), "  ".join(tags), "  ".join(tags[:2]), labelled[0],
                     tags[0]):
            if 2 * pad + mfm.horizontalAdvance(text) <= plot.width() - 12:
                return [text], 0
        return [fields[0][2]], 0                             # too narrow even for that: the number alone, never cut

    def _paint_cursor(self, p: QPainter, lay: SimpleNamespace, to_px) -> None:
        if len(self._pins) >= 2:
            self._paint_pins(p, lay, to_px)
            return
        i = self._hover if self._hover is not None else self._pin
        if i is None or not 0 <= i < len(self._x):
            return
        pinned = i == self._pin
        plot = lay.plot
        xv = float(self._x[i])
        gx = round(to_px(xv, lay.y0).x())
        p.save()
        p.setClipRect(plot.adjusted(-2, -2, 2, 2))           # as the curves: the end points sit on the border
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setBrush(Qt.BrushStyle.NoBrush)
        if pinned:                                           # two pixels wide on the pixel grid
            p.setPen(QPen(_CURSOR_PIN, 2.0))
            p.drawLine(QPointF(gx, plot.top()), QPointF(gx, plot.bottom()))
        else:
            p.setPen(QPen(_CURSOR_LIVE, 1.5))
            p.drawLine(QPointF(gx + 0.5, plot.top()), QPointF(gx + 0.5, plot.bottom()))
        r = 5.4 if pinned else 4.6
        for k in range(len(self._series)):
            yv = float(self._values[k, i])
            if math.isfinite(yv):
                p.setPen(QPen(QColor("#ffffff"), 1.6))
                p.setBrush(QColor(SERIES_COLORS[k % len(SERIES_COLORS)]))
                p.drawEllipse(to_px(xv, yv), r, r)
        p.restore()

        mono = self._mono_font()
        mfm = QFontMetrics(mono)
        lh, pad, margin = mfm.height(), 6, 6
        fields = self._fields(i)
        rows, swatch = self._readout_rows(fields, mfm, plot, pad)
        w = 2 * pad + swatch + max(mfm.horizontalAdvance(t) for t in rows)
        h = 2 * pad + lh * len(rows)
        if h > 0.85 * plot.height():                         # no room for even one line: the line and dots say it
            return
        left = QRectF(plot.left() + margin, plot.top() + margin, w, h)
        right = QRectF(plot.right() - margin - w, plot.top() + margin, w, h)

        def over_line(b: QRectF) -> bool:
            return b.left() - 8 <= gx <= b.right() + 8

        box = right if over_line(left) and not over_line(right) else left       # never hide the point itself
        inset = 1.0 if pinned else 0.5                                          # the border's centre, on the pixel grid
        p.setPen(QPen(_CURSOR_PIN if pinned else _AXIS, 2.0 if pinned else 1.0))
        p.setBrush(READOUT_BG)
        p.drawRoundedRect(box.adjusted(inset, inset, -inset, -inset), 3, 3)
        p.setFont(mono)
        for n, text in enumerate(rows):
            row = QRectF(box.left() + pad + swatch, box.top() + pad + n * lh, w - 2 * pad - swatch, lh)
            p.setPen(_TEXT)
            p.drawText(row, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), text)
            if swatch and fields[n][0] is not None:
                p.fillRect(QRectF(box.left() + pad, row.center().y() - 2, 11, 4), fields[n][0])

    # several pins: a line and dots each, a number on each line, one readout box that lists them
    def _draw_reading(self, p: QPainter, lay: SimpleNamespace, to_px, i: int, pinned: bool) -> None:
        """The cursor's line at point ``i`` and a dot on every curve there (a pinned point's drawn thicker)."""
        plot = lay.plot
        gx = round(to_px(float(self._x[i]), lay.y0).x())
        if pinned:
            p.setPen(QPen(_CURSOR_PIN, 2.0))
            p.drawLine(QPointF(gx, plot.top()), QPointF(gx, plot.bottom()))
        else:
            p.setPen(QPen(_CURSOR_LIVE, 1.5))
            p.drawLine(QPointF(gx + 0.5, plot.top()), QPointF(gx + 0.5, plot.bottom()))
        r = 5.4 if pinned else 4.6
        for k in range(len(self._series)):
            yv = float(self._values[k, i])
            if math.isfinite(yv):
                p.setPen(QPen(QColor("#ffffff"), 1.6))
                p.setBrush(QColor(SERIES_COLORS[k % len(SERIES_COLORS)]))
                p.drawEllipse(to_px(float(self._x[i]), yv), r, r)

    def _list_lines(self, entries: list[tuple[str, int]], mfm: QFontMetrics, plot: QRectF, pad: int) -> list[str]:
        """One line per reading, ``tag  label = number ...``, the numbers of each column lined up: every field when
        the axes are wide enough, else the cascade the single readout has (fewer labels, then the first two fields)."""
        fields = [self._fields(i) for _tag, i in entries]
        width = max(len(t) for t, _i in entries)

        def build(names: list[str], count: int) -> list[str]:
            cols = []
            for f in range(count):
                numbers = [fs[f][2] for fs in fields]
                w = max(len(n) for n in numbers)
                cols.append([f"{names[f]}{'' if not names[f] else ' = '}{n.rjust(w)}" for n in numbers])
            return [f"{t.ljust(width)}  " + "  ".join(c[r] for c in cols) for r, (t, _i) in enumerate(entries)]

        labels = [f[1] for f in fields[0]]
        tags = ["x"] + (["y"] if len(labels) == 2 else [f"y{k}" for k in range(1, len(labels))])
        n = len(labels)
        for names, count in ((labels, n), (labels, min(2, n)), (tags, n), (tags, min(2, n)), (["x"], 1)):
            lines = build(names, count)
            if 2 * pad + max(mfm.horizontalAdvance(t) for t in lines) <= plot.width() - 12:
                return lines
        return build([""], 1)                                  # too narrow even for that: the x alone, never cut

    def _paint_pins(self, p: QPainter, lay: SimpleNamespace, to_px) -> None:
        plot = lay.plot
        pins = [i for i in self._pins if 0 <= i < len(self._x)]
        hover = self._hover if self._hover is not None and self._hover not in pins and 0 <= self._hover < len(self._x) else None
        p.save()
        p.setClipRect(plot.adjusted(-2, -2, 2, 2))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setBrush(Qt.BrushStyle.NoBrush)
        if hover is not None:
            self._draw_reading(p, lay, to_px, hover, pinned=False)
        for i in pins:
            self._draw_reading(p, lay, to_px, i, pinned=True)
        p.restore()

        mono = self._mono_font()
        mfm = QFontMetrics(mono)
        lh, pad, margin = mfm.height(), 6, 6
        entries = [(str(k + 1), i) for k, i in enumerate(pins)] + ([(">", hover)] if hover is not None else [])
        lines = self._list_lines(entries, mfm, plot, pad)
        rows = max(0, int((0.85 * plot.height() - 2 * pad) // lh))
        shown = lines[-rows:] if rows else []
        top_box = False
        if shown:
            w = 2 * pad + max(mfm.horizontalAdvance(t) for t in shown)
            h = 2 * pad + lh * len(shown)
            spots = [QPointF(to_px(float(self._x[i]), float(self._values[k, i])))
                     for _t, i in entries for k in range(len(self._series)) if math.isfinite(float(self._values[k, i]))]
            corners = (QRectF(plot.left() + margin, plot.top() + margin, w, h),
                       QRectF(plot.right() - margin - w, plot.top() + margin, w, h),
                       QRectF(plot.left() + margin, plot.bottom() - margin - h, w, h),
                       QRectF(plot.right() - margin - w, plot.bottom() - margin - h, w, h))
            cost = [sum(r.adjusted(-8, -8, 8, 8).contains(q) for q in spots) for r in corners]
            box = corners[cost.index(min(cost))]               # the corner that hides the fewest of the readings' dots
            top_box = box.top() < plot.center().y()
            p.setPen(QPen(_CURSOR_PIN, 2.0))
            p.setBrush(READOUT_BG)
            p.drawRoundedRect(box.adjusted(1.0, 1.0, -1.0, -1.0), 3, 3)
            p.setFont(mono)
            for n, text in enumerate(shown):
                p.setPen(_TEXT)
                p.drawText(QRectF(box.left() + pad, box.top() + pad + n * lh, w - 2 * pad, lh),
                           int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), text)
        # a number on each pinned line: at the edge of the axes the box is not on, else the other, else a little in; never
        # over a dot, the box or another number
        bold = self._fonts()[1]
        bfm = QFontMetrics(bold)
        th = bfm.height() + 2
        dots = [to_px(float(self._x[i]), float(self._values[k, i])) for i in pins for k in range(len(self._series))
                if math.isfinite(float(self._values[k, i]))]
        placed: list[QRectF] = [box] if shown else []
        p.setFont(bold)
        up, down = plot.bottom() - 2 - th, plot.top() + 2
        for k, i in enumerate(pins):
            text = str(k + 1)
            tw = max(th, bfm.horizontalAdvance(text) + 10)
            cx = min(max(round(to_px(float(self._x[i]), lay.y0).x()), plot.left() + tw / 2 + 1), plot.right() - tw / 2 - 1)
            first, other = (up, down) if top_box else (down, up)
            ys = [first + (-j if first == up else j) * (th + 2) for j in range(4)] + \
                 [other + (-j if other == up else j) * (th + 2) for j in range(4)]
            rect = QRectF(cx - tw / 2, ys[0], tw, th)
            for y in ys:
                cand = QRectF(cx - tw / 2, y, tw, th)
                grown = cand.adjusted(-7, -7, 7, 7)
                if not any(cand.intersects(q) for q in placed) and not any(grown.contains(d) for d in dots):
                    rect = cand
                    break
            placed.append(rect)
            p.setPen(QPen(_CURSOR_PIN, 1.5))
            p.setBrush(QColor(10, 12, 24, 232))
            p.drawRoundedRect(rect, 3, 3)
            p.setPen(QColor("#ffffff"))
            p.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), text)

    @staticmethod
    def _draw_series(p: QPainter, x: np.ndarray, y: np.ndarray, color: QColor, to_px) -> None:
        ok = np.isfinite(x) & np.isfinite(y)
        if not ok.any():
            return
        path = QPainterPath()
        pen_down = False
        for xv, yv, good in zip(x.tolist(), y.tolist(), ok.tolist()):
            if not good:
                pen_down = False            # a gap, not a line through zero
                continue
            pt = to_px(xv, yv)
            if pen_down:
                path.lineTo(pt)
            else:
                path.moveTo(pt)
                pen_down = True
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(color, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(path)
        # points that a line cannot show: every point of a short series, and any lone one
        prev = np.concatenate(([False], ok[:-1]))
        nxt = np.concatenate((ok[1:], [False]))
        dots = ok if int(ok.sum()) <= 60 else ok & ~prev & ~nxt
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        for xv, yv in zip(x[dots].tolist(), y[dots].tolist()):
            p.drawEllipse(to_px(xv, yv), 3.2, 3.2)
