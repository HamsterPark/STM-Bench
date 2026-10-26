"""PlotView: a plain line plot for the desktop console (spectra, line profiles, any column pair).

White background, 1-2-5 axis ticks with light grid lines, several series in colours that stay
distinct for the common colour-vision deficiencies, a legend strip when there is more than one
series. No interaction of any kind: the picture is read from a screenshot, so it favours large
text (never below :data:`MIN_FONT_PX`) and thick lines over detail. Non-finite points are
skipped, never drawn as zero; a series with nothing finite, or no series at all, shows a message.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

MIN_FONT_PX = 14
#: Okabe-Ito colours (no yellow: it vanishes on white); more series than colours cycle
SERIES_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")
_BACKGROUND = QColor("#ffffff")
_TEXT = QColor("#111111")
_MUTED = QColor("#555b63")
_GRID = QColor("#e1e5ea")
_AXIS = QColor("#3d434b")
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


def format_number(v: float, decimals: int) -> str:
    """Fixed-point text without a negative zero; scientific notation beyond 1e7 or below 1e-5."""
    if not math.isfinite(v):
        return "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")
    if abs(v) >= 1e7 or (v != 0 and abs(v) < 1e-5):
        return f"{v:.3g}"
    s = f"{v:.{decimals}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def format_ticks(ticks: list[float], step: float) -> list[str]:
    d = tick_decimals(step)
    return [format_number(t, d) for t in ticks]


def _as_floats(a) -> np.ndarray:
    try:
        return np.asarray(a, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return np.empty(0)


class PlotView(QWidget):
    """Line plot of one shared x and any number of named y series. See the module docstring."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._x = np.empty(0)
        self._series: list[tuple[str, np.ndarray]] = []
        self._x_label = ""
        self._y_label = ""
        self._title = ""
        self._message = ""
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)

    # ── data ──
    def set_series(self, x, series: list[tuple[str, "array-like"]], x_label: str, y_label: str,
                   title: str = "") -> None:
        """Plot each ``(name, y)`` against ``x``. A series longer than ``x`` is cut to it (and the
        reverse); NaN and infinities leave gaps; nothing finite shows ``No data``."""
        xs = _as_floats(x)
        self._x = xs
        self._series = [(str(name), _as_floats(y)[:len(xs)]) for name, y in series]
        self._x_label, self._y_label, self._title = str(x_label), str(y_label), str(title)
        self._message = "" if self._extent() is not None else _NO_DATA
        self.update()

    def clear(self, message: str = "") -> None:
        """Empty the plot; ``message`` (default none) is shown in its place."""
        self._x = np.empty(0)
        self._series = []
        self._x_label = self._y_label = self._title = ""
        self._message = str(message)
        self.update()

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

    # ── painting ──
    def _fonts(self) -> tuple[QFont, QFont]:
        regular = QFont(self.font())
        regular.setPixelSize(self._fpx())
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

    def _layout(self, ext, fm: QFontMetrics, bold_fm: QFontMetrics, fpx: int) -> SimpleNamespace:
        W, H = self.width(), self.height()
        pad = max(6, fpx // 2)
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
        title_h = bold_fm.height() + 4 if self._title else 0
        legend = self._legend_rows(fm, W - 2 * pad)
        legend_h = len(legend) * (fh + 3) + (3 if legend else 0)
        top = pad + title_h + legend_h
        bottom = pad + tick_len + gap + fh + ((gap + fh) if self._x_label else 0)
        plot_h = max(20, H - top - bottom)
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
        return SimpleNamespace(W=W, H=H, pad=pad, fh=fh, tick_len=tick_len, gap=gap, plot=plot, legend=legend,
                               title_h=title_h, x0=x0, x1=x1, y0=y0, y1=y1, xt=xt, xlabels=xlabels,
                               yt=yt, ylabels=ylabels, top=top)

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), _BACKGROUND)
        ext = self._extent()
        regular, bold = self._fonts()
        if ext is None:
            if self._message:
                p.setFont(regular)
                p.setPen(_MUTED)
                p.drawText(QRectF(self.rect()).adjusted(12, 12, -12, -12),
                           int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap), self._message)
            p.end()
            return
        fm, bold_fm = QFontMetrics(regular), QFontMetrics(bold)
        lay = self._layout(ext, fm, bold_fm, self._fpx())
        plot = lay.plot
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setFont(regular)

        def to_px(xv: float, yv: float) -> QPointF:
            return QPointF(plot.left() + (xv - lay.x0) / (lay.x1 - lay.x0) * plot.width(),
                           plot.bottom() - (yv - lay.y0) / (lay.y1 - lay.y0) * plot.height())

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
        if self._x_label:
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
        if self._title:
            r = QRectF(lay.pad, lay.pad, lay.W - 2 * lay.pad, lay.title_h)
            p.drawText(r, int(Qt.AlignmentFlag.AlignCenter),
                       bold_fm.elidedText(self._title, Qt.TextElideMode.ElideRight, int(r.width())))

        # legend strip between the title and the axes
        p.setFont(regular)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        y = lay.pad + lay.title_h
        for row in lay.legend:
            x = lay.pad
            for name, color, w in row:
                p.setPen(QPen(color, 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                p.drawLine(QPointF(x + 2, y + lay.fh / 2 + 1), QPointF(x + 24, y + lay.fh / 2 + 1))
                p.setPen(_TEXT)
                p.drawText(QRectF(x + 32, y, w + 4, lay.fh), int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), name)
                x += 26 + 6 + w + 18
            y += lay.fh + 3
        p.end()

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
