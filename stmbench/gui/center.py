"""The middle column: the viewer, its controls, the markers and the strip of recent frames.

Everything here is read off the data the instrument gave: the cursor's coordinates, a marker's value,
a profile along a line. Nothing is found, fitted or suggested: no atom finder, no drift estimate. A
marker is the pixel the reader clicked, kept in scan coordinates at that pixel's centre, with the numbers
instrument software shows for any point (position, value, distance to the previous one): the position and
the value belong to the same point, so a value read at a marker's coordinates is the value the instrument
measured there. The three ``← marker`` buttons copy a marker's coordinates, as the table shows them, into the
fields of the action or report that takes them; they run nothing.
"""
from __future__ import annotations

import math
import re

import numpy as np
from PySide6.QtCore import QEvent, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QPushButton, QSizePolicy,
                               QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from . import fmt, style
from .controls import ElidedLabel, Segmented, ThumbButton, polish
from .labels import S, one, tr
from .widgets.frame_view import FrameView, frame_thumbnail
from .widgets.plot_view import PlotView

#: spectrum columns as the viewer plots them: (button text, factor from the stored unit, unit); a move trace's
#: own are the distance along the move and the time since it began
SPECTRUM_COLUMNS = {
    "v": ("V", 1.0, "V"), "i": ("I", 1e12, "pA"), "didv": ("dI/dV", 1e9, "nS"), "z_rel": ("z rel", 1e12, "pm"),
    "z_abs": ("z", 1e9, "nm"), "df": ("Δf", 1.0, "Hz"), "amp": ("amplitude", 1e12, "pm"),
    "dist": ("distance", 1e9, "nm"), "t": ("time", 1.0, "s"),
}
_X_FIRST = ("v", "z_rel", "z_abs")
_Y_FIRST = ("didv", "df", "i")
#: a move trace is read along the move, or in time, against what the instrument recorded there: Z, then current
MOVE_X, MOVE_Y = ("dist", "t"), ("z_abs", "i")
#: the columns that are words, not symbols, with the label that names them in the window's language
_COLUMN_WORDS = {"dist": "col_dist", "t": "col_t"}
#: a spectrum has at least this many points (STS and Z sweeps take eight or more), a move trace two
_SPECTRUM_MIN, _MOVE_MIN = 8, 2
#: the thumbnails of the strip: at most this many at once, at least this many whatever the window's height (the strip
#: shows as many as its height holds, whole)
MAX_THUMBS = 8
MIN_THUMBS = 3
THUMB_PX = 64
#: the paging buttons' least height (the style sheet's small button)
NAV_PX = 28
#: the marker table shows this many rows before it scrolls: its strip is exactly a header, the rows and the frame
#: (a viewport that is not a whole number of rows leaves the first row cut off under the header when scrolled)
MARKER_ROWS = 6
ROW_PX, HEADER_PX = 26, 24
#: the marker table's columns, by their place: the start time of the frame a marker was placed on is the sixth
START_COLUMN = 5
MARKER_STRIP_PX = HEADER_PX + MARKER_ROWS * ROW_PX + 2


def column_label(key: str, lang: str = "en") -> str:
    name, _, unit = SPECTRUM_COLUMNS[key]
    if key in _COLUMN_WORDS:
        name = tr(_COLUMN_WORDS[key], lang)
    return name if name == unit else f"{name} ({unit})"


def spectrum_columns(spec: dict) -> list[str]:
    """The plottable columns of a spectrum (or of a move trace, which may be as short as two points), in a
    fixed order."""
    least = _MOVE_MIN if spec.get("kind") == "move" else _SPECTRUM_MIN
    return [k for k in SPECTRUM_COLUMNS if k in spec and np.size(spec[k]) >= least]


def axis_columns(spec: dict) -> tuple[list[str], list[str]]:
    """The columns offered for the x axis and for the y axis: every plottable column for either, except a move
    trace's, whose x is the distance along the move or the time and whose y is Z or the current."""
    cols = spectrum_columns(spec)
    if spec.get("kind") == "move":
        return [c for c in MOVE_X if c in cols], [c for c in MOVE_Y if c in cols]
    return cols, cols


def default_axes(spec: dict) -> tuple[str, str]:
    """The sweep axis against the first signal that varies (a move trace: Z against the distance along it)."""
    if spec.get("kind") == "move":
        xs, ys = axis_columns(spec)
        return xs[0], ys[0]
    cols = spectrum_columns(spec)
    x = next((c for c in _X_FIRST if c in cols), cols[0])
    ys = [c for c in cols if c != x]
    varying = [c for c in _Y_FIRST if c in ys and np.ptp(np.nan_to_num(np.asarray(spec[c], float))) > 0]
    y = varying[0] if varying else (next((c for c in _Y_FIRST if c in ys), ys[0] if ys else x))
    return x, y


def thumb_pixmap(frame: dict, size: int) -> QPixmap:
    """A frame's thumbnail with the last digits of its name in one corner."""
    pm = QPixmap.fromImage(frame_thumbnail(frame, size))
    m = re.search(r"(\d+)(?=\.\w+$)", str(frame.get("frame", "")))
    text = (m.group(1)[-3:] if m else str(frame.get("frame", ""))[:3])
    p = QPainter(pm)
    p.setFont(style.sans_font(15, bold=True))
    w = p.fontMetrics().horizontalAdvance(text) + 6
    h = p.fontMetrics().height() + 2
    p.fillRect(QRect(0, pm.height() - h, w, h), QColor(0, 0, 0, 190))
    p.setPen(QColor("#ffffff"))
    p.drawText(QRect(0, pm.height() - h, w, h), Qt.AlignmentFlag.AlignCenter, text)
    p.end()
    return pm


class MarkerTable(QTableWidget):
    """The marker list. Every row is reachable: a notch of the wheel over it moves it exactly one row (an item
    view's own wheel moves as many rows as the desktop's setting says, three by default, which an agent that counts
    notches cannot know), finer wheels add up to whole rows, and Ctrl with the wheel does nothing. The scroll
    steps are rows, whatever the style's default (a table that scrolls by pixel would leave the first row cut off
    under the header, which the strip's height rule above is made to prevent)."""

    NOTCH = 120                                   # a wheel's angle delta for one notch

    def __init__(self, rows: int, cols: int, parent: QWidget | None = None):
        super().__init__(rows, cols, parent)
        self._wheel = 0
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerItem)
        self.verticalScrollBar().setSingleStep(1)

    def first_row(self) -> int:
        """The row at the top of what is shown (-1 when there are no rows)."""
        return self.rowAt(0)

    def last_row(self) -> int:
        """The last row that is shown whole (-1 when there are no rows)."""
        return self.rowAt(self.viewport().height() - 1)

    def wheelEvent(self, ev) -> None:
        ev.accept()                               # taken whatever happens: a parent never scrolls
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            return
        self._wheel += ev.angleDelta().y()
        notches = int(self._wheel / self.NOTCH)   # toward zero: a partial notch waits for the rest of it
        if not notches:
            return
        self._wheel -= notches * self.NOTCH
        bar = self.verticalScrollBar()
        before = bar.value()
        bar.setValue(before - notches)
        if bar.value() == before:                 # at an end: the turns beyond it are not saved up
            self._wheel = 0


class _MarkedThumb(ThumbButton):
    """A thumbnail that says, in the window's accent colour, that its frame is the one on show (the toolkit's own
    checked look is a slightly darker edge, which in a 1280x720 picture is not seen)."""

    def paintEvent(self, ev) -> None:
        super().paintEvent(ev)
        if self.isChecked():
            w = float(style.px(3))
            pen = QPen(QColor(style.COLORS["accent"]), w)
            pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
            p = QPainter(self)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(self.rect()).adjusted(w / 2, w / 2, -w / 2, -w / 2))
            p.end()


class FrameStrip(QWidget):
    """The thumbnails beside the viewer: every frame of the episode, newest first, a window of them at a time.

    The strip used to hold the newest six frames and nothing else; an older frame was reached from the Data tab
    (screen trial 5, steps 42-45: "scrolling the wheel over it did nothing"). Now the window moves: a notch of the
    wheel over the strip is one thumbnail (finer wheels add up, Ctrl does nothing), and when more frames exist than
    the strip shows, two buttons move it a page and say how many frames lie beyond their end. How many thumbnails
    show is what the strip's height holds, whole, between the two buttons (their room is kept when they are not
    there, so that the thumbnails do not move when the seventh frame comes).

    The frame on show is marked when it is in the window, and it is brought into it when it changes (a new scan, a
    frame picked from the Data tab); a read of the data list leaves the reader's place where it is."""

    picked = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("view.strip")
        self._thumb = style.px(THUMB_PX + 8)
        self.setFixedWidth(self._thumb)                  # always as wide as a thumbnail: the image does not jump
        self._items: list[tuple[str, QPixmap]] = []
        self._first = 0                                  # index of the thumbnail at the top
        self._current = ""                               # the frame on show
        self._revealed = ""                              # the one that was last brought into the window
        self._wheel = 0
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(style.px(3))
        self.newer = self._nav("strip.newer", lambda: self.scroll_by(-self.capacity()))
        self.older = self._nav("strip.older", lambda: self.scroll_by(self.capacity()))
        self.thumbs = [_MarkedThumb() for _ in range(MAX_THUMBS)]
        col.addWidget(self.newer)
        for i, b in enumerate(self.thumbs):
            b.setObjectName(f"thumb.{i}")
            b.clicked.connect(lambda _=False, btn=b: self.picked.emit(btn.name))
            b.hide()
            col.addWidget(b)
        col.addStretch(1)
        col.addWidget(self.older)
        for b in (self.newer, self.older, *self.thumbs):
            b.installEventFilter(self)                   # a button does not pass the wheel on to the strip: take it from it
        # the strip must not ask the column for the height of a full window of thumbnails: it shows as many as fit
        self.setMinimumHeight(2 * style.px(NAV_PX) + MIN_THUMBS * (self._thumb + style.px(3)))

    def _nav(self, name: str, fn) -> QPushButton:
        b = QPushButton()
        b.setObjectName(name)
        b.setProperty("role", "small")
        b.clicked.connect(lambda _=False: fn())
        sp = b.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        b.setSizePolicy(sp)
        b.hide()
        return b

    # ── what it shows ──
    def capacity(self) -> int:
        """How many thumbnails the strip's height holds, whole, between its two buttons."""
        gap = self.layout().spacing()
        room = self.height() - 2 * self._nav_height() - gap
        return max(MIN_THUMBS, min(MAX_THUMBS, room // (self._thumb + gap)))

    def _nav_height(self) -> int:
        self.newer.ensurePolished()                      # the style sheet's minimum height is in the size hint only then
        return max(self.newer.sizeHint().height(), style.px(NAV_PX))

    def first_index(self) -> int:
        """How many frames are newer than the one at the top."""
        return self._first

    def names(self) -> list[str]:
        return [n for n, _ in self._items]

    def set_items(self, items: list[tuple[str, QPixmap]], current: str = "") -> None:
        """Every frame that has a thumbnail, newest first; ``current`` is the frame on show (or none)."""
        anchor = self._items[self._first][0] if self._first < len(self._items) else ""
        self._items = list(items)
        names = self.names()
        if self._first and anchor in names:              # the window stays on the same frames when new ones come first
            self._first = names.index(anchor)
        self._current = current
        self._refresh()

    def set_current(self, name: str) -> None:
        self._current = name
        self._refresh()

    def scroll_by(self, k: int) -> None:
        """Move the window ``k`` thumbnails toward the older frames (negative: toward the newer), as far as the ends."""
        self._first += int(k)
        self._refresh()

    def _refresh(self) -> None:
        n = self.capacity()
        names = self.names()
        if self._current and self._current != self._revealed and self._current in names:
            idx = names.index(self._current)             # a frame that has just come on show: into the window
            if idx < self._first:
                self._first = idx
            elif idx >= self._first + n:
                self._first = idx - n + 1
            self._revealed = self._current
        elif not self._current:
            self._revealed = ""
        self._first = max(0, min(self._first, len(names) - n))
        for i, b in enumerate(self.thumbs):
            k = self._first + i
            if i < n and k < len(names):
                b.set_thumb(names[k], self._items[k][1], style.px(THUMB_PX))
                b.setChecked(names[k] == self._current)
            else:
                b.setChecked(False)
                b.set_thumb("", None, style.px(THUMB_PX))
        newer, older = self._first, max(0, len(names) - self._first - n)
        for b, count, glyph in ((self.newer, newer, "▲"), (self.older, older, "▼")):
            b.setText(f"{glyph} {count}" if count else glyph)
            b.setEnabled(count > 0)
            b.setVisible(len(names) > n)

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._refresh()

    def eventFilter(self, obj, ev) -> bool:
        if ev.type() == QEvent.Type.Wheel and obj is not self:
            self.wheelEvent(ev)                          # over a thumbnail or a paging button it is the strip's wheel too
            return True
        return False

    def wheelEvent(self, ev) -> None:
        ev.accept()                                      # taken whatever happens: nothing behind it scrolls
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            return
        self._wheel += ev.angleDelta().y()
        notches = int(self._wheel / 120)
        if not notches:
            return
        self._wheel -= notches * 120
        before = self._first
        self.scroll_by(-notches)                         # up: toward the newer frames, which are at the top
        if self._first == before:                        # at an end: the turns beyond it are not saved up
            self._wheel = 0


class CenterColumn(QWidget):
    view_changed = Signal()                       # the reader toggled channel or direction
    thumb_clicked = Signal(str)
    tip_from_marker = Signal(float, float)
    scan_from_marker = Signal(float, float)
    report_from_marker = Signal(float, float, str)
    notice = Signal(str)

    def __init__(self, lang: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.lang = lang
        self._spec: dict | None = None
        self._has_frame = False                       # a frame has been shown: before that a click on the view marks nothing
        self._frame_start: dict[str, float] = {}      # frame name -> instrument time its scan began (s)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(style.px(4))

        # ── what is shown ──
        # what is shown, in two lines that each fit the column (the frame's name, and its geometry): one line
        # long enough for the geometry was cut off with an ellipsis on a wider font
        self.info = QLabel()
        self.info.setObjectName("view.info")
        self.info.setFont(style.sans_font(17))
        self.info.setWordWrap(True)
        self.info.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.info.setFixedHeight(2 * QFontMetrics(self.info.font()).lineSpacing() + style.px(4))
        v.addWidget(self.info)

        self.toolbars = QStackedWidget()
        self.toolbars.setObjectName("view.toolbars")
        self.toolbars.addWidget(self._frame_toolbar())
        self.toolbars.addWidget(self._spectrum_toolbar())
        v.addWidget(self.toolbars)
        v.addWidget(self._tools_row())

        # ── the image (or the spectrum), with the last frames in a column beside it ──
        view = QHBoxLayout()
        view.setSpacing(style.px(4))
        self.stack = QStackedWidget()
        self.stack.setObjectName("view.stack")
        self.frame_view = FrameView()
        self.frame_view.setObjectName("view.frame")
        self.frame_view.set_placeholder(tr("no_view", lang))
        self.plot = PlotView()
        self.plot.setObjectName("view.plot")
        self.stack.addWidget(self.frame_view)
        self.stack.addWidget(self.plot)
        # while a scan runs (or waits) the viewer says so over whatever it shows; the pointer passes through it
        self.overlay = QLabel(tr("scan_running", lang), self.stack)
        self.overlay.setObjectName("view.busy")
        self.overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.overlay.setFont(style.sans_font(30, bold=True))
        self.overlay.setProperty("role", "overlay")
        self.overlay.hide()
        self.stack.installEventFilter(self)
        self.frame_view.installEventFilter(self)
        view.addWidget(self.stack, 1)
        self.strip = FrameStrip()
        self.strip.picked.connect(self.thumb_clicked)
        self.thumbs = self.strip.thumbs
        view.addWidget(self.strip)
        v.addLayout(view, 1)
        self.readout = ElidedLabel()
        self.readout.setObjectName("view.readout")
        self.readout.setFont(style.mono_font(17))
        self.readout.setMinimumHeight(style.px(26))
        v.addWidget(self.readout)

        # ── markers and the profile ──
        # (the table's font is in the style sheet: a font set on a widget was lost when an ancestor was re-parented)
        strip = QWidget()
        strip.setFixedHeight(style.px(MARKER_STRIP_PX))
        bottom = QHBoxLayout(strip)
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(style.px(8))
        self.table = self._marker_table(strip)
        bottom.addWidget(self.table, 7)                  # seven columns want almost four fifths of the strip
        self.small_plot = PlotView()
        self.small_plot.setObjectName("view.profile")
        self.small_plot.setMinimumWidth(style.px(180))
        self.small_plot.clear(tr("profile", lang))
        bottom.addWidget(self.small_plot, 2)
        v.addWidget(strip)
        buttons = QHBoxLayout()
        buttons.setSpacing(style.px(8))
        self.to_tip = self._marker_button("markers.tip", "mk_tip")
        self.to_scan = self._marker_button("markers.scan", "mk_scan")
        self.to_report = self._marker_button("markers.report", "mk_report")
        self.delete_marker = self._marker_button("markers.delete", "delete_marker")
        for b in (self.to_tip, self.to_scan, self.to_report, self.delete_marker):
            buttons.addWidget(b, len(b.text().split(chr(10))[-1]))        # room by the length of the English text
        v.addLayout(buttons)
        self.to_tip.clicked.connect(lambda: self._copy_marker(lambda m: self.tip_from_marker.emit(m["x_nm"], m["y_nm"])))
        self.to_scan.clicked.connect(lambda: self._copy_marker(lambda m: self.scan_from_marker.emit(m["x_nm"], m["y_nm"])))
        self.to_report.clicked.connect(
            lambda: self._copy_marker(lambda m: self.report_from_marker.emit(m["x_nm"], m["y_nm"], m.get("frame", ""))))
        self.delete_marker.clicked.connect(self._delete_selected)

        fv = self.frame_view
        fv.zoom_changed.connect(self.zoom.set_value)       # the wheel zooms too: the buttons show where it is
        fv.cursor_moved.connect(self._cursor)
        fv.markers_changed.connect(self._markers_changed)
        fv.marker_added.connect(self._marker_added)
        fv.profile_changed.connect(self._profile)
        self._show_readout("")                             # the hint, until the pointer is over a frame

    def set_scan_pending(self, pending: bool) -> None:
        """Show or hide "scan running..." over the viewer."""
        self.overlay.setGeometry(self.stack.rect())
        self.overlay.setVisible(pending)
        if pending:
            self.overlay.raise_()

    def eventFilter(self, obj, ev) -> bool:
        if obj is self.stack and ev.type() == QEvent.Type.Resize:
            self.overlay.setGeometry(self.stack.rect())
        elif (obj is self.frame_view and ev.type() == QEvent.Type.MouseButtonPress and not self._has_frame
              and ev.button() == Qt.MouseButton.LeftButton):
            self.notice.emit(tr("no_frame_click", self.lang))        # said, not lost: nothing was marked
        return False

    # ── building ──
    def _seg(self, options, name: str, value) -> Segmented:
        # the viewer's toolbar names its buttons in one language: with the Contrast switch the second row would
        # not fit the column with Chinese and English on every button
        s = Segmented([(val, one(key, self.lang), f"{name}.{val}") for val, key in options], value)
        s.setObjectName(name)
        return s

    def _row(self, *items) -> QHBoxLayout:
        h = QHBoxLayout()
        h.setSpacing(style.px(8))
        h.setContentsMargins(0, 0, 0, 0)
        for it in items:
            h.addWidget(it)
        h.addStretch(1)
        return h

    def _caption(self, key: str) -> QLabel:
        lab = QLabel(one(key, self.lang))
        lab.setProperty("role", "label")
        return lab

    def _frame_toolbar(self) -> QWidget:
        w = QWidget()
        col = QVBoxLayout(w)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(style.px(4))
        self.channel = self._seg([("z", "ch_z"), ("current", "ch_current"), ("didv", "ch_didv")], "view.channel", "z")
        self.direction = self._seg([("forward", "dir_forward"), ("backward", "dir_backward")], "view.direction",
                                   "forward")
        self.flatten = self._seg([("line", "flat_line"), ("plane", "flat_plane"), ("none", "flat_none")],
                                 "view.flatten", "line")
        self.contrast = self._seg([("auto", "contrast_auto"), ("full", "contrast_full"), ("visible", "contrast_visible")],
                                  "view.contrast", "auto")
        self.zoom = self._seg([(1, "zoom_fit"), (2, "zoom_2"), (4, "zoom_4"), (8, "zoom_8")], "view.zoom", 1)
        # two rows: what is shown, and how its grey scale and size are set (the third row, what a drag on the image
        # does and the markers, is outside this page: see _tools_row)
        col.addLayout(self._row(self.channel, self.direction, self._caption("lbl_flatten"), self.flatten))
        col.addLayout(self._row(self._caption("lbl_contrast"), self.contrast, self._caption("lbl_zoom"), self.zoom))
        self.channel.changed.connect(lambda _v: self.view_changed.emit())
        self.direction.changed.connect(lambda _v: self.view_changed.emit())
        self.flatten.changed.connect(lambda m: self.frame_view.set_flatten(m))
        self.contrast.changed.connect(lambda m: self.frame_view.set_contrast(m))
        self.zoom.changed.connect(self._zoom_clicked)
        return w

    def _zoom_clicked(self, level: int) -> None:
        """A zoom button: about the selected marker (the table's selection) when there is one, which the view is
        then centred on (screen trial 4, step 84: it always centred on the frame, and the atom was cut at the
        edge); else about the centre of the view as it is."""
        m = self.frame_view.selected_marker()
        self.frame_view.set_zoom(level, focus=(m["x_nm"], m["y_nm"]) if m is not None else None)

    def _tools_row(self) -> QWidget:
        """What a drag on the image does, and Clear markers: a row that stays whatever the viewer shows.

        It is outside the pages of the toolbar stack because the markers are not the frame's: they are kept in
        scan coordinates and come back on the next frame. In screen trial 5 (step 39) the agent clicked where Clear
        markers had been while a move trace was on show, the frame toolbar was not there, the click reached empty
        toolbar, and the old markers came back on the next frame. Here the button is in the same place in every view."""
        w = QWidget()
        w.setObjectName("view.tools")
        self.tool = self._seg([("marker", "tool_marker"), ("profile", "tool_profile"), ("pan", "tool_pan")],
                              "view.tool", "marker")
        self.clear_markers = QPushButton(one("clear_markers", self.lang))
        self.clear_markers.setObjectName("view.clear_markers")
        # the readings pinned on a plot, which the plot's own toolbar (the axes) has no room for: the same place always
        self.clear_readings = QPushButton(one("clear_readings", self.lang))
        self.clear_readings.setObjectName("view.clear_readings")
        w.setLayout(self._row(self._caption("lbl_tool"), self.tool, self.clear_markers, self.clear_readings))
        self.tool.changed.connect(lambda t: (self.frame_view.set_tool(t),
                                             self.readout.property("tone") == "note" and self._show_readout("")))
        self.clear_markers.clicked.connect(lambda: (self.frame_view.clear_markers(), self.frame_view.clear_profile()))
        self.clear_readings.clicked.connect(lambda: (self.plot.clear_pins(), self.small_plot.clear_pins()))
        return w

    def _spectrum_toolbar(self) -> QWidget:
        w = QWidget()
        col = QVBoxLayout(w)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(style.px(4))
        self.spec_x_row, self.spec_y_row = QHBoxLayout(), QHBoxLayout()
        self.spec_x: Segmented | None = None
        self.spec_y: Segmented | None = None
        for row, key in ((self.spec_x_row, "lbl_x"), (self.spec_y_row, "lbl_y")):
            row.setSpacing(style.px(8))
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(self._caption(key))
            row.addStretch(1)
            col.addLayout(row)
        return w

    def _marker_table(self, parent: QWidget) -> QTableWidget:
        cols = ["mk_num", "mk_x", "mk_y", "mk_value", "mk_frame", "mk_start", "mk_dist"]
        t = MarkerTable(0, len(cols), parent)
        t.setObjectName("markers.table")
        # one line each, in one language: a header of two lines took a third of the strip in Chinese, and
        # the table holds five rows (30 px each under a 24 px header) in the strip's 176 px
        t.setHorizontalHeaderLabels([one(c, self.lang) for c in cols])
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        t.setAlternatingRowColors(True)
        t.verticalHeader().hide()
        t.verticalHeader().setMinimumSectionSize(1)       # the header's own minimum (about a font's height) would make rows taller
        t.verticalHeader().setDefaultSectionSize(style.px(ROW_PX))
        t.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        t.horizontalHeader().setFixedHeight(style.px(HEADER_PX))
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        t.horizontalHeader().setStretchLastSection(True)
        t.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # never a bar along the bottom: it would take rows' room, and the strip's rows must stay whole
        t.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        t.cellClicked.connect(lambda r, _c: self.frame_view.select_marker(r))
        return t

    def _marker_button(self, name: str, key: str) -> QPushButton:
        en, zh = S[key]
        # in Chinese the two names go on two lines: four of these side by side must fit a column that is
        # not wide, each as wide as its own text needs and the rest shared
        b = QPushButton(en if self.lang == "en" else f"{zh}\n{en}")
        b.setObjectName(name)
        b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if self.lang != "en":
            b.setFixedHeight(style.px(52))
        return b

    # ── what is shown ──
    def current_channel(self) -> str:
        return self.channel.value() or "z"

    def current_direction(self) -> str:
        return self.direction.value() or "forward"

    def show_frame(self, frame: dict, channels: list[str] | None = None, meta: dict | None = None) -> None:
        """Show an ``Instrument.frame()`` dict. ``channels`` are the ones the frame has; ``meta`` its entry
        of the data list (its start time)."""
        self.toolbars.setCurrentIndex(0)
        self.stack.setCurrentIndex(0)
        self._has_frame = True
        chans = channels or ["z", "current"]
        self.channel.set_enabled_values(chans)
        self.channel.set_value(frame.get("channel", "z"))
        self.direction.set_value(frame.get("direction", "forward"))
        self.frame_view.set_frame(frame)
        t0 = (meta or {}).get("t_start_s", frame.get("t_start_s"))
        cx, cy = frame["center_nm"]
        self.info.setText(
            f"{frame['frame']}   start {fmt.hms(t0) if t0 is not None else '-'}\n"
            f"centre ({fmt.num(cx, 5)}, {fmt.num(cy, 5)}) nm · size {fmt.num(frame['size_nm'], 5)} nm · "
            f"{frame['pixels']} px · angle {fmt.num(frame['angle_deg'], 4)}°")
        self._show_readout("")
        self.strip.set_current(frame["frame"])

    def show_spectrum(self, spec: dict, axes: tuple[str, str] | None = None) -> None:
        """Show a spectrum, or a move trace (``kind`` "move"), plotted against the columns the reader picks."""
        self._spec = spec
        xcols, ycols = axis_columns(spec)
        if not xcols or not ycols:
            self.plot.clear(tr("no_view", self.lang))
            self.toolbars.setCurrentIndex(1)
            self.stack.setCurrentIndex(1)
            self._show_readout("")
            return
        xk, yk = axes or default_axes(spec)
        self._rebuild_axes(xcols, ycols, xk, yk)
        self.toolbars.setCurrentIndex(1)
        self.stack.setCurrentIndex(1)
        self.strip.set_current("")
        self._show_readout("")
        self.info.setText(self._spectrum_info(spec, int(np.size(spec[xcols[0]]))))
        self._draw_spectrum(xk, yk)

    @staticmethod
    def _spectrum_info(spec: dict, points: int) -> str:
        """Two lines: the record's name and when it was taken, then what it is. A move trace also says what the
        instrument recorded of the move (where from and to, how fast, at what bias and setpoint, feedback on or
        off), the file's own header, and nothing that was read off the curves."""
        first = f"{spec['name']}   at {fmt.hms(spec['t_s'])}"
        if spec.get("kind") != "move":
            x, y = spec["tip_nm"]
            return f"{first}\n{spec['kind']} · tip ({x:.3f}, {y:.3f}) nm · {points} points"
        junction = [fmt.volts(spec["bias_v"]) if "bias_v" in spec else "",
                    fmt.picoamps(spec["setpoint_pa"]) if "setpoint_pa" in spec else "",
                    f"feedback {fmt.on_off(spec['feedback'])}" if "feedback" in spec else ""]
        path = "move"
        if "from_nm" in spec and "to_nm" in spec:                  # five digits like the frame's centre: the line must fit
            (x0, y0), (x1, y1) = spec["from_nm"], spec["to_nm"]    # the narrowest centre column even at (-150.12, 150.12)
            path += f" · ({fmt.num(x0, 5)}, {fmt.num(y0, 5)}) → ({fmt.num(x1, 5)}, {fmt.num(y1, 5)}) nm"
        if "speed_nm_s" in spec:
            path += f" · {fmt.num(spec['speed_nm_s'], 4)} nm/s"
        return " · ".join([first, *[j for j in junction if j]]) + f"\n{path} · {points} points"

    def _rebuild_axes(self, xcols: list[str], ycols: list[str], xk: str, yk: str) -> None:
        for attr, row, name, cols, val in (("spec_x", self.spec_x_row, "view.spec_x", xcols, xk),
                                           ("spec_y", self.spec_y_row, "view.spec_y", ycols, yk)):
            old = getattr(self, attr)
            if old is not None:
                row.removeWidget(old)
                old.deleteLater()
            seg = Segmented([(c, column_label(c, self.lang), f"{name}.{c}") for c in cols], val)
            seg.setObjectName(name)
            seg.changed.connect(lambda _v: self._axes_clicked())
            row.insertWidget(1, seg)
            setattr(self, attr, seg)

    def _axes_clicked(self) -> None:
        if self._spec is not None and self.spec_x and self.spec_y:
            self._draw_spectrum(self.spec_x.value(), self.spec_y.value())

    def _draw_spectrum(self, xk: str, yk: str) -> None:
        spec = self._spec
        x = np.asarray(spec[xk], float) * SPECTRUM_COLUMNS[xk][1]
        y = np.asarray(spec[yk], float) * SPECTRUM_COLUMNS[yk][1]
        xl, yl = column_label(xk, self.lang), column_label(yk, self.lang)
        self.plot.set_series(x, [(yl, y)], xl, yl, spec["name"])

    def current_spectrum_axes(self) -> tuple[str, str] | None:
        if self._spec is None or not (self.spec_x and self.spec_y):
            return None
        return self.spec_x.value(), self.spec_y.value()

    def set_thumbs(self, items: list[tuple[str, QPixmap]], current: str = "") -> None:
        """Every frame that has a thumbnail, newest first; ``current`` is the frame on show (none: ``""``)."""
        self.strip.set_items(items, current)

    # ── readouts ──
    def _cursor(self, c: dict) -> None:
        if not c:
            self._show_readout("")
            return
        val = c.get("value")
        # the pixel under the pointer, then where its centre lies and what it holds: the numbers a click there marks
        text = f"pixel col {c['col']}, row {c['row']}   centre x {c['x_nm']:.3f} nm   y {c['y_nm']:.3f} nm"
        if val is not None and math.isfinite(float(val)):
            text += f"   {float(val):.2f} {c.get('unit', '')}"
        self._show_readout(text)

    def _hint(self) -> str:
        """What the readout line says while it has no reading: how the viewer is used (what the pointer and a click
        do in the view on show, and with the tool picked), nothing about the experiment. One language: the line is
        a single line of the width of the column, which both at once would not fit."""
        if self.stack.currentIndex() == 1:
            return one("hint_plot", self.lang)
        tool = self.tool.value()
        return one({"profile": "hint_image_profile", "pan": "hint_image_pan"}.get(tool, "hint_image"), self.lang)

    def _show_readout(self, text: str) -> None:
        """The reading, or, with none, the usage hint in the muted colour (the line keeps its font and height, so
        nothing moves when the pointer enters or leaves)."""
        self.readout.set_full_text(text or self._hint())
        tone = "" if text else "note"
        if (self.readout.property("tone") or "") != tone:
            self.readout.setProperty("tone", tone)
            polish(self.readout)

    def _marker_added(self, _m: dict) -> None:
        n = self.table.rowCount()
        if n:
            self.table.selectRow(n - 1)
            self.table.scrollToItem(self.table.item(n - 1, 0))        # the new one is in view even past the fifth

    def set_frame_starts(self, starts: dict[str, float]) -> None:
        """When each saved frame's scan began, for the table's start column: the frame a marker was placed on,
        as the Data tab gives it. Only that column is written: the data list is read after every action, and a
        table that scrolled back to its selection each time would undo the reader's own scrolling."""
        self._frame_start = dict(starts)
        for i, m in enumerate(self.frame_view.markers()):
            item = self.table.item(i, START_COLUMN)
            if item is not None:
                item.setText(self._started(m))

    def _started(self, m: dict) -> str:
        started = self._frame_start.get(str(m.get("frame", "")))
        return fmt.hms(started) if started is not None else "-"

    def _markers_changed(self, markers: list) -> None:
        t = self.table
        t.setRowCount(len(markers))
        prev = None
        for i, m in enumerate(markers):
            d = "" if prev is None else f"{math.hypot(m['x_nm'] - prev['x_nm'], m['y_nm'] - prev['y_nm']):.3f}"
            val = m.get("value")
            vtxt = f"{float(val):.2f} {m.get('unit', '')}" if val is not None and math.isfinite(float(val)) else "-"
            for c, text in enumerate([str(i + 1), f"{m['x_nm']:.3f}", f"{m['y_nm']:.3f}", vtxt, str(m.get("frame", "")),
                                      self._started(m), d]):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | (Qt.AlignmentFlag.AlignLeft if c == 4
                                                                       else Qt.AlignmentFlag.AlignRight))
                t.setItem(i, c, item)
            prev = m
        sel = self.frame_view.selected_marker()
        if sel is not None and markers:
            idx = next((i for i, m in enumerate(markers) if m["x_nm"] == sel["x_nm"] and m["y_nm"] == sel["y_nm"]),
                       len(markers) - 1)
            t.selectRow(idx)
            t.scrollToItem(t.item(idx, 0))
        else:
            t.clearSelection()

    def _profile(self, p: dict) -> None:
        if not p:
            self.small_plot.clear(tr("profile", self.lang))
            return
        length = float(p["s_nm"][-1]) if p.get("s_nm") else 0.0
        # the frame is the one on show (a profile drawn on another is dropped), so the title is the length alone:
        # the plot is a third of the column wide
        self.small_plot.set_series(p["s_nm"], [("profile", p["values"])], "distance (nm)", p.get("unit", ""),
                                   f"A→B {length:.2f} nm")

    def show_trace(self, title: str, lists: dict[str, list[float]]) -> None:
        """Plot the equal-length numeric lists of an action's reply: the first against the others when
        there are several, else against the index. Any reply is plotted the same way."""
        names = list(lists)
        if len(names) >= 2:
            x, series, xl = lists[names[0]], [(n, lists[n]) for n in names[1:]], names[0]
        else:
            x, series, xl = list(range(len(lists[names[0]]))), [(names[0], lists[names[0]])], "index"
        self.small_plot.set_series(x, series, xl, "", f"{tr('trace', self.lang)}: {title}")

    def selected_row(self) -> int | None:
        """The marker the table has selected (0-based), if any."""
        rows = self.table.selectionModel().selectedRows()
        return rows[0].row() if rows else None

    def _delete_selected(self) -> None:
        row = self.selected_row()
        if row is None or self.frame_view.selected_marker() is None:
            self.notice.emit(tr("mk_none", self.lang))
            return
        self.frame_view.remove_marker(row)

    def _copy_marker(self, deliver) -> None:
        m = self.frame_view.selected_marker()
        if m is None:
            self.notice.emit(tr("mk_none", self.lang))
            return
        deliver(m)
