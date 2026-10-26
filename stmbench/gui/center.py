"""The middle column: the viewer, its controls, the markers and the strip of recent frames.

Everything here is read off the data the instrument gave: the cursor's coordinates, a marker's value,
a profile along a line. Nothing is found, fitted or suggested: no atom finder, no drift estimate. A
marker is a point the reader chose, kept in scan coordinates, with the numbers instrument software
shows for any point (position, value, distance to the previous one). The three ``← marker`` buttons copy
a marker's coordinates into the fields of the action or report that takes them; they run nothing.
"""
from __future__ import annotations

import math
import re

import numpy as np
from PySide6.QtCore import QEvent, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QPushButton, QSizePolicy,
                               QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from . import fmt, style
from .controls import ElidedLabel, Segmented, ThumbButton
from .labels import S, both, one, tr
from .widgets.frame_view import FrameView, frame_thumbnail
from .widgets.plot_view import PlotView

#: spectrum columns as the viewer plots them: (button text, factor from the stored unit, unit)
SPECTRUM_COLUMNS = {
    "v": ("V", 1.0, "V"), "i": ("I", 1e12, "pA"), "didv": ("dI/dV", 1e9, "nS"), "z_rel": ("z rel", 1e12, "pm"),
    "z_abs": ("z", 1e9, "nm"), "df": ("Δf", 1.0, "Hz"), "amp": ("amplitude", 1e12, "pm"),
}
_X_FIRST = ("v", "z_rel", "z_abs")
_Y_FIRST = ("didv", "df", "i")
N_THUMBS = 6
THUMB_PX = 64
#: the marker table shows this many rows before it scrolls: its strip is exactly a header, the rows and the frame
#: (a viewport that is not a whole number of rows leaves the first row cut off under the header when scrolled)
MARKER_ROWS = 6
ROW_PX, HEADER_PX = 26, 24
MARKER_STRIP_PX = HEADER_PX + MARKER_ROWS * ROW_PX + 2


def column_label(key: str) -> str:
    name, _, unit = SPECTRUM_COLUMNS[key]
    return name if name == unit else f"{name} ({unit})"


def spectrum_columns(spec: dict) -> list[str]:
    """The plottable columns of a spectrum, in a fixed order."""
    return [k for k in SPECTRUM_COLUMNS if k in spec and np.size(spec[k]) >= 8]


def default_axes(spec: dict) -> tuple[str, str]:
    """The sweep axis against the first signal that varies."""
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
        column = QVBoxLayout()
        column.setSpacing(style.px(3))
        column.setContentsMargins(0, 0, 0, 0)
        strip_w = QWidget()                              # always as wide as a thumbnail: the image does not jump
        strip_w.setFixedWidth(style.px(THUMB_PX + 8))
        strip_w.setLayout(column)
        self.thumbs = [ThumbButton() for _ in range(N_THUMBS)]
        for i, b in enumerate(self.thumbs):
            b.setObjectName(f"thumb.{i}")
            b.clicked.connect(lambda _=False, btn=b: self.thumb_clicked.emit(btn.name))
            b.hide()
            column.addWidget(b)
        column.addStretch(1)
        view.addWidget(strip_w)
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
        fv.cursor_moved.connect(self._cursor)
        fv.markers_changed.connect(self._markers_changed)
        fv.marker_added.connect(self._marker_added)
        fv.profile_changed.connect(self._profile)

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
        s = Segmented([(val, both(*S[key], self.lang), f"{name}.{val}") for val, key in options], value)
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
        lab = QLabel(tr(key, self.lang))
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
        self.tool = self._seg([("marker", "tool_marker"), ("profile", "tool_profile")], "view.tool", "marker")
        self.clear_markers = QPushButton(tr("clear_markers", self.lang))
        self.clear_markers.setObjectName("view.clear_markers")
        col.addLayout(self._row(self.channel, self.direction, self.clear_markers))
        col.addLayout(self._row(self._caption("lbl_flatten"), self.flatten, self._caption("lbl_tool"), self.tool))
        self.channel.changed.connect(lambda _v: self.view_changed.emit())
        self.direction.changed.connect(lambda _v: self.view_changed.emit())
        self.flatten.changed.connect(lambda m: self.frame_view.set_flatten(m))
        self.tool.changed.connect(lambda t: self.frame_view.set_tool(t))
        self.clear_markers.clicked.connect(lambda: (self.frame_view.clear_markers(), self.frame_view.clear_profile()))
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
        t = QTableWidget(0, len(cols), parent)
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
        self.readout.set_full_text("")
        for b in self.thumbs:
            b.setChecked(b.name == frame["frame"])

    def show_spectrum(self, spec: dict, axes: tuple[str, str] | None = None) -> None:
        self._spec = spec
        cols = spectrum_columns(spec)
        if not cols:
            self.plot.clear(tr("no_view", self.lang))
            self.toolbars.setCurrentIndex(1)
            self.stack.setCurrentIndex(1)
            return
        xk, yk = axes or default_axes(spec)
        self._rebuild_axes(cols, xk, yk)
        self.toolbars.setCurrentIndex(1)
        self.stack.setCurrentIndex(1)
        for b in self.thumbs:
            b.setChecked(False)
        self.readout.set_full_text("")
        self.info.setText(f"{spec['name']}   at {fmt.hms(spec['t_s'])}\n"
                          f"{spec['kind']} · tip ({spec['tip_nm'][0]:.3f}, {spec['tip_nm'][1]:.3f}) nm · "
                          f"{np.size(spec[cols[0]])} points")
        self._draw_spectrum(xk, yk)

    def _rebuild_axes(self, cols: list[str], xk: str, yk: str) -> None:
        for attr, row, name, val in (("spec_x", self.spec_x_row, "view.spec_x", xk),
                                     ("spec_y", self.spec_y_row, "view.spec_y", yk)):
            old = getattr(self, attr)
            if old is not None:
                row.removeWidget(old)
                old.deleteLater()
            seg = Segmented([(c, column_label(c), f"{name}.{c}") for c in cols], val)
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
        self.plot.set_series(x, [(column_label(yk), y)], column_label(xk), column_label(yk), spec["name"])

    def current_spectrum_axes(self) -> tuple[str, str] | None:
        if self._spec is None or not (self.spec_x and self.spec_y):
            return None
        return self.spec_x.value(), self.spec_y.value()

    def set_thumbs(self, items: list[tuple[str, QPixmap]], current: str = "") -> None:
        """The last frames, newest first."""
        size = style.px(THUMB_PX)
        for i, b in enumerate(self.thumbs):
            if i < len(items):
                b.set_thumb(items[i][0], items[i][1], size)
                b.setChecked(items[i][0] == current)
            else:
                b.set_thumb("", None, size)

    # ── readouts ──
    def _cursor(self, c: dict) -> None:
        if not c:
            self.readout.set_full_text("")
            return
        val = c.get("value")
        text = f"x {c['x_nm']:.3f} nm   y {c['y_nm']:.3f} nm"
        if val is not None and math.isfinite(float(val)):
            text += f"   {float(val):.2f} {c.get('unit', '')}"
        self.readout.set_full_text(text + f"   (col {c['col']}, row {c['row']})")

    def _marker_added(self, _m: dict) -> None:
        n = self.table.rowCount()
        if n:
            self.table.selectRow(n - 1)
            self.table.scrollToItem(self.table.item(n - 1, 0))        # the new one is in view even past the fifth

    def set_frame_starts(self, starts: dict[str, float]) -> None:
        """When each saved frame's scan began, for the table's start column: the frame a marker was placed on,
        as the Data tab gives it."""
        self._frame_start = dict(starts)
        self._markers_changed(self.frame_view.markers())

    def _markers_changed(self, markers: list) -> None:
        t = self.table
        t.setRowCount(len(markers))
        prev = None
        for i, m in enumerate(markers):
            d = "" if prev is None else f"{math.hypot(m['x_nm'] - prev['x_nm'], m['y_nm'] - prev['y_nm']):.3f}"
            val = m.get("value")
            vtxt = f"{float(val):.2f} {m.get('unit', '')}" if val is not None and math.isfinite(float(val)) else "-"
            started = self._frame_start.get(str(m.get("frame", "")))
            stxt = fmt.hms(started) if started is not None else "-"
            for c, text in enumerate([str(i + 1), f"{m['x_nm']:.3f}", f"{m['y_nm']:.3f}", vtxt, str(m.get("frame", "")),
                                      stxt, d]):
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
