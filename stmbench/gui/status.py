"""The status strip across the top: three lines of large text.

1. the program, the scenario's id (never the seed), READY / BUSY with what runs and how many wait,
   instrument time used against the budget, the action count;
2. the readings the instrument reports after every action, in the units of the fields that set them
   (Chinese labels only in ``zh``: eleven readings in two languages would not fit one line);
3. what runs while the instrument is busy (the call, with its parameters, and how many wait, with a
   button that drops the ones that have not started), else the last result in the instrument's words:
   green with a tick, or red with a cross and the refusal.

The strip is built for a reader looking at a downscaled screenshot: one line each, 18 px text.

While the console holds (a call of the operator's was refused and the refusal is not acknowledged, see
:mod:`stmbench.gui.worker`) a red band takes the place of lines 2 and 3: the instrument's words, the call that was
refused with its parameters, the calls that did not run because of it, and the button that acknowledges it. The
band is exactly as tall as the two lines it replaces, so nothing below the strip moves when it comes or goes (a
reader's later clicks of the same step were aimed at the old picture), and its button ends where the place of
Cancel queued begins: no control is under the button when nothing is refused (a click there by habit does
nothing), and a click aimed at Cancel queued does not acknowledge a refusal.
"""
from __future__ import annotations

from html import escape

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton, QSizePolicy, QVBoxLayout,
                               QWidget)

from . import fmt, style
from .controls import ElidedLabel, polish, strong
from .jobs import Refusal
from .labels import S, both, one, tr

#: how many of the calls that did not run the band names before it counts the rest
BAND_NAMES = 8


class StatusStrip(QFrame):
    cancel_requested = Signal()
    acknowledge_requested = Signal()                 # the click on the band's button

    def __init__(self, scenario_id: str, lang: str = "zh", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("strip")
        self.lang = lang
        self._budget: float | None = None
        self._qplus = True                                # until the instrument's facts say there is no sensor
        self._st: dict | None = None
        self._plain: dict[str, str] = {}
        self._last: tuple[bool | None, str] = (None, "")           # the last result: shown whenever nothing runs
        self._busy: str | None = None
        self._waiting = 0
        self._refusal: Refusal | None = None
        v = QVBoxLayout(self)
        v.setContentsMargins(style.px(12), style.px(3), style.px(12), style.px(3))
        v.setSpacing(style.px(1))
        big = style.sans_font(18)

        top = QHBoxLayout()
        top.setSpacing(style.px(14))
        self.title = QLabel(f"{tr('app', lang)}  ·  {scenario_id}")
        self.title.setObjectName("status.title")
        self.title.setFont(style.sans_font(18, bold=True))
        self.state = QLabel()
        self.state.setObjectName("status.state")
        self.state.setProperty("role", "pill")
        self.state.setFont(style.sans_font(18, bold=True))
        self.time = QLabel()
        self.time.setObjectName("status.time")
        self.time.setFont(style.mono_font(18))
        self.bar = QProgressBar()
        self.bar.setObjectName("status.bar")
        self.bar.setTextVisible(False)
        self.bar.setFixedWidth(style.px(260))
        self.count = QLabel()
        self.count.setObjectName("status.actions")
        self.count.setFont(style.mono_font(18))
        top.addWidget(self.title)
        top.addWidget(self.state)
        top.addStretch(1)
        top.addWidget(self.time)
        top.addWidget(self.bar)
        top.addWidget(self.count)
        v.addLayout(top)

        self.readings = QLabel()
        self.readings.setObjectName("status.readings")
        self.readings.setTextFormat(Qt.TextFormat.RichText)
        self.readings.setFont(style.sans_font(17))
        self.readings.setWordWrap(True)                  # a second line, never a clipped reading, if the fonts are wide
        self.readings.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        v.addWidget(self.readings)

        self.result_row = QWidget()                      # line 3, in a widget of its own so that the band can take its place
        self.result_row.setObjectName("status.result_row")
        line = QHBoxLayout(self.result_row)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(style.px(10))
        self.result = ElidedLabel()
        self.result.setObjectName("status.result")
        self.result.setFont(big)
        self.result.setMinimumHeight(style.px(28))               # the row keeps its height whether the button is there or not
        self.cancel = QPushButton(tr("cancel_queued", lang))
        self.cancel.setObjectName("status.cancel")
        self.cancel.setProperty("role", "small")
        self.cancel.setAutoDefault(False)
        self.cancel.clicked.connect(self.cancel_requested.emit)
        self.cancel.hide()
        line.addWidget(self.result, 1)
        line.addWidget(self.cancel)
        v.addWidget(self.result_row)

        # the band of a refusal that is not acknowledged: in place of lines 2 and 3 while the console holds
        self.band = QFrame()
        self.band.setObjectName("refusal.band")
        self.band.setProperty("role", "band")
        band = QHBoxLayout(self.band)
        band.setSpacing(style.px(12))
        self.band_text = QLabel()
        self.band_text.setObjectName("refusal.text")
        self.band_text.setProperty("role", "band")
        self.band_text.setTextFormat(Qt.TextFormat.RichText)
        self.band_text.setWordWrap(True)
        self.band_text.setFont(big)
        self.band_text.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.ack = QPushButton(tr("ack", lang))
        self.ack.setObjectName("refusal.ack")
        self.ack.setAutoDefault(False)
        strong(self.ack, "ack")
        self.ack.setMinimumWidth(style.px(150))
        self.ack.clicked.connect(self.acknowledge_requested.emit)
        band.addWidget(self.band_text, 1)
        band.addWidget(self.ack, 0, Qt.AlignmentFlag.AlignVCenter)
        # the button ends where Cancel queued begins: a click aimed at Cancel queued (which is under the band's lower
        # right while nothing is refused) must not reach it, whatever the heights of the rows are in this font
        band.setContentsMargins(style.px(10), 0, style.px(8) + self.cancel.sizeHint().width() + style.px(10), 0)
        self.band.hide()
        v.addWidget(self.band)
        self.set_state(False, "", 0, False)
        self.set_status(None)
        self.set_result(None, tr("no_call", lang))

    # ── line 1 ──
    def set_state(self, busy: bool, running: str, waiting: int, finished: bool, held: bool = False) -> None:
        """The pill: FINISHED, else HELD while a refusal is not acknowledged, else BUSY with what runs, else READY."""
        if finished:
            text, tone = tr("finished", self.lang), "done"
        elif held:
            text, tone = tr("held", self.lang), "held"
        elif busy or waiting:
            text = f"{tr('busy', self.lang)}: {running}" if running else tr("busy", self.lang)
            text += f" (+{waiting} {both(*S['queued'], self.lang)})" if waiting else ""
            tone = "busy"
        else:
            text, tone = tr("ready", self.lang), ""
        self.state.setText(text)
        self.state.setProperty("tone", tone)
        polish(self.state)
        self._plain["state"] = text

    def set_has_qplus(self, has: bool) -> None:
        """Whether the rig has a qPlus sensor; without one the strip has no oscillation reading."""
        if has != self._qplus:
            self._qplus = has
            self.set_status(self._st)

    def set_status(self, st: dict | None) -> None:
        """Readings from ``status()``: instrument time against the budget, the action count, the readings."""
        if st is None:
            self.time.setText(f"{tr('time', self.lang)} -")
            self.count.setText(f"{tr('actions', self.lang)} -")
            self.readings.setText("")
            return
        self._st = st
        used, self._budget = float(st["instrument_time_s"]), st.get("budget_s")
        text = f"{tr('time', self.lang)} {fmt.hms(used)}" + (f" / {fmt.hms(self._budget)}" if self._budget else "")
        self.time.setText(text)
        self.bar.setMaximum(1000)
        self.bar.setValue(min(1000, int(1000 * used / self._budget)) if self._budget else 0)
        self.count.setText(f"{tr('actions', self.lang)} {st['actions']}")
        mono, muted = style.mono_family(), style.COLORS["muted"]
        parts, plain = [], []
        for key, value in fmt.reading_items(st, self.lang, self._qplus):
            label = one(key, self.lang)                  # eleven readings in two languages would not fit one line
            parts.append(f"<nobr><span style='color:{muted}'>{label}</span> "
                         f"<b style=\"font-family:'{mono}'\">{value.replace('<', '&lt;')}</b></nobr>")
            plain.append(f"{label} {value}")
        self.readings.setText("&nbsp; ".join(parts))
        self._plain.update(time=text, readings="   ".join(plain), actions=self.count.text())

    # ── line 3 ──
    def set_result(self, ok: bool | None, text: str) -> None:
        """The last result; shown at once unless something runs, then as soon as nothing does."""
        self._last = (ok, text)
        self._show_line()

    def set_busy(self, running: str | None, waiting: int, cancellable: int = 0) -> None:
        """``running``: what the instrument is doing (None: nothing); ``waiting``: jobs behind it;
        ``cancellable`` of them can still be dropped, which shows the Cancel queued button."""
        self._busy, self._waiting = running, waiting
        self.cancel.setVisible(cancellable > 0)
        self._show_line()

    def _show_line(self) -> None:
        if self._busy is not None:
            text = f"{tr('running', self.lang)}: {self._busy}"
            if self._waiting:
                text += f" (+{self._waiting} {both(*S['queued'], self.lang)})"
            tone = "busy"
        else:
            ok, body = self._last
            text = ("" if ok is None else ("✓ " if ok else "✗ ")) + body
            tone = "" if ok is None else ("ok" if ok else "bad")
        self.result.set_full_text(text)
        self.result.setProperty("tone", tone)
        polish(self.result)
        self._plain["result"] = text

    # ── the band ──
    def show_refusal(self, refusal: Refusal | None) -> None:
        """Show the band for ``refusal`` in place of lines 2 and 3, or (None) take it away and bring the lines back.
        Called again when more calls are turned away, to name them."""
        if refusal is None:
            self._refusal = None
            self.band.hide()
            self.readings.show()
            self.result_row.show()
            self._plain.pop("band", None)
            return
        if self._refusal is None:
            # as tall as the two lines it replaces (the last result's line without the Cancel queued button that
            # may still be showing for the moment: nothing waits once a call was refused)
            lower = self.readings.height() + self.layout().spacing() + self.result.minimumHeight()
            self.band.setMinimumHeight(lower)
        self._refusal = refusal
        self.readings.hide()
        self.result_row.hide()
        self.band.show()
        self.band_text.setText(self._band_html(refusal))
        self._plain["band"] = self.band_plain(refusal)

    def _band_lines(self, r: Refusal) -> tuple[tuple[str, str, str], str]:
        """The two lines of the band as ``((head, call, words), second line)``, without markup."""
        head = "✗ " + tr("refused_head" if r.refused else "error_head", self.lang)
        shown = r.not_run[:BAND_NAMES]
        more = len(r.not_run) - len(shown)
        names = ", ".join(shown) + (f" … +{more}" if more > 0 else "")
        second = tr("held_note", self.lang) + (f"  {tr('band_not_run', self.lang)}: {names}" if r.not_run else "")
        return (head, r.call, r.words), second

    def _band_html(self, r: Refusal) -> str:
        (head, call, words), second = self._band_lines(r)
        mono = style.mono_family()
        first = (f"<b>{escape(head)}</b> &nbsp;<span style=\"font-family:'{mono}'\"><b>{escape(call)}</b></span>"
                 f" &nbsp;—&nbsp; {escape(words)}")
        return f"{first}<br>{escape(second)}"

    def band_plain(self, r: Refusal) -> str:
        (head, call, words), second = self._band_lines(r)
        return f"{head} {call} — {words}\n{second}"

    def plain_text(self) -> str:
        """What the strip says, without markup: for tests and for logs."""
        return "\n".join(filter(None, [self.title.text(), self._plain.get("state", ""), self._plain.get("time", ""),
                                       self._plain.get("actions", ""), self._plain.get("readings", ""),
                                       self._plain.get("result", ""), self._plain.get("band", "")]))
