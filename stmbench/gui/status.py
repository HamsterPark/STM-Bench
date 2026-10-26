"""The status strip across the top: three lines of large text.

1. the program, the scenario's id (never the seed), READY / BUSY with what runs and how many wait,
   instrument time used against the budget, the action count;
2. the readings the instrument reports after every action, in the units of the fields that set them
   (Chinese labels only in ``zh``: eleven readings in two languages would not fit one line);
3. the last result, in the instrument's words: green with a tick, or red with a cross and the refusal.

The strip is built for a reader looking at a downscaled screenshot: one line each, 18 px text.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QSizePolicy, QVBoxLayout, QWidget

from . import fmt, style
from .controls import ElidedLabel, polish
from .labels import S, both, one, tr


class StatusStrip(QFrame):
    def __init__(self, scenario_id: str, lang: str = "zh", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("strip")
        self.lang = lang
        self._budget: float | None = None
        self._plain: dict[str, str] = {}
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

        self.result = ElidedLabel()
        self.result.setObjectName("status.result")
        self.result.setFont(big)
        v.addWidget(self.result)
        self.set_state(False, "", 0, False)
        self.set_status(None)
        self.set_result(None, tr("no_call", lang))

    # ── line 1 ──
    def set_state(self, busy: bool, running: str, waiting: int, finished: bool) -> None:
        if finished:
            text, tone = tr("finished", self.lang), "done"
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

    def set_status(self, st: dict | None) -> None:
        """Readings from ``status()``: instrument time against the budget, the action count, the readings."""
        if st is None:
            self.time.setText(f"{tr('time', self.lang)} -")
            self.count.setText(f"{tr('actions', self.lang)} -")
            self.readings.setText("")
            return
        used, self._budget = float(st["instrument_time_s"]), st.get("budget_s")
        text = f"{tr('time', self.lang)} {fmt.hms(used)}" + (f" / {fmt.hms(self._budget)}" if self._budget else "")
        self.time.setText(text)
        self.bar.setMaximum(1000)
        self.bar.setValue(min(1000, int(1000 * used / self._budget)) if self._budget else 0)
        self.count.setText(f"{tr('actions', self.lang)} {st['actions']}")
        mono, muted = style.mono_family(), style.COLORS["muted"]
        parts, plain = [], []
        for key, value in fmt.reading_items(st, self.lang):
            label = one(key, self.lang)                  # eleven readings in two languages would not fit one line
            parts.append(f"<nobr><span style='color:{muted}'>{label}</span> "
                         f"<b style=\"font-family:'{mono}'\">{value.replace('<', '&lt;')}</b></nobr>")
            plain.append(f"{label} {value}")
        self.readings.setText("&nbsp; ".join(parts))
        self._plain.update(time=text, readings="   ".join(plain), actions=self.count.text())

    # ── line 3 ──
    def set_result(self, ok: bool | None, text: str) -> None:
        mark = "" if ok is None else ("✓ " if ok else "✗ ")
        self.result.set_full_text(mark + text)
        self.result.setProperty("tone", "" if ok is None else ("ok" if ok else "bad"))
        polish(self.result)
        self._plain["result"] = mark + text

    def plain_text(self) -> str:
        """What the strip says, without markup: for tests and for logs."""
        return "\n".join(filter(None, [self.title.text(), self._plain.get("state", ""), self._plain.get("time", ""),
                                       self._plain.get("actions", ""), self._plain.get("readings", ""),
                                       self._plain.get("result", "")]))
