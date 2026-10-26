"""The left column: Task | Manual | Report | Data | Log.

The task text with its rule card and the manual are what every agent is told, shown unchanged. The
report form is the result channel: one radio button per reportable claim, with the line the rule card
says about it; nothing is chosen for the reader except the unit once a claim is picked. Data lists what
the instrument saved; Log keeps every call, newest first, queued ones marked until they run (and the
ones dropped before they did, as cancelled).
"""
from __future__ import annotations

from html import escape

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QFrame, QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QPushButton, QRadioButton, QScrollArea, QVBoxLayout, QWidget)

from . import fmt, style
from .controls import QuietBrowser, QuietText, SelectAllLineEdit, Tabs, polish, strong
from .labels import S, tr, two_lines


_CHANNEL = {"z": "Z", "current": "I", "didv": "dI/dV"}


def _text_view(text: str, mono: bool, size: int) -> QuietText:
    v = QuietText()
    v.setFont(style.mono_font(size) if mono else style.sans_font(size))
    v.setPlainText(text)
    return v


def _field_label(key: str, lang: str, required: bool = False) -> str:
    """A report field as ReportResult and the rule card name it (``frame``, ``evidence``, ``x_nm`` ...), then
    what it is, as rich text with the red asterisk the action panels put on a field that is required."""
    en, zh = S[f"rep_{key}"]
    desc = en if lang == "en" else zh
    html = f"{key} <span style='color:{style.COLORS['muted']}'>({desc})</span>" if desc else key
    return html + (f" <span style='color:{style.COLORS['bad']}'>*</span>" if required else "")


class _ClaimLabel(QLabel):
    """The rule card's line about a claim; a click on it picks the claim."""

    def __init__(self, text: str, radio: QRadioButton):
        super().__init__(text)
        self._radio = radio
        self.setWordWrap(True)
        self.setProperty("role", "claim")
        self.setContentsMargins(style.px(26), 0, 0, style.px(2))

    def mousePressEvent(self, ev) -> None:
        self._radio.click()


class ReportTab(QWidget):
    """Report a result, and finish the episode (two clicks, no dialog)."""

    submitted = Signal(dict)
    finish_requested = Signal()

    def __init__(self, claims: list[dict], lang: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.lang = lang
        self._claims = {c["id"]: c for c in claims}
        self._viewed = ""
        v = QVBoxLayout(self)
        v.setContentsMargins(style.px(8), style.px(4), style.px(8), style.px(4))
        v.setSpacing(style.px(3))
        self.title = self._title(f"{tr('report_title', lang)} (ReportResult)")
        self.title.setObjectName("report.title")
        v.addWidget(self.title)
        self.radios: dict[str, QRadioButton] = {}
        for c in claims:
            r = QRadioButton(c["id"])
            r.setObjectName(f"report.claim.{c['id']}")
            r.setFont(style.mono_font(17, bold=True))
            r.toggled.connect(lambda on, cid=c["id"]: on and self._claim_picked(cid))
            self.radios[c["id"]] = r
            v.addWidget(r)
            say = c["say_en"] if lang == "en" and c.get("say_en") else c["say"]
            v.addWidget(_ClaimLabel(say, r))

        grid = QGridLayout()
        grid.setHorizontalSpacing(style.px(8))
        grid.setVerticalSpacing(style.px(3))
        self.fields: dict[str, SelectAllLineEdit] = {}

        self.labels: dict[str, QLabel] = {}

        def field(key: str, row: int, col: int, span: int = 1, beside: QWidget | None = None) -> None:
            lab = QLabel(_field_label(key, lang, key == "value"))
            lab.setObjectName(f"report.label.{key}")
            lab.setTextFormat(Qt.TextFormat.RichText)
            lab.setProperty("role", "label")
            self.labels[key] = lab
            e = SelectAllLineEdit()
            e.setObjectName(f"report.{key}")
            e.setFont(style.mono_font(18))
            e.returnPressed.connect(self._submit)                # Enter in any field sends the report, like the button
            self.fields[key] = e
            grid.addWidget(lab, row * 2, col, 1, span)
            if beside is None:
                grid.addWidget(e, row * 2 + 1, col, 1, span)
                return
            line = QHBoxLayout()                             # a button at the end of the field's own line
            line.setSpacing(style.px(8))
            line.addWidget(e, 1)
            line.addWidget(beside)
            grid.addLayout(line, row * 2 + 1, col, 1, span)

        # the button that appends the viewed item's name sits at the end of the evidence field's line
        self.viewed_button = QPushButton("+ " + tr("rep_viewed", lang))
        self.viewed_button.setObjectName("report.viewed")
        self.viewed_button.clicked.connect(self._append_viewed)
        field("value", 0, 0)
        field("x_nm", 0, 1)
        field("y_nm", 0, 2)
        field("frame", 1, 0, 3)
        field("evidence", 2, 0, 3, beside=self.viewed_button)
        field("unit", 3, 0)
        field("sigma", 3, 1)
        field("note", 3, 2)
        for c in range(3):
            grid.setColumnStretch(c, 1)
        v.addLayout(grid)

        self.submit = QPushButton(tr("rep_submit", lang))
        self.submit.setObjectName("report.submit")
        strong(self.submit, "primary")
        self.submit.clicked.connect(self._submit)
        v.addWidget(self.submit)
        self.reply = QLabel()
        self.reply.setObjectName("report.reply")
        self.reply.setWordWrap(True)
        self.reply.setProperty("role", "label")
        v.addWidget(self.reply)

        v.addSpacing(style.px(10))
        self.finish_begin = QPushButton(tr("finish", lang))
        self.finish_begin.setObjectName("finish.begin")
        strong(self.finish_begin, "danger")
        self.finish_begin.clicked.connect(self._ask)
        v.addWidget(self.finish_begin)
        self.finish_ask = QWidget()
        ask = QVBoxLayout(self.finish_ask)
        ask.setContentsMargins(0, 0, 0, 0)
        ask_text = QLabel(tr("finish_ask", lang))
        ask_text.setWordWrap(True)
        ask_text.setProperty("role", "label")
        ask.addWidget(ask_text)
        row = QHBoxLayout()
        self.finish_yes = QPushButton(tr("finish_yes", lang))
        self.finish_yes.setObjectName("finish.confirm")
        strong(self.finish_yes, "danger")
        self.finish_yes.clicked.connect(self._confirm)
        self.finish_no = QPushButton(tr("finish_no", lang))
        self.finish_no.setObjectName("finish.cancel")
        self.finish_no.clicked.connect(self._cancel)
        row.addWidget(self.finish_yes)
        row.addWidget(self.finish_no)
        ask.addLayout(row)
        self.finish_ask.hide()
        v.addWidget(self.finish_ask)
        self.finish_message = QLabel()
        self.finish_message.setObjectName("finish.message")
        self.finish_message.setWordWrap(True)
        self.finish_message.setProperty("tone", "ok")
        self.finish_message.setProperty("role", "label")
        self.finish_message.hide()                       # nothing to say until the end
        v.addWidget(self.finish_message)
        v.addStretch(1)

    @staticmethod
    def _title(text: str) -> QLabel:
        lab = QLabel(text)
        strong(lab)
        return lab

    # ── the form ──
    def claim_id(self) -> str:
        return next((cid for cid, r in self.radios.items() if r.isChecked()), "")

    def _claim_picked(self, cid: str) -> None:
        self.fields["unit"].setText(self._claims[cid]["unit"])          # the one thing prefilled
        self._mark_required(cid)

    def required_fields(self) -> set[str]:
        """The fields the picked claim needs: the value always, and the position (x_nm, y_nm) and the frame it
        was read in when the claim is about a place."""
        cid = self.claim_id()
        needs_position = bool(cid) and bool(self._claims[cid].get("position"))
        return {"value"} | ({"x_nm", "y_nm", "frame"} if needs_position else set())

    def _mark_required(self, _cid: str = "") -> None:
        need = self.required_fields()
        for key, lab in self.labels.items():
            lab.setText(_field_label(key, self.lang, key in need))

    def _submit(self) -> None:
        body = {"claim_id": self.claim_id()}
        body.update({k: e.text().strip() for k, e in self.fields.items()})
        self.submitted.emit(body)

    def set_viewed(self, name: str) -> None:
        self._viewed = name

    def _append_viewed(self) -> None:
        if not self._viewed:
            return
        have = [t.strip() for t in self.fields["evidence"].text().replace("，", ",").split(",") if t.strip()]
        if self._viewed not in have:
            have.append(self._viewed)
        self.fields["evidence"].setText(", ".join(have))

    def fill_position(self, x_nm: float, y_nm: float, frame: str) -> None:
        self.fields["x_nm"].setText(f"{x_nm:.3f}")
        self.fields["y_nm"].setText(f"{y_nm:.3f}")
        self.fields["frame"].setText(frame)

    def set_reply(self, ok: bool, text: str) -> None:
        self.reply.setText(("✓ " if ok else "✗ ") + text)
        self.reply.setProperty("tone", "ok" if ok else "bad")
        polish(self.reply)

    # ── finishing ──
    def _ask(self) -> None:
        self.finish_begin.hide()
        self.finish_ask.show()

    def _cancel(self) -> None:
        self.finish_ask.hide()
        self.finish_begin.show()

    def _confirm(self) -> None:
        self.finish_ask.hide()
        self.finish_requested.emit()

    def set_finished(self) -> None:
        self.finish_ask.hide()
        self.finish_begin.hide()
        self.finish_message.setText(tr("over", self.lang))
        self.finish_message.show()


class DataTab(QWidget):
    """What the instrument saved, newest first: each frame or spectrum is two lines (its name, start time
    and channels; its geometry), which carry every column of a table in the width of this column. A click
    shows the item in the viewer."""

    frame_clicked = Signal(str)
    spectrum_clicked = Signal(str)

    def __init__(self, lang: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.lang = lang
        v = QVBoxLayout(self)
        v.setContentsMargins(style.px(8), style.px(8), style.px(8), style.px(8))
        v.setSpacing(style.px(4))
        self.frames_title = QLabel()
        strong(self.frames_title)
        v.addWidget(self.frames_title)
        self.frames = self._list("data.frames")
        v.addWidget(self.frames, 3)
        self.spectra_title = QLabel()
        strong(self.spectra_title)
        v.addWidget(self.spectra_title)
        self.spectra = self._list("data.spectra")
        v.addWidget(self.spectra, 2)
        self.frames.itemClicked.connect(lambda it: self.frame_clicked.emit(it.data(Qt.ItemDataRole.UserRole)))
        self.spectra.itemClicked.connect(lambda it: self.spectrum_clicked.emit(it.data(Qt.ItemDataRole.UserRole)))
        self.set_data([], [])

    def _list(self, name: str) -> QListWidget:
        w = QListWidget()
        w.setObjectName(name)
        w.setAlternatingRowColors(True)
        w.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        w.setFont(style.sans_font(17))
        w.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        w.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        return w

    def set_data(self, frames: list[dict], spectra: list[dict]) -> None:
        self.frames_title.setText(f"{tr('frames', self.lang)} ({len(frames)})")
        self.spectra_title.setText(f"{tr('spectra', self.lang)} ({len(spectra)})")
        self._fill(self.frames, [self._frame_lines(f) for f in reversed(frames)])
        self._fill(self.spectra, [self._spectrum_lines(s) for s in reversed(spectra)])

    @staticmethod
    def _frame_lines(f: dict) -> tuple[str, str, str]:
        """``(name, first line, second line)``: name, start and channels; then the geometry."""
        channels = " ".join(_CHANNEL[c] for c in f.get("channels", []))
        cx, cy = f["center_nm"]
        return (f["frame"], f"{f['frame']}   {fmt.hms(f['t_start_s'])}   {channels}",
                f"{fmt.num(f['size_nm'], 5)} nm · {f['pixels']} px · centre ({fmt.num(cx, 5)}, {fmt.num(cy, 5)}) nm · "
                f"{fmt.num(f['angle_deg'], 4)}°")

    @staticmethod
    def _spectrum_lines(s: dict) -> tuple[str, str, str]:
        return (s["spectrum"], f"{s['spectrum']}   {s['kind']}   {fmt.hms(s['t_s'])}",
                f"tip ({s['tip_nm'][0]:.3f}, {s['tip_nm'][1]:.3f}) nm · {s['points']} points")

    @staticmethod
    def _fill(w: QListWidget, rows: list[tuple[str, str, str]]) -> None:
        w.clear()
        for name, first, second in rows:
            item = QListWidgetItem(f"{first}\n{second}")
            item.setData(Qt.ItemDataRole.UserRole, name)
            w.addItem(item)

    def select(self, kind: str, name: str) -> None:
        """Mark the item now in the viewer (no signal)."""
        for w, k in ((self.frames, "frame"), (self.spectra, "spectrum")):
            w.clearSelection()
            if k == kind:
                for r in range(w.count()):
                    if w.item(r).data(Qt.ItemDataRole.UserRole) == name:
                        w.setCurrentRow(r)


class LogTab(QuietBrowser):
    """Every call, newest first: instrument time, the action with its parameters, the reply or the
    refusal (red). A call that has not run yet says so until it does."""

    def __init__(self, lang: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("log.view")
        self.lang = lang
        self._entries: list[dict] = []
        self.setFont(style.sans_font(17))
        self._render()

    def add(self, job_id: int, label: str, args: dict, t_s: float | None) -> None:
        self._entries.append({"id": job_id, "label": label, "args": fmt.params_text(args), "state": "queued",
                              "t": t_s, "ok": None, "text": ""})
        self._render()

    def running(self, job_id: int) -> None:
        self._set(job_id, state="running")

    def cancelled(self, job_id: int) -> None:
        """A call dropped before it started: it stays in the log, said to be cancelled."""
        self._set(job_id, state="cancelled")

    def done(self, job_id: int, ok: bool, text: str, t_s: float | None) -> None:
        self._set(job_id, state="done", ok=ok, text=text, t=t_s)

    def _set(self, job_id: int, **kw) -> None:
        for e in self._entries:
            if e["id"] == job_id:
                e.update(kw)
        self._render()

    def plain(self) -> str:
        return self.toPlainText()

    def _render(self) -> None:
        c = style.COLORS
        mono = style.mono_family()
        if not self._entries:
            self.setHtml(f"<p style='color:{c['muted']}'>{escape(tr('log_empty', self.lang))}</p>")
            return
        out = []
        for e in reversed(self._entries):
            when = fmt.hms(e["t"]) if e["t"] is not None else "-"
            head = (f"<span style=\"font-family:'{mono}';color:{c['muted']}\">{when}</span> "
                    f"<b style=\"font-family:'{mono}'\">{escape(e['label'])}</b> "
                    f"<span style=\"font-family:'{mono}'\">{escape(e['args'])}</span>")
            if e["state"] == "queued":
                tail = f"<span style='color:{c['warn']}'>{escape(tr('queued', self.lang))}</span>"
            elif e["state"] == "running":
                tail = f"<span style='color:{c['warn']}'>{escape(tr('log_running', self.lang))}</span>"
            elif e["state"] == "cancelled":
                tail = f"<span style='color:{c['muted']}'>{escape(tr('cancelled', self.lang))}</span>"
            else:
                color = c["ok"] if e["ok"] else c["bad"]
                tail = (f"<span style=\"color:{color};font-family:'{mono}'\">"
                        f"{'✓' if e['ok'] else '✗'} {escape(e['text'])}</span>")
            out.append(f"<div style='margin-bottom:7px'>{head}<br>{tail}</div>")
        self.setHtml("".join(out))


class FactsTable(QFrame):
    """The instrument's fixed facts (``info()``), read once when the window opens: what an operator finds
    registered on the instrument, in readable units. A grid of labels, so that a long value wraps in its
    cell and the table is exactly as tall as its text (a table widget's rows are sized before it knows
    its width, and cut a wrapped line off)."""

    def __init__(self, lang: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("facts.table")
        self.setProperty("role", "panel")
        self.lang = lang
        self.rows: list[tuple[QLabel, QLabel]] = []
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(1, 1, 1, 1)
        self._grid.setSpacing(0)
        self._grid.setColumnStretch(0, 11)
        self._grid.setColumnStretch(1, 9)

    def set_info(self, info: dict) -> None:
        for lab, val in self.rows:
            for w in (lab, val):
                self._grid.removeWidget(w)
                w.deleteLater()
        self.rows = []
        for r, (key, value) in enumerate(fmt.facts_items(info)):
            lab, val = QLabel(tr(key, self.lang)), QLabel(value)
            lab.setObjectName(f"facts.label.{key}")
            val.setObjectName(f"facts.value.{key}")
            lab.setFont(style.sans_font(17))
            val.setFont(style.mono_font(17))
            for col, w in enumerate((lab, val)):
                w.setWordWrap(True)
                w.setContentsMargins(style.px(6), style.px(3), style.px(6), style.px(3))
                w.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
                w.setProperty("stripe", r % 2 == 1)
                self._grid.addWidget(w, r, col)
            self.rows.append((lab, val))

    def facts(self) -> dict[str, str]:
        """``{label as shown: value as shown}``, for tests and logs."""
        return {lab.text(): val.text() for lab, val in self.rows}


class ManualTab(QWidget):
    """The instrument's facts above the manual: everything an agent is told about the instrument."""

    def __init__(self, manual: str, lang: str, parent: QWidget | None = None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(style.px(6), style.px(6), style.px(6), style.px(6))
        v.setSpacing(style.px(4))
        title = QLabel(tr("facts_title", lang))
        strong(title)
        v.addWidget(title)
        self.facts = FactsTable(lang)
        v.addWidget(self.facts)
        self.text = _text_view(manual, mono=True, size=17)
        self.text.setObjectName("manual.text")
        v.addWidget(self.text, 1)


class Sidebar(Tabs):
    """The five tabs; ``tab`` finds one by key."""

    KEYS = ("task", "manual", "report", "data", "log")

    def __init__(self, task: str, manual: str, claims: list[dict], lang: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.lang = lang
        self.task = _text_view(task, mono=False, size=18)
        self.task.setObjectName("task.text")
        self.manual = ManualTab(manual, lang)
        self.report = ReportTab(claims, lang)
        # in a scroll area, as a last resort: the claims' wrapped lines are taller than the layout's minimum
        # says, and a form squeezed below that overlaps itself (P1 in English: four long claims)
        self.report_scroll = QScrollArea()
        self.report_scroll.setObjectName("report.scroll")
        self.report_scroll.setWidgetResizable(True)
        self.report_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.report_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.report_scroll.setWidget(self.report)
        self.data = DataTab(lang)
        self.log = LogTab(lang)
        for key, w in zip(self.KEYS, (self.task, self.manual, self.report_scroll, self.data, self.log)):
            self.addTab(w, two_lines(f"tab_{key}", lang))
        self.setDocumentMode(False)
        self.tabBar().setObjectName("sidebar.tabs")
        self.tabBar().setExpanding(True)
        self.tabBar().setUsesScrollButtons(False)

    @property
    def facts(self) -> FactsTable:
        return self.manual.facts

    def show_tab(self, key: str) -> None:
        self.setCurrentIndex(self.KEYS.index(key))

    def tab_key(self) -> str:
        return self.KEYS[self.currentIndex()]
