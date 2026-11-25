"""The left column: Task | Manual | Report | Data | Log.

The task text with its rule card and the manual are what every agent is told, shown unchanged. The
report form is the result channel: one radio button per reportable claim, with the line the rule card
says about it; nothing is chosen for the reader except the unit once a claim is picked, and the claim itself
when there is only one to report (P4). The value, the position and the evidence are the reader's. Data lists what
the instrument saved; Log keeps every call, newest first, queued ones marked until they run (and the
ones dropped before they did, as cancelled).
"""
from __future__ import annotations

from html import escape

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QBoxLayout, QButtonGroup, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QPushButton, QRadioButton, QScrollArea, QSizePolicy, QStyle, QVBoxLayout,
                               QWidget)

from . import fmt, style
from .controls import QuietBrowser, QuietText, SelectAllLineEdit, StepList, Tabs, WrapField, polish, strong
from .labels import S, one, tr, two_lines


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
        self.setContentsMargins(0, 0, 0, 0)

    def mousePressEvent(self, ev) -> None:
        self._radio.click()


#: the lines reserved under the evidence field for the full list of cited files (two: the lists of the trials were one to three files)
EVIDENCE_LINES = 2


class ReportTab(QWidget):
    """Report a result, and finish the episode (two clicks, no dialog)."""

    submitted = Signal(dict)
    finish_requested = Signal()
    notice = Signal(str)                                   # something the form wants said among the recent results

    def __init__(self, claims: list[dict], lang: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.lang = lang
        self._claims = {c["id"]: c for c in claims}
        self._viewed = ""
        self._entries: dict[str, dict[str, str]] = {}          # per result: what its fields held when it was last left
        self._current = ""                                     # the result whose entries the fields hold
        v = QVBoxLayout(self)
        v.setContentsMargins(style.px(8), style.px(4), style.px(8), style.px(4))
        v.setSpacing(style.px(1))                        # the claims of P1 in English fill the page: see the form below
        self.title = self._title(f"{tr('report_title', lang)} (ReportResult)")
        self.title.setObjectName("report.title")
        v.addWidget(self.title)
        self.radios: dict[str, QRadioButton] = {}
        self.claim_labels: dict[str, QLabel] = {}
        # in a group of their own: a lone auto-exclusive radio button can be clicked off, and P4 has one claim;
        # in an exclusive QButtonGroup the one that is checked stays checked when it is clicked again
        self.claim_group = QButtonGroup(self)
        self.claim_group.setExclusive(True)
        for c in claims:
            r = QRadioButton(c["id"])
            r.setObjectName(f"report.claim.{c['id']}")
            r.setFont(style.mono_font(17, bold=True))
            r.toggled.connect(lambda on, cid=c["id"]: on and self._claim_picked(cid))
            self.radios[c["id"]] = r
            self.claim_group.addButton(r)
            v.addWidget(r)
            say = c["say_en"] if lang == "en" and c.get("say_en") else c["say"]
            lab = _ClaimLabel(say, r)
            lab.setObjectName(f"report.say.{c['id']}")
            self.claim_labels[c["id"]] = lab
            v.addWidget(lab)

        form = QVBoxLayout()
        form.setSpacing(style.px(3))
        self.fields: dict[str, SelectAllLineEdit | WrapField] = {}

        self.labels: dict[str, QLabel] = {}
        # the full list of cited files, in grey under the field, when the field is too narrow to show it: the one line
        # showed only the end of "2.dat, sim004.sxm", so which spectrum was cited could not be read (p3check1, finding 2)
        self._evidence_full = QLabel()
        self._evidence_full.setObjectName("report.evidence_full")
        # its space is reserved for good: EVIDENCE_LINES lines, empty while the field shows everything, so that nothing below it
        # (unit, sigma, Submit, Finish) moves while the reader types (screen trial 23: Submit jumped by 28 px and two clicks
        # landed on labels). A longer list fills the lines and ends in "+N more"; it never grows.
        self._evidence_full.setWordWrap(False)
        self._evidence_full.setProperty("role", "label")
        self._evidence_full.setProperty("tone", "note")
        self._evidence_full.setFont(style.mono_font(16))
        self._evidence_full.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._evidence_full.setMinimumWidth(0)
        self._evidence_full.setFixedHeight(EVIDENCE_LINES * self._evidence_full.fontMetrics().lineSpacing() + style.px(2))

        def cell(key: str, beside: QWidget | None = None, wrap: bool = False) -> QWidget:
            """A field with its label above it (and a button at the end of the label's line, if ``beside``)."""
            lab = QLabel(_field_label(key, lang, key == "value"))
            lab.setObjectName(f"report.label.{key}")
            lab.setTextFormat(Qt.TextFormat.RichText)
            lab.setProperty("role", "label")
            self.labels[key] = lab
            e = WrapField(3) if wrap else SelectAllLineEdit()   # the note wraps: a line edit a third of the form wide showed a few words
            e.setObjectName(f"report.{key}")
            e.setFont(style.mono_font(18))
            if wrap:
                e.fit_height()
            e.returnPressed.connect(self._submit)                # Enter in any field sends the report, like the button
            self.fields[key] = e
            box = QWidget()
            col = QVBoxLayout(box)
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(style.px(3))
            if beside is None:
                col.addWidget(lab)
            else:
                line = QHBoxLayout()                             # a button at the end of the label's line
                line.setSpacing(style.px(8))
                line.addWidget(lab, 1)
                line.addWidget(beside)
                col.addLayout(line)
            col.addWidget(e)
            if key == "evidence":                                # the whole list, if the field cannot show it
                col.addWidget(self._evidence_full)
            return box

        def row(*cells: tuple[QWidget, int]) -> None:
            """Cells side by side, as wide as their stretch says."""
            h = QHBoxLayout()
            h.setSpacing(style.px(8))
            for w, stretch in cells:
                h.addWidget(w, stretch)
            form.addLayout(h)

        # the button that appends the name of the file the viewer shows sits at the end of the evidence label's line, so
        # that the field has the whole width of the form; its label says what it adds ("+ shown file"), and it is in one
        # language: the line is not wide. The field's own description is its placeholder.
        self.viewed_button = QPushButton("+ " + one("rep_viewed", lang))
        self.viewed_button.setProperty("role", "tiny")             # as tall as a label's line: the form is no taller for it
        self.viewed_button.setObjectName("report.viewed")
        self.viewed_button.clicked.connect(self._append_viewed)
        # four rows: the unit and the uncertainty stand in the frame's row, and the note has the last row to itself, as wide
        # as the form (so three lines show the longest note of the tests). The space reserved under the evidence field for its list
        # (EVIDENCE_LINES lines, so that Submit never moves: screen trial 23) is paid for here: P1 in English leaves the page 49 px at
        # 1920x1053 (27 in DejaVu) and the reservation takes 44.
        row((cell("value"), 1), (cell("x_nm"), 1), (cell("y_nm"), 1))
        row((cell("frame"), 2), (cell("unit"), 1), (cell("sigma"), 1))
        row((cell("evidence", beside=self.viewed_button), 1))
        ev = self.fields["evidence"]
        ev.setPlaceholderText(one("rep_evidence_hint", lang))
        ev.textChanged.connect(lambda _t: self._show_full_evidence())
        row((cell("note", wrap=True), 1))
        v.addSpacing(style.px(4))
        v.addLayout(form)
        v.addSpacing(style.px(4))

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

        v.addSpacing(style.px(4))
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
        if len(self.radios) == 1:                        # one claim to report (P4): there is nothing to choose between
            next(iter(self.radios.values())).setChecked(True)

    @staticmethod
    def _title(text: str) -> QLabel:
        lab = QLabel(text)
        lab.setWordWrap(True)                  # "报告结果 Report a result (ReportResult)" is wider than the column in DejaVu
        strong(lab)
        return lab

    # ── the form ──
    def claim_id(self) -> str:
        return next((cid for cid, r in self.radios.items() if r.isChecked()), "")

    def _claim_picked(self, cid: str) -> None:
        """Each result keeps its own entries (screen trial 23: value, evidence, note and position carried over from one result
        to the next, and a radius would have been submitted with the files of another result). What the fields hold is kept
        for the result being left, and the picked one shows what it held when it was left (as typed or as submitted); a result not
        opened before shows empty fields and its unit, the one thing prefilled."""
        if self._current and self._current != cid:
            self._entries[self._current] = {k: e.text() for k, e in self.fields.items()}
        self._current = cid
        saved = self._entries.get(cid)
        for key, e in self.fields.items():
            e.setText(saved[key] if saved else "")
        if not saved:
            self.fields["unit"].setText(self._claims[cid]["unit"])
        self._mark_required(cid)
        self._enable_position(not self._takes_no_position(cid))

    def _takes_no_position(self, cid: str) -> bool:
        return bool(cid) and not self._claims[cid].get("position")

    def _enable_position(self, on: bool) -> None:
        """Grey out the position fields, and keep them empty, for a claim the rule card gives no position: they only
        lost their asterisk before, and a reader of P3's form could not tell whether x_nm and y_nm were wanted
        (p3check1, finding 3). Picking a claim that names a place enables them again."""
        for key in ("x_nm", "y_nm", "frame"):
            if not on:
                self.fields[key].setText("")
            self.fields[key].setEnabled(on)
            self.labels[key].setEnabled(on)

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
        if not self._viewed:                               # nothing in the viewer: say so rather than do nothing
            self.notice.emit(tr("rep_nothing_shown", self.lang))
            return
        have = [t.strip() for t in self.fields["evidence"].text().replace("，", ",").split(",") if t.strip()]
        if self._viewed not in have:
            have.append(self._viewed)
        self.fields["evidence"].setText(", ".join(have))

    def _show_full_evidence(self) -> None:
        """The cited files, in grey under the field, when the field cannot show them all: as many as fit in the reserved lines,
        then "+N more". The label keeps its place and its height whether it shows anything or not."""
        ev = self.fields["evidence"]
        text = ev.text().strip()
        room = ev.width() - 2 * style.px(8)
        wide = bool(text) and ev.fontMetrics().horizontalAdvance(text) > room
        self._evidence_full.setText(self._evidence_lines(text, room) if wide else "")

    def _evidence_lines(self, text: str, room: int) -> str:
        """The files of ``text`` packed into at most EVIDENCE_LINES lines of ``room`` px; what does not fit is told as "+N more"."""
        fm = self._evidence_full.fontMetrics()
        files = [t.strip() for t in text.replace("，", ",").split(",") if t.strip()]
        lines: list[str] = []
        i = 0
        while i < len(files) and len(lines) < EVIDENCE_LINES:
            line = ""
            while i < len(files):
                trial = f"{line}, {files[i]}" if line else files[i]
                if line and fm.horizontalAdvance(trial + ",") > room:
                    break
                line = trial
                i += 1
            if not line:
                break
            lines.append(line)
        left = len(files) - i
        if left > 0:
            more = f" +{left} more"
            last = lines[-1] if lines else ""
            while last and fm.horizontalAdvance(last + "…" + more) > room:
                last = last[:-1]
            lines[-1:] = [last + "…" + more]
        else:
            lines = [ln + "," if k < len(lines) - 1 else ln for k, ln in enumerate(lines)]
        return "\n".join(lines)

    def resizeEvent(self, ev) -> None:
        super().resizeEvent(ev)
        self._show_full_evidence()

    def fill_position(self, x_nm: float, y_nm: float, frame: str) -> None:
        if self._takes_no_position(self.claim_id()):       # greyed and empty: say why nothing was copied
            self.notice.emit(tr("rep_no_position", self.lang))
            return
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

    def finish_not_run(self) -> None:
        """The end was asked for but did not happen (the call was refused or not run): the button is back."""
        self.finish_ask.hide()
        self.finish_begin.show()

    def set_finished(self) -> None:
        self.finish_ask.hide()
        self.finish_begin.hide()
        self.finish_message.setText(tr("over", self.lang))
        self.finish_message.show()


class DataTab(QWidget):
    """What the instrument saved, newest first, in three lists: frames, spectra and move traces (the Z and
    current recorded while the tip moved; the instrument keeps them with the spectra, and so does the viewer,
    but a list of their own says what they are). Each item is three short lines (its name, start time and
    channels; its size; where it is), which carry every column of a table in the width of this column without
    cutting any of them off. A click shows the item in the viewer; a wheel notch moves a list by one item
    (:class:`~stmbench.gui.controls.StepList`)."""

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
        self.traces_title = QLabel()
        strong(self.traces_title)
        v.addWidget(self.traces_title)
        self.traces = self._list("data.traces")
        v.addWidget(self.traces, 2)
        self.frames.itemClicked.connect(lambda it: self.frame_clicked.emit(it.data(Qt.ItemDataRole.UserRole)))
        for w in (self.spectra, self.traces):                 # a trace is read from the instrument as a spectrum is
            w.itemClicked.connect(lambda it: self.spectrum_clicked.emit(it.data(Qt.ItemDataRole.UserRole)))
        self._junction: dict[str, dict] = {}
        self.set_data([], [])

    def _list(self, name: str) -> StepList:
        w = StepList()                                    # a wheel notch moves it by one entry
        w.setObjectName(name)
        w.setAlternatingRowColors(True)
        w.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        w.setFont(style.sans_font(17))
        w.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        w.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        w.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)      # no line is longer than the column
        w.setWordWrap(True)                                                       # and one that is wraps rather than hides
        w.setTextElideMode(Qt.TextElideMode.ElideNone)
        return w

    def set_data(self, frames: list[dict], spectra: list[dict], junction: dict[str, dict] | None = None) -> None:
        """``spectra`` is the instrument's list: the move traces are in it, with ``kind`` "move". ``junction`` is what
        the window knows of how each spectrum was taken (:func:`~stmbench.gui.fmt.sweep_text` and
        :func:`~stmbench.gui.fmt.junction_text` read it), by name."""
        self._junction = junction or {}
        traces = [s for s in spectra if s.get("kind") == "move"]
        spectra = [s for s in spectra if s.get("kind") != "move"]
        self.frames_title.setText(f"{tr('frames', self.lang)} ({len(frames)})")
        self.spectra_title.setText(f"{tr('spectra', self.lang)} ({len(spectra)})")
        self.traces_title.setText(f"{tr('move_traces', self.lang)} ({len(traces)})")
        self._fill(self.frames, [self._frame_lines(f) for f in reversed(frames)])
        self._fill(self.spectra, [self._spectrum_lines(s, self._junction.get(s["spectrum"])) for s in reversed(spectra)])
        self._fill(self.traces, [self._trace_lines(s) for s in reversed(traces)])

    @staticmethod
    def _frame_lines(f: dict) -> tuple[str, list[str]]:
        """``(name, lines)``: name, start and channels; the size; the centre."""
        channels = " ".join(_CHANNEL[c] for c in f.get("channels", []))
        cx, cy = f["center_nm"]
        return (f["frame"], [f"{f['frame']}   {fmt.hms(f['t_start_s'])}   {channels}",
                             f"size {fmt.num(f['size_nm'], 5)} nm · {f['pixels']} px · angle {fmt.num(f['angle_deg'], 4)}°",
                             f"centre ({fmt.num(cx, 5)}, {fmt.num(cy, 5)}) nm"])

    @staticmethod
    def _spectrum_lines(s: dict, junction: dict | None = None) -> tuple[str, list[str]]:
        """Name; kind and time; the sweep and the junction it started from (when the window knows them); the tip and
        the points."""
        extra = [t for t in (fmt.sweep_text(junction), fmt.junction_text(junction)) if t]
        return (s["spectrum"], [s["spectrum"], f"{s['kind']} · {fmt.hms(s['t_s'])}", *extra,
                                f"tip ({s['tip_nm'][0]:.3f}, {s['tip_nm'][1]:.3f}) nm · {s['points']} points"])

    @staticmethod
    def _trace_lines(s: dict) -> tuple[str, list[str]]:
        """A move trace: its name; when it began; where the move ended and how many points were recorded."""
        end = f"to ({s['tip_nm'][0]:.3f}, {s['tip_nm'][1]:.3f}) nm · " if s.get("tip_nm") else ""
        return (s["spectrum"], [s["spectrum"], f"move trace · {fmt.hms(s['t_s'])}", f"{end}{s['points']} points"])

    @staticmethod
    def _fill(w: QListWidget, rows: list[tuple[str, list[str]]]) -> None:
        w.clear()
        for name, lines in rows:
            item = QListWidgetItem(chr(10).join(lines))
            item.setData(Qt.ItemDataRole.UserRole, name)
            w.addItem(item)

    def select(self, kind: str, name: str) -> None:
        """Mark the item now in the viewer (no signal)."""
        for w, k in ((self.frames, "frame"), (self.spectra, "spectrum"), (self.traces, "spectrum")):
            w.clearSelection()
            if k == kind:
                for r in range(w.count()):
                    if w.item(r).data(Qt.ItemDataRole.UserRole) == name:
                        w.setCurrentRow(r)


class LogTab(QuietBrowser):
    """Every call, newest first: instrument time, the action with its parameters, the reply or the
    refusal (red). A call that has not run yet says so until it does; one that was dropped or turned away
    because of a refusal says that. The acknowledgement of a refusal is a line of its own, in grey: a window
    event, not a call."""

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

    def expand(self, job_id: int, calls: list[tuple[int, str, dict]], t_s: float | None) -> None:
        """Replace the entry of one job by one entry per call it made (a move with during-move values), in the same place
        of the log, each waiting to be told what it answered."""
        new = [{"id": i, "label": label, "args": fmt.params_text(args), "state": "running", "t": t_s, "ok": None, "text": ""}
               for i, label, args in calls]
        at = next((k for k, e in enumerate(self._entries) if e["id"] == job_id), None)
        if at is None:
            self._entries += new
        else:
            self._entries[at:at + 1] = new
        self._render()

    def cancelled(self, job_id: int) -> None:
        """A call dropped before it started: it stays in the log, said to be cancelled."""
        self._set(job_id, state="cancelled")

    def not_run(self, job_id: int, held: bool = False) -> None:
        """A call that was not run because an earlier call was refused: it was waiting behind the refusal, or
        (``held``) it was clicked afterwards, while the refusal was not acknowledged."""
        self._set(job_id, state="held" if held else "skipped")

    def note(self, text: str, t_s: float | None) -> None:
        """A window event that is not a call (the acknowledgement of a refusal): it is in this log and in no
        ledger or export."""
        self._entries.append({"id": None, "label": "", "args": "", "state": "note", "t": t_s, "ok": None, "text": text})
        self._render()

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
            if e["state"] == "note":                      # a window event: grey, on one line, no call behind it
                out.append(f"<div style='margin-bottom:7px'><span style=\"font-family:'{mono}';color:{c['muted']}\">"
                           f"{when}</span> <span style='color:{c['muted']}'>{escape(e['text'])}</span></div>")
                continue
            head = (f"<span style=\"font-family:'{mono}';color:{c['muted']}\">{when}</span> "
                    f"<b style=\"font-family:'{mono}'\">{escape(e['label'])}</b> "
                    f"<span style=\"font-family:'{mono}'\">{escape(e['args'])}</span>")
            if e["state"] == "queued":
                tail = f"<span style='color:{c['warn']}'>{escape(tr('queued', self.lang))}</span>"
            elif e["state"] == "running":
                tail = f"<span style='color:{c['warn']}'>{escape(tr('log_running', self.lang))}</span>"
            elif e["state"] == "cancelled":
                tail = f"<span style='color:{c['muted']}'>{escape(tr('cancelled', self.lang))}</span>"
            elif e["state"] == "skipped":
                tail = f"<span style='color:{c['bad']}'>{escape(tr('not_run', self.lang))}</span>"
            elif e["state"] == "held":
                tail = f"<span style='color:{c['bad']}'>{escape(tr('not_run_held', self.lang))}</span>"
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
    """The instrument's facts above the manual: everything an agent is told about the instrument.

    Why a wide view. In the 360 px column the manual's 64 lines wrap into about a hundred, and a page of the pane
    shows 27 of them: screen trials 4 and 5 spent five to seven steps scrolling it. The button at the top of the tab
    asks the window for the whole row under the status strip (:meth:`MainWindow._set_wide`): the facts then stand
    beside the text instead of above it, and the same text, the one an MCP agent gets, is two screens or fewer. The
    button only asks: the window hides the other columns and gives them back."""

    wide_toggled = Signal(bool)

    def __init__(self, manual: str, lang: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.lang = lang
        self._wide = False
        self._box = QBoxLayout(QBoxLayout.Direction.TopToBottom, self)       # left to right in the wide view
        self._box.setContentsMargins(style.px(6), style.px(6), style.px(6), style.px(6))
        self._box.setSpacing(style.px(8))
        self.side = QWidget()                            # the button and the facts
        side = QVBoxLayout(self.side)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(style.px(4))
        self.toggle = QPushButton()
        self.toggle.setObjectName("manual.wide")
        self.toggle.setAutoDefault(False)
        self.toggle.clicked.connect(self._clicked)
        title = QLabel(tr("facts_title", lang))
        strong(title)
        self.facts = FactsTable(lang)
        side.addWidget(self.toggle)
        side.addWidget(title)
        side.addWidget(self.facts)
        side.addStretch(1)
        self.text = _text_view(manual, mono=True, size=17)
        self.text.setObjectName("manual.text")
        self._box.addWidget(self.side)
        self._box.addWidget(self.text, 1)
        self._label()

    @property
    def wide(self) -> bool:
        return self._wide

    def _label(self) -> None:
        self.toggle.setText(tr("manual_narrow" if self._wide else "manual_wide", self.lang))

    def _clicked(self) -> None:
        self.set_wide(not self._wide)
        self.wide_toggled.emit(self._wide)

    def set_wide(self, on: bool) -> None:
        """Lay the tab out for the wide view (the facts beside the text, as wide as the column was) or for the column."""
        if on == self._wide:
            return
        if on:
            self.side.setFixedWidth(self.side.width())
        else:
            self.side.setMinimumWidth(0)
            self.side.setMaximumWidth(16777215)
        self._wide = on
        self._box.setDirection(QBoxLayout.Direction.LeftToRight if on else QBoxLayout.Direction.TopToBottom)
        self._label()


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

    def minimumSizeHint(self) -> QSize:
        """Wide enough for the report form too. The form sits in a scroll area (below), whose own minimum says
        nothing of the form's, so the window sized the column by the other pages and the form was cut at the right
        edge: on Linux (DejaVu), 392 px (English) and 403 px (Chinese) of form in 358 px of column; trials 9 and
        10 lost the y_nm field, a label, the "+ shown file" button and the right of Submit and Finish. The form's
        own lines are kept short enough to fit the set 360 px (the two field labels, the title); this is for a font
        or a Qt version in which they do not, where the column grows instead of the form being cut. The pane's frame
        is added on both sides."""
        hint = super().minimumSizeHint()
        frame = self.style().pixelMetric(QStyle.PixelMetric.PM_DefaultFrameWidth, None, self)
        return QSize(max(hint.width(), self.report.minimumSizeHint().width() + 2 * frame), hint.height())

    @property
    def facts(self) -> FactsTable:
        return self.manual.facts

    def show_tab(self, key: str) -> None:
        self.setCurrentIndex(self.KEYS.index(key))

    def tab_key(self) -> str:
        return self.KEYS[self.currentIndex()]
