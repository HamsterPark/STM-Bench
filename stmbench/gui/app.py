"""The desktop console's window and its entry point (``stmbench.cli gui``).

One window, three columns under a status strip: the task, the report form, the data and the log on
the left; the viewer with its markers in the middle; the instrument's actions on the right, with the
last five results under them. Every
click that touches the instrument becomes a job on the worker thread (:mod:`stmbench.gui.worker`);
this thread only draws what the jobs bring back. The window adds no hint: it shows what the instrument
said, in the instrument's words, and the instrument is the only validator.

A refused call of the operator's holds the console: the calls queued behind it are dropped, and every operator
call clicked afterwards is turned away, until the reader clicks the button of the red band that the refusal puts
on the status strip (:mod:`stmbench.gui.worker`, :mod:`stmbench.gui.status`). The acknowledgement is a window
event: it is in the Log and in no ledger or export.

Two more things live in this module rather than in a column: the line beside the right tabs that tells the range
of the numeric field entered last (:meth:`ActionPanel.field_help`), and the Manual tab's wide view, in which the left
column takes the whole row under the strip and the viewer and the right column are hidden until the view is left
(:meth:`MainWindow._set_wide`).

Two ways to run it. *Dev*: a scenario and a seed on the command line, a run directory (the usual full
ledger at the end) with the export beside it; closing the window finishes the episode, judged as it
stands. *Sealed*: the seed arrives on stdin and appears nowhere else; the window opens full screen (a
maximized window keeps the title bar GNOME draws, so its size depends on the window manager and a
full-screen window's is the screen's) and its close button only says that the program stays open (the
process ends when it is killed, and everything the agent did is already in the export).
"""
from __future__ import annotations

import signal
import sys
import time
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QVBoxLayout, QWidget

from . import fmt, jobs, layout, style
from .center import THUMB_PX, CenterColumn, thumb_pixmap
from .controls import ElidedLabel, Tabs
from .datacopy import DataCopier
from .labels import one, two_lines, tr, with_console_notes
from .panels import ActionPanel
from .recent import RecentList
from .session import GuiSession, SpecError, evidence_files, make_private_dir, read_spec
from .sidebar import Sidebar
from .status import StatusStrip
from .worker import Job, Worker

#: the side columns' widths at ui scale 1: the right one (the action tabs and the recent results) is 704 px wide on a
#: 1920 px screen, which leaves the centre column 824 px; a column that its contents need more of gets it
#: (:func:`column_widths`)
LEFT_PX, RIGHT_PX = 360, 704
#: what a window manager's title bar may take from a maximized window's height (GNOME: 37 px, so 1016 of a 1080 screen)
FRAME_PX = 37
#: frames kept for instant re-viewing (arrays of at most 1024 x 1024 doubles): a byte budget
CACHE_BYTES = 256 * 1024 * 1024


def column_needs(sidebar: QWidget, panels: Iterable[ActionPanel]) -> tuple[int, int]:
    """The width the left and the right column need with the fonts this machine has: the narrowest the sidebar
    can be, and the widest line of an action panel (the Operate tab's, or the Spectroscopy tab's) with the tab
    pane's margins."""
    right = max(p.widget().minimumSizeHint().width() for p in panels) + 2 * style.px(3)
    return sidebar.minimumSizeHint().width(), right


def column_widths(sidebar_need: int, right_need: int) -> tuple[int, int]:
    """The left and the right column's widths in pixels: the set ones (:data:`LEFT_PX`, :data:`RIGHT_PX`), or what
    their contents need when that is more. The centre column keeps the rest of the window, which at 1920 px has
    143 px to give before it is at its own minimum (681 px in DejaVu; the window's own fonts leave more).

    Why a column grows instead of the window shrinking. Every size of the window scales together, the column's
    width as much as the line that does not fit it, so a shrink leaves the line as wide as the column. Measured
    on 2026-10-27 on Windows, offscreen, with DejaVu in front of the system fonts: the Chinese labels (each one
    Chinese, then English) needed 745 px of the right column's 704, and after fit_scale had shrunk every size
    to 0.733 they needed 537 of 516; the English ones needed 642. Taking the width from the centre column
    fits them at scale 1.0."""
    return max(style.px(LEFT_PX), sidebar_need), max(style.px(RIGHT_PX), right_need)


class MainWindow(QMainWindow):
    _turned_away = Signal(object)                        # clicks the worker did not accept, to be told in order

    def __init__(self, session: GuiSession, *, lang: str = "zh", sealed: bool = False, data_dir: str | Path | None = None):
        super().__init__()
        self.session = session
        # with --data-dir: plain-text copies of what the Data tab lists (stmbench.gui.datacopy); without it nothing is written
        self._copier = DataCopier(session, data_dir) if data_dir else None
        self.lang = lang
        self.sealed = sealed
        self.setWindowTitle(f"{tr('app', lang)} · {session.scenario_id}")
        self._open = 0                                   # jobs submitted whose outcome is not drawn yet
        self._running = ""
        self._running_job: Job | None = None
        self._refusal: jobs.Refusal | None = None        # the refusal the band shows: the console holds while it is there
        self._state: dict | None = None                  # the last status(): what the On / Off buttons show
        self._status: dict | None = None                 # the last status() read: the junction the next call starts from
        self._junction: dict[str, dict] = {}             # per spectrum: the sweep sent and the bias and setpoint it started at
        self._finished = False
        self._t_s = 0.0                                  # the last instrument time the window was told
        self._data: dict = {"frames": [], "spectra": []}
        self._cache: OrderedDict[tuple, dict] = OrderedDict()
        self._cache_bytes = 0
        self._thumbs: dict[str, QPixmap] = {}
        self._want: tuple | None = None                  # what the reader last asked to see
        self._viewed: tuple[str, str] | None = None

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        col = QVBoxLayout(root)
        col.setContentsMargins(style.px(8), style.px(5), style.px(8), style.px(6))
        col.setSpacing(style.px(5))
        self.status = StatusStrip(session.scenario_id, lang)
        col.addWidget(self.status)
        row = QHBoxLayout()
        row.setSpacing(style.px(8))
        col.addLayout(row, 1)
        self._row = row

        self.sidebar = Sidebar(session.task_text(), with_console_notes(session.manual(), lang, data_dir), session.claims(), lang)
        self.center = CenterColumn(lang)
        self.operate = ActionPanel(layout.OPERATE, self.run_action, lang, submit_move=self.run_move_during)
        self.spectroscopy = ActionPanel(layout.SPECTROSCOPY, self.run_action, lang)
        left_px, right_px = column_widths(*column_needs(self.sidebar, (self.operate, self.spectroscopy)))
        self.sidebar.setFixedWidth(left_px)
        self._left_px = left_px
        self._wide = False                               # the Manual tab's wide view: the left column has the whole row
        self.right = Tabs()
        self.right.setObjectName("right.tabs")
        self.right.addTab(self.operate, two_lines("tab_operate", lang).replace("\n", " "))
        self.right.addTab(self.spectroscopy, two_lines("tab_spectroscopy", lang).replace("\n", " "))
        self.right.tabBar().setObjectName("right.tabbar")
        self.right.tabBar().setExpanding(True)
        self.right.setFixedWidth(right_px)
        # the range of the numeric field entered last, in the part of the tab row the two tabs leave empty: no height
        # of its own, so that nothing below it moves, and nothing to hover over
        self.field_help = ElidedLabel()
        self.field_help.setObjectName("field.help")
        self.field_help.setProperty("role", "help")
        self.field_help.setFont(style.sans_font(17, bold=True))          # bold: it is read in a picture of 1280 px
        corner = QWidget()
        corner_row = QHBoxLayout(corner)
        corner_row.setContentsMargins(style.px(8), 0, 0, 0)
        corner_row.addWidget(self.field_help)
        # the tab widget gives a corner widget the size of its hint, which a layout takes from the label's own width
        self.field_help.setFixedWidth(max(style.px(200), right_px - self.right.tabBar().sizeHint().width() - style.px(14)))
        self.right.setCornerWidget(corner, Qt.Corner.TopRightCorner)
        corner.show()                                    # setting the parent hid it
        self._help_field: QWidget | None = None          # the field the line above describes
        for panel in (self.operate, self.spectroscopy):
            panel.field_entered.connect(lambda a, k, p=panel: self._field_entered(p, a, k))
        # the line describes the field entered last: it goes with the tab (screen trial 24: it still read "tip y" on the
        # Spectroscopy tab) and with the focus, when that moves to another widget of this window
        self.right.currentChanged.connect(lambda _i: self._clear_field_help())
        QApplication.instance().focusChanged.connect(self._focus_changed)
        self.recent = RecentList(lang)
        self.right_column = QWidget()                    # the tabs, and under them what the last calls answered
        rc = QVBoxLayout(self.right_column)
        rc.setContentsMargins(0, 0, 0, 0)
        rc.setSpacing(style.px(6))
        rc.addWidget(self.right, 1)
        rc.addWidget(self.recent)
        row.addWidget(self.sidebar)
        row.addWidget(self.center, 1)
        row.addWidget(self.right_column)

        # the reader's gestures
        self.sidebar.report.submitted.connect(self._report)
        self.sidebar.report.finish_requested.connect(self._finish)
        self.sidebar.report.notice.connect(lambda text: self.recent.push("note", text))
        self.sidebar.data.frame_clicked.connect(self._pick_frame)
        self.sidebar.data.spectrum_clicked.connect(self.request_spectrum)
        self.center.thumb_clicked.connect(self._pick_frame)
        self.center.view_changed.connect(self._view_changed)
        self.center.tip_from_marker.connect(lambda x, y: self._fill("move_tip", "x_nm", "y_nm", x, y))
        self.center.scan_from_marker.connect(lambda x, y: self._fill("scan", "center_x_nm", "center_y_nm", x, y))
        self.center.report_from_marker.connect(self._report_from_marker)
        self.center.notice.connect(lambda text: self.recent.push("note", text))
        self.status.cancel_requested.connect(self._cancel_queued)
        self.status.acknowledge_requested.connect(self._acknowledge)
        self.sidebar.manual.wide_toggled.connect(self._set_wide)
        self.sidebar.currentChanged.connect(self._tab_changed)

        self.worker = Worker(session, self)
        self.worker.started_job.connect(self._on_started)
        self.worker.finished_job.connect(self._on_finished)
        self.worker.queue_changed.connect(self._on_queue)
        self.worker.dropped.connect(self._on_dropped)
        self._turned_away.connect(self._turned_away_now, Qt.ConnectionType.QueuedConnection)
        self.worker.start()
        self._submit(Job("boot", "start", {}, jobs.run_boot), log=False)

    # ── the range line beside the right tabs ──
    def _field_entered(self, panel: ActionPanel, action: str, param: str) -> None:
        self._help_field = panel.controls.get((action, param))
        self.field_help.set_full_text(panel.field_help(action, param))

    def _clear_field_help(self) -> None:
        self._help_field = None
        self.field_help.set_full_text("")

    def _focus_changed(self, old: QWidget | None, new: QWidget | None) -> None:
        """Focus left the described field for another widget of this window outside the action panels (a report field, a
        viewer control): the line is cleared. Another numeric field shows its own range as it is entered, and a button of the
        panels (Set setpoint after a refusal) leaves the line on the field it belongs to. Focus that leaves the window
        (deactivation) or goes to a popup is not a move to another widget of this window, and the line stays."""
        if self._help_field is None or old is not self._help_field or new is None or new is old:
            return
        if new.window() is not self.window() or new.window() is not self:
            return
        if self.operate.isAncestorOf(new) or self.spectroscopy.isAncestorOf(new):
            return                                       # a field of the panels sets the line itself; a button of theirs keeps it
        self._clear_field_help()

    # ── jobs ──
    def _submit(self, job: Job, log: bool = True) -> int:
        self._open += 1
        if log:
            self.sidebar.log.add(job.id, job.label, job.args, self._t_s)
        if not self.worker.submit(job):                  # a click while a refusal is not acknowledged: not run
            self._open -= 1
            self._turned_away.emit([job])                # queued: after the refusal's own result has been drawn
        return job.id

    def _not_run(self, jobs: list[Job], held: bool = False) -> None:
        """Calls that were not run because an earlier call was refused: said in the log, in the recent results
        and on the band. ``held``: they were clicked after the refusal, while it was not acknowledged; else they
        were waiting behind it. The refusal itself stays on the strip's last line, which nothing that did not run
        replaces."""
        why = tr("not_run_held" if held else "not_run", self.lang)
        for job in jobs:
            self.sidebar.log.not_run(job.id, held)
            if job.kind == "finish":
                self.sidebar.report.finish_not_run()
            elif job.kind == "report":
                self.sidebar.report.set_reply(False, why)
        self.recent.not_run([j.label for j in jobs], why, key="not_run_held" if held else "not_run")
        if self._refusal is not None:
            self._refusal.not_run += [j.label for j in jobs]
            self.status.show_refusal(self._refusal)

    def _turned_away_now(self, jobs: list[Job]) -> None:
        """Clicks the worker turned away, told after the refusal's own result was drawn."""
        self._not_run(jobs, held=True)

    def _on_dropped(self, jobs: list[Job]) -> None:
        self._open -= len(jobs)
        self._not_run(jobs)
        self._show_state()

    def _hold(self, job: Job, out: jobs.Outcome) -> None:
        """A call of the operator's failed: the worker holds, and the band says what was refused."""
        self._refusal = jobs.Refusal(fmt.call_text(job.label, job.args), out.text, out.refused)
        self.status.show_refusal(self._refusal)

    def _acknowledge(self) -> None:
        """The reader has read the refusal: operator calls are accepted again. A window event, not an instrument
        call; the Log lists it, no ledger or export does."""
        self.worker.acknowledge()
        refusal, self._refusal = self._refusal, None
        self.status.show_refusal(None)
        if refusal is not None:
            self.sidebar.log.note(f"{tr('acknowledged', self.lang)} {refusal.call}", self._t_s)
        self._show_state()

    def run_action(self, name: str, args: dict) -> int:
        """Queue one action with the values the fields held at the click."""
        args = dict(args)
        return self._submit(Job("act", name, args, lambda s, n=name, a=args: jobs.run_act(s, n, a)))

    def run_move_during(self, args: dict, during: dict) -> int:
        """Queue a move of the tip with values set around it (the Tip group's "during move" fields): one job that runs
        the calls an operator would click one by one (:func:`stmbench.gui.jobs.run_move_during`). The Log shows the
        job as the move until it has run, then lists every call it made."""
        args = {k: v for k, v in args.items()}
        during = {name: dict(a) for name, a in during.items()}
        shown = {**args, "during_move": "; ".join(f"{name} {fmt.params_text(a)}" for name, a in during.items())}
        return self._submit(Job("act", "move_tip", shown, lambda s, a=args, d=during: jobs.run_move_during(s, a, d)))

    def _apply_steps(self, job: Job, out: jobs.Outcome) -> None:
        """A move with during-move values has run: each call it made is drawn as if it had been clicked (a Log entry,
        a recent result, the readings), in the order it ran. A refused call holds the console as a click would; the
        last refusal is the one the band shows (a refused putting back, after a refused move, is the one that ends it)."""
        calls = [Job("act", s.label, s.args, lambda sess, s=s: s) for s in out.steps]
        self.sidebar.log.expand(job.id, [(c.id, c.label, c.args) for c in calls], self._t_s)
        refused = any(not s.ok for s in out.steps)
        for call, step in zip(calls, out.steps):
            # the Log and the readings get every call; the band and the recent results get the move's own reply once (screen
            # trial 24: the band ended on the last set-back and one move filled all five rows), a refusal as it is, and, when
            # a call was refused, the move's reply too if the move went through (its trace name is what a drag cites)
            self._apply(call, step, quiet=step.ok and not (refused and step.label == "move_tip"))
        if not refused:
            move = next((s for s in out.steps if s.label == "move_tip"), None)
            if move is not None:
                text = fmt.reply_summary("move_tip", move.reply)
                back = self._set_back_text(out.steps)
                self._succeed(f"{text} · {back}" if back else text)

    def _set_back_text(self, steps: list) -> str:
        """What a move with during-move values put back, from the calls that ran after the move: "set back: current range
        10 nA, setpoint 50 pA". Empty when nothing was put back."""
        names = {"set_current_range": ("sb_range", "full_scale_na", "nA"), "set_setpoint": ("sb_setpoint", "current_pa", "pA"),
                 "set_bias": ("sb_bias", "bias_v", "V")}
        moved = False
        parts = []
        for s in steps:
            if s.label == "move_tip":
                moved = True
            elif moved and s.ok and s.label in names:
                key, param, unit = names[s.label]
                parts.append(f"{one(key, self.lang)} {fmt.num(s.args.get(param))} {unit}")
        return f"{one('set_back', self.lang)}: {', '.join(parts)}" if parts else ""

    def _report(self, body: dict) -> None:
        sent = {k: v for k, v in body.items() if v != ""}
        self._submit(Job("report", "report_result", sent, lambda s, b=dict(body): jobs.run_report(s, b)))

    def _finish(self) -> None:
        self._submit(Job("finish", "finish", {}, jobs.run_finish))

    def request_frame(self, name: str, channel: str = "z", direction: str = "forward") -> None:
        """Show a saved frame: at once when its pixels are at hand, else after the instrument gives them."""
        self._want = ("frame", name, channel, direction)
        frame = self._cache.get((name, channel, direction))
        if frame is not None:
            self._cache.move_to_end((name, channel, direction))
            self._display_frame(frame)
            return
        self._submit(Job("view_frame", name, {"channel": channel, "direction": direction},
                         lambda s: jobs.run_view_frame(s, name, channel, direction)), log=False)

    def request_spectrum(self, name: str) -> None:
        self._want = ("spectrum", name)
        self._submit(Job("view_spectrum", name, {}, lambda s: jobs.run_view_spectrum(s, name)), log=False)

    def _pick_frame(self, name: str) -> None:
        meta = next((f for f in self._data["frames"] if f["frame"] == name), {})
        channel = self.center.current_channel()
        if channel not in meta.get("channels", ["z", "current"]):
            channel = "z"
        self.request_frame(name, channel, self.center.current_direction())

    def _view_changed(self) -> None:
        if self._viewed and self._viewed[0] == "frame":
            self.request_frame(self._viewed[1], self.center.current_channel(), self.center.current_direction())

    # ── worker signals ──
    def _on_queue(self, _waiting: int) -> None:
        self._show_state()

    def _on_started(self, job: Job) -> None:
        self._running = job.label
        self._running_job = job
        self._show_state()
        self.sidebar.log.running(job.id)

    def _show_state(self) -> None:
        # the queue is read from the worker, not from a signal's payload: signals from two threads can
        # arrive out of order
        queued = self.worker.queued()
        job = self._running_job if self._running else None
        self.status.set_state(bool(self._running), self._running, len(queued), self._finished,
                              held=self._refusal is not None)
        self.status.set_busy(fmt.describe_job(job.kind, job.label, job.args) if job else None, len(queued),
                             sum(1 for j in queued if j.kind != "boot"))
        scans = [job, *queued]
        self.center.set_scan_pending(any(j is not None and j.kind == "act" and j.label == "scan" for j in scans))

    def _cancel_queued(self) -> None:
        """Drop the calls that have not started: they are not sent to the instrument, so they are on no
        record but this window's log, which says they were cancelled."""
        dropped = self.worker.cancel_queued()
        for job in dropped:
            self._open -= 1
            self.sidebar.log.cancelled(job.id)
        if dropped:
            names = ", ".join(j.label for j in dropped)
            self.recent.push("note", f"{tr('cancelled_note', self.lang)}: {names}")
        self._show_state()

    def _on_finished(self, job: Job, out: jobs.Outcome) -> None:
        try:
            self._apply(job, out)
        finally:
            self._open -= 1
            self._running = ""
            self._running_job = None
            self._show_state()

    def _apply(self, job: Job, out: jobs.Outcome, quiet: bool = False) -> None:
        """Draw what a job brought back. ``quiet``: a call of a move with during-move values that went through is not told
        on its own (the band and the list of recent results get one entry for the move: :meth:`_apply_steps`)."""
        if out.steps:
            self._apply_steps(job, out)
            return
        before = self._status                            # what the junction was when this call began
        if out.status:
            self._status = out.status
            self._t_s = float(out.status["instrument_time_s"])
            self.status.set_status(out.status)
            self._state = fmt.instrument_state(out.status)
        self._show_instrument_state()                    # also after a refusal: a clicked button goes back to the truth
        if out.data:
            self._apply_data(out.data)
        if out.kind == "boot":
            if out.info:
                self.sidebar.facts.set_info(out.info)
                has_qplus = bool(out.info.get("qplus"))      # an etched-tip STM has no PLL to drive
                self.spectroscopy.set_group_visible("pll", has_qplus)
                self.status.set_has_qplus(has_qplus)
            if not out.ok:
                self._fail(out.text)
            return
        if out.kind == "view_frame":
            if out.ok:
                self._remember(out.frame)
                if self._want == ("frame", out.frame["frame"], out.frame["channel"], out.frame["direction"]):
                    self._display_frame(out.frame)
            else:
                self._fail(("refused: " if out.refused else "") + out.text)
            return
        if out.kind == "view_spectrum":
            if out.ok and self._want == ("spectrum", out.label):
                self._display_spectrum(out.spectrum)
            elif not out.ok:
                self._fail(("refused: " if out.refused else "") + out.text)
            return
        t = float(out.reply.get("instrument_time_s", self._t_s)) if isinstance(out.reply, dict) else self._t_s
        if not out.ok:
            text = ("refused: " if out.refused else "error: ") + out.text
            self._fail(text)
            self.sidebar.log.done(job.id, False, text, self._t_s)
            if out.kind == "report":
                self.sidebar.report.set_reply(False, text)
            elif out.kind == "finish":
                self.sidebar.report.finish_not_run()
            self._hold(job, out)                              # the worker holds from this refusal on: the band says so
            return
        if out.kind == "act":
            summary = fmt.reply_summary(job.label, out.reply)
            if not quiet:
                self._succeed(summary)
            self.sidebar.log.done(job.id, True, fmt.reply_full(out.reply), t)
            if out.frame is not None:
                self._remember(out.frame)
                self._want = ("frame", out.frame["frame"], "z", "forward")
                self._display_frame(out.frame)
            elif out.spectrum is not None:
                self._note_junction(out, before)
                self._want = ("spectrum", out.spectrum["name"])
                self._display_spectrum(out.spectrum)
            lists = fmt.numeric_lists(out.reply)
            if lists:
                self.center.show_trace(job.label, lists)
            if self._copier is not None:                      # what the action saved, as text, written off this thread
                self._copier.save(out.reply, self._junction)
            if out.text:                                      # a read after the action failed
                self._fail(out.text)
        elif out.kind == "report":
            self._succeed(str(out.reply))
            self.sidebar.report.set_reply(True, str(out.reply))
            self.sidebar.log.done(job.id, True, str(out.reply), self._t_s)
            if self.sidebar.report.claim_cites(str(job.args.get("claim_id") or "")) and not evidence_files(job.args.get("evidence")):
                # what was sent, nothing more: a window event, in no ledger or export (screen trial 24: the green reply was the
                # same with and without files)
                self.recent.push("warn", tr("no_evidence_note", self.lang))
                self.sidebar.report.set_reply_note(tr("no_evidence_note", self.lang))
        elif out.kind == "finish":
            self._finished = True
            self.sidebar.report.set_finished()
            self._succeed(tr("over", self.lang))
            self.sidebar.log.done(job.id, True, str(out.reply), self._t_s)

    def _note_junction(self, out: jobs.Outcome, before: dict | None) -> None:
        """Remember how a spectrum was taken, from what this window sent (the sweep) and had read (the bias and setpoint
        before the call), so that the viewer and the Data page can tell spectra apart without the Log (p3check1,
        finding 4). Nothing is asked of the instrument: a move trace is shown as it is."""
        keys = {"sts": ("v_start", "v_end"), "zspec": ("sweep_nm",)}.get(out.label)
        if keys is None:
            return
        info = {k: out.args[k] for k in keys if k in out.args}
        if before:
            info.update(bias_v=before["bias_v"], setpoint_pa=before["setpoint_pa"])
        self._junction[out.spectrum["name"]] = info
        self.sidebar.data.set_data(self._data["frames"], self._data["spectra"], self._junction)

    def _show_instrument_state(self) -> None:
        if self._state is not None:
            for panel in (self.operate, self.spectroscopy):
                panel.show_state(self._state)

    def _succeed(self, text: str) -> None:
        """What the last call answered: the strip's last line, and the list of recent results."""
        self.status.set_result(True, text)
        self.recent.push("ok", text)

    def _fail(self, text: str) -> None:
        self.status.set_result(False, text)
        self.recent.push("bad", text)

    # ── data and views ──
    def _apply_data(self, data: dict) -> None:
        self._data = data
        self.center.set_frame_starts({f["frame"]: f["t_start_s"] for f in data["frames"]})
        self.sidebar.data.set_data(data["frames"], data["spectra"], self._junction)
        self._refresh_thumbs()

    def _refresh_thumbs(self) -> None:
        names = [f["frame"] for f in reversed(self._data["frames"]) if f["frame"] in self._thumbs]
        cur = self._viewed[1] if self._viewed and self._viewed[0] == "frame" else ""
        self.center.set_thumbs([(n, self._thumbs[n]) for n in names], cur)

    def _remember(self, frame: dict) -> None:
        key = (frame["frame"], frame["channel"], frame["direction"])
        if key in self._cache:
            return
        self._cache[key] = frame
        self._cache_bytes += int(frame["array"].nbytes)
        while self._cache_bytes > CACHE_BYTES and len(self._cache) > 1:
            _, old = self._cache.popitem(last=False)
            self._cache_bytes -= int(old["array"].nbytes)
        if frame["frame"] not in self._thumbs:
            self._thumbs[frame["frame"]] = thumb_pixmap(frame, style.px(THUMB_PX))

    def _display_frame(self, frame: dict) -> None:
        meta = next((f for f in self._data["frames"] if f["frame"] == frame["frame"]), {})
        self._viewed = ("frame", frame["frame"])
        self.center.show_frame(frame, meta.get("channels"), meta)
        self.sidebar.report.set_viewed(frame["frame"])
        self.sidebar.data.select("frame", frame["frame"])
        self._refresh_thumbs()

    def _display_spectrum(self, spec: dict) -> None:
        self._viewed = ("spectrum", spec["name"])
        self.center.show_spectrum(spec, junction=self._junction.get(spec["name"]))
        self.sidebar.report.set_viewed(spec["name"])
        self.sidebar.data.select("spectrum", spec["name"])
        self._refresh_thumbs()

    # ── the manual's wide view ──
    def _set_wide(self, on: bool) -> None:
        """The Manual tab's wide view: the left column takes the whole row under the status strip (the viewer and the
        right column are hidden, not closed: nothing about them changes, and a job that finishes meanwhile is drawn
        into them), or gives it back. Why it is a whole row: with the centre column too the text is 820 px wide and the
        manual 2.1 to 2.3 screens (measured with Consolas and DejaVu Sans Mono), a third screenshot for its last few
        lines; with the whole row it is 1.5."""
        if on == self._wide:
            return
        if on:
            self._wide_px = self._left_px + self._row.spacing() + self.center.width() + self._row.spacing() \
                + self.right_column.width()
        self._wide = on
        # the order matters: a window grows to the minimum its layout asks for and never shrinks back, so the columns are
        # hidden before the left one is widened, and it is narrowed before they are shown
        if on:
            self.center.hide()
            self.right_column.hide()
            self.sidebar.setFixedWidth(self._wide_px)
        else:
            self.sidebar.setFixedWidth(self._left_px)
            self.center.show()
            self.right_column.show()
        self.sidebar.manual.set_wide(on)

    def _tab_changed(self, _index: int) -> None:
        """The wide view belongs to the Manual tab: any other tab of the left column has the window as it was."""
        if self._wide and self.sidebar.tab_key() != "manual":
            self._set_wide(False)

    # ── copying a marker into a form ──
    def _panel_for(self, action: str) -> ActionPanel:
        return self.operate if self.operate.has_action(action) else self.spectroscopy

    def _fill(self, action: str, kx: str, ky: str, x: float, y: float) -> None:
        panel = self._panel_for(action)
        self.right.setCurrentWidget(panel)
        panel.set_field(action, kx, f"{x:.3f}")
        panel.set_field(action, ky, f"{y:.3f}")

    def _report_from_marker(self, x: float, y: float, frame: str) -> None:
        self.sidebar.show_tab("report")
        self.sidebar.report.fill_position(x, y, frame)

    # ── what the window shows (for tests and screenshots) ──
    @property
    def viewed(self) -> tuple[str, str] | None:
        """``("frame", name)`` or ``("spectrum", name)`` for what the viewer shows, or None."""
        return self._viewed

    def cached_frame(self, name: str, channel: str = "z", direction: str = "forward") -> dict:
        return self._cache[(name, channel, direction)]

    # ── life ──
    def is_idle(self) -> bool:
        """Every job submitted has run and been drawn."""
        return self._open == 0

    def wait_idle(self, timeout: float = 120.0) -> bool:
        """Run the event loop until the queue is empty and drawn (for tests and screenshots)."""
        end = time.monotonic() + timeout
        app = QApplication.instance()
        while time.monotonic() < end:
            app.processEvents()
            if self.is_idle():
                app.processEvents()
                return True
            time.sleep(0.005)
        return False

    def shutdown(self, finish: bool = False) -> None:
        """Stop the worker after the jobs already queued; with ``finish`` end the episode first."""
        if self._copier is not None:                        # the copies of what was saved are written before the end
            self._copier.close()
        if not self.worker.isRunning():
            return
        for sig in (self.worker.started_job, self.worker.finished_job, self.worker.queue_changed, self.worker.dropped):
            try:
                sig.disconnect()
            except (RuntimeError, TypeError):
                pass
        if finish and not self.session.finished:
            self.worker.submit(Job("finish", "finish", {}, jobs.run_finish), force=True)
        self.worker.stop()
        self.worker.wait()

    def closeEvent(self, ev) -> None:
        if self.sealed:
            ev.ignore()
            self.status.set_result(None, tr("stays_open", self.lang))
            return
        self.shutdown(finish=True)
        ev.accept()


def _open_session(scenario, seed, out, lang, sealed, private_dir, export_dir, stdin) -> GuiSession | None:
    """The episode behind the window, or None after a message on stderr. In the sealed mode the seed
    comes from the one line on stdin and is dropped when this function returns: no message, here or
    later, repeats it."""
    secret = None
    try:
        if sealed:
            if private_dir is None or export_dir is None:
                raise SystemExit("--sealed needs --private-dir and --export-dir")
            scenario, secret = read_spec(stdin if stdin is not None else sys.stdin)
            return GuiSession(scenario, secret, run_dir=make_private_dir(private_dir), export_dir=export_dir, lang=lang)
        if scenario is None or out is None:
            raise SystemExit("--scenario and --out are needed (or --sealed with --private-dir and --export-dir)")
        return GuiSession(scenario, seed, run_dir=out, export_dir=Path(out) / "export", lang=lang)
    except (SpecError, FileExistsError, ValueError) as exc:
        text = str(exc)
        print("error: " + (text.replace(str(secret), "<seed>") if secret is not None else text), file=sys.stderr)
        return None


def fit_scale(app: QApplication, lang: str, requested: float, available: QSize, frame_px: int = FRAME_PX) -> float:
    """The largest ui scale, not above ``requested``, at which the window fits ``available`` with the fonts
    this machine has, and leaves that scale applied.

    Why. The layout is tuned with one family of fonts; another (a wider one on the virtual machine) makes
    a row of the Operate tab wider than its column, and a window never narrower than its contents either
    grows past the screen or hides what overflows. The window is measured - the three columns at the
    widths they need (:func:`column_widths`: a side column wider than its set width takes the width from the
    centre one, which a shrink of every size could not give it), and the Operate tab's height - and every size
    is shrunk by what the screen cannot hold. With the fonts it was tuned for, and with Chinese beside English
    in DejaVu on a 1920x1080 screen, the answer is ``requested``.

    ``frame_px`` is what a window manager's title bar takes from ``available``: :data:`FRAME_PX` for a
    maximized window, 0 for a full-screen one (which is all of the screen)."""
    scale = float(requested)
    for _ in range(4):
        style.apply(app, scale)
        panels = [ActionPanel(groups, lambda *_: None, lang) for _, groups in layout.TABS]
        center, strip, tabs, sidebar = CenterColumn(lang), StatusStrip("-", lang), Tabs(), Sidebar("", "", [], lang)
        tabs.addTab(QWidget(), "x")
        recent = RecentList(lang)
        left_px, right_px = column_widths(*column_needs(sidebar, panels))
        height_need = max(p.widget().minimumSizeHint().height() for p in panels)
        # the window manager's title bar is not in `available` and takes about frame_px of the height
        height_have = (available.height() - frame_px - strip.minimumSizeHint().height()
                       - tabs.tabBar().sizeHint().height() - style.px(5 + 6 + 5) - 2 * style.px(2)
                       - recent.sizeHint().height() - style.px(6))
        total = left_px + right_px + center.minimumSizeHint().width() + 4 * style.px(8)
        ratio = min(1.0, available.width() / total, height_have / height_need)
        for w in (*panels, center, strip, tabs, sidebar, recent):
            w.deleteLater()
        if ratio >= 1.0:
            return scale
        scale *= max(0.7, ratio * 0.99)
    style.apply(app, scale)                    # the loop's last shrink was not applied yet: leave the scale returned
    return scale


def show_window(win: MainWindow) -> None:
    """Open the window as its mode wants: the sealed one full screen, whatever the window manager would
    draw around a maximized window; a dev one maximized, which leaves the desktop's own bars."""
    if win.sealed:
        win.showFullScreen()
    else:
        win.showMaximized()


def announce_when_shown(win: MainWindow, path: str | Path, *, poll_ms: int = 50, give_up_s: float = 60.0) -> QTimer:
    """Write ``path`` (one line ``ready``, mode 0644, atomically) as soon as the window is exposed, that is, drawn
    on the screen: a launcher in the virtual machine waits for it before it tells the agent's account the console
    is up. Nothing is written if the window is not exposed within ``give_up_s``; the file never names the seed
    or the scenario. Returns the timer (parented to the window, so the caller need not keep it)."""
    path = Path(path)
    timer = QTimer(win)
    started = time.monotonic()

    def check() -> None:
        handle = win.windowHandle()
        if handle is not None and handle.isExposed():
            timer.stop()
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text("ready\n", encoding="utf-8")
            try:
                tmp.chmod(0o644)
            except OSError:
                pass
            tmp.replace(path)
        elif time.monotonic() - started > give_up_s:
            timer.stop()

    timer.timeout.connect(check)
    timer.start(poll_ms)
    return timer


def run(*, scenario: str | None = None, seed: int = 0, out: str | Path | None = None, lang: str = "zh",
        ui_scale: float = 1.0, sealed: bool = False, private_dir: str | Path | None = None,
        export_dir: str | Path | None = None, stdin: TextIO | None = None,
        ready_file: str | Path | None = None, data_dir: str | Path | None = None) -> int:
    """Open the window and run until it is closed (dev) or the process is killed (sealed). With ``ready_file``
    that file is written once the window is exposed (see :func:`announce_when_shown`)."""
    session = _open_session(scenario, seed, out, lang, sealed, private_dir, export_dir, stdin)
    if session is None:
        return 2
    app = QApplication.instance() or QApplication(["stmbench-gui"])
    screen = app.primaryScreen()
    # a full-screen window covers the whole screen, the desktop's top bar included; a maximized one has what
    # is left of it, less its own title bar
    fit_scale(app, lang, ui_scale, (screen.geometry() if sealed else screen.availableGeometry()).size(),
              0 if sealed else FRAME_PX)
    win = MainWindow(session, lang=lang, sealed=sealed, data_dir=data_dir)
    if not sealed:
        signal.signal(signal.SIGINT, lambda *_: app.quit())
        ticker = QTimer()                                # lets Python see Ctrl-C while Qt's loop runs
        ticker.start(250)
        ticker.timeout.connect(lambda: None)
    show_window(win)
    if ready_file is not None:
        announce_when_shown(win, ready_file)
    try:
        return int(app.exec())
    finally:
        if not sealed:
            win.shutdown(finish=True)
