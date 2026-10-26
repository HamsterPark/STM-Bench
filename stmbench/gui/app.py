"""The desktop console's window and its entry point (``stmbench.cli gui``).

One window, three columns under a status strip: the task, the report form, the data and the log on
the left; the viewer with its markers in the middle; the instrument's actions on the right. Every
click that touches the instrument becomes a job on the worker thread (:mod:`stmbench.gui.worker`);
this thread only draws what the jobs bring back. The window adds no hint: it shows what the instrument
said, in the instrument's words, and the instrument is the only validator.

Two ways to run it. *Dev*: a scenario and a seed on the command line, a run directory (the usual full
ledger at the end) with the export beside it; closing the window finishes the episode, judged as it
stands. *Sealed*: the seed arrives on stdin and appears nowhere else; the window opens maximized and
its close button only says that the program stays open (the process ends when it is killed, and
everything the agent did is already in the export).
"""
from __future__ import annotations

import signal
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import TextIO

from PySide6.QtCore import QSize, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QVBoxLayout, QWidget

from . import fmt, jobs, layout, style
from .center import THUMB_PX, CenterColumn, thumb_pixmap
from .controls import Tabs
from .labels import two_lines, tr
from .panels import ActionPanel
from .session import GuiSession, SpecError, make_private_dir, read_spec
from .sidebar import Sidebar
from .status import StatusStrip
from .worker import Job, Worker

LEFT_PX, RIGHT_PX = 400, 720
#: what a window manager's title bar may take from a maximized window's height
FRAME_PX = 30
#: frames kept for instant re-viewing (arrays of at most 1024 x 1024 doubles): a byte budget
CACHE_BYTES = 256 * 1024 * 1024


class MainWindow(QMainWindow):
    def __init__(self, session: GuiSession, *, lang: str = "zh", sealed: bool = False):
        super().__init__()
        self.session = session
        self.lang = lang
        self.sealed = sealed
        self.setWindowTitle(f"{tr('app', lang)} · {session.scenario_id}")
        self._open = 0                                   # jobs submitted whose outcome is not drawn yet
        self._running = ""
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
        col.setContentsMargins(style.px(8), style.px(6), style.px(8), style.px(8))
        col.setSpacing(style.px(6))
        self.status = StatusStrip(session.scenario_id, lang)
        col.addWidget(self.status)
        row = QHBoxLayout()
        row.setSpacing(style.px(8))
        col.addLayout(row, 1)

        self.sidebar = Sidebar(session.task_text(), session.manual(), session.claims(), lang)
        self.sidebar.setFixedWidth(style.px(LEFT_PX))
        self.center = CenterColumn(lang)
        self.operate = ActionPanel(layout.OPERATE, self.run_action, lang)
        self.spectroscopy = ActionPanel(layout.SPECTROSCOPY, self.run_action, lang)
        self.right = Tabs()
        self.right.setObjectName("right.tabs")
        self.right.addTab(self.operate, two_lines("tab_operate", lang).replace("\n", " "))
        self.right.addTab(self.spectroscopy, two_lines("tab_spectroscopy", lang).replace("\n", " "))
        self.right.tabBar().setObjectName("right.tabbar")
        self.right.tabBar().setExpanding(True)
        self.right.setFixedWidth(style.px(RIGHT_PX))
        row.addWidget(self.sidebar)
        row.addWidget(self.center, 1)
        row.addWidget(self.right)

        # the reader's gestures
        self.sidebar.report.submitted.connect(self._report)
        self.sidebar.report.finish_requested.connect(self._finish)
        self.sidebar.data.frame_clicked.connect(self._pick_frame)
        self.sidebar.data.spectrum_clicked.connect(self.request_spectrum)
        self.center.thumb_clicked.connect(self._pick_frame)
        self.center.view_changed.connect(self._view_changed)
        self.center.tip_from_marker.connect(lambda x, y: self._fill("move_tip", "x_nm", "y_nm", x, y))
        self.center.scan_from_marker.connect(lambda x, y: self._fill("scan", "center_x_nm", "center_y_nm", x, y))
        self.center.report_from_marker.connect(self._report_from_marker)
        self.center.notice.connect(lambda text: self.status.set_result(False, text))

        self.worker = Worker(session, self)
        self.worker.started_job.connect(self._on_started)
        self.worker.finished_job.connect(self._on_finished)
        self.worker.queue_changed.connect(self._on_queue)
        self.worker.start()
        self._submit(Job("boot", "start", {}, jobs.run_boot), log=False)

    # ── jobs ──
    def _submit(self, job: Job, log: bool = True) -> int:
        self._open += 1
        if log:
            self.sidebar.log.add(job.id, job.label, job.args, self._t_s)
        self.worker.submit(job)
        return job.id

    def run_action(self, name: str, args: dict) -> int:
        """Queue one action with the values the fields held at the click."""
        args = dict(args)
        return self._submit(Job("act", name, args, lambda s, n=name, a=args: jobs.run_act(s, n, a)))

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
        self._show_state()
        self.sidebar.log.running(job.id)

    def _show_state(self) -> None:
        # the count is read from the worker, not from a signal's payload: signals from two threads can
        # arrive out of order
        self.status.set_state(bool(self._running), self._running, self.worker.waiting, self._finished)

    def _on_finished(self, job: Job, out: jobs.Outcome) -> None:
        try:
            self._apply(job, out)
        finally:
            self._open -= 1
            self._running = ""
            self._show_state()

    def _apply(self, job: Job, out: jobs.Outcome) -> None:
        if out.status:
            self._t_s = float(out.status["instrument_time_s"])
            self.status.set_status(out.status)
        if out.data:
            self._apply_data(out.data)
        if out.kind == "boot":
            if not out.ok:
                self.status.set_result(False, out.text)
            return
        if out.kind == "view_frame":
            if out.ok:
                self._remember(out.frame)
                if self._want == ("frame", out.frame["frame"], out.frame["channel"], out.frame["direction"]):
                    self._display_frame(out.frame)
            else:
                self.status.set_result(False, ("refused: " if out.refused else "") + out.text)
            return
        if out.kind == "view_spectrum":
            if out.ok and self._want == ("spectrum", out.label):
                self._display_spectrum(out.spectrum)
            elif not out.ok:
                self.status.set_result(False, ("refused: " if out.refused else "") + out.text)
            return
        t = float(out.reply.get("instrument_time_s", self._t_s)) if isinstance(out.reply, dict) else self._t_s
        if not out.ok:
            text = ("refused: " if out.refused else "error: ") + out.text
            self.status.set_result(False, text)
            self.sidebar.log.done(job.id, False, text, self._t_s)
            if out.kind == "act":
                self._panel_for(job.label).set_reply(job.label, False, text)
            elif out.kind == "report":
                self.sidebar.report.set_reply(False, text)
            return
        if out.kind == "act":
            summary = fmt.reply_summary(job.label, out.reply)
            self._panel_for(job.label).set_reply(job.label, True, summary)
            self.status.set_result(True, summary)
            self.sidebar.log.done(job.id, True, fmt.reply_full(out.reply), t)
            if out.frame is not None:
                self._remember(out.frame)
                self._want = ("frame", out.frame["frame"], "z", "forward")
                self._display_frame(out.frame)
            elif out.spectrum is not None:
                self._want = ("spectrum", out.spectrum["name"])
                self._display_spectrum(out.spectrum)
            lists = fmt.numeric_lists(out.reply)
            if lists:
                self.center.show_trace(job.label, lists)
            if out.text:                                      # a read after the action failed
                self.status.set_result(False, out.text)
        elif out.kind == "report":
            self.status.set_result(True, str(out.reply))
            self.sidebar.report.set_reply(True, str(out.reply))
            self.sidebar.log.done(job.id, True, str(out.reply), self._t_s)
        elif out.kind == "finish":
            self._finished = True
            self.sidebar.report.set_finished()
            self.status.set_result(True, tr("over", self.lang))
            self.sidebar.log.done(job.id, True, str(out.reply), self._t_s)

    # ── data and views ──
    def _apply_data(self, data: dict) -> None:
        self._data = data
        self.sidebar.data.set_data(data["frames"], data["spectra"])
        self._refresh_thumbs()

    def _refresh_thumbs(self) -> None:
        names = [f["frame"] for f in reversed(self._data["frames"]) if f["frame"] in self._thumbs]
        cur = self._viewed[1] if self._viewed and self._viewed[0] == "frame" else ""
        self.center.set_thumbs([(n, self._thumbs[n]) for n in names[:6]], cur)

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
        self.center.show_spectrum(spec)
        self.sidebar.report.set_viewed(spec["name"])
        self.sidebar.data.select("spectrum", spec["name"])
        self._refresh_thumbs()

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
        if not self.worker.isRunning():
            return
        for sig in (self.worker.started_job, self.worker.finished_job, self.worker.queue_changed):
            try:
                sig.disconnect()
            except (RuntimeError, TypeError):
                pass
        if finish and not self.session.finished:
            self.worker.submit(Job("finish", "finish", {}, jobs.run_finish))
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


def fit_scale(app: QApplication, lang: str, requested: float, available: QSize) -> float:
    """The largest ui scale, not above ``requested``, at which the window fits ``available`` with the fonts
    this machine has, and leaves that scale applied.

    Why. The layout is tuned with one family of fonts; another (a wider one on the virtual machine) makes
    a row of the Operate tab wider than its column, and a window never narrower than its contents either
    grows past the screen or hides what overflows. Everything scales together, so the window is
    measured - the three columns' narrowest widths, and the Operate tab's height - and every size is
    shrunk by what is missing. With the fonts it was tuned for, the answer is ``requested``."""
    scale = float(requested)
    for _ in range(4):
        style.apply(app, scale)
        panels = [ActionPanel(groups, lambda *_: None, lang) for _, groups in layout.TABS]
        center, strip, tabs, sidebar = CenterColumn(lang), StatusStrip("-", lang), Tabs(), Sidebar("", "", [], lang)
        tabs.addTab(QWidget(), "x")
        sidebar_need = sidebar.minimumSizeHint().width()
        right_need = max(p.widget().minimumSizeHint().width() for p in panels) + 2 * style.px(3)
        height_need = max(p.widget().minimumSizeHint().height() for p in panels)
        # the window manager's title bar is not in `available` and takes about FRAME_PX of the height
        height_have = (available.height() - FRAME_PX - strip.minimumSizeHint().height()
                       - tabs.tabBar().sizeHint().height() - style.px(6 + 6 + 8) - 2 * style.px(3))
        total = style.px(LEFT_PX) + style.px(RIGHT_PX) + center.minimumSizeHint().width() + 4 * style.px(8)
        ratio = min(1.0, available.width() / total, style.px(RIGHT_PX) / right_need, height_have / height_need,
                    style.px(LEFT_PX) / max(sidebar_need, 1))
        for w in (*panels, center, strip, tabs, sidebar):
            w.deleteLater()
        if ratio >= 1.0:
            break
        scale *= max(0.7, ratio * 0.99)
    return scale


def run(*, scenario: str | None = None, seed: int = 0, out: str | Path | None = None, lang: str = "zh",
        ui_scale: float = 1.0, sealed: bool = False, private_dir: str | Path | None = None,
        export_dir: str | Path | None = None, stdin: TextIO | None = None) -> int:
    """Open the window and run until it is closed (dev) or the process is killed (sealed)."""
    session = _open_session(scenario, seed, out, lang, sealed, private_dir, export_dir, stdin)
    if session is None:
        return 2
    app = QApplication.instance() or QApplication(["stmbench-gui"])
    fit_scale(app, lang, ui_scale, app.primaryScreen().availableGeometry().size())
    win = MainWindow(session, lang=lang, sealed=sealed)
    if not sealed:
        signal.signal(signal.SIGINT, lambda *_: app.quit())
        ticker = QTimer()                                # lets Python see Ctrl-C while Qt's loop runs
        ticker.start(250)
        ticker.timeout.connect(lambda: None)
    win.showMaximized()
    try:
        return int(app.exec())
    finally:
        if not sealed:
            win.shutdown(finish=True)
