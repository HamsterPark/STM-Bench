"""The one thread that talks to the instrument.

Every instrument call - actions, observations, reports, the end - is a :class:`Job` that waits its
turn in a first-in, first-out queue and runs on this thread; the window's own thread only draws. A click
puts its job in the queue at once, with the values the fields held when it was clicked, and the buttons
stay enabled while the instrument is busy: an agent's single step often clicks Set bias, Set setpoint
and Move tip in quick succession, and a disabled button would drop the second click without a word.
The jobs run in the order they were clicked.
"""
from __future__ import annotations

import itertools
import queue
import threading
from dataclasses import dataclass, field
from typing import Callable

from PySide6.QtCore import QThread, Signal

from .jobs import Outcome
from .session import GuiSession


@dataclass
class Job:
    kind: str                                  # act | report | finish | view_frame | view_spectrum | boot
    label: str                                 # what the Log and the status strip call it
    args: dict
    fn: Callable[[GuiSession], Outcome]
    id: int = field(default_factory=itertools.count(1).__next__)


class Worker(QThread):
    """Runs jobs one by one. ``started_job`` and ``finished_job`` are emitted from this thread and
    reach the window's slots on its own thread."""

    started_job = Signal(object)                # Job
    finished_job = Signal(object, object)       # Job, Outcome
    queue_changed = Signal(int)                 # jobs waiting, not counting the one running

    def __init__(self, session: GuiSession, parent=None):
        super().__init__(parent)
        self.session = session
        self._jobs: queue.SimpleQueue = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._waiting = 0
        self._running: Job | None = None

    def submit(self, job: Job) -> None:
        with self._lock:
            self._waiting += 1
            n = self._waiting
        self._jobs.put(job)
        self.queue_changed.emit(n)

    def stop(self) -> None:
        """Let the jobs already queued run, then end the thread."""
        self._jobs.put(None)

    @property
    def idle(self) -> bool:
        with self._lock:
            return self._waiting == 0 and self._running is None

    @property
    def waiting(self) -> int:
        with self._lock:
            return self._waiting

    def run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            with self._lock:
                self._waiting -= 1
                self._running = job
                n = self._waiting
            self.queue_changed.emit(n)
            self.started_job.emit(job)
            try:
                out = job.fn(self.session)
            except Exception as exc:  # noqa: BLE001 - a job function reports its own faults; this is the last net
                out = Outcome(job.kind, job.label, job.args, ok=False, text=f"{type(exc).__name__}: {exc}")
            with self._lock:
                self._running = None
            self.finished_job.emit(job, out)
