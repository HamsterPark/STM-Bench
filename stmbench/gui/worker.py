"""The one thread that talks to the instrument.

Every instrument call - actions, observations, reports, the end - is a :class:`Job` that waits its
turn in a first-in, first-out queue and runs on this thread; the window's own thread only draws. A click
puts its job in the queue at once, with the values the fields held when it was clicked, and the buttons
stay enabled while the instrument is busy: an agent's single step often clicks Set bias, Set setpoint
and Move tip in quick succession, and a disabled button would drop the second click without a word.
The jobs run in the order they were clicked. Jobs that have not started can be cancelled, all at once
(a click on Cancel queued): the one running is never touched, and nothing that was cancelled reaches the
instrument, so it is on no record but the window's own log.
"""
from __future__ import annotations

import itertools
import threading
from collections import deque
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
        self._lock = threading.Lock()
        self._turn = threading.Condition(self._lock)
        self._pending: deque[Job | None] = deque()       # None ends the thread
        self._running: Job | None = None

    def _count(self) -> int:
        return sum(1 for j in self._pending if j is not None)

    def submit(self, job: Job) -> None:
        with self._turn:
            self._pending.append(job)
            n = self._count()
            self._turn.notify()
        self.queue_changed.emit(n)

    def stop(self) -> None:
        """Let the jobs already queued run, then end the thread."""
        with self._turn:
            self._pending.append(None)
            self._turn.notify()

    def cancel_queued(self) -> list[Job]:
        """Drop every job that has not started, except the first read of the window, and return them in the
        order they were clicked. Under the lock that hands the worker its next job, so a job is either
        dropped here or run, never both."""
        with self._turn:
            dropped = [j for j in self._pending if j is not None and j.kind != "boot"]
            self._pending = deque(j for j in self._pending if j is None or j.kind == "boot")
            n = self._count()
        if dropped:
            self.queue_changed.emit(n)
        return dropped

    @property
    def idle(self) -> bool:
        with self._lock:
            return self._count() == 0 and self._running is None

    @property
    def waiting(self) -> int:
        with self._lock:
            return self._count()

    @property
    def running(self) -> Job | None:
        with self._lock:
            return self._running

    def queued(self) -> list[Job]:
        """The jobs waiting for their turn, in order."""
        with self._lock:
            return [j for j in self._pending if j is not None]

    def run(self) -> None:
        while True:
            with self._turn:
                while not self._pending:
                    self._turn.wait()
                job = self._pending.popleft()
                if job is None:
                    return
                self._running = job
                n = self._count()
            self.queue_changed.emit(n)
            self.started_job.emit(job)
            try:
                out = job.fn(self.session)
            except Exception as exc:  # noqa: BLE001 - a job function reports its own faults; this is the last net
                out = Outcome(job.kind, job.label, job.args, ok=False, text=f"{type(exc).__name__}: {exc}")
            with self._lock:
                self._running = None
            self.finished_job.emit(job, out)
