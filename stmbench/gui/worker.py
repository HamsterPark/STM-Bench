"""The one thread that talks to the instrument.

Every instrument call - actions, observations, reports, the end - is a :class:`Job` that waits its
turn in a first-in, first-out queue and runs on this thread; the window's own thread only draws. A click
puts its job in the queue at once, with the values the fields held when it was clicked, and the buttons
stay enabled while the instrument is busy: an agent's single step often clicks Set bias, Set setpoint
and Move tip in quick succession, and a disabled button would drop the second click without a word.
The jobs run in the order they were clicked. Jobs that have not started can be cancelled, all at once
(a click on Cancel queued): the one running is never touched, and nothing that was cancelled reaches the
instrument, so it is on no record but the window's own log.

A call that fails (the instrument refuses it) ends the step it was in. An agent clicks a dozen buttons
and looks at the screen once, and a refused Set setpoint followed by a drag would run the drag at a
junction resistance nobody chose. So when a call of the operator's (an action, a report, the end) fails, the
operator calls waiting behind it are dropped before the worker takes the next one, and for ``guard_s``
seconds after it (renewed by every click that is turned away) a new operator call is not accepted either:
the clicks of the same step that come after the refusal was already answered. A click after a pause is a new
step. Reads (a frame, a spectrum) are never dropped. Nothing dropped reaches the instrument.
"""
from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from PySide6.QtCore import QThread, Signal

from .jobs import Outcome
from .session import GuiSession


#: the calls that change the instrument or the record: a refusal of one of them stops the rest of its step
OPERATOR_KINDS = ("act", "report", "finish")
#: seconds after a refusal during which a click that would start another operator call is turned away
GUARD_S = 3.0


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
    dropped = Signal(object)                    # list[Job]: operator calls that were waiting when a call failed

    def __init__(self, session: GuiSession, parent=None, clock: Callable[[], float] = time.monotonic,
                 guard_s: float = GUARD_S):
        super().__init__(parent)
        self.session = session
        self.guard_s = guard_s
        self._clock = clock
        self._guard_until = 0.0
        self._lock = threading.Lock()
        self._turn = threading.Condition(self._lock)
        self._pending: deque[Job | None] = deque()       # None ends the thread
        self._running: Job | None = None

    def _count(self) -> int:
        return sum(1 for j in self._pending if j is not None)

    def submit(self, job: Job, force: bool = False) -> bool:
        """Queue a job. False when it is turned away: an operator call within ``guard_s`` seconds of a failed
        one (``force`` queues it anyway: the end of the episode when the window is closed)."""
        with self._turn:
            if not force and job.kind in OPERATOR_KINDS and self._clock() < self._guard_until:
                self._guard_until = self._clock() + self.guard_s          # the step is still going on
                return False
            self._pending.append(job)
            n = self._count()
            self._turn.notify()
        self.queue_changed.emit(n)
        return True

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
            if job.kind in OPERATOR_KINDS and not out.ok:
                with self._turn:
                    self._running = None
                    # before the next job is taken: the calls of this step that were already waiting do not run
                    gone = [j for j in self._pending if j is not None and j.kind in OPERATOR_KINDS]
                    self._pending = deque(j for j in self._pending if j is None or j.kind not in OPERATOR_KINDS)
                    self._guard_until = self._clock() + self.guard_s
                    n = self._count()
                    # said under the lock that turns a click away, so that the window hears of the refusal before
                    # it hears of any click that was turned away because of it (the order of the recent results)
                    self.finished_job.emit(job, out)
                    if gone:
                        self.dropped.emit(gone)
                if gone:
                    self.queue_changed.emit(n)
                continue
            with self._lock:
                self._running = None
            self.finished_job.emit(job, out)
