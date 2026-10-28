"""The console's hold at the worker's level (stmbench.gui.worker), without a window or an instrument: after a call of
the operator's fails, the calls waiting behind it are dropped and every operator call that is submitted afterwards is
turned away until the refusal is acknowledged. Reads, Cancel queued and the end of the episode when the window is closed
(``force``) are never held. The window's own side (the band, the pill, the lists) is in test_gui_app.py."""
from __future__ import annotations

import os
import threading
import time

import pytest

pytest.importorskip("PySide6")
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QCoreApplication, Qt  # noqa: E402

from stmbench.gui.jobs import Outcome  # noqa: E402
from stmbench.gui.worker import OPERATOR_KINDS, Job, Worker  # noqa: E402


@pytest.fixture
def worker():
    QCoreApplication.instance() or QCoreApplication(["test"])
    w = Worker(session=None)
    w.start()
    yield w
    w.stop()
    w.wait(10_000)


def job(kind: str, ok: bool = True, label: str | None = None, gate: threading.Event | None = None,
        ran: list | None = None) -> Job:
    label = label or kind

    def fn(_session):
        if gate is not None:
            gate.wait(10)
        if ran is not None:
            ran.append(label)
        return Outcome(kind, label, {}, ok=ok, refused=not ok, text="" if ok else "refused by the test")
    return Job(kind, label, {}, fn)


def until(check, seconds: float = 10.0) -> None:
    end = time.monotonic() + seconds
    while not check():
        assert time.monotonic() < end, "the worker did not get there"
        time.sleep(0.005)


def test_a_failed_operator_call_holds_every_operator_call_until_it_is_acknowledged_however_long_that_takes(worker):
    ran: list[str] = []
    assert not worker.held and worker.submit(job("act", ok=False, label="refused", ran=ran))
    until(lambda: worker.held and worker.idle)
    for kind in OPERATOR_KINDS:                                       # an action, a report, the end: all turned away
        assert not worker.submit(job(kind, label=f"turned_away_{kind}", ran=ran))
    time.sleep(0.5)                                                   # nothing but the reader's click lifts it
    assert not worker.submit(job("act", label="later", ran=ran)) and worker.held
    assert worker.waiting == 0 and ran == ["refused"]                 # none of them reached the queue
    assert worker.acknowledge() is True and not worker.held
    assert worker.acknowledge() is False                              # nothing was held any more
    assert worker.submit(job("act", label="after", ran=ran))
    until(lambda: ran == ["refused", "after"] and worker.idle)


@pytest.mark.parametrize("kind", ["act", "report", "finish"])
def test_each_kind_of_operator_call_that_fails_holds(worker, kind):
    assert worker.submit(job(kind, ok=False))
    until(lambda: worker.held and worker.idle)
    assert worker.acknowledge()


def test_reads_are_never_held_and_neither_is_the_end_of_a_closing_window(worker):
    ran: list[str] = []
    worker.submit(job("act", ok=False, ran=ran))
    until(lambda: worker.held and worker.idle)
    assert worker.submit(job("view_frame", label="read_a_frame", ran=ran))
    assert worker.submit(job("view_spectrum", label="read_a_spectrum", ran=ran))
    assert worker.submit(job("finish", label="close", ran=ran), force=True)       # the window is closing: the episode ends
    until(lambda: ran == ["act", "read_a_frame", "read_a_spectrum", "close"] and worker.idle)
    assert worker.held                                                # a read does not lift the hold


def test_the_calls_waiting_behind_a_refusal_are_dropped_and_the_reads_are_not(worker):
    ran: list[str] = []
    dropped: list[Job] = []
    worker.dropped.connect(lambda jobs: dropped.extend(jobs), Qt.ConnectionType.DirectConnection)
    gate = threading.Event()
    worker.submit(job("act", ok=False, label="refused", gate=gate, ran=ran))
    until(lambda: worker.running is not None)
    for kind, label in (("act", "a"), ("view_frame", "read"), ("report", "b"), ("finish", "c")):
        assert worker.submit(job(kind, label=label, ran=ran))                     # not yet refused: all accepted
    assert worker.waiting == 4
    gate.set()
    until(lambda: worker.idle and worker.held)
    assert ran == ["refused", "read"] and [j.label for j in dropped] == ["a", "b", "c"]


def test_cancel_queued_drops_what_has_not_started_whether_or_not_the_console_holds(worker):
    ran: list[str] = []
    gate = threading.Event()
    worker.submit(job("act", ok=False, label="refused", ran=ran))
    until(lambda: worker.held and worker.idle)                        # a refusal that is not acknowledged
    worker.submit(job("view_frame", label="running", gate=gate, ran=ran))
    until(lambda: worker.running is not None)
    worker.submit(job("view_frame", label="r1", ran=ran))
    worker.submit(job("view_spectrum", label="r2", ran=ran))
    assert [j.label for j in worker.cancel_queued()] == ["r1", "r2"] and worker.waiting == 0
    gate.set()
    until(lambda: worker.idle)
    assert ran == ["refused", "running"] and worker.held


def test_a_fault_that_is_not_a_refusal_holds_too(worker):
    def boom(_session):
        raise RuntimeError("the instrument fell over")
    assert worker.submit(Job("act", "boom", {}, boom))
    until(lambda: worker.held and worker.idle)
    assert not worker.submit(job("act"))
