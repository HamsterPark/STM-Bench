"""What the window asks of the instrument, one job at a time (no Qt).

A job runs on the worker thread and returns an :class:`Outcome`. After an action the job reads what the
window shows next, as a person's eye would: the readings (``status``), and after a scan the data list and
the new frame (Z, forward), after a spectrum or a tip move the spectrum or the move's trace. Those reads
are observations: they spend no instrument time and draw no random number, and they land in the ledger
like an agent's reads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..instrument import ActionError
from .session import GuiSession

#: what the instrument (or the call's own arguments) refuses; any other exception is a fault
REFUSALS = (ActionError, ValueError, TypeError, KeyError)


@dataclass
class Outcome:
    kind: str                       # act | report | finish | view_frame | view_spectrum | boot
    label: str                      # the action's name, or what the job was for
    args: dict = field(default_factory=dict)
    ok: bool = True
    refused: bool = False           # the instrument refused: nothing happened, no time passed
    text: str = ""                  # the refusal's words, or a fault's description
    reply: Any = None               # an action's reply (dict) or a report's acknowledgement (str)
    status: dict | None = None
    info: dict | None = None
    data: dict | None = None
    frame: dict | None = None
    spectrum: dict | None = None
    finished: bool = False


def _fail(out: Outcome, exc: Exception) -> Outcome:
    out.ok = False
    if isinstance(exc, REFUSALS):
        out.refused = True
        out.text = str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)
    else:
        out.text = f"{type(exc).__name__}: {exc}"
    return out


def run_boot(sess: GuiSession) -> Outcome:
    out = Outcome("boot", "start")
    try:
        out.status, out.info, out.data = sess.status(), sess.info(), sess.data()
    except Exception as exc:  # noqa: BLE001 - shown in the status strip; the window stays up
        return _fail(out, exc)
    return out


def run_act(sess: GuiSession, name: str, args: dict) -> Outcome:
    out = Outcome("act", name, dict(args))
    try:
        out.reply = sess.act(name, args)
    except Exception as exc:  # noqa: BLE001
        return _fail(out, exc)
    try:
        out.status = sess.status()
        reply = out.reply
        if reply.get("frame"):
            out.data = sess.data()
            out.frame = sess.frame(reply["frame"], "z", "forward")
        elif reply.get("spectrum") or reply.get("trace"):         # a move's trace is saved and read like a spectrum
            out.data = sess.data()
            out.spectrum = sess.spectrum(reply.get("spectrum") or reply["trace"])
    except Exception as exc:  # noqa: BLE001 - the action itself succeeded; say what could not be read
        out.text = f"{type(exc).__name__}: {exc}"
    return out


def run_view_frame(sess: GuiSession, name: str, channel: str, direction: str) -> Outcome:
    out = Outcome("view_frame", name, {"channel": channel, "direction": direction})
    try:
        out.frame = sess.frame(name, channel, direction)
    except Exception as exc:  # noqa: BLE001
        return _fail(out, exc)
    return out


def run_view_spectrum(sess: GuiSession, name: str) -> Outcome:
    out = Outcome("view_spectrum", name)
    try:
        out.spectrum = sess.spectrum(name)
    except Exception as exc:  # noqa: BLE001
        return _fail(out, exc)
    return out


def run_report(sess: GuiSession, fields: dict) -> Outcome:
    out = Outcome("report", str(fields.get("claim_id") or "report"), dict(fields))
    try:
        out.reply = sess.report(fields)
    except Exception as exc:  # noqa: BLE001
        return _fail(out, exc)
    return out


def run_finish(sess: GuiSession) -> Outcome:
    out = Outcome("finish", "finish")
    try:
        sess.finish()
        out.reply = "The episode is over; the record is written."
        out.finished = True
    except Exception as exc:  # noqa: BLE001
        return _fail(out, exc)
    return out
