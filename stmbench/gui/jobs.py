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

from ..instrument import ACTIONS, ActionError
from ..instrument.core import validate
from .session import GuiSession

#: what the instrument (or the call's own arguments) refuses; any other exception is a fault
REFUSALS = (ActionError, ValueError, TypeError, KeyError)


@dataclass
class Refusal:
    """A refused operator call, as the band on the status strip shows it until the reader acknowledges it: the
    call with the parameters it was clicked with, the instrument's words, and the calls that did not run because
    of it (the ones that were waiting behind it, then every click turned away while the console held)."""
    call: str
    words: str
    refused: bool = True            # False: a fault, not the instrument's refusal
    not_run: list[str] = field(default_factory=list)


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
    steps: list = field(default_factory=list)    # a move with during-move values: the Outcome of every call it made, in order


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


#: the setters a move may run around itself, in the order they run (the Junction group's order: range, bias,
#: setpoint); each is ``(action, parameter, key of the instrument's status() that reports it)``
DURING = (("set_current_range", "full_scale_na", "current_full_scale_na"),
          ("set_bias", "bias_v", "bias_v"),
          ("set_setpoint", "current_pa", "setpoint_pa"))


def run_move_during(sess: GuiSession, args: dict, during: dict[str, dict]) -> Outcome:
    """A move of the tip with values set around it, as the same calls an operator would click one by one.

    ``during`` maps a setter's name to its arguments, for the ones that are set. Every value is checked against
    the action table first (the move's own fields with them): a value that is refused ends it with nothing run. Then
    the setters run in the order range, bias, setpoint, except when the range is lowered (below what ``status()``
    reported when this began): then setpoint, bias, range, so that the range never stands below the setpoint the
    junction still has (the instrument's ``crash_over_range`` hits the tip when the loop's setpoint is over the range, which
    screen trial 22 met by hand); then the move with ``args`` as typed; then, in the reverse of the order the setters
    ran, each value that a setter changed goes back to what ``status()`` reported when this began. If a call is refused (or
    faults) the setters that were applied are put back all the same, and the outcome says what was refused; if a
    putting back is refused, the rest is left and that refusal is the one that is shown. Every call is a normal
    action of the session, so each is in the Log, the export and the ledger as if it had been clicked; the outcome
    holds each call's own outcome in ``steps``. (The read of the state at the start is an observation like the reads
    the window makes after each action.)"""
    out = Outcome("act", "move_tip", dict(args))
    plan = [(name, dict(during[name])) for name, _, _ in DURING if name in during] + [("move_tip", dict(args))]
    for name, a in plan:                                          # first: every value, before anything runs
        try:
            validate(ACTIONS[name], a)                            # the table's own check: nothing runs, nothing is recorded
        except Exception as exc:  # noqa: BLE001
            out.steps = [_fail(Outcome("act", name, a), exc)]
            return _summed(out)
    try:
        before = sess.status()                                    # read now, not at the click: earlier clicks may still have been queued
    except Exception as exc:  # noqa: BLE001
        return _fail(out, exc)
    keys = {name: (param, key) for name, param, key in DURING}
    lowered = "set_current_range" in during and isinstance(before.get("current_full_scale_na"), (int, float))         and during["set_current_range"]["full_scale_na"] < before["current_full_scale_na"]
    if lowered:                                                   # decided now that the state is read: the plan runs the setters the other way round
        plan = [(name, dict(during[name])) for name, _, _ in reversed(DURING) if name in during] + [("move_tip", dict(args))]
    applied: list[str] = []
    for name, a in plan:
        step = run_act(sess, name, a)
        out.steps.append(step)
        if not step.ok:
            break
        if name != "move_tip":
            param, key = keys[name]
            if not _equal(a[param], before.get(key)):
                applied.append(name)                              # a setter that changed nothing has nothing to put back
    for name in reversed(applied):
        param, key = keys[name]
        step = run_act(sess, name, {param: before[key]})
        out.steps.append(step)
        if not step.ok:
            break
    return _summed(out)


def _equal(a, b) -> bool:
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def _summed(out: Outcome) -> Outcome:
    """The composite's own outcome: whether every call went through, and the words of the last one that did not."""
    bad = [s for s in out.steps if not s.ok]
    out.ok = not bad
    if bad:
        out.refused, out.text = bad[-1].refused, bad[-1].text
    if out.steps and out.steps[-1].status:
        out.status = out.steps[-1].status
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
