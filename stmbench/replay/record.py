"""What a run records for its replay: the instrument clock on every step, and the tip's history.

:mod:`stmbench.replay.timeline` reads both back (``driver.json`` steps stamped with
``sim_s``; ``tip_timeline.json``).
"""
from __future__ import annotations


def sim_stamped(world, on_event=None):
    """``on_event`` for a driver that first writes the instrument's clock into the record.

    The driver appends the record to its event list before calling this, so the stamp lands
    in ``driver.json`` too. With it a replay puts each model step on the instrument's
    timeline directly; without it the clock has to be rebuilt from wall times (the clock
    stands still while the model thinks). Reading the clock costs nothing."""
    def _hook(rec: dict) -> None:
        try:
            rec["sim_s"] = float(world.clock.sim())
        except Exception:  # noqa: BLE001 — a stamp must never break the episode
            pass
        if on_event is not None:
            on_event(rec)
    return _hook


def tip_timeline(world, initial: dict | None = None) -> dict:
    """The tip's history for a replay: ``initial`` (the ledger's ``truth_before["tip"]``)
    and every change after it (spontaneous, pulse, poke, crash, pick-up, drop) with the
    state it left.

    Spontaneous changes are drawn when a frame is rendered at scan start and carry the time
    of the row they hit, so ``world.tip`` read in the middle of a frame is already ahead of
    the picture; this list, played against the clock, is not. ``i`` is the index that
    ``scan_start`` events cite in ``tip_events``."""
    return {"initial": initial if initial is not None else world.tip.snapshot(),
            "events": [{"i": k, "sim_s": float(e.sim_s), "kind": e.kind, "detail": e.detail,
                        "state": e.state} for k, e in enumerate(world.tip.events)]}
