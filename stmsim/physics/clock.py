"""Two clocks (docs/DESIGN.md §4.1).

* ``wall`` — real elapsed seconds. Every *hardware-side* process (tip shaper, bias pulse,
  oscilloscope screens, auto-approach cycles, spectroscopy sweeps) runs on it, because
  MAST's read-back skills sample with ``perf_counter`` and judge segment lengths in wall
  seconds.
* ``sim`` — virtual seconds = wall × ``time_scale`` (+ offset). Only scan-line progress and
  slow physics (creep, drift, tip-change hazard) use it, and the ``.sxm`` header stamps it.

``wall()`` is the ONLY time source in stmsim, so freezing it freezes everything derived
from it (``sim``, ``slow``, transients, approach cycles). ``pause()`` / ``resume()`` do that:
the seconds spent paused are accumulated and subtracted from ``wall()``, so resuming
continues seamlessly from where the clock stopped. Pauses nest — two ``pause()`` calls
need two ``resume()`` calls before the clock runs again.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import time


class Clock:
    def __init__(self, time_scale: float = 1.0,
                 sim_epoch: _dt.datetime | None = None, slow_scale: float = 1.0):
        self.time_scale = float(time_scale)
        # Slow physics — thermal drift, piezo creep, the sample creeping toward the tip —
        # runs on its own scale, which the World ties to ``time_scale``.
        #
        # It used to default to 1× wall on the reasoning that MAST judges "Z has settled" in
        # wall seconds, so a 20× clock would make a 1 nm/min drift look like 20. That reasoning
        # does not hold: everything MAST times in wall seconds (transients, ring-up, tip-shaper
        # segments) reads ``wall()`` directly and never touches this clock. What this clock
        # does drive is how far the sample walks during an experiment — and the budget an
        # experiment is given is denominated in *sim* hours. Leaving drift on the wall clock
        # charged a three-hour measurement three minutes' worth of drift at 20×, which quietly
        # deleted the one thing a long experiment has to cope with.
        self.slow_scale = float(slow_scale)
        self._t0 = time.monotonic()
        self._sim_offset = 0.0
        self._slow_offset = 0.0
        self.sim_epoch = sim_epoch or _dt.datetime(2026, 9, 1, 9, 0, 0)
        # pause bookkeeping: depth counter (nested), monotonic instant the outermost pause
        # began, and the total seconds already spent in *finished* pauses
        self._pause_depth = 0
        self._pause_started: float | None = None
        self._paused_total = 0.0

    # ── pause / resume ──
    @property
    def is_paused(self) -> bool:
        return self._pause_depth > 0

    @property
    def paused_total_s(self) -> float:
        """Seconds the clock has spent frozen so far (including the current pause)."""
        total = self._paused_total
        if self._pause_started is not None:
            total += time.monotonic() - self._pause_started
        return total

    def pause(self) -> None:
        """Freeze ``wall()`` (and therefore ``sim()`` / ``slow()``). Nests: each ``pause()``
        must be matched by a ``resume()``."""
        if self._pause_depth == 0:
            self._pause_started = time.monotonic()
        self._pause_depth += 1

    def resume(self) -> None:
        """Undo one ``pause()``. The clock runs again only when every pause is undone;
        unmatched ``resume()`` calls are ignored."""
        if self._pause_depth == 0:
            return
        self._pause_depth -= 1
        if self._pause_depth == 0 and self._pause_started is not None:
            self._paused_total += time.monotonic() - self._pause_started
            self._pause_started = None

    @contextlib.contextmanager
    def paused(self):
        """``with clock.paused(): ...`` — freeze for the block, resume on exit (even on error)."""
        self.pause()
        try:
            yield self
        finally:
            self.resume()

    # ── readings ──
    def wall(self) -> float:
        if self._pause_started is not None:
            # frozen: report the instant the outermost pause began
            return self._pause_started - self._t0 - self._paused_total
        return time.monotonic() - self._t0 - self._paused_total

    def sim(self) -> float:
        return self.wall() * self.time_scale + self._sim_offset

    def slow(self) -> float:
        """Seconds of slow-physics time elapsed."""
        return self.wall() * self.slow_scale + self._slow_offset

    def slow_from_sim(self, t_sim: float) -> float:
        """Map a sim-time instant (e.g. a scan row) onto the slow-physics clock."""
        wall = (float(t_sim) - self._sim_offset) / self.time_scale
        return wall * self.slow_scale + self._slow_offset

    def advance_sim(self, seconds: float) -> None:
        """Jump both virtual clocks (scenario setup, e.g. 'six hours into the night')."""
        self._sim_offset += float(seconds)
        self._slow_offset += float(seconds)

    def sim_datetime(self, sim_s: float | None = None) -> _dt.datetime:
        s = self.sim() if sim_s is None else sim_s
        return self.sim_epoch + _dt.timedelta(seconds=float(s))
