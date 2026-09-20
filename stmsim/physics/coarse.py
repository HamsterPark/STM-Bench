"""Coarse motion and the auto-approach state machine.

* ``coarse_gap_m`` — distance from the sample reference plane to the tip apex when the
  fine Z is at 0. Z-motor steps reduce it (with per-step scatter); XY steps move to a new
  site (new terrain) with no position feedback.
* Auto-approach (controller "Safe" / woodpecker mode): retract fine Z, N motor pulses,
  extend fine Z with the feedback on while watching the current; landed when the loop
  finds a gap inside the fine range. Runs on the **wall** clock; each cycle takes the
  extend time (range / I-gain) plus the motor delay. Every pulse injects the crosstalk
  spike into the current, which is how a low setpoint gets a false landing when the
  noise is high (docs/DESIGN.md §4.2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class CoarseStage:
    coarse_gap_m: float = 40e-6
    z_step_m: float = 0.28e-6           # per pulse (≈ 0.84 µm per 3-pulse cycle)
    z_step_scatter: float = 0.15
    xy_step_m: float = 0.5e-6
    steps_done: int = 0
    xy_steps: dict[str, int] = field(default_factory=lambda: {"X+": 0, "X-": 0, "Y+": 0, "Y-": 0})
    freq_hz: float = 300.0
    amp_v: float = 180.0

    def step_z(self, n: int, rng, direction: int = -1) -> float:
        """Move ``n`` pulses (direction −1 = approach, +1 = retract). Returns Δgap."""
        total = 0.0
        for _ in range(n):
            d = self.z_step_m * (1 + rng.normal(0, self.z_step_scatter))
            total += d * (1 if direction > 0 else -1)
        self.coarse_gap_m = self.coarse_gap_m + total
        self.steps_done += n
        return total


@dataclass
class AutoApproach:
    running: bool = False
    mode: str = "safe"
    pulses_per_cycle: int = 3
    delay_after_move_s: float = 0.02
    stop_condition_pct: float = 0.0
    t_next_cycle_wall: float = 0.0
    cycles: int = 0
    landed: bool = False
    aborted: bool = False
    t_started_wall: float = 0.0
    extend_time_s: float = 2.0          # fine-Z full extension at the approach I-gain
    retract_time_s: float = 1.0

    def cycle_time_s(self) -> float:
        return self.extend_time_s + self.retract_time_s + self.delay_after_move_s + 0.3
