"""Fault library + scheduler (docs/DESIGN.md §4.3).

A fault is ``{kind, at_sim_s | at_wall_s | when: <condition>, params}``. The scheduler is
ticked from the wire dispatcher's ``fault_hook`` (i.e. on every controller command) and from
``World`` polling paths, so faults fire without their own thread.

Kinds (physical):
  thermal_drift_to_limit   sample creeps toward the tip until the Z loop pins at the retract limit
  tip_change_burst         λ multiplied for a window (dirty / unstable tip)
  tip_damage               immediate: blunt / multi / low_phi / dead
  feedback_oscillation     I-gain multiplied (loop rings / limit-cycles)
  contaminate_site         low-φ blobs appended around the tip
  surface_spent            damage markers sprinkled around the tip
  preamp_range             gain index change (saturation level moves)
  approach_noise           crosstalk-spike scale + false-landing probability
Kinds (communication, need the server handle):
  comms_latency            extra seconds per reply (>5 s exceeds the client receive timeout)
  comms_drop               close the connection on the next N commands
  module_unload            a module starts answering NeedModule
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Fault:
    kind: str
    params: dict = field(default_factory=dict)
    at_sim_s: float | None = None
    at_wall_s: float | None = None
    when: str | None = None          # "after_first_scan" | "after_n_pokes:3" | "on_command:Bias.Pulse"
    fired: bool = False
    fired_at_sim_s: float | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Fault":
        return cls(kind=d["kind"], params=dict(d.get("params", {})), at_sim_s=d.get("at_sim_s"),
                   at_wall_s=d.get("at_wall_s"), when=d.get("when"))


class FaultScheduler:
    def __init__(self, world, faults: list[Fault] | None = None, server=None):
        self.world = world
        self.faults: list[Fault] = list(faults or [])
        self.server = server
        self.log: list[dict] = []
        self._n_pokes_seen = 0

    # ── triggering ──
    def tick(self, command: str | None = None) -> None:
        w = self.world
        for f in self.faults:
            if f.fired:
                continue
            if self._due(f, command):
                self.apply(f)

    def _due(self, f: Fault, command: str | None) -> bool:
        w = self.world
        if f.at_sim_s is not None and w.clock.sim() >= f.at_sim_s:
            return True
        if f.at_wall_s is not None and w.clock.wall() >= f.at_wall_s:
            return True
        if f.when:
            if f.when == "after_first_scan":
                return any(e["kind"] == "scan_saved" for e in w.events)
            if f.when.startswith("after_n_pokes:"):
                n = int(f.when.split(":", 1)[1])
                return sum(1 for e in w.events if e["kind"] == "poke") >= n
            if f.when.startswith("on_command:") and command is not None:
                return command == f.when.split(":", 1)[1]
        return False

    # ── effects ──
    def apply(self, f: Fault) -> None:
        w = self.world
        p = f.params
        k = f.kind
        with w.lock:
            if k == "thermal_drift_to_limit":
                # sample approaches the tip until the loop pins at +limit
                w.drift_v_m_per_s = np.array([w.drift_v_m_per_s[0], w.drift_v_m_per_s[1], 0.0])
                if p.get("immediate"):
                    # The range is already eaten when the fault fires. World._tick_slow
                    # integrates a creep as ``coarse_gap_m -= v·dt``; apply the equivalent
                    # jump in one go — 1.2 × the full fine-Z span (limit_low → limit_high),
                    # so the loop is pinned at the rail wherever Z sat before. No creep is
                    # left running: the drift is the jump, and a correct recovery (coarse
                    # retract + re-approach) must not be eaten again within the episode.
                    span = float(w.zctrl.limit_high_m - w.zctrl.limit_low_m)
                    jump = float(p.get("jump_factor", 1.2)) * span
                    w._tick_slow()                       # settle any pending creep first
                    w.coarse.coarse_gap_m = w.coarse.coarse_gap_m - jump
                    w._coarse_creep_v = 0.0
                    if w.zctrl.on and not w.withdrawn:
                        w.achievable_z_tip()             # refresh zctrl.z_n onto the rail now
                else:
                    # creep at v (m/s) on the slow clock (World._tick_slow)
                    w._coarse_creep_v = float(p.get("v_m_per_s", 2e-11))
            elif k == "tip_change_burst":
                w.tip.lambda_per_s = max(w.tip.lambda_per_s, float(p.get("lambda_per_s", 0.02)))
                w.tip.metastable = True
            elif k == "tip_damage":
                mode = p.get("mode", "blunt")
                from ..physics.tip import Apex
                if mode == "blunt":
                    w.tip.radius_m = float(p.get("radius_nm", 8.0)) * 1e-9
                    w.tip._refresh_apex(w.rng)
                elif mode == "multi":
                    w.tip.apexes = [Apex(0, 0, 0, 1.0), Apex(float(p.get("dx_nm", 2.5)) * 1e-9,
                                                              float(p.get("dy_nm", -1.5)) * 1e-9,
                                                              -0.2354e-9, float(p.get("w", 0.8)))]
                elif mode == "low_phi":
                    w.tip.phi_ev = float(p.get("phi_ev", 1.0))
                elif mode == "unstable":
                    w.tip.lambda_per_s = float(p.get("lambda_per_s", 0.02))
                elif mode == "dead":
                    w.tip.dead = True
                    w.tip.radius_m = 60e-9
                    w.tip._refresh_apex(w.rng)
            elif k == "feedback_oscillation":
                w.zctrl.i_m_per_s *= float(p.get("i_gain_factor", 40.0))
            elif k == "contaminate_site":
                s = w.surface.site
                sx, sy = w.sample_xy()
                for _ in range(int(p.get("n", 6))):
                    s.phi_blobs.append((sx + float(w.rng.normal(0, 150e-9)), sy + float(w.rng.normal(0, 150e-9)),
                                        float(w.rng.uniform(80e-9, 300e-9)), float(p.get("factor", 0.25))))
            elif k == "surface_spent":
                from ..physics.surface import Feature
                sx, sy = w.sample_xy()
                for _ in range(int(p.get("n", 12))):
                    w.surface.add_feature(Feature(sx + float(w.rng.normal(0, 300e-9)), sy + float(w.rng.normal(0, 300e-9)),
                                                  float(w.rng.uniform(0.5e-9, 2e-9)), float(w.rng.uniform(3e-9, 10e-9)),
                                                  kind="crater", born_sim_s=w.clock.sim()))
            elif k == "preamp_range":
                w.preamp.full_scale_a = float(p.get("full_scale_a", 1e-9))
            elif k == "approach_noise":
                w.approach_noise = float(p.get("scale", 1.0))
                w.false_landing_p = float(p.get("false_landing_p", 0.6))
            elif k == "comms_latency" and self.server is not None:
                self.server.latency_all = float(p.get("seconds", 6.0))
                if p.get("commands"):
                    self.server.latency_all = 0.0
                    for c in p["commands"]:
                        self.server.latency[c] = float(p.get("seconds", 6.0))
            elif k == "comms_drop" and self.server is not None:
                self.server.drop_commands = set(p.get("commands", ["Scan.Action"]))
            elif k == "module_unload" and self.server is not None:
                self.server.dispatcher.unloaded_modules.add(str(p.get("module", "TipShaper")))
            elif k == "clear_comms" and self.server is not None:
                self.server.latency_all = 0.0
                self.server.latency.clear()
                self.server.drop_commands.clear()
            else:
                raise ValueError(f"unknown fault kind {k!r}")
        f.fired = True
        f.fired_at_sim_s = w.clock.sim()
        entry = {"sim_s": f.fired_at_sim_s, "kind": "fault", "fault": k, **p}
        w.events.append(entry)
        self.log.append(entry)
