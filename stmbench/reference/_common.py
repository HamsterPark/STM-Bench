"""What every privileged reference solution does the same way.

Navigation reads the truth (where things are on the sample, how far it has drifted, what state
the tip is in); measurement goes through instrument actions only.
"""
from __future__ import annotations

import math

import numpy as np

from ..analysis.drift import frame_shift
from ..instrument import Instrument


class Session:
    """An instrument episode plus truth-based navigation."""

    def __init__(self, scenario: str, seed: int, out_dir=None):
        self.inst = Instrument(scenario, seed, out_dir)
        self.w = self.inst.world

    def scan_xy(self, sx: float, sy: float) -> tuple[float, float]:
        """Where a sample-frame point (m) is in scan coordinates now, in nm (privileged: the
        drift is read from the world)."""
        ox, oy, _ = self.w.drift_offset(self.w.clock.sim())
        return (sx - ox) * 1e9, (sy - oy) * 1e9

    def go(self, sx: float, sy: float) -> None:
        """Put the tip over a sample-frame point."""
        x, y = self.scan_xy(sx, sy)
        self.inst.act("move_tip", x_nm=x, y_nm=y)


def is_bare(w, x: float, y: float, level, clear: float) -> bool:
    """On terrace ``level`` (any terrace if None), no step and no scatterer within ``clear``."""
    surf = w.surface
    if level is not None and surf.terrace_level(np.array([x]), np.array([y]))[0] != level:
        return False
    return surf.step_free_window(x, y, 2 * clear) and surf.nearest_scatterer(x, y)[1] >= clear


def bare_spot(w, x0: float, y0: float, *, r_min: float = 2.5e-9, r_max: float = 12e-9,
              clear: float = 2.2e-9, same_terrace: bool = True):
    """The nearest bare spot to (x0, y0), at least ``r_min`` away."""
    level = w.surface.terrace_level(np.array([x0]), np.array([y0]))[0] if same_terrace else None
    for r in np.arange(r_min, r_max, 0.5e-9):
        for ang in np.linspace(0.0, 2 * math.pi, 24, endpoint=False):
            x, y = x0 + r * math.cos(ang), y0 + r * math.sin(ang)
            if is_bare(w, x, y, level, clear):
                return x, y
    raise RuntimeError("no bare spot nearby")


def bare_pair(w, x0: float, y0: float, *, r_min: float = 2.5e-9, clear: float = 2.2e-9):
    """Two bare spots on opposite sides of (x0, y0) at the same distance, on its terrace:
    averaging a background over them cancels a sample tilt under the point."""
    level = w.surface.terrace_level(np.array([x0]), np.array([y0]))[0]
    for r in np.arange(r_min, 12e-9, 0.5e-9):
        for ang in np.linspace(0.0, math.pi, 12, endpoint=False):
            c, s = r * math.cos(ang), r * math.sin(ang)
            a, b = (x0 + c, y0 + s), (x0 - c, y0 - s)
            if is_bare(w, *a, level, clear) and is_bare(w, *b, level, clear):
                return a, b
    raise RuntimeError("no pair of bare spots around the point")


def tip_needs_poke(w, *, clean_ldos: bool = False, imaging: bool = False) -> bool:
    """Privileged diagnosis: an unstable tip; for spectroscopy also one carrying an adsorbate;
    for imaging also a blunt (radius > 2 nm) or multiple one."""
    t = w.tip
    return bool(t.metastable or t.lambda_per_s > 2e-3
                or (clean_ldos and t.snapshot().get("ldos") == "featured")
                or (imaging and (t.radius_m > 2e-9 or t.n_apex > 1)))


def condition_tip(s: Session, x0: float, y0: float, *, clean_ldos: bool = False, imaging: bool = False,
                  r_min: float = 10e-9, r_max: float | None = None, clear: float = 4e-9,
                  same_terrace: bool = False, tries: int | None = None) -> int:
    """Condition the tip on bare surface ``r_min``–``r_max`` from (x0, y0) until the diagnosis
    is clean; returns how many rounds. A blunt or multiple tip (``imaging``) gets a 4 V, 50 ms
    pulse first — pulses reshape the apex, a shallow poke only sharpens it now and then — and
    every round ends in a shallow poke (0.4 nm at the present bias), which stabilises the
    metastable apex a pulse leaves and buries an adsorbate. A poke leaves a cluster where it
    lands and a pulse sprays clusters ~40 nm around (σ), hence ``r_min``: pulses belong
    ~150 nm from anything that is to be imaged."""
    rounds, limit = 0, tries if tries is not None else (20 if imaging else 3)
    while tip_needs_poke(s.w, clean_ldos=clean_ldos, imaging=imaging) and rounds < limit:
        s.go(*bare_spot(s.w, x0, y0, r_min=r_min, r_max=r_max if r_max is not None else r_min + 25e-9,
                        clear=clear, same_terrace=same_terrace))
        t = s.w.tip
        if imaging and (t.radius_m > 2e-9 or t.n_apex > 1):
            s.inst.act("pulse", bias_v=4.0, width_ms=50.0)
        s.inst.act("poke", depth_nm=0.4, dwell_s=0.5)
        rounds += 1
    return rounds


def lateral_drift(s: Session, x0: float, y0: float, *, size_nm: float = 12.0) -> tuple[float, float]:
    """Two frames over (x0, y0) 30 s apart: the features' velocity in the scan frame (nm/min)
    by cross-correlation — the measurement ``drift_comp`` takes."""
    inst = s.inst
    s.go(x0, y0)
    cx, cy = inst.status()["tip_nm"]
    frames = []
    for i in range(2):
        if i:
            inst.act("wait", seconds=30.0)
        r = inst.act("scan", center_x_nm=cx, center_y_nm=cy, size_nm=size_nm, pixels=128, line_time_s=0.1)
        frames.append(inst.frame(r["frame"]))
    dx, dy = frame_shift(frames[0]["array"], frames[1]["array"], frames[0]["nm_per_px"])
    dt_min = (frames[1]["t_start_s"] - frames[0]["t_start_s"]) / 60.0
    return dx / dt_min, dy / dt_min


def z_rate_pm_per_min(inst, seconds: float = 60.0) -> float:
    """The loop's Z drift with the feedback holding the setpoint."""
    z0 = inst.status()["z_nm"]
    inst.act("wait", seconds=seconds)
    return (inst.status()["z_nm"] - z0) * 1e3 / (seconds / 60.0)


def compensate_z_drift(inst, vx: float, vy: float) -> float:
    """With the lateral compensation running, measure the Z drift, program it, measure again;
    keep the sign that cancels (the documented "measure again" step). Returns the programmed
    rate (pm/min). The lateral drift has to be stopped first: a tip sliding over a tilted
    surface reads as Z drift."""
    inst.act("wait", seconds=5.0)
    rate = z_rate_pm_per_min(inst)
    best = (abs(rate), 0.0)
    for sign in (-1.0, 1.0):
        v = sign * rate
        inst.act("drift_comp", on=True, vx_nm_per_min=vx, vy_nm_per_min=vy, vz_pm_per_min=v)
        left = abs(z_rate_pm_per_min(inst))
        if left < best[0]:
            best = (left, v)
        if left < 0.2 * abs(rate):
            break
    inst.act("drift_comp", on=True, vx_nm_per_min=vx, vy_nm_per_min=vy, vz_pm_per_min=best[1])
    return best[1]


def claims_ok(inst) -> dict:
    """Per claim: reported value within tolerance and its evidence found."""
    claims = (inst.verdict.details or {}).get("claims", {})
    return {k: bool(d.get("value_ok")) and bool(d.get("evidence_ok", True)) for k, d in claims.items()}
