"""P5 privileged reference solution: Sader–Jarvis force spectroscopy over one Fe atom.

Privileged means the simulator's truth is used to *navigate* — which atom to use, which nearby
spot is bare surface on the same terrace, how far the sample has drifted, whether the tip needs
a shallow poke first — and never for a reported number. The three numbers come from Δf(z)
curves the instrument measured, through :mod:`stmbench.analysis.force`. What a run establishes
is how precisely a careful operator can measure P5 on this instrument (for the tolerances) and
how much instrument time that takes (for the budget).

Protocol:

1. if the tip is unstable, a shallow poke (0.4 nm, at the 20 mV imaging bias, which does not
   ring the sensor up) on bare surface well away from the atoms, at most three times;
2. two 12 nm images over the atom 30 s apart; their cross-correlation gives the lateral drift,
   which the compensation then follows (:mod:`stmbench.analysis.drift`);
3. over bare surface: the loop's Z drift is measured and compensated (measure, program, measure
   again), then the oscillation is switched on (the frequency shift is zeroed there);
4. curves alternate bare, atom, bare, … ending on bare, the bare ones on two spots opposite
   each other across the atom: a slow Z drift between curves and a sample tilt under the atom
   both average out of the background. Repeats are combined with the median, which a
   frequency jump in one curve does not survive. The feedback height over the atom is about 0.4 nm, already inside the
   bond's well, so every curve starts 0.75–0.95 nm farther away and ends 0.15–0.2 nm closer;
5. Δf over bare surface is subtracted at equal loop Z, the difference inverted, and F_min, the
   decay length and E_b read off (:func:`stmbench.analysis.force.p5_numbers`).

The oscillation amplitude is 100 pm. Sader–Jarvis is an approximation whose error peaks when
the amplitude is comparable with the force's decay length (50–125 pm here): on noise-free
curves from the force law (20 seeds) it reads F_min 6.7 % low on average at 50 pm (4.5–7.4 %)
and 1.6 % low at 100 pm (from 5.9 % low to 2.1 % high). With the drift compensated, 30 seeds
(0–29, the draws of 2026-10-12) all reproduced, |error| p90 5.4 % (F_min), 2.0 % (decay
length) and 7.5 % (E_b), in about 354 s of instrument time; sixteen sweeps instead of four
changed nothing on eight seeds, so noise is not what limits it.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..analysis.drift import frame_shift
from ..analysis.force import Curve, p5_numbers
from ..instrument import Instrument

SCENARIO = "P5_sader_jarvis_qplus"
#: curves over the atom: start this far above the feedback height, sweep this far (nm)
ATOM_OFFSET_NM, ATOM_SWEEP_NM = 0.75, 0.95
#: curves over bare surface: wider, so that they span the atom's curves in absolute Z (the
#: feedback height over the atom is higher by the atom's height, ~0.1 nm)
BG_OFFSET_NM, BG_SWEEP_NM = 0.95, 1.10
POINTS = 512


def _isolated_atom(w):
    """The atom nearest the scan origin with no other atom within 2.5 nm, ≥ 1.5 nm from a step,
    and room for a pair of bare spots on both sides of it (:func:`_bare_pair`)."""
    reg = w.surface.site.adatoms
    xy = reg.positions()
    cands = []
    for a in reg.atoms.values():
        x, y = reg.site_xy(a.i, a.j, a.sub)
        if sorted(np.hypot(xy[:, 0] - x, xy[:, 1] - y))[1] < 2.5e-9:
            continue
        if not w.surface.step_free_window(x, y, 3e-9):
            continue
        cands.append((math.hypot(x, y), x, y))
    for _, x, y in sorted(cands):
        try:
            _bare_pair(w, x, y)
        except RuntimeError:
            continue
        return x, y
    raise RuntimeError("no isolated atom with bare surface on both sides")


def _is_bare(w, x: float, y: float, level, clear: float) -> bool:
    surf = w.surface
    return (surf.terrace_level(np.array([x]), np.array([y]))[0] == level
            and surf.step_free_window(x, y, 2 * clear) and surf.nearest_scatterer(x, y)[1] >= clear)


def _bare_spot(w, ax: float, ay: float, *, r_min: float = 2.5e-9, clear: float = 2.2e-9):
    """The nearest spot to the atom on the same terrace with no scatterer within ``clear``."""
    level = w.surface.terrace_level(np.array([ax]), np.array([ay]))[0]
    for r in np.arange(r_min, 12e-9, 0.5e-9):
        for ang in np.linspace(0.0, 2 * math.pi, 24, endpoint=False):
            x, y = ax + r * math.cos(ang), ay + r * math.sin(ang)
            if _is_bare(w, x, y, level, clear):
                return x, y
    raise RuntimeError("no bare spot near the atom")


def _bare_pair(w, ax: float, ay: float, *, r_min: float = 2.5e-9, clear: float = 2.2e-9):
    """Two bare spots on opposite sides of the atom at the same distance: averaging the
    background over them cancels a sample tilt (and any linear height trend) under the atom.
    Over a blunt tip's van der Waals force, 7 pm of plane offset is ~6 % of F_min."""
    level = w.surface.terrace_level(np.array([ax]), np.array([ay]))[0]
    for r in np.arange(r_min, 12e-9, 0.5e-9):
        for ang in np.linspace(0.0, math.pi, 12, endpoint=False):
            c, s_ = r * math.cos(ang), r * math.sin(ang)
            a, b = (ax + c, ay + s_), (ax - c, ay - s_)
            if _is_bare(w, *a, level, clear) and _is_bare(w, *b, level, clear):
                return a, b
    raise RuntimeError("no pair of bare spots around the atom")


def _z_rate_pm_per_min(inst, seconds: float = 60.0) -> float:
    """The loop's Z drift with the feedback holding the setpoint over bare surface."""
    z0 = inst.status()["z_nm"]
    inst.act("wait", seconds=seconds)
    return (inst.status()["z_nm"] - z0) * 1e3 / (seconds / 60.0)


def _lateral_drift(s: "_Session", ax: float, ay: float) -> tuple[float, float]:
    """Two 12 nm frames over the atom 30 s apart: the features' velocity in the scan frame
    (nm/min) by cross-correlation — the measurement ``drift_comp`` takes."""
    inst = s.inst
    s.go(ax, ay)
    cx, cy = inst.status()["tip_nm"]
    frames = []
    for i in range(2):
        if i:
            inst.act("wait", seconds=30.0)
        r = inst.act("scan", center_x_nm=cx, center_y_nm=cy, size_nm=12.0, pixels=128, line_time_s=0.1)
        frames.append(inst.frame(r["frame"]))
    dx, dy = frame_shift(frames[0]["array"], frames[1]["array"], frames[0]["nm_per_px"])
    dt_min = (frames[1]["t_start_s"] - frames[0]["t_start_s"]) / 60.0
    return dx / dt_min, dy / dt_min


def _compensate_z_drift(inst, vx: float, vy: float) -> float:
    """With the lateral compensation running, measure the Z drift, program it, measure again;
    keep the sign that cancels (the documented "measure again" step). Returns the programmed
    rate (pm/min).

    Without it the cold rig's ~60 pm/min shifts consecutive curves by tens of pm and, with
    long averaging, the curves no longer overlap where the force minimum is. The lateral drift
    has to be stopped first: a tip sliding over a tilted surface reads as Z drift."""
    inst.act("wait", seconds=5.0)
    rate = _z_rate_pm_per_min(inst)
    best = (abs(rate), 0.0)
    for sign in (-1.0, 1.0):
        v = sign * rate
        inst.act("drift_comp", on=True, vx_nm_per_min=vx, vy_nm_per_min=vy, vz_pm_per_min=v)
        left = abs(_z_rate_pm_per_min(inst))
        if left < best[0]:
            best = (left, v)
        if left < 0.2 * abs(rate):
            break
    inst.act("drift_comp", on=True, vx_nm_per_min=vx, vy_nm_per_min=vy, vz_pm_per_min=best[1])
    return best[1]


class _Session:
    def __init__(self, seed: int, out_dir=None):
        self.inst = Instrument(SCENARIO, seed, out_dir)
        self.w = self.inst.world

    def go(self, sx: float, sy: float) -> None:
        """Put the tip over a sample-frame point (privileged: the drift is read from the world)."""
        ox, oy, _ = self.w.drift_offset(self.w.clock.sim())
        self.inst.act("move_tip", x_nm=(sx - ox) * 1e9, y_nm=(sy - oy) * 1e9)

    def curve(self, offset_nm: float, sweep_nm: float, sweeps: int) -> tuple[str, dict]:
        r = self.inst.act("zspec", sweep_nm=sweep_nm, z_offset_nm=offset_nm, points=POINTS,
                          sweeps=sweeps, backward=False)
        return r["spectrum"], self.inst.spectrum(r["spectrum"])


def run(seed: int, *, out_dir: str | Path | None = None, amplitude_pm: float = 100.0,
        repeats: int = 3, sweeps: int = 4) -> dict:
    """One P5 episode by the reference protocol; returns what it reported and how it scored."""
    s = _Session(seed, out_dir)
    inst, w = s.inst, s.w
    ax, ay = _isolated_atom(w)
    bare_spots = _bare_pair(w, ax, ay)
    pokes = 0
    while (w.tip.metastable or w.tip.lambda_per_s > 2e-3) and pokes < 3:
        px, py = _bare_spot(w, ax, ay, r_min=10e-9, clear=4e-9)
        s.go(px, py)
        inst.act("poke", depth_nm=0.4, dwell_s=0.5)
        pokes += 1
    vx, vy = _lateral_drift(s, ax, ay)
    inst.act("drift_comp", on=True, vx_nm_per_min=vx, vy_nm_per_min=vy)
    s.go(*bare_spots[0])
    vz = _compensate_z_drift(inst, vx, vy)
    inst.act("pll", on=True, amplitude_pm=amplitude_pm)
    atom, bare, names = [], [], []
    for i in range(repeats + 1):
        s.go(*bare_spots[i % 2])
        n, sp = s.curve(BG_OFFSET_NM, BG_SWEEP_NM, sweeps)
        bare.append(Curve.from_spectrum(sp))
        names.append(n)
        if i == repeats:
            break
        s.go(ax, ay)
        n, sp = s.curve(ATOM_OFFSET_NM, ATOM_SWEEP_NM, sweeps)
        atom.append(Curve.from_spectrum(sp))
        names.append(n)
        amp_m = float(np.nanmedian(sp["amp"]))
    q = inst.info()["qplus"]
    got = p5_numbers(atom, bare, f0=q["f0_hz"], k=q["k_n_per_m"], amplitude=amp_m)
    units = {"f_min_pn": "pN", "f_decay_pm": "pm", "e_bind_mev": "meV"}
    for c in inst.scenario.claims:
        cid = str(c["id"])
        inst.report(cid, float(got[cid]), unit=units.get(cid, ""), evidence=names)
    out = inst.finish()
    truth = w.truth()["force"]
    claims = inst.verdict.details.get("claims", {})
    return {"paper": "P5", "seed": int(seed), "success": bool(out["success"]),
            "partial": float(out.get("partial", 0.0)),
            "values": {k: float(got[k]) for k in units},
            "truth": {k: float(truth[k]) for k in units},
            "rel_err": {k: float(got[k] / truth[k] - 1.0) for k in units},
            "claims_ok": {k: bool(d.get("value_ok")) and bool(d.get("evidence_ok", True))
                          for k, d in claims.items()},
            "instrument_s": float(inst.used_s), "tip_pokes": pokes, "curves": len(names),
            "drift_comp": {"vx_nm_per_min": float(vx), "vy_nm_per_min": float(vy), "vz_pm_per_min": float(vz)},
            "tip_condition": inst.scenario.draw_hidden(seed).get("tip", {}).get("condition")}
