"""What the scripted baselines share: the instrument as an agent gets it, and the routine an
operator runs on it — frames, drift, judging and conditioning the tip, finding a work area again.

Nothing here reads the simulator. A baseline gets a :class:`Blind` view of the instrument: the
actions, the frames and spectra it took, the status and the instrument's description, the task
and its rule card. That is what an agent gets over the MCP server (``stmbench episode``).
"""
from __future__ import annotations

import math

import numpy as np

from ..analysis.adatoms import locate, row_jumps
from ..analysis.drift import frame_shift
from ..analysis.tip import diagnose

#: fastest line speed used: above 450 nm/s a tip of more than 1.5 nm scratches the surface
MAX_SPEED_NM_S = 400.0
#: the survey that judges the tip: 40 nm at 0.1 nm/px over home, where the task promises atoms
JUDGE_NM, JUDGE_PX = 40.0, 400


class Blind:
    """The instrument as an agent sees it. ``act``, ``frame``, ``spectrum``, ``status``,
    ``info``, ``task`` and ``report`` pass through; the world behind it is not reachable from
    here."""

    def __init__(self, inst):
        self.__inst = inst

    def act(self, name: str, **args) -> dict:
        return self.__inst.act(name, **args)

    def frame(self, name: str, channel: str = "z", direction: str = "forward") -> dict:
        return self.__inst.frame(name, channel=channel, direction=direction)

    def spectrum(self, name: str) -> dict:
        return self.__inst.spectrum(name)

    def status(self) -> dict:
        return self.__inst.status()

    def info(self) -> dict:
        return self.__inst.info()

    def task(self) -> str:
        return self.__inst.task()

    def report(self, claim_id: str, value: float, **kw) -> str:
        return self.__inst.report(claim_id, value, **kw)


class Ops:
    """A session at the instrument: frames (each one checked for the tip changing in it),
    the row jumps seen so far, the imaging conditions to come back to, and ``home``, the place
    the task points to."""

    def __init__(self, v: Blind, log: dict, *, home: tuple[float, float] = (0.0, 0.0)):
        self.v, self.log, self.home = v, log, (float(home[0]), float(home[1]))
        st = v.status()
        self.imaging_v, self.imaging_pa = float(st["bias_v"]), float(st["setpoint_pa"])
        self.budget_s = float(v.info()["budget_s"])
        self.jumps: list[float] = []                       # instrument times of tip changes seen
        self.n_pokes = 0

    def now(self) -> float:
        return float(self.v.status()["instrument_time_s"])

    def frame(self, cx: float, cy: float, size: float, px: int, line_s: float, *, retakes: int = 2):
        """A frame and the instrument time at its middle row. One in which the tip changed (a row
        jump) is taken again, up to ``retakes`` times, and the change remembered. Jumps that
        recur at the same rows in the retake are the surface — a step running along the rows
        steps the whole row, as a tip change does — and are neither retaken nor remembered
        (seed 159: a step in every frame of the work area had the tip poked five times)."""
        line_s = max(float(line_s), size / MAX_SPEED_NM_S)
        prev = None
        for k in range(retakes + 1):
            r = self.v.act("scan", center_x_nm=cx, center_y_nm=cy, size_nm=size, pixels=px, line_time_s=line_s)
            f = self.v.frame(r["frame"])
            jumps = row_jumps(f["array"])
            t = float(f["t_start_s"]) + 0.5 * float(r["duration_s"])
            if not jumps:
                if prev is not None:
                    self.jumps.append(prev[1])                 # the earlier frame's jump was the tip
                return f, t
            if prev is not None and _same_rows(prev[0], jumps):
                self.log["surface_jumps"] = self.log.get("surface_jumps", 0) + 1
                return f, t
            if prev is not None:
                self.jumps.append(prev[1])
            prev = (jumps, float(f["t_start_s"]))
            if k == retakes:
                self.jumps.append(prev[1])
                return f, t
        raise AssertionError("unreachable")

    def unstable(self, window_s: float = 900.0, n: int = 2) -> bool:
        """``n`` tip changes within the last ``window_s`` of instrument time."""
        t = self.now()
        return sum(1 for j in self.jumps if t - j <= window_s) >= n

    def go(self, x: float, y: float) -> None:
        self.v.act("move_tip", x_nm=x, y_nm=y)

    def drift(self, x: float, y: float, *, size_nm: float = 20.0) -> tuple[float, float]:
        """Lateral drift: two frames 30 s apart around (x, y), with the compensation off (with it
        on, they would give only what it leaves); then compensated. The shift is the atoms'
        own (the same ones found again within 3 nm); cross-correlation only when the frames
        show none: on a bare frame it locks onto noise, and a wrong velocity makes the
        compensation push the image away, and shears every frame after it. The first frame
        also re-centres the pair on the atoms it shows: after a long move the creep has
        carried the sample nanometres from where an atom was last seen."""
        self.v.act("drift_comp", on=False)
        px = int(round(size_nm / 0.1))
        f0, _ = self.frame(x, y, size_nm, px, 0.05)
        a = locate(f0)
        if a:
            ax, ay = np.median(np.array([p[:2] for p in a]), axis=0)
            x, y = float(ax), float(ay)
        frames = []
        for i in range(2):
            if i:
                self.v.act("wait", seconds=30.0)
            f, _ = self.frame(x, y, size_nm, px, 0.05)
            frames.append(f)
        a, b = locate(frames[0]), locate(frames[1])
        n = 0
        if a and b:
            try:
                dx, dy, n = find_shift(a, b, within=0.3, max_shift=3.0)
            except RuntimeError:
                n = 0
        if not n:
            dx, dy = frame_shift(frames[0]["array"], frames[1]["array"], frames[0]["nm_per_px"])
        self.log.setdefault("drift_atoms", []).append(int(n))
        dt_min = (frames[1]["t_start_s"] - frames[0]["t_start_s"]) / 60.0
        vx, vy = dx / dt_min, dy / dt_min
        self.v.act("drift_comp", on=True, vx_nm_per_min=vx, vy_nm_per_min=vy)
        return vx, vy

    def poke_near(self, n: int = 3, *, away_nm: float = 60.0) -> None:
        """Shallow pokes (0.4 nm) on fresh ground ``away_nm`` below home. The contact stabilises
        the apex (an unstable tip changes every few minutes, under a manipulation current
        within seconds), now and then sharpens it, and buries what it carries; it does not
        blunt it or add an apex. A poke leaves one cluster where it lands and sprays none."""
        hx, hy = self.home
        for _ in range(n):
            k = self.n_pokes
            self.n_pokes += 1
            self.go(hx + 3.0 * (k % 10) - 13.5, hy - away_nm - 3.0 * (k // 10))
            self.v.act("poke", depth_nm=0.4, dwell_s=0.5)
        self.go(hx, hy)
        self.jumps.clear()
        self.log["pokes"] = self.log.get("pokes", 0) + n

    def condition(self, *, rounds: int = 20, away_nm: float = 150.0) -> tuple[dict, dict]:
        """Reshape a blunt or double tip: a 4 V, 50 ms pulse ``away_nm`` above home on fresh
        ground each round (pulses spray clusters ~40 nm around), two shallow pokes beside it
        (a pulse leaves the apex metastable), then back home, where the task promises atoms,
        for a 40 nm survey that judges it; until it images well. Returns the last diagnosis
        and its survey; the creep of the long moves is the caller's to wait out."""
        hx, hy = self.home
        d, f, n = {"verdict": "unknown"}, None, 0
        for n in range(1, rounds + 1):
            x0 = hx + 8.0 * n
            self.go(x0, hy + away_nm)
            self.v.act("pulse", bias_v=4.0, width_ms=50.0)
            for k in (1, 2):
                self.go(x0 + 2.0 * k, hy + away_nm)
                self.v.act("poke", depth_nm=0.4, dwell_s=0.5)
            self.go(hx, hy)
            f, _ = self.frame(hx, hy, JUDGE_NM, JUDGE_PX, 0.1)
            d = diagnose(f)
            if d["verdict"] == "ok":
                break
        self.log["tip_rounds"] = self.log.get("tip_rounds", 0) + n
        self.jumps.clear()
        return d, f


def _same_rows(a: list[int], b: list[int], *, tol: int = 3) -> bool:
    """Do two frames of one place step at the same rows (to ``tol``, the residual drift)?"""
    return bool(a) and len(a) == len(b) and all(min(abs(x - y) for y in b) <= tol for x in a)


def survey_atoms(frame: dict) -> list[tuple[float, float, float]]:
    """The adatoms of a frame (scan coordinates), compact bumps only."""
    return [(x, y, h) for x, y, h in locate(frame)]


def find_shift(ref, now, *, within: float = 0.4, max_shift: float = 25.0) -> tuple[float, float, int]:
    """How far the atoms of an earlier frame have moved in a new one (drift, creep, a changed
    apex): of the shifts one atom pair proposes, the one most atoms agree with, refined by
    their median. Returns (dx, dy, atoms agreeing)."""
    a = np.asarray([p[:2] for p in ref], float)
    b = np.asarray([p[:2] for p in now], float)
    if not len(a) or not len(b):
        raise RuntimeError("no atoms to register")
    best, best_n = None, 0
    for p in a:
        for q in b:
            dv = q - p
            if math.hypot(*dv) > max_shift:
                continue
            dist = np.min(np.hypot(b[None, :, 0] - (a[:, 0] + dv[0])[:, None], b[None, :, 1] - (a[:, 1] + dv[1])[:, None]), axis=1)
            n = int(np.count_nonzero(dist <= within))
            if n > best_n:
                best, best_n = dv, n
    if best is None:
        raise RuntimeError("the atoms were not found again")
    pairs = []
    for p in a:
        q = b[int(np.argmin(np.hypot(*(b - (p + best)).T)))]
        if math.hypot(*(q - p - best)) <= within:
            pairs.append(q - p)
    dx, dy = np.median(np.array(pairs), axis=0)
    return float(dx), float(dy), best_n
