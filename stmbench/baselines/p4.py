"""P4 scripted baseline: move one atom by d onto its lattice site with nothing but what the
instrument shows — no truth, no MAST.

What the reference (:mod:`stmbench.reference.p4`) reads from the truth, this reads from frames:

* **the tip**: a 40 nm survey of the area the task points to judges it
  (:func:`stmbench.analysis.tip.diagnose`: wide, low atoms or none are a blunt tip, a ghost
  beside every atom a double one). Three shallow pokes 45 nm away come first whatever it
  shows — an unstable or dirty tip does not show in one frame, and a shallow contact only
  stabilises, sharpens and cleans an apex — and a blunt or double tip is pulsed 150 nm away
  and judged again on the survey, until it images well. Later, row jumps in two frames within
  15 minutes mean the tip keeps changing: three more pokes;
* **which atom**: the survey's own atoms. One with no other within 3 nm, nothing within 2 nm of
  its path or 3 nm of its goal, no step near the path (a step leaves the background uneven by
  far more than a terrace does), start and goal 5 nm inside the survey, and three other atoms
  within 14 nm to register frames on;
* **the lattice**: the adatom ruler on the survey's atoms
  (:func:`stmbench.analysis.adatoms.lattice_angle`), as in the reference;
* **where the atom is, and the drift**: two frames 30 s apart for the drift, then frames
  centred where the last one put the atom; what an atom that sits still seems to move is
  drift the compensation left, and the goal moves with it
  (:class:`stmbench.reference._manip.Track`);
* **whether it moved**: current spikes, then a work survey against the fiducials (atoms that
  match the first work survey after one shift); **whether it is still on the surface**: that
  survey shows it nowhere along its path. An atom taken by the tip is buried with the pokes and
  another one is taken, more than 11 nm from it.

The manipulation itself is the reference's: the resistance ladder from 400 kΩ down by 0.8 at
5 mV, drags at 0.25 nm/s, the verification frame 6.4 nm at 0.1 nm/px. What the baseline
reaches is what a script reaches from the observations alone (``stmbench.cli baseline``).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from ..analysis.adatoms import lattice_angle, lattice_vector, locate
from ..analysis.tip import BLUNT_HEIGHT_PM, BLUNT_SIGMA_NM, diagnose
from ..instrument import Instrument
# pure measurement helpers and settings of the reference; none of them reads the world
from ..reference._manip import (DRAG_NM_S, LADDER_FACTOR, LADDER_KOHM, LADDER_MIN_KOHM, MANIP_MAX_NA,
                                MANIP_V, SLOW_LINE_S, SLOW_NM, SLOW_PX, VERIFY_LINE_S, VERIFY_NM, VERIFY_PX,
                                Track, nearest, register)
from ._common import JUDGE_NM, JUDGE_PX, Blind, Ops, find_shift

SCENARIO = "P4_atom_positioning_cu111"
LINE_S = 0.1
#: the survey around start and goal that places the atom against its fiducials
WORK_NM, WORK_PX = 28.0, 280
ISOLATION_NM, PATH_NM, GOAL_NM, EDGE_NM, CROWD_NM = 3.0, 2.0, 3.0, 5.0, 14.0
#: an atom taken by the tip, or lost otherwise: the next one starts and ends farther than this
AVOID_NM = 11.0
ON_GOAL_NM, NEARLY_NM = 0.12, 0.6
MAX_ATTEMPTS = 30
#: what the verification frame and the report still need when the ladder stops (s)
RESERVE_S = 120.0
#: where atoms are looked for, around home (nm): the task promises an isolated one within 30 nm
SEARCH = ((0.0, 0.0), (25.0, 0.0), (-25.0, 0.0), (0.0, 25.0), (0.0, -25.0))


class Lost(Exception):
    """The atom is no longer where it can be followed: taken by the tip, or gone."""


def task_vector(text: str) -> tuple[float, float]:
    """The displacement the task asks for, as its rule card states it."""
    m = re.search(r"平移 \(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\) nm", text)
    if m is None:
        raise ValueError("the task does not say how far to move the atom")
    return float(m.group(1)), float(m.group(2))


def _levelled(frame: dict) -> np.ndarray:
    """The topograph with the rows levelled against each other (as :func:`bumps` does)."""
    z = np.asarray(frame["array"], float)
    z = np.where(np.isfinite(z), z, np.nanmedian(z))
    return z - np.concatenate([[0.0], np.cumsum(np.median(np.diff(z, axis=0), axis=1))])[:, None]


def _grid_nm(frame: dict) -> tuple[np.ndarray, np.ndarray]:
    """Scan coordinates (nm) of every pixel of an unrotated frame."""
    cx, cy = (float(v) for v in frame["center_nm"])
    size, (ny, nx) = float(frame["size_nm"]), frame["array"].shape
    x = cx - size / 2 + (np.arange(nx) + 0.5) * size / nx
    y = cy + size / 2 - (np.arange(ny) + 0.5) * size / ny
    return np.meshgrid(x, y)


def step_free(frame: dict, a, b, atoms, *, band_nm: float = 1.5, max_pm: float = 70.0) -> bool:
    """No step within ``band_nm`` of the path a→b (and as far beyond its ends): the background
    there, atoms left out, stays within ``max_pm`` of a plane. A Cu(111) step is 208 pm high,
    an atom 50-90 pm; a terrace's own ripples and noise stay within a few pm."""
    X, Y = _grid_nm(frame)
    z = _levelled(frame)
    (ax, ay), (bx, by) = a, b
    L = math.hypot(bx - ax, by - ay) or 1e-9
    ux, uy = (bx - ax) / L, (by - ay) / L
    s = (X - ax) * ux + (Y - ay) * uy
    w = -(X - ax) * uy + (Y - ay) * ux
    band = (s >= -band_nm) & (s <= L + band_nm) & (np.abs(w) <= band_nm)
    for x, y, _ in atoms:
        band &= np.hypot(X - x, Y - y) > 1.0
    if np.count_nonzero(band) < 30:
        return False
    A = np.column_stack([np.ones(np.count_nonzero(band)), X[band], Y[band]])
    coef, *_ = np.linalg.lstsq(A, z[band], rcond=None)
    r = z[band] - A @ coef
    return float(np.percentile(r, 99) - np.percentile(r, 1)) * 1e12 < max_pm


def candidates(frame: dict, atoms, d, *, avoid=(), why: dict | None = None,
               obstacles=None) -> list[tuple[float, float, float]]:
    """Atoms of a survey that can be moved by ``d`` without touching another one (see the
    module's docstring), those with three other atoms within 14 nm first, then nearest the
    frame's centre. Bumps much higher or lower than the typical atom (a dimer, a cluster, a
    deposit) are neither taken nor counted as clear of the path. ``obstacles``: every bump of
    the frame, compact or not, which the clearances are checked against — two atoms 0.9 nm
    apart merge into one elongated bump, which the compactness test that makes a bump an atom
    turns away (seed 185: the moved atom had two such neighbours 1.8 and 2.6 nm off, and was
    not isolated). ``why`` counts the reasons the others were passed over."""
    why = {} if why is None else why

    def no(reason: str) -> None:
        why[reason] = why.get(reason, 0) + 1

    pts = np.array([a[:2] for a in atoms], float).reshape(-1, 2)
    if not len(pts):
        return []
    obs = pts if obstacles is None else np.array([a[:2] for a in obstacles], float).reshape(-1, 2)
    heights = np.array([a[2] for a in atoms], float)
    typical = float(np.median(heights))
    cx, cy = (float(v) for v in frame["center_nm"])
    half = float(frame["size_nm"]) / 2
    dd = d[0] * d[0] + d[1] * d[1]
    out = []
    for k, (x, y, h) in enumerate(atoms):
        gx, gy = x + d[0], y + d[1]
        if not 0.6 * typical <= h <= 1.5 * typical:
            no("height")
            continue
        if max(abs(x - cx), abs(gx - cx), abs(y - cy), abs(gy - cy)) > half - EDGE_NM:
            no("edge")
            continue
        others = obs[np.hypot(obs[:, 0] - x, obs[:, 1] - y) > 0.3] if len(obs) else obs
        if len(others):
            if np.min(np.hypot(others[:, 0] - x, others[:, 1] - y)) <= ISOLATION_NM:
                no("neighbour")
                continue
            t = np.clip(((others[:, 0] - x) * d[0] + (others[:, 1] - y) * d[1]) / dd, 0.0, 1.0)
            if np.min(np.hypot(others[:, 0] - (x + t * d[0]), others[:, 1] - (y + t * d[1]))) <= PATH_NM:
                no("path")
                continue
            if np.min(np.hypot(others[:, 0] - gx, others[:, 1] - gy)) <= GOAL_NM:
                no("goal")
                continue
        if any(min(math.hypot(px - x, py - y), math.hypot(px - gx, py - gy)) <= AVOID_NM for px, py in avoid):
            no("avoid")
            continue
        if not step_free(frame, (x, y), (gx, gy), atoms):
            no("step")
            continue
        mates = np.delete(pts, k, axis=0)
        crowd = int(np.count_nonzero(np.hypot(mates[:, 0] - x, mates[:, 1] - y) <= CROWD_NM)) if len(mates) else 0
        out.append((crowd < 3, math.hypot(x - cx, y - cy), (float(x), float(y), float(h))))
    out.sort(key=lambda c: (c[0], c[1]))
    return [c[2] for c in out]


class Mover:
    """Manipulation and imaging settings, and the frames that place the atom."""

    def __init__(self, ops: Ops):
        self.ops, self.v = ops, ops.v

    def manipulate(self, r_ohm: float) -> None:
        v = min(MANIP_V, r_ohm * MANIP_MAX_NA * 1e-9)
        self.v.act("set_current_range", full_scale_na=100.0)
        self.v.act("set_bias", bias_v=v)
        self.v.act("set_setpoint", current_pa=v / r_ohm * 1e12)

    def image(self) -> None:
        self.v.act("set_setpoint", current_pa=self.ops.imaging_pa)
        self.v.act("set_bias", bias_v=self.ops.imaging_v)
        self.v.act("set_current_range", full_scale_na=10.0)

    def drag(self, frm, to, r_ohm: float) -> int:
        self.v.act("move_tip", x_nm=frm[0], y_nm=frm[1])
        self.manipulate(r_ohm)
        r = self.v.act("move_tip", x_nm=to[0], y_nm=to[1], speed_nm_s=DRAG_NM_S)
        self.image()
        return int(r["current_spikes"])

    def measure(self, guess, *, within: float = 1.0):
        """The atom in a slow frame centred on ``guess``: (place, time), or (None, time)."""
        f, t = self.ops.frame(guess[0], guess[1], SLOW_NM, SLOW_PX, SLOW_LINE_S)
        return nearest(locate(f), guess[0], guess[1], within), t

    def verify(self, guess) -> dict | None:
        """The verification frame over the atom and its place in it."""
        f, _ = self.ops.frame(guess[0], guess[1], VERIFY_NM, VERIFY_PX, VERIFY_LINE_S)
        hit = nearest(locate(f), guess[0], guess[1], 0.8)
        return None if hit is None else {"frame": f, "xy": hit}


class Work:
    """One atom's move: the first work survey (its fiducials and the atom in it), the lattice
    vector, and where the next survey is centred."""

    def __init__(self, ops: Ops, mv: Mover, start, L, survey_atoms, *, size_nm: float = WORK_NM):
        """``size_nm``: the work survey's side; 40 nm instead of 28 where atoms are sparse."""
        self.ops, self.mv, self.L = ops, mv, (float(L[0]), float(L[1]))
        self.size, self.start = float(size_nm), (float(start[0]), float(start[1]))
        self.centre = (start[0] + 0.5 * L[0], start[1] + 0.5 * L[1])
        f, _ = ops.frame(*self.centre, self.size, int(round(self.size / 0.1)), LINE_S)
        atoms = locate(f)
        # the big survey's atoms inside this one place it: a changed apex, the creep and the
        # loop's lag at another speed shift a whole frame, by more than a fixed window allows
        half = self.size / 2 - 0.5
        inside = [a for a in survey_atoms if abs(a[0] - self.centre[0]) < half and abs(a[1] - self.centre[1]) < half]
        sx = sy = 0.0
        if len(inside) >= 3:
            try:
                sx, sy, _ = find_shift(inside, atoms, max_shift=5.0)
            except RuntimeError:
                sx, sy, _ = find_shift(inside, atoms, max_shift=25.0)
        me = nearest(atoms, start[0] + sx, start[1] + sy, 0.6)
        if me is None:
            raise Lost("the atom is not in the work survey")
        self.me = me
        self.others = [a[:2] for a in atoms if math.hypot(a[0] - me[0], a[1] - me[1]) > 1.0]
        if len(self.others) < 2:
            raise RuntimeError("too few fiducials around the atom")
        self.shift = (0.0, 0.0)

    def survey(self, *, big: bool = False):
        """The work survey where the fiducials were last seen: (frame, atoms, diagnosis).
        ``big``: 40 nm instead of 28, for when they were not found there — a change of the
        imaging apex shifts the whole image by its offset, and a later change can take that
        shift back."""
        cx, cy = self.centre[0] + self.shift[0], self.centre[1] + self.shift[1]
        size = self.size * 10.0 / 7.0 if big else self.size          # 28 → 40 nm, 40 → 57 nm
        f, _ = self.ops.frame(cx, cy, size, int(round(size / 0.1)), LINE_S)
        return f, locate(f), diagnose(f)

    def place(self, atoms, last, *, wide: bool = False):
        """From a work survey's atoms: (the vector from the atom to its goal, the atom). The goal
        comes from the fiducials — their shift since the first survey takes up drift, creep and
        a changed apex alike — and both ends are read in one fast frame, so the loop's lag
        drops out. ``wide``: after a long move, the shift is searched up to 25 nm, and ``last``
        moved with it. The atom is the bump that is no fiducial nearest ``last`` or the stretch
        from there to the goal. Raises ``RuntimeError`` when the fiducials are not found,
        :class:`Lost` when the atom is not."""
        if wide:
            dx, dy, _ = find_shift(self.others, atoms, max_shift=25.0)
            sx, sy = register([(x + dx, y + dy) for x, y in self.others], atoms)
            sx, sy = sx + dx, sy + dy
            ax, ay = last[0] + sx - self.shift[0], last[1] + sy - self.shift[1]
        else:
            sx, sy = register([(x + self.shift[0], y + self.shift[1]) for x, y in self.others], atoms)
            sx, sy = sx + self.shift[0], sy + self.shift[1]
            ax, ay = last
        self.shift = (sx, sy)
        fid = np.array([(x + sx, y + sy) for x, y in self.others])
        goal = (self.me[0] + self.L[0] + sx, self.me[1] + self.L[1] + sy)
        free = [a for a in atoms if np.min(np.hypot(fid[:, 0] - a[0], fid[:, 1] - a[1])) > 0.5]
        ux, uy = goal[0] - ax, goal[1] - ay
        uu = ux * ux + uy * uy or 1e-12

        def off_path(a):
            t = min(max(((a[0] - ax) * ux + (a[1] - ay) * uy) / uu, 0.0), 1.0)
            return math.hypot(a[0] - (ax + t * ux), a[1] - (ay + t * uy))

        hit = min(free, key=off_path, default=None)
        if hit is None or off_path(hit) > 1.0:
            raise Lost("the atom is not along its path")
        return (goal[0] - hit[0], goal[1] - hit[1]), (float(hit[0]), float(hit[1]))


def _work(ops: Ops, mv: Mover, cands, L, atoms, log: dict):
    """The work survey for the first candidate that has two fiducials in it: 28 nm around the
    first four in turn, then 40 nm around the first (a sparse field); ``None`` when none has."""
    for start in cands[:4]:
        try:
            return Work(ops, mv, start, L, atoms)
        except RuntimeError as exc:                   # too few fiducials around this one
            log.setdefault("skipped", []).append(str(exc))
        except Lost as exc:
            exc.start = start
            raise
    try:
        return Work(ops, mv, cands[0], L, atoms, size_nm=JUDGE_NM)
    except RuntimeError as exc:
        log.setdefault("skipped", []).append(f"{exc} (40 nm)")
    return None


def _anchor(frame: dict, home, *, avoid=()) -> tuple[float, float]:
    """An atom of a frame to measure the drift on: the one nearest home, away from atoms
    lost. Home itself when the frame shows none."""
    atoms = [a for a in locate(frame) if all(math.hypot(a[0] - p[0], a[1] - p[1]) > 2.0 for p in avoid)]
    best = min(atoms, key=lambda a: math.hypot(a[0] - home[0], a[1] - home[1]), default=None)
    return (float(best[0]), float(best[1])) if best is not None else (float(home[0]), float(home[1]))


def _recover(ops: Ops, at, log: dict, *, why: str) -> tuple[float, float]:
    """After the tip was conditioned or poked: the creep of the moves waited out and the
    drift measured again on the atom being moved (``at``, where it was last seen)."""
    log.setdefault("recoveries", []).append(why)
    ops.v.act("wait", seconds=300.0 if why == "conditioned" else 120.0)
    return ops.drift(*at)


def gone_bad(diag: dict, atoms) -> bool:
    """Has the tip gone bad since it was judged fit? A double tip, a clearly blunt one (with a
    margin over the threshold that passed it: on a work survey's few atoms the median width
    wanders by a few hundredths of a nm), or no atom at all where the fiducials were (a third
    apex can smear every atom below detection). A frame that shows a few sharp atoms but not
    the fiducials has moved, not gone bad."""
    if diag["verdict"] == "double" or not atoms:
        return True
    return bool(diag.get("sigma_nm", 0.0) > BLUNT_SIGMA_NM + 0.05 or diag.get("height_pm", 1e9) < BLUNT_HEIGHT_PM - 5.0)


def _locate(ops: Ops, work: Work, last, log: dict, *, wide: bool = False):
    """The work survey again: (vector to the goal, the atom, the new drift compensation or
    ``None``). A survey that shows the tip gone bad — blunt, double, or no atoms where the
    fiducials were (a manipulation current changes the tip, and a third apex can smear every
    atom below detection) — or whose fiducials are not found even widely, has the tip
    conditioned first, the creep waited out and the drift measured again."""
    comp = None
    for _ in range(3):
        for big in (False, True):
            f, atoms, diag = work.survey(big=big)
            if gone_bad(diag, atoms):
                break
            for w in ((True,) if wide or big else (False, True)):
                try:
                    g, at = work.place(atoms, last, wide=w)
                    return g, at, comp
                except RuntimeError:
                    continue
        log["tip_mid"] = log.get("tip_mid", 0) + 1
        d, _ = ops.condition()
        if d["verdict"] != "ok":
            raise RuntimeError("the tip could not be conditioned")
        comp = _recover(ops, last, log, why="conditioned")
        wide = True
    raise RuntimeError("the work area was not found again")


def ladder(ops: Ops, mv: Mover, work: Work, comp, log: dict) -> dict | None:
    """The resistance ladder on the work's atom; the verification once it is on its goal.
    Returns the verification, ``None`` when the budget ran out first; raises :class:`Lost`."""
    pos, t = mv.measure(work.me)
    if pos is None:
        raise Lost("the atom is not in the slow frame centred on it")
    track = Track(ops.v, comp, (pos[0] + work.L[0], pos[1] + work.L[1]), t)
    attempts = log.setdefault("attempts", [])
    r = LADDER_KOHM * 1e3
    for _ in range(MAX_ATTEMPTS):
        if ops.now() > ops.budget_s - RESERVE_S - 60.0:
            return None
        if r < LADDER_MIN_KOHM * 1e3:
            raise Lost("the atom did not follow down to the lowest resistance")
        if ops.unstable():
            ops.poke_near()
            comp = _recover(ops, pos, log, why="poked")
            g, at, c2 = _locate(ops, work, pos, log, wide=True)
            comp = c2 or comp
            pos, t = mv.measure(at)
            if pos is None:
                raise Lost("the atom is not where the survey put it")
            track = Track(ops.v, comp, (pos[0] + g[0], pos[1] + g[1]), t)
        spikes = mv.drag(pos, track.goal, r)
        attempts.append({"r_kohm": round(r / 1e3, 1), "spikes": spikes})
        if spikes == 0:
            new, t_new = mv.measure(pos)
            if new is not None:
                track.still(pos, t, new, t_new)
                pos, t = new, t_new
                r *= LADDER_FACTOR
                continue
            # not where it sat: taken by the tip without a hop, or the tip changed
        (gx, gy), at, c2 = _locate(ops, work, pos, log)
        if c2 is not None:
            comp = c2
        off = math.hypot(gx, gy)
        attempts[-1]["off_goal_nm"] = round(off, 3)
        if off <= ON_GOAL_NM:
            log["r_success_kohm"] = r / 1e3
            got = mv.verify(at)
            if got is None:
                raise Lost("the atom is not in its verification frame")
            return got
        if off > NEARLY_NM:
            r *= LADDER_FACTOR
        pos, t = mv.measure(at)
        if pos is None:
            raise Lost("the atom is not where the survey put it")
        track = Track(ops.v, comp, (pos[0] + gx, pos[1] + gy), t) if c2 is not None else track
        track.goal, track.t = (pos[0] + gx, pos[1] + gy), t
    raise Lost("the atom did not reach its goal")


def play(v: Blind, log: dict) -> dict | None:
    """One P4 session through ``v``: the verification of the moved atom, or ``None``."""
    d = task_vector(v.task())
    ops = Ops(v, log)
    mv = Mover(ops)
    home = ops.home
    f, _ = ops.frame(*home, JUDGE_NM, JUDGE_PX, LINE_S)
    diag = diagnose(f)
    log["tip_first"] = diag["verdict"]
    ops.poke_near()
    waited = 120.0
    if diag["verdict"] != "ok":
        diag, f = ops.condition()
        waited = 300.0
        if diag["verdict"] != "ok":
            raise RuntimeError("the tip could not be conditioned")
    v.act("wait", seconds=waited)
    comp = ops.drift(*_anchor(f, home))
    lost: list[tuple[float, float]] = []
    for _ in range(4):
        survey, atoms, bad, seen, work, lost_on = None, [], False, [], None, None
        for centre in SEARCH:
            survey, _ = ops.frame(home[0] + centre[0], home[1] + centre[1], JUDGE_NM, JUDGE_PX, LINE_S)
            if diagnose(survey)["verdict"] in ("blunt", "double"):
                bad = True
                break
            atoms = locate(survey)
            seen.append(atoms)
            why: dict = {}
            cands = candidates(survey, atoms, d, avoid=lost, why=why, obstacles=locate(survey, compact_frac=1.0))
            log.setdefault("passed_over", []).append(why)
            if not cands:
                continue
            # the ruler: the survey of this round with the most atoms (vectors within one frame)
            ruler = max(seen, key=len)
            phi, rms, n = lattice_angle(ruler)
            L = lattice_vector(d, phi)
            log.update(lattice_deg=math.degrees(phi), lattice_rms_nm=rms, lattice_vectors=n, ruler_atoms=len(ruler),
                       survey_atoms=len(atoms), candidates=len(cands))
            try:
                work = _work(ops, mv, cands, L, atoms, log)
            except Lost as exc:
                lost_on = (exc, getattr(exc, "start", cands[0]))
                break
            if work is not None:
                break
        if bad:
            diag, f = ops.condition()
            if diag["verdict"] != "ok":
                raise RuntimeError("the tip could not be conditioned")
            v.act("wait", seconds=300.0)
            comp = ops.drift(*_anchor(f, home))
            continue
        try:
            if work is None:
                if lost_on is not None:
                    raise lost_on[0]
                raise RuntimeError(f"no atom near home can be moved by d ({log['passed_over']})")
            log["start_nm"] = list(work.start)
            return ladder(ops, mv, work, comp, log)
        except Lost as exc:
            log.setdefault("lost", []).append(str(exc))
            lost.append(work.start if work is not None else lost_on[1][:2])
            ops.poke_near()                        # buries an atom the tip may carry
            v.act("wait", seconds=120.0)
            comp = ops.drift(*_anchor(survey, home, avoid=lost))
            log["atoms_tried"] = log.get("atoms_tried", 1) + 1
    return None


def run(seed: int, *, out_dir: str | Path | None = None, scenario: str = SCENARIO) -> dict:
    """One P4 episode by the baseline; what it reported, how it scored, and (from the truth,
    after the episode only) where it went wrong."""
    inst = Instrument(scenario, seed, out_dir)
    log: dict = {"atoms_tried": 1}
    got, error = None, None
    try:
        got = play(Blind(inst), log)
    except Exception as exc:  # noqa: BLE001 — a failed session is a result
        error = f"{type(exc).__name__}: {exc}"
    if got is not None:
        x, y = got["xy"]
        inst.report("target_site", x, unit="nm", x_nm=x, y_nm=y, frame=got["frame"]["frame"],
                    evidence=[got["frame"]["frame"]])
    out = inst.finish()
    det = (inst.verdict.details or {}).get("claims", {})
    ts = det.get("target_site", {})
    truth = ts.get("truth") if isinstance(ts.get("truth"), dict) else {}
    placed, now = ts.get("sample_xy_nm"), truth.get("now_nm")
    err = float(math.hypot(placed[0] - now[0], placed[1] - now[1])) if placed and now else float("nan")
    return {"paper": "P4", "seed": int(seed), "success": bool(out["success"]),
            "partial": float(out.get("partial", 0.0)), "error": error,
            "claims_ok": {k: bool(c.get("value_ok")) and bool(c.get("evidence_ok", True)) for k, c in det.items()},
            "on_goal": truth.get("on_goal"), "n_bystanders": truth.get("n_bystanders"),
            # how far the reported place is from the atom (nm), against report_nm
            "abs_err": {"target_site": err},
            "instrument_s": float(inst.used_s),
            "tip_condition": inst.scenario.draw_hidden(seed).get("tip", {}).get("condition"), **log}
