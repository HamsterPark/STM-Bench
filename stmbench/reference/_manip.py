"""Moving atoms with the instrument: what the reference solutions that build things share
(P4, P3's repair).

The truth navigates — where an atom is, whether it is still on the surface, the tip's state —
and the manipulation itself is measured: the resistance at which an atom follows is found by
lowering it until current spikes say so, and where an atom has to go, and whether it got there,
is read from frames. What the P4 trials taught is built in:

* the Z loop lags the surface along the fast axis (0.15 nm at 280 nm/s), so places that
  decide a drag's end come from a slow frame, or from vectors between two things in one fast
  frame, where the lag drops out;
* the creep of a long move decays as 1/(τ + t) and outruns a constant compensation within
  minutes, so an atom that sits still is measured again before each rung, the goal moves with
  it, and the compensation takes up the residual velocity (:class:`Track`);
* the manipulation current changes the tip, so every frame is checked (:func:`scan`) and a tip
  unfit to image ends the attempt (:class:`LostTrack`) until it is conditioned again;
* drift is measured with the compensation off: with it on, two frames give only what it
  leaves.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from ..analysis.adatoms import locate, row_jumps
from ._common import Session, lateral_drift, tip_needs_poke

#: manipulation at 5 mV; below 62.5 kΩ the bias comes down instead of the current going above
#: 80 nA (the setpoint's range ends at 100 nA, and every nA raises the tip's change rate)
MANIP_V, MANIP_MAX_NA = 0.005, 80.0
LADDER_KOHM, LADDER_FACTOR, LADDER_MIN_KOHM = 400.0, 0.8, 25.0
DRAG_NM_S = 0.25
#: slow frames: 20 nm/s, where the loop's lag is ~0.01 nm
SLOW_NM, SLOW_PX, SLOW_LINE_S = 5.0, 50, 0.25
VERIFY_NM, VERIFY_PX, VERIFY_LINE_S = 6.4, 64, 0.32
MAX_ATTEMPTS = 30


class LostTrack(Exception):
    """The tip has to be conditioned in the middle of a manipulation."""


def scan(inst, cx: float, cy: float, size: float, px: int, line_s: float) -> tuple[dict, float]:
    """A frame and the instrument time at its middle row. Raises :class:`LostTrack` when the
    tip is not fit to image by the end of it: a second apex puts a ghost beside every atom, and
    places read from such a frame are wrong. A frame in which the tip changed
    (:func:`stmbench.analysis.adatoms.row_jumps`) is taken again: its rows before and after the
    change disagree by the apex's sideways jump."""
    for _ in range(3):
        r = inst.act("scan", center_x_nm=cx, center_y_nm=cy, size_nm=size, pixels=px, line_time_s=line_s)
        if tip_needs_poke(inst.world, imaging=True):
            raise LostTrack()
        f = inst.frame(r["frame"])
        if not row_jumps(f["array"]):
            break
    return f, float(f["t_start_s"]) + 0.5 * float(r["duration_s"])


def nearest(atoms, x: float, y: float, within: float):
    best = min(atoms, key=lambda a: math.hypot(a[0] - x, a[1] - y), default=None)
    if best is None or math.hypot(best[0] - x, best[1] - y) > within:
        return None
    return float(best[0]), float(best[1])


def register(ref, now, *, within: float = 0.3, max_shift: float = 4.0) -> tuple[float, float]:
    """The shift that carries fiducials of an earlier frame onto the atoms of a new one: of the
    shifts one fiducial-atom pair proposes, the one that most fiducials agree with (to
    ``within``), refined by their median. A conditioned tip images everything shifted by its
    new apex, by more than any fixed matching radius would allow."""
    ref = np.asarray([p[:2] for p in ref], float)
    now = np.asarray([p[:2] for p in now], float)
    best, best_n = None, 0
    for f in ref:
        for a in now:
            d = a - f
            if math.hypot(*d) > max_shift:
                continue
            dist = np.min(np.hypot(*(now[None, :, :] - (ref + d)[:, None, :]).transpose(2, 0, 1)), axis=1)
            n = int(np.count_nonzero(dist <= within))
            if n > best_n:
                best, best_n = d, n
    if best is None or best_n < min(2, len(ref)):
        raise RuntimeError("the fiducials were not found again")
    moved = ref + best
    pairs = [(now[int(np.argmin(np.hypot(*(now - m).T)))] - f) for f, m in zip(ref, moved)
             if np.min(np.hypot(*(now - m).T)) <= within]
    sx, sy = np.median(np.array(pairs), axis=0)
    return float(sx), float(sy)


class Track:
    """The goal in scan coordinates, kept where the sample has drifted it.

    Drift shows on an atom that sits still: whatever it seems to have moved between two slow
    frames moved the goal too, and the compensation takes up the residual velocity."""

    def __init__(self, inst, comp: tuple[float, float], goal, t: float):
        self.inst, self.comp = inst, [float(comp[0]), float(comp[1])]
        self.goal, self.t = (float(goal[0]), float(goal[1])), float(t)

    def still(self, before, t0: float, after, t1: float) -> None:
        """The same atom, unmoved, measured at t0 and t1."""
        sx, sy = after[0] - before[0], after[1] - before[1]
        self.goal, self.t = (self.goal[0] + sx, self.goal[1] + sy), t1
        dt = t1 - t0
        if dt >= 15.0:
            self.comp = [self.comp[0] + sx / dt * 60.0, self.comp[1] + sy / dt * 60.0]
            self.inst.act("drift_comp", on=True, vx_nm_per_min=self.comp[0], vy_nm_per_min=self.comp[1])


class Mover:
    """The instrument settings and frames of a manipulation session."""

    def __init__(self, s: Session, *, imaging_v: float, imaging_pa: float, compact_frac: float = 0.6,
                 apex_aware: bool = False):
        """``compact_frac``: how compact a bump has to be to be taken for the atom being looked
        for (:func:`stmbench.analysis.adatoms.bumps`). ``apex_aware``: look where the atom shows
        through the tip's apex (:meth:`Session.image_xy`) rather than where it is; the apex's
        offset walks with the tip's changes, past the 0.8 nm a frame is searched on some
        seeds."""
        self.s, self.inst = s, s.inst
        self.imaging_v, self.imaging_pa = float(imaging_v), float(imaging_pa)
        self.compact_frac = float(compact_frac)
        self._where = s.image_xy if apex_aware else s.scan_xy

    def manipulate(self, r_ohm: float) -> None:
        """Manipulation conditions: the preamp's range first, then the bias, then the setpoint."""
        v = min(MANIP_V, r_ohm * MANIP_MAX_NA * 1e-9)
        self.inst.act("set_current_range", full_scale_na=100.0)
        self.inst.act("set_bias", bias_v=v)
        self.inst.act("set_setpoint", current_pa=v / r_ohm * 1e12)

    def image(self) -> None:
        """Back to imaging, in the reverse order: the setpoint comes down before the range does."""
        self.inst.act("set_setpoint", current_pa=self.imaging_pa)
        self.inst.act("set_bias", bias_v=self.imaging_v)
        self.inst.act("set_current_range", full_scale_na=10.0)

    def drag(self, frm, to, r_ohm: float) -> int:
        """The tip onto ``frm`` at the imaging setpoint, to ``to`` at ``r_ohm``; the current
        spikes on the way (one per hop of an atom)."""
        self.inst.act("move_tip", x_nm=frm[0], y_nm=frm[1])
        self.manipulate(r_ohm)
        r = self.inst.act("move_tip", x_nm=to[0], y_nm=to[1], speed_nm_s=DRAG_NM_S)
        self.image()
        return int(r["current_spikes"])

    def drift(self, x0: float, y0: float) -> tuple[float, float]:
        """Lateral drift at a sample point, measured with the compensation off (with it on, two
        frames give only what it leaves), then compensated."""
        self.inst.act("drift_comp", on=False)
        vx, vy = lateral_drift(self.s, x0, y0)
        self.inst.act("drift_comp", on=True, vx_nm_per_min=vx, vy_nm_per_min=vy)
        return vx, vy

    def measure(self, atom):
        """The atom from a slow frame centred where the truth has it: (place, time). Raises
        :class:`LostTrack` when it cannot be seen because the tip went bad."""
        reg = self.s.w.surface.site.adatoms
        x, y = self._where(*reg.site_xy(atom.i, atom.j, atom.sub))
        f, t = scan(self.inst, x, y, SLOW_NM, SLOW_PX, SLOW_LINE_S)
        hit = nearest(locate(f, compact_frac=self.compact_frac), x, y, 0.8)
        if hit is None:
            if tip_needs_poke(self.s.w, imaging=True):
                raise LostTrack()
            raise RuntimeError("the atom is not in the frame centred on it")
        return hit, t

    def verified(self, atom) -> dict:
        """A slow frame over an atom that is where it should be: the verification frame a claim
        asks for (at least 6 nm, at most 0.2 nm/px, after the last hop), read without a fast
        frame's lag."""
        reg = self.s.w.surface.site.adatoms
        x, y = self._where(*reg.site_xy(atom.i, atom.j, atom.sub))
        f, _ = scan(self.inst, x, y, VERIFY_NM, VERIFY_PX, VERIFY_LINE_S)
        hit = nearest(locate(f, compact_frac=self.compact_frac), x, y, 0.8)
        if hit is None:
            if tip_needs_poke(self.s.w, imaging=True):
                raise LostTrack()
            raise RuntimeError("the atom is not in its verification frame")
        return {"frame": f, "xy": hit}


def ladder(mv: Mover, atom, *, pos, t: float, goal, comp: tuple[float, float],
           resurvey: Callable[[], tuple[float, float]], r_ohm: float, log: dict,
           on_goal_nm: float, nearly_nm: float = 0.6, done: Callable[[], dict] | None = None) -> dict | None:
    """Move ``atom`` (now at ``pos``, measured at ``t``) to ``goal`` (scan coordinates, same
    time) by the resistance ladder: from ``r_ohm`` down by :data:`LADDER_FACTOR` per rung.

    No current spikes: the atom stayed; a slow frame measures it again and the goal moves with
    whatever drift that shows (:class:`Track`). Spikes: ``resurvey()`` gives the vector from
    the atom to the goal, read in one fast frame against fiducials; within ``on_goal_nm`` the
    atom is there and ``done()`` (default: :meth:`Mover.verified`) closes the move; short of it,
    or a site off, the drag resumes from where a slow frame puts the atom (one rung lower
    unless it got within ``nearly_nm``).

    Returns what ``done`` returns, or ``None`` when the atom went onto the tip. Raises
    :class:`LostTrack` when the tip needs conditioning; ``log["r_resume"]`` is then the rung to
    resume from."""
    s, inst, w = mv.s, mv.inst, mv.s.w
    done = done or (lambda: mv.verified(atom))
    attempts = log.setdefault("attempts", [])
    track = Track(inst, comp, goal, t)
    for _ in range(MAX_ATTEMPTS):
        if r_ohm < LADDER_MIN_KOHM * 1e3:
            raise RuntimeError("the atom did not follow down to the lowest resistance")
        log["r_resume"] = r_ohm
        if tip_needs_poke(w, imaging=True):
            raise LostTrack()
        spikes = mv.drag(pos, track.goal, r_ohm)
        attempts.append({"r_kohm": round(r_ohm / 1e3, 1), "spikes": spikes})
        if atom.status != "on_surface":
            return None                                  # it went onto the tip
        log["r_resume"] = r_ohm * LADDER_FACTOR if spikes == 0 else r_ohm
        if tip_needs_poke(w, imaging=True):
            # the manipulation current changed the tip (a second apex doubles every atom):
            # condition it, then find the goal again from the fiducials
            raise LostTrack()
        if spikes == 0:
            new, t_new = mv.measure(atom)
            track.still(pos, t, new, t_new)
            pos, t = new, t_new
            r_ohm *= LADDER_FACTOR
            continue
        # it moved: a survey says how far it is from the goal, against the fiducials
        gx, gy = resurvey()
        off = math.hypot(gx, gy)
        attempts[-1]["off_goal_nm"] = round(off, 3)
        if off <= on_goal_nm:
            log["r_success_kohm"] = r_ohm / 1e3
            return done()
        if off > nearly_nm:
            r_ohm *= LADDER_FACTOR
        pos, t = mv.measure(atom)
        track.goal, track.t = (pos[0] + gx, pos[1] + gy), t
    raise RuntimeError("the atom did not reach its goal")
