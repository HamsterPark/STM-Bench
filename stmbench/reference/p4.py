"""P4 privileged reference solution: move one atom by d onto its lattice site, as Eigler and
Schweizer (1990) did.

The truth navigates — which atom is isolated, where it is, whether it is still on the surface,
how far the sample has drifted, what state the tip is in. It does not do the manipulation: the
resistance at which the atom follows is found the way the 1990 paper describes, by lowering it
until the atom comes along (no atom's threshold is read), and the site the atom has to land on
is computed from a lattice the reference measures. The reported position is read from the
frame that verified it.

Protocol:

1. a tip that is blunt, multiple, unstable or carries something is conditioned ~150 nm from
   the atom (pulses spray clusters ~40 nm around); back at the atom, five minutes for the creep
   of that long move;
2. lateral drift from two frames 30 s apart with the compensation off, then compensated;
3. a 28 nm survey around the start and the goal. Every atom in it sits on an fcc hollow, so the
   vectors between them are lattice vectors, and they give the lattice's orientation
   (:func:`stmbench.analysis.adatoms.lattice_angle`, an adatom ruler, which also takes out the
   shear the drift left during the frame); with fewer than eight atoms a 40 nm frame is taken
   for the ruler. The survey's other atoms are also fiducials. Here and below, a frame in which
   the tip changed (a jump of whole rows) is taken again, and a tip no longer fit to image is
   conditioned first;
4. the atom's start from a slow 5 nm frame. The Z loop lags the surface by v·τ along the fast
   axis — 0.15 nm on the survey at 240 nm/s, ~0.01 nm at 20 nm/s — and the Voronoi cell of a
   hollow site is only 0.13 nm across its narrowest. The goal is the start plus the lattice
   vector nearest d;
5. the resistance ladder, at 5 mV (less below 62.5 kΩ, so that the current stays at 80 nA): R
   from 400 kΩ down by a factor 0.8, the preamp's range switched up before the setpoint goes
   above it. At each step the tip goes onto the atom at the imaging setpoint, the setpoint is
   raised, the tip moves to the goal at 0.25 nm/s and the setpoint is restored.

   No current spikes: the atom stayed, and a slow frame measures it again. Whatever it seems
   to have moved is drift the compensation left — the creep of the long move back decays as
   1/(τ + t) and outruns any constant compensation within minutes — so the goal moves by the
   same, and the compensation takes up the residual velocity.

   Spikes: the survey again. Drift, creep and a changed apex shift every atom of one frame
   alike, and the loop's lag is the same for all of them, so against the fiducials the frame
   says how far the atom is from the goal. On the goal (within 0.12 nm), the ladder ends;
   short of it or a site off, the drag resumes from where a slow frame puts the atom, by that
   vector (one step lower unless it was nearly there). A tip that the current changed is
   conditioned again and the ladder resumes from the fiducials; an atom that went onto the
   tip is buried with a poke far away and another isolated atom is taken;
6. a slow 6.4 nm frame over the atom at 0.1 nm/px: the verification frame the claim asks for
   (at least 6 nm, at most 0.2 nm/px, after the last hop); the reported place is the atom's
   in it.

On seeds 0–149 (the draws of 2026-10-13): 150/150 reproduced, 147 on the nearest site and 3 on
a site tied with it (``site_tie_nm``); reported place |error| median 0.04 nm, p90 0.16 nm,
max 0.41 nm (the judge allows 0.6; most of it is the apex's own offset, which images and drags
share); the atom followed at 54–400 kΩ (median 164 kΩ) after a median of five drags;
instrument time median 879 s, p90 1428 s, max 2003 s. The tip was conditioned first on 100
seeds and again mid-ladder on 32. No atom went onto the tip on these seeds; the recovery is
tested on one made to (``tests/test_reference_p4.py``).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..analysis.adatoms import lattice_angle, lattice_vector, locate, row_jumps
from ._common import Session, claims_ok, condition_tip, lateral_drift, tip_needs_poke

SCENARIO = "P4_atom_positioning_cu111"
IMAGING_V, IMAGING_PA = 0.05, 200.0
#: manipulation at 5 mV; below 62.5 kΩ the bias comes down instead of the current going above
#: 80 nA (the setpoint's range ends at 100 nA, and every nA raises the tip's change rate)
MANIP_V, MANIP_MAX_NA = 0.005, 80.0
LADDER_KOHM, LADDER_FACTOR, LADDER_MIN_KOHM = 400.0, 0.8, 25.0
DRAG_NM_S = 0.25
SURVEY_NM, SURVEY_PX, SURVEY_LINE_S = 28.0, 280, 0.1
#: the ruler wants eight atoms (:func:`stmbench.analysis.adatoms.lattice_angle`); short of
#: that, a 40 nm frame (twice the area, so about twice the atoms) is taken for it, at 400 nm/s:
#: above 450 nm/s a tip of more than 1.5 nm scratches the surface (World.maybe_scratch)
RULER_MIN_ATOMS, RULER_NM, RULER_PX = 8, 40.0, 400
#: slow frames: 20 nm/s, where the loop's lag is ~0.01 nm
SLOW_NM, SLOW_PX, SLOW_LINE_S = 5.0, 50, 0.25
VERIFY_NM, VERIFY_PX, VERIFY_LINE_S = 6.4, 64, 0.32
#: the atom is taken to sit on the goal within this (nm): the hollow sites are 0.255 nm apart,
#: and one survey places the atom against its fiducials to ~0.03 nm
ON_GOAL_NM = 0.12
#: closer than this to the goal after a drag, it followed at this resistance: nudge at the same
NEARLY_NM = 0.6
MAX_ATTEMPTS = 30


class LostTrack(Exception):
    """The tip has to be conditioned in the middle of a manipulation."""


def _scan(inst, cx: float, cy: float, size: float, px: int, line_s: float) -> tuple[dict, float]:
    """A frame and the instrument time at its middle row. Raises :class:`LostTrack` when the
    tip is not fit to image by the end of it: a second apex puts a ghost beside every atom, and
    places read from such a frame (fiducials, a ruler, a start) are wrong. A frame in which
    the tip changed (:func:`stmbench.analysis.adatoms.row_jumps`) is taken again: its rows
    before and after the change disagree by the apex's sideways jump."""
    for _ in range(3):
        r = inst.act("scan", center_x_nm=cx, center_y_nm=cy, size_nm=size, pixels=px, line_time_s=line_s)
        if tip_needs_poke(inst.world, imaging=True):
            raise LostTrack()
        f = inst.frame(r["frame"])
        if not row_jumps(f["array"]):
            break
    return f, float(f["t_start_s"]) + 0.5 * float(r["duration_s"])


def _nearest(atoms, x: float, y: float, within: float):
    best = min(atoms, key=lambda a: math.hypot(a[0] - x, a[1] - y), default=None)
    if best is None or math.hypot(best[0] - x, best[1] - y) > within:
        return None
    return float(best[0]), float(best[1])


def _measure(s: Session, atom):
    """The atom from a slow frame centred where the truth has it: (place, time). Raises
    :class:`LostTrack` when it cannot be seen because the tip went bad."""
    reg = s.w.surface.site.adatoms
    x, y = s.scan_xy(*reg.site_xy(atom.i, atom.j, atom.sub))
    f, t = _scan(s.inst, x, y, SLOW_NM, SLOW_PX, SLOW_LINE_S)
    hit = _nearest(locate(f), x, y, 0.8)
    if hit is None:
        if tip_needs_poke(s.w, imaging=True):
            raise LostTrack()
        raise RuntimeError("the atom is not in the frame centred on it")
    return hit, t


def _manipulate(inst, r_ohm: float) -> None:
    """Manipulation conditions: the preamp's range first, then the bias, then the setpoint."""
    v = min(MANIP_V, r_ohm * MANIP_MAX_NA * 1e-9)
    inst.act("set_current_range", full_scale_na=100.0)
    inst.act("set_bias", bias_v=v)
    inst.act("set_setpoint", current_pa=v / r_ohm * 1e12)


def _image(inst) -> None:
    """Back to imaging, in the reverse order: the setpoint comes down before the range does."""
    inst.act("set_setpoint", current_pa=IMAGING_PA)
    inst.act("set_bias", bias_v=IMAGING_V)
    inst.act("set_current_range", full_scale_na=10.0)


def _drag(inst, frm, to, r_ohm: float) -> int:
    inst.act("move_tip", x_nm=frm[0], y_nm=frm[1])
    _manipulate(inst, r_ohm)
    r = inst.act("move_tip", x_nm=to[0], y_nm=to[1], speed_nm_s=DRAG_NM_S)
    _image(inst)
    return int(r["current_spikes"])


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


def _survey(s: Session, atom, d_nm) -> dict:
    """The fast survey around start and goal, centred on the same sample point every time: the
    atoms in it and which of them is ``atom`` (one frame, so its uniform lag drops out of every
    difference)."""
    reg = s.w.surface.site.adatoms
    x0, y0 = atom.x0, atom.y0
    cx, cy = s.scan_xy(x0 + 0.5 * d_nm[0] * 1e-9, y0 + 0.5 * d_nm[1] * 1e-9)
    f, t = _scan(s.inst, cx, cy, SURVEY_NM, SURVEY_PX, SURVEY_LINE_S)
    found = locate(f)
    ex, ey = s.scan_xy(*reg.site_xy(atom.i, atom.j, atom.sub))
    me = _nearest(found, ex, ey, 1.0)
    if me is None:
        if tip_needs_poke(s.w, imaging=True):
            raise LostTrack()
        raise RuntimeError("the atom is not in the survey")
    others = [(a[0], a[1]) for a in found if math.hypot(a[0] - me[0], a[1] - me[1]) > 1.0]
    return {"frame": f, "centre": (cx, cy), "t": t, "atoms": found, "me": me, "others": others}


def _register(ref, now, *, within: float = 0.3, max_shift: float = 4.0) -> tuple[float, float]:
    """The shift that carries the fiducials of the first survey onto the atoms of a new one:
    of the shifts one fiducial-atom pair proposes, the one that most fiducials agree with
    (to ``within``), refined by their median. A conditioned tip images everything shifted by
    its new apex, by more than any fixed matching radius would allow."""
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


def _resurvey(s: Session, atom, d_nm, ref: dict):
    """A new survey, and in it the vector from the atom to the goal. The goal comes from the
    fiducials (their shift since the first survey takes up drift, creep and a changed apex
    alike), and both ends are read in the same fast frame, so the loop's lag, the same for
    every atom in it, drops out. (An absolute place from a fast frame would need the lag, and
    a lag measured against a slow frame a minute later carries that minute's drift.)"""
    now = _survey(s, atom, d_nm)
    sx, sy = _register(ref["others"], now["atoms"])
    goal = (ref["me"][0] + ref["L"][0] + sx, ref["me"][1] + ref["L"][1] + sy)
    return now, (goal[0] - now["me"][0], goal[1] - now["me"][1])


def _verified(s: Session, atom, log: dict, r_ohm: float) -> dict:
    """The atom is on its goal: a slow frame over it, the verification frame the claim asks for
    (at least 6 nm, at most 0.2 nm/px, after the last hop), read without a fast frame's lag."""
    log["r_success_kohm"] = r_ohm / 1e3
    x, y = s.scan_xy(*s.w.surface.site.adatoms.site_xy(atom.i, atom.j, atom.sub))
    f, _ = _scan(s.inst, x, y, VERIFY_NM, VERIFY_PX, VERIFY_LINE_S)
    hit = _nearest(locate(f), x, y, 0.8)
    if hit is None:
        if tip_needs_poke(s.w, imaging=True):
            raise LostTrack()
        raise RuntimeError("the atom is not in its verification frame")
    return {"frame": f, "xy": hit}


def _ladder(s: Session, atom, d_nm, log: dict, r_ohm: float, survey: dict | None):
    """Survey (unless resuming), start, then the ladder. Returns the verification, ``None``
    when the atom went onto the tip; raises :class:`LostTrack` when the tip needs conditioning."""
    inst, w = s.inst, s.w
    reg = w.surface.site.adatoms
    x0, y0 = reg.site_xy(atom.i, atom.j, atom.sub)
    # measured with the compensation off: with it on, two frames give only what it leaves
    inst.act("drift_comp", on=False)
    vx, vy = lateral_drift(s, x0, y0)
    inst.act("drift_comp", on=True, vx_nm_per_min=vx, vy_nm_per_min=vy)
    attempts = log.setdefault("attempts", [])
    log["resume"] = (survey, r_ohm)
    if survey is None:
        survey = _survey(s, atom, d_nm)
        ruler = survey["atoms"]
        if len(ruler) < RULER_MIN_ATOMS:
            # too few atoms to be sure of the orientation: a wider frame, for the ruler only
            big, _ = _scan(inst, *survey["centre"], RULER_NM, RULER_PX, SURVEY_LINE_S)
            ruler = max(ruler, locate(big), key=len)
        phi, rms, nvec = lattice_angle(ruler)
        survey["L"] = lattice_vector(d_nm, phi)
        log.update(lattice_deg=math.degrees(phi), lattice_rms_nm=rms, lattice_vectors=nvec,
                   ruler_atoms=len(ruler), survey_atoms=len(survey["atoms"]))
        pos, t = _measure(s, atom)
        goal = (pos[0] + survey["L"][0], pos[1] + survey["L"][1])
        log["resume"] = (survey, r_ohm)
    else:
        # back after a conditioning: the drag before it may already have put the atom there
        _, (gx, gy) = _resurvey(s, atom, d_nm, survey)
        if math.hypot(gx, gy) <= ON_GOAL_NM:
            return _verified(s, atom, log, r_ohm)
        pos, t = _measure(s, atom)
        goal = (pos[0] + gx, pos[1] + gy)
    track = Track(inst, (vx, vy), goal, t)
    for _ in range(MAX_ATTEMPTS):
        if r_ohm < LADDER_MIN_KOHM * 1e3:
            raise RuntimeError("the atom did not follow down to the lowest resistance")
        log["resume"] = (survey, r_ohm)
        if tip_needs_poke(w, imaging=True):
            raise LostTrack()
        spikes = _drag(inst, pos, track.goal, r_ohm)
        attempts.append({"r_kohm": round(r_ohm / 1e3, 1), "spikes": spikes})
        if atom.status != "on_surface":
            return None                                  # it went onto the tip
        log["resume"] = (survey, r_ohm * LADDER_FACTOR if spikes == 0 else r_ohm)
        if tip_needs_poke(w, imaging=True):
            # the manipulation current changed the tip (a second apex doubles every atom):
            # condition it, then find the goal again from the fiducials
            raise LostTrack()
        if spikes == 0:
            new, t_new = _measure(s, atom)
            track.still(pos, t, new, t_new)
            pos, t = new, t_new
            r_ohm *= LADDER_FACTOR
            continue
        # it moved: a survey says how far it is from the goal, against the fiducials
        _, (gx, gy) = _resurvey(s, atom, d_nm, survey)
        off = math.hypot(gx, gy)
        attempts[-1]["off_goal_nm"] = round(off, 3)
        if off <= ON_GOAL_NM:
            return _verified(s, atom, log, r_ohm)
        if off > NEARLY_NM:
            r_ohm *= LADDER_FACTOR
        pos, t = _measure(s, atom)
        track.goal, track.t = (pos[0] + gx, pos[1] + gy), t
    raise RuntimeError("the atom did not reach its goal")


def _clean_tip(s: Session, x0: float, y0: float, log: dict) -> None:
    if tip_needs_poke(s.w, imaging=True) or s.w.tip.carried:
        # an atom on the apex reads as an adsorbate (clean_ldos): a shallow poke buries it
        log["tip_rounds"] += condition_tip(s, x0, y0, imaging=True, clean_ldos=bool(s.w.tip.carried),
                                           r_min=150e-9, r_max=200e-9, clear=5e-9)
        s.go(x0, y0)
        s.inst.act("wait", seconds=300.0)


def _candidates(w, d_nm, *, avoid: list[tuple[float, float]]) -> list:
    """Isolated atoms that can be moved by d without touching another one: none within 3 nm of
    the start, none within 2 nm of the path or 3 nm of the goal, no step within 1.5 nm of the
    path, and ``avoid`` (the starts of atoms already lost) more than 11 nm from start and goal."""
    reg = w.surface.site.adatoms
    births = np.array([(a.x0, a.y0) for a in reg.atoms.values()], float)
    now = reg.positions()
    dx, dy = d_nm[0] * 1e-9, d_nm[1] * 1e-9
    out = []
    for a in reg.atoms.values():
        if a.status != "on_surface" or a.history:
            continue
        x, y = reg.site_xy(a.i, a.j, a.sub)
        gx, gy = x + dx, y + dy
        if np.count_nonzero(np.hypot(births[:, 0] - x, births[:, 1] - y) <= 3e-9) > 1:
            continue
        t = np.clip(((now[:, 0] - x) * dx + (now[:, 1] - y) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
        perp = np.hypot(now[:, 0] - (x + t * dx), now[:, 1] - (y + t * dy))
        if np.count_nonzero(perp <= 2e-9) > 1 or np.count_nonzero(np.hypot(now[:, 0] - gx, now[:, 1] - gy) <= 3e-9):
            continue
        if not all(w.surface.step_free_window(x + f * dx, y + f * dy, 3e-9) for f in (0.0, 0.5, 1.0)):
            continue
        if any(min(math.hypot(px - x, py - y), math.hypot(px - gx, py - gy)) <= 11e-9 for px, py in avoid):
            continue
        out.append(a)
    return out


def run(seed: int, *, out_dir: str | Path | None = None, scenario: str = SCENARIO) -> dict:
    """One P4 episode by the reference protocol; returns what it reported and how it scored."""
    s = Session(scenario, seed, out_dir)
    inst, w = s.inst, s.w
    reg = w.surface.site.adatoms
    d_nm = tuple(float(v) for v in reg.target["d_nm"])
    atom = reg.atoms[reg.target["atom_id"]]
    log: dict = {"atoms_tried": 1, "pickups": 0, "tip_rounds": 0, "track_lost": 0}
    lost: list[tuple[float, float]] = []
    got, survey, r_ohm = None, None, LADDER_KOHM * 1e3
    for _ in range(8):
        x0, y0 = reg.site_xy(atom.i, atom.j, atom.sub)
        _clean_tip(s, x0, y0, log)
        try:
            got = _ladder(s, atom, d_nm, log, r_ohm, survey)
        except LostTrack:
            log["track_lost"] += 1
            survey, r_ohm = log.pop("resume")
            continue
        if got is not None:
            break
        log["pickups"] += 1
        lost.append((atom.x0, atom.y0))
        nxt = _candidates(w, d_nm, avoid=lost)
        if not nxt or log["atoms_tried"] >= 4:
            break
        # the nearest, preferring one with three other atoms within the survey's reach: they
        # are its fiducials and the ruler's first atoms
        now = reg.positions()
        crowd = {a.id: int(np.count_nonzero(np.hypot(now[:, 0] - a.x0, now[:, 1] - a.y0) <= 14e-9)) - 1
                 for a in nxt}
        atom = min(nxt, key=lambda a: (crowd[a.id] < 3, math.hypot(a.x0 - x0, a.y0 - y0)))
        log["atoms_tried"] += 1
        survey, r_ohm = None, LADDER_KOHM * 1e3
    log.pop("resume", None)
    if got is not None:
        x, y = got["xy"]
        inst.report("target_site", x, unit="nm", x_nm=x, y_nm=y, frame=got["frame"]["frame"],
                    evidence=[got["frame"]["frame"]])
    out = inst.finish()
    snap = w.truth()["adatoms"]["target"]
    cand = next((c for c in snap["candidates"] if c["id"] == atom.id), None)
    det = (inst.verdict.details or {}).get("claims", {})
    placed = det.get("target_site", {}).get("sample_xy_nm")
    now = (cand or {}).get("now_nm")
    err = float(math.hypot(placed[0] - now[0], placed[1] - now[1])) if placed and now else float("nan")
    return {"paper": "P4", "seed": int(seed), "success": bool(out["success"]),
            "partial": float(out.get("partial", 0.0)),
            "claims_ok": claims_ok(inst), "atom_id": int(atom.id),
            "on_goal": bool(cand and cand["on_goal"]),
            "bystander_max_shift_nm": float(cand["bystander_max_shift_nm"]) if cand else float("nan"),
            # how far the reported place is from the atom (nm), against report_nm
            "abs_err": {"target_site": err},
            "instrument_s": float(inst.used_s), **log,
            "tip_condition": inst.scenario.draw_hidden(seed).get("tip", {}).get("condition")}
