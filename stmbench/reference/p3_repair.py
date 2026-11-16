"""P3's repair variant, privileged reference solution: close the gap in a quantum corral by
lateral manipulation, then measure the corral as P3 does.

The truth navigates — where the corral is, which atoms are the spares and where each one is
now, whether it is still on the surface, the drift, the tip's state. The repair is measured:
the ring, its gap and the sites that close it are read from a survey
(:mod:`stmbench.analysis.corral`), each spare is moved by the resistance ladder
(:mod:`stmbench.reference._manip`), and whether a site is filled is read from the ring.

Protocol:

1. a tip unfit to image is conditioned ~150 nm away, then five minutes for the creep;
2. lateral drift compensated at the corral;
3. a 36 nm survey over the corral at 0.1 nm/px (360 nm/s: above 450 nm/s a blunt tip scratches
   the surface). The ring by a vote of its bright pixels (:func:`find_ring`; a circle fitted
   through everything is pulled 1-2 nm off by the spares), the spares as the isolated atoms
   1.5-7 nm outside it — the task says there are as many as atoms missing, and the ring's
   atoms, 0.6-0.9 nm apart, are too close to count one by one — and the missing sites evenly
   in the stretch where the ring's cross-section falls to half (:func:`ring_gap`);
4. spares paired with sites in angular order. Each spare goes to a staging point 1.5 nm
   radially outside its site, where it is still an isolated bump, and then in; before every
   drag a survey reads the atom and its goal in one frame, and the resistance comes down a rung
   whenever the atom did not move (no threshold is read). The site is filled when the ring's
   cross-section there is back to its level;
5. the tip conditioned for spectroscopy if it needs to be, and P3's measurement
   (:func:`stmbench.reference.p3.measure`): the radius, the band bottom, the resonances.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..analysis.adatoms import locate
from ..analysis.corral import find_ring, ring_gap, ring_profile
from . import p3
from ._common import Session, claims_ok, condition_tip, tip_needs_poke
from ._manip import LADDER_FACTOR, LADDER_KOHM, LADDER_MIN_KOHM, LostTrack, Mover, nearest, scan

SCENARIO = "P3_corral_repair_cu111"
SURVEY_NM, SURVEY_PX, SURVEY_LINE_S = 36.0, 360, 0.1
#: spares are the isolated atoms this far outside the ring (nm); the scenario puts them 3-6 nm
#: out and keeps the rest of the deposit 8 nm away
SPARE_BAND_NM = (1.5, 7.0)
#: the staging point, radially outside the site: far enough that the atom is still an isolated
#: bump there (0.7 nm out from it the ring must stay below 0.6 of its height)
STAGE_NM = 1.5
#: an atom looked for where the truth has it, beside the ring: its bump may fall to 0.9 of its
#: height 0.7 nm out on the ring's side (the spares themselves are found with the usual 0.6)
NEAR_COMPACT = 0.9
STAGE_ON_GOAL_NM = 0.35
#: every drag starts this far behind the atom, so that the tip runs over it and carries it
RUN_UP_NM = 0.5
MAX_STEPS = 24


def _unit(theta: float) -> np.ndarray:
    return np.array([math.cos(theta), math.sin(theta)])


def _survey(s: Session, centre) -> dict:
    """A fast frame over the corral (centred on the same sample point every time) and what is in
    it, in scan coordinates: the ring's centre and radius, the isolated atoms, the ring's
    cross-section along it."""
    cx, cy = s.scan_xy(*centre)
    f, t = scan(s.inst, cx, cy, SURVEY_NM, SURVEY_PX, SURVEY_LINE_S)
    npx = float(f["nm_per_px"])
    ring = find_ring(f["array"], npx)
    x0, y0 = cx - SURVEY_NM / 2, cy - SURVEY_NM / 2
    rc = np.array([x0 + ring["centre_nm"][0], y0 + ring["centre_nm"][1]])
    th, prof = ring_profile(f["array"], npx, ring["centre_nm"], ring["radius_nm"])
    return {"frame": f, "t": t, "centre": rc, "R": float(ring["radius_nm"]), "atoms": locate(f),
            "near": locate(f, compact_frac=NEAR_COMPACT), "angles": th, "profile": prof}


def _cross_section(sv: dict, theta: float, half_nm: float = 0.2) -> float:
    """The ring's cross-section at ``theta``, averaged over ``half_nm`` of arc either side."""
    th, prof = sv["angles"], sv["profile"]
    d = np.abs((th - theta + np.pi) % (2 * np.pi) - np.pi) * sv["R"]
    return float(np.mean(prof[d <= half_nm]))


def _filled(sv: dict, theta: float, level: float) -> bool:
    return _cross_section(sv, theta) >= 0.5 * level


def _plan(sv: dict) -> dict:
    """The spares and the sites they go to, from the first survey."""
    rc, R = sv["centre"], sv["R"]
    spares = [a for a in sv["atoms"]
              if SPARE_BAND_NM[0] < math.hypot(a[0] - rc[0], a[1] - rc[1]) - R < SPARE_BAND_NM[1]]
    g = ring_gap(sv["angles"], sv["profile"], R, missing=len(spares))
    return {"spares": spares, "sites": list(g["sites_rad"]), "level": g["level"],
            "gap_width_nm": g["gap_width_nm"], "spacing_nm": g["spacing_nm"]}


def _pair(s: Session, sv: dict, plan: dict) -> list:
    """(atom, site angle) in angular order: the registry atom under each spare seen in the
    survey (navigation), sorted by angle, against the sites sorted by angle."""
    reg = s.w.surface.site.adatoms
    rc = sv["centre"]
    on = [a for a in reg.atoms.values() if a.status == "on_surface"]
    atoms = []
    for x, y, _h in plan["spares"]:
        a = min(on, key=lambda q: math.hypot(s.image_xy(*reg.site_xy(q.i, q.j))[0] - x,
                                               s.image_xy(*reg.site_xy(q.i, q.j))[1] - y))
        atoms.append((math.atan2(y - rc[1], x - rc[0]), a))
    ref = plan["sites"][0] if plan["sites"] else 0.0

    def unwrap(t: float) -> float:
        return (t - ref + np.pi) % (2 * np.pi) - np.pi

    atoms.sort(key=lambda q: unwrap(q[0]))
    sites = sorted(plan["sites"], key=unwrap)
    return [(a, t) for (_, a), t in zip(atoms, sites)]


def _place(s: Session, mv: Mover, centre, atom, theta: float, plan: dict, log: dict, r_ohm: float):
    """One spare to its site, a survey before every drag: the atom and its goal are read in the
    same fast frame and the drag goes from the one to the other, first to the staging point
    1.5 nm outside the site, then in. A rung lower whenever it did not move (no spikes).

    The drag's vector comes from the survey (atom and goal in one frame, so the fast frame's
    lag drops out) and it starts with the tip on the atom; drift between the survey and the
    drag moves atom and goal alike. No drift tracking on the atom, unlike P4: beside the ring the bump
    nearest where an atom should be can be a ring atom, and a still-atom shift read off the
    wrong bump put the drift compensation off by nanometres a minute.

    Returns the resistance that moved it, or ``None`` when it went onto the tip."""
    reg = s.w.surface.site.adatoms
    attempts = log.setdefault("attempts", [])
    staged = False
    for _ in range(MAX_STEPS):
        log["r_resume"] = r_ohm
        if r_ohm < LADDER_MIN_KOHM * 1e3:
            raise RuntimeError("the spare did not follow down to the lowest resistance")
        sv = _survey(s, centre)
        rc, R = sv["centre"], sv["R"]
        # the spare is an isolated bump off the ring: a bump on the ring within reach of where
        # it should be is the gap's edge, and a vector from it sent an atom past its site
        off_ring = [a for a in sv["near"] if abs(math.hypot(a[0] - rc[0], a[1] - rc[1]) - R) > 0.6]
        me = nearest(off_ring, *s.image_xy(*reg.site_xy(atom.i, atom.j, atom.sub)), 1.0)
        if _filled(sv, theta, plan["level"]) and (me is None or math.hypot(*(np.array(me) - rc)) - R < 0.6):
            return r_ohm
        if me is None:
            raise RuntimeError("the spare is not an isolated bump where it should be")
        stage = rc + (R + STAGE_NM) * _unit(theta)
        site = rc + R * _unit(theta)
        out = math.hypot(me[0] - rc[0], me[1] - rc[1]) - R
        if not staged and (math.hypot(me[0] - stage[0], me[1] - stage[1]) <= STAGE_ON_GOAL_NM
                           or out < STAGE_NM):
            staged = True                                    # there, or already past it
        goal = site if staged else stage
        vec = np.array([goal[0] - me[0], goal[1] - me[1]])   # within one frame: no lag
        # the drag runs over the atom (navigation puts the tip on it) from RUN_UP_NM behind it
        # to the goal: started on the atom, a drag of a few tenths of a nanometre hardly moved it
        # (the atom hops only once the apex is 0.6 site spacings away), and one started where
        # the fast frame shows it sat behind the atom by the lag, so the atom went back and forth
        pos = np.array(s.image_xy(*reg.site_xy(atom.i, atom.j, atom.sub)))
        u = vec / max(float(np.hypot(*vec)), 1e-9)
        start = pos - RUN_UP_NM * u
        spikes = mv.drag((float(start[0]), float(start[1])), (float(pos[0] + vec[0]), float(pos[1] + vec[1])), r_ohm)
        attempts.append({"r_kohm": round(r_ohm / 1e3, 1), "spikes": spikes, "in": staged})
        if atom.status != "on_surface":
            return None                                      # it went onto the tip
        if tip_needs_poke(s.w, imaging=True):
            raise LostTrack()
        if spikes == 0:
            r_ohm *= LADDER_FACTOR
    raise RuntimeError("the spare did not reach its site")


def _clean_tip(s: Session, centre, radius_m: float, log: dict, *, spectroscopy: bool = False) -> bool:
    """Condition the tip far from the corral if it needs it; True when it did (the long move
    back creeps, and the drift has to be measured again)."""
    w = s.w
    if tip_needs_poke(w, imaging=True, clean_ldos=spectroscopy) or w.tip.carried:
        log["tip_rounds"] += condition_tip(s, *centre, imaging=True, clean_ldos=spectroscopy or bool(w.tip.carried),
                                           r_min=radius_m + 150e-9, r_max=radius_m + 200e-9, clear=5e-9)
        s.go(*centre)
        s.inst.act("wait", seconds=300.0)
        return True
    return False


def run(seed: int, *, out_dir: str | Path | None = None, scenario: str = SCENARIO, repeats: int = 3,
        integration_ms: float = 20.0) -> dict:
    """One P3-repair episode by the reference protocol; returns what it reported and how it
    scored."""
    s = Session(scenario, seed, out_dir)
    inst, w = s.inst, s.w
    ring = w.surface.corral_snapshot()
    centre = tuple(v * 1e-9 for v in ring["centre_nm"])
    radius_m = float(ring["radius_nm"]) * 1e-9
    st = inst.status()
    mv = Mover(s, imaging_v=st["bias_v"], imaging_pa=st["setpoint_pa"], compact_frac=NEAR_COMPACT,
               apex_aware=True)
    log: dict = {"tip_rounds": 0, "track_lost": 0, "pickups": 0, "placed": 0}
    _clean_tip(s, centre, radius_m, log)
    mv.drift(*centre)
    sv = _survey(s, centre)
    plan = _plan(sv)
    pairs = _pair(s, sv, plan)
    log.update(spares_seen=len(plan["spares"]), sites=len(plan["sites"]),
               gap_width_nm=plan["gap_width_nm"], spacing_nm=plan["spacing_nm"],
               ring_radius_fit_nm=sv["R"])
    r_next = LADDER_KOHM * 1e3
    for atom, theta in pairs:
        r_ohm = r_next
        for _ in range(6):
            if _clean_tip(s, centre, radius_m, log):
                mv.drift(*centre)
            if atom.status != "on_surface":
                break
            try:
                got = _place(s, mv, centre, atom, theta, plan, log, r_ohm)
            except LostTrack:
                log["track_lost"] += 1
                r_ohm = log.pop("r_resume", r_ohm)
                continue
            if got is None:
                log["pickups"] += 1
            else:
                log["placed"] += 1
                r_next = min(LADDER_KOHM * 1e3, 1.6 * got)
            break
    _clean_tip(s, centre, radius_m, log, spectroscopy=True)
    m = p3.measure(s, centre, radius_m, integration_ms=integration_ms, repeats=repeats, compensated=True)
    got = m.pop("values")
    out = inst.finish()
    want, err = p3.truth_errors(inst, got, m.pop("radius_claims"))
    snap = w.truth()["corral"]
    log.pop("r_resume", None)
    return {"paper": "P3-repair", "seed": int(seed), "success": bool(out["success"]),
            "partial": float(out.get("partial", 0.0)),
            "values": got, "truth": want, "abs_err": err, "claims_ok": claims_ok(inst),
            "ring_occupancy": float(snap["ring_occupancy"]), "gap_sites": len(ring["gap_sites"]),
            "radius_nm": float(ring["radius_nm"]), "instrument_s": float(inst.used_s), **log, **m,
            "tip_condition": inst.scenario.draw_hidden(seed).get("tip", {}).get("condition")}
