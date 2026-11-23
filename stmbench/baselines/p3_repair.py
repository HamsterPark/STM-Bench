"""P3-repair scripted baseline: close the gap in a quantum corral and measure it, with nothing but
what the instrument shows — no truth.

What the privileged reference (:mod:`stmbench.reference.p3_repair`) takes from the truth, this
reads from observations, through :class:`stmbench.baselines._common.Blind` only:

* **where the corral is**: a 60 nm survey around the origin, where the task promises it; the
  bright pixels vote for the circle (:func:`stmbench.analysis.corral.find_ring`);
* **the spares and how many sites are open**: the compact bumps 1.5-7 nm outside the ring in a
  36 nm frame centred on it. The task says there are as many spares as atoms missing, so the
  gap (where the ring's cross-section falls to half, :func:`ring_gap`) is divided among that
  many sites;
* **the tip**: three shallow pokes far away first (an unstable or dirty tip does not show in
  one frame), then the survey judges it (:func:`judge_tip`: the ring's cross-section and its
  ghost, the spares seen); a blunt or double tip is pulsed far from the corral and judged
  again. Row jumps in later frames, or no spare in sight, send it back there;
* **where an atom is, and where its site is**: both in one fast frame, so the loop's lag drops
  out of the vector from one to the other. A slow frame centred on the atom gives the place
  the tip has to be put at, and the offset between the two places for the same atom is the lag
  that the ring in the fast frame carries too. The drift is measured on atoms and compensated
  (:meth:`stmbench.baselines._common.Ops.drift`);
* **whether an atom moved**: the trace the drag recorded (:func:`stmbench.analysis.trace.count_hops`);
  **whether a site is filled**: the ring's cross-section there in a frame after the last drag.

The manipulation is the reference's: the resistance ladder from 400 kOhm down by 0.8 at 5 mV,
drags at 0.25 nm/s, each spare first to a staging point 1.5 nm outside its site, then in. The
measurement afterwards is the reference's protocol (radius from a frame, band bottom on bare
terrace, three spectra at the centre, resonances labelled with the box levels).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..analysis.adatoms import locate
from ..analysis.corral import find_ring, ghost_strength, label_levels, ring_gap, ring_profile, ring_width_nm
from ..analysis.trace import count_hops
from ..instrument import Instrument
from ..reference._manip import DRAG_NM_S, LADDER_FACTOR, LADDER_KOHM, LADDER_MIN_KOHM, MANIP_MAX_NA, MANIP_V
from ..reference.p3 import _peak_values
from ._common import Blind
from .p3 import Corral  # the measurement part (radius, band bottom, spectra, labels) is the P3 baseline's

SCENARIO = "P3_corral_repair_cu111"
#: the survey that finds the ring: 60 nm at 0.1 nm/px (180 s)
FIND_NM, FIND_PX, FIND_LINE_S = 60.0, 600, 0.15
#: the survey around the ring: 36 nm at 0.1 nm/px, 360 nm/s (36 s)
SURVEY_NM, SURVEY_PX, SURVEY_LINE_S = 36.0, 360, 0.1
#: the slow frame that puts the tip on an atom: 6 nm at 0.1 nm/px, 20 nm/s
SLOW_NM, SLOW_PX, SLOW_LINE_S = 6.0, 60, 0.3
SPARE_BAND_NM = (1.0, 7.0)
STAGE_NM = 1.5
#: the gap's edges are looked for in a cross-section smoothed over this half-width (nm)
GAP_SMOOTH_NM = 0.4
#: the last approach is repeated until the atom is within this of its site (nm), at most MAX_FINE drags
FINE_NM, MAX_FINE = 0.2, 4
#: the slow frame that reads the atom beside the ring's: 8 nm at 0.1 nm/px
FINE_FRAME_NM, FINE_FRAME_PX, FINE_FRAME_LINE_S = 8.0, 80, 0.4
MAX_STEPS = 40
PARTIAL_MAX = 2
POKE_AWAY_NM, PULSE_AWAY_NM = 130.0, 170.0
MAX_ROUNDS = 20
SWEEP_V = (-0.55, 0.15)
STAB_V, STAB_PA = 0.1, 1000.0
POINTS = 351
#: what the report and the measurement still need (s)
RESERVE_S = 1600.0


class Lost(Exception):
    """The spare cannot be followed: taken by the tip, or not found."""


class BadTip(Exception):
    """The tip does not image well enough to go on."""


def _unit(theta: float) -> np.ndarray:
    return np.array([math.cos(theta), math.sin(theta)])


def _to_scan(frame: dict, c) -> tuple[float, float]:
    """Scan coordinates of a point given as (x from the frame's left edge, y up from its
    bottom edge), nm."""
    size = float(frame["size_nm"])
    return float(frame["center_nm"][0]) - size / 2 + float(c[0]), float(frame["center_nm"][1]) - size / 2 + float(c[1])


def ring_of(frame: dict, *, search_nm: float = 3.0) -> dict:
    """The ring of a frame centred on it within ``search_nm``: ``{centre (scan nm), radius_nm,
    centre_frame_nm}``."""
    r = find_ring(frame["array"], float(frame["nm_per_px"]), search_nm=search_nm)
    return {"centre_frame_nm": list(r["centre_nm"]), "centre": _to_scan(frame, r["centre_nm"]),
            "radius_nm": float(r["radius_nm"])}


def find_corral(frame: dict) -> dict:
    """The ring in the big survey, wherever in it: the vote over a binned copy (0.3 nm), refined on
    the full frame."""
    z = np.asarray(frame["array"], float)
    npx = float(frame["nm_per_px"])
    b = max(1, int(round(0.3 / npx)))
    ny, nx = z.shape[0] // b * b, z.shape[1] // b * b
    zb = np.where(np.isfinite(z), z, np.nanmedian(z))[:ny, :nx].reshape(ny // b, b, nx // b, b).mean(axis=(1, 3))
    size = float(frame["size_nm"])
    coarse = find_ring(zb, npx * b, search_nm=size / 2 - 11.0, radius_range_nm=(3.5, 10.5))
    near = {"array": zb, "nm_per_px": npx * b}
    cx, cy = coarse["centre_nm"]
    return {"centre": _to_scan(frame, (cx, cy)), "radius_nm": float(coarse["radius_nm"]), "pixels": coarse["pixels"],
            "_": near}


def spares_of(frame: dict, centre, radius: float) -> list[tuple[float, float, float]]:
    """The compact bumps 1.5-7 nm beyond the ring, as high as the typical one (scan nm)."""
    atoms = [a for a in locate(frame)
             if SPARE_BAND_NM[0] < math.hypot(a[0] - centre[0], a[1] - centre[1]) - radius < SPARE_BAND_NM[1]]
    if len(atoms) >= 3:
        typical = float(np.median([a[2] for a in atoms]))
        atoms = [a for a in atoms if 0.5 * typical <= a[2] <= 2.0 * typical]
    return atoms


#: ring cross-section above this (nm) is a blunt tip: sharp tips 0.75-1.05, blunt 1.15-1.25
BLUNT_FWHM_NM = 1.12
#: ghost-ring autocorrelation at or above this is a double tip (good tips at most 0.07)
GHOST_MIN = 0.085
#: and the same once the repair has begun
GHOST_MIN_REPAIR = 0.17
#: more spares than this in sight: a double tip draws each of them twice
MAX_SPARES = 6


def judge_tip(frame: dict, ring: dict, spares, expected: int | None = None, *, ghost_min: float = GHOST_MIN) -> dict:
    """The tip as a survey of the corral shows it. A blunt tip widens the ring's cross-section and
    smears the spares below detection; a double one draws the ring twice (the autocorrelation of
    the ring has a second maximum) and every spare twice. ``expected``: how many spares are still
    to be seen (the survey's count at the start, less those placed); ``None`` before it is known.
    Once atoms have been moved the ring is no longer the one the thresholds were set on: a ring
    with a gap and a few new atoms reads 0.085-0.11 in the autocorrelation with a good tip, so
    ``ghost_min`` is raised for the surveys of the repair."""
    npx, c, r = float(frame["nm_per_px"]), ring["centre_frame_nm"], ring["radius_nm"]
    width = ring_width_nm(frame["array"], npx, c, r)
    ghost, _ = ghost_strength(frame["array"], npx, c, r)
    # the spares say something about the tip only before the repair has begun, when their count is
    # the thing to be found out; afterwards a spare left outside or one moved says nothing of it
    many = expected is None and len(spares) > MAX_SPARES
    smeared = expected is None and not spares
    verdict = "double" if ghost >= ghost_min or many else "blunt" if width > BLUNT_FWHM_NM or smeared else "ok"
    return {"verdict": verdict, "width_nm": round(float(width), 2), "ghost": round(float(ghost), 3), "spares": len(spares)}


def cross_section(angles, profile, radius: float, theta: float, half_nm: float = 0.2) -> float:
    d = np.abs((angles - theta + np.pi) % (2 * np.pi) - np.pi) * radius
    return float(np.mean(profile[d <= half_nm]))


class Session:
    """One episode at the instrument."""

    def __init__(self, v: Blind, log: dict, ops):
        self.v, self.log = v, log
        self.ops = ops
        self.centre = (0.0, 0.0)
        self.R = 0.0
        self.comp: tuple[float, float] | None = None
        self.expected: int | None = None      # spares still to be seen, once known
        self.ghost_min = GHOST_MIN

    # ── frames ──
    def survey(self):
        """The 36 nm frame centred on the ring: (frame, ring, spares, profile, diagnosis of the tip).
        The ring is read again each time and the centre follows it."""
        cx, cy = self.centre
        f, _ = self.ops.frame(cx, cy, SURVEY_NM, SURVEY_PX, SURVEY_LINE_S)
        try:
            r = ring_of(f)
        except ValueError as exc:
            raise BadTip(f"no ring in the survey ({exc})") from exc
        if math.hypot(r["centre"][0] - cx, r["centre"][1] - cy) < 8.0:
            self.centre = r["centre"]
        sp = spares_of(f, r["centre"], r["radius_nm"])
        th, prof = ring_profile(f["array"], float(f["nm_per_px"]), r["centre_frame_nm"], r["radius_nm"])
        d = judge_tip(f, r, sp, self.expected, ghost_min=self.ghost_min)
        if d["verdict"] != "ok":
            self.log.setdefault("not_ok", []).append({"t": round(self.ops.now()), **d})
        return f, r, sp, (th, prof), d

    # ── the tip ──
    def poke(self, n: int = 3) -> None:
        self.ops.poke_near(n, away_nm=POKE_AWAY_NM)
        self.v.act("wait", seconds=120.0)

    def condition(self) -> None:
        """Pulse and poke far from the corral until the spares image well; the creep of the long
        moves is waited out and the drift measured again by the caller."""
        cx, cy = self.centre
        acts = self.log.setdefault("tip_actions", [])
        ghosts: list[float] = []
        for k in range(1, MAX_ROUNDS + 1):
            x0, y0 = cx + 8.0 * k, cy + PULSE_AWAY_NM
            self.ops.go(x0, y0)
            self.v.act("pulse", bias_v=4.0, width_ms=50.0)
            for j in (1, 2):
                self.ops.go(x0 + 2.0 * j, y0)
                self.v.act("poke", depth_nm=0.4, dwell_s=0.5)
            self.ops.go(cx, cy)
            self.ops.jumps.clear()
            try:
                _, _, sp, _, d = self.survey()
            except BadTip:
                acts.append("no_ring")
                continue
            acts.append(d["verdict"])
            ghosts.append(d["ghost"])
            # a ghost that three pulses leave where it was (within 0.03) belongs to the sample (seed 26: a
            # good tip, a ring with 0.21 at every round), not to the tip; the width and the spares still count
            steady = (d["verdict"] == "double" and len(ghosts) >= 3 and max(ghosts[-3:]) - min(ghosts[-3:]) < 0.03
                      and d["width_nm"] <= BLUNT_FWHM_NM and d["spares"] <= MAX_SPARES)
            if steady:
                self.log["ghost_steady"] = ghosts[-1]
                self.ghost_min = max(self.ghost_min, ghosts[-1] + 0.05)
            if d["verdict"] == "ok" or steady:
                self.log["tip_rounds"] = self.log.get("tip_rounds", 0) + k
                self.v.act("wait", seconds=300.0)
                return
        raise RuntimeError("the tip could not be conditioned")

    def fit_tip(self) -> None:
        """Judge the tip on a survey and condition it when it is not fit."""
        _, _, sp, _, d = self.survey()
        self.log.setdefault("judged", []).append(d["verdict"])
        if d["verdict"] != "ok" or self.ops.unstable():
            if d["verdict"] == "ok" and self.ops.unstable():
                self.poke()
                self.ops.jumps.clear()
            else:
                self.condition()

    # ── manipulation ──
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
        self.v.act("move_tip", x_nm=float(frm[0]), y_nm=float(frm[1]))
        self.manipulate(r_ohm)
        r = self.v.act("move_tip", x_nm=float(to[0]), y_nm=float(to[1]), speed_nm_s=DRAG_NM_S)
        self.image()
        return count_hops(self.v.spectrum(r["trace"]))

    def slow(self, guess):
        """The bump nearest ``guess`` in a slow frame centred on it, or ``None``."""
        f, _ = self.ops.frame(guess[0], guess[1], SLOW_NM, SLOW_PX, SLOW_LINE_S)
        atoms = locate(f)
        best = min(atoms, key=lambda a: math.hypot(a[0] - guess[0], a[1] - guess[1]), default=None)
        if best is None or math.hypot(best[0] - guess[0], best[1] - guess[1]) > 1.2:
            return None
        return np.array(best[:2], float)

    def fine_frame(self, at):
        f, _ = self.ops.frame(at[0], at[1], FINE_FRAME_NM, FINE_FRAME_PX, FINE_FRAME_LINE_S)
        return [np.array(a[:2], float) for a in locate(f, compact_frac=0.95)]

    def finish(self, aF, site, r_ohm: float, others):
        """The last approach: the spare, at the staging point, to its site, closed in a loop on
        the places a slow frame gives. One frame before the drag shows the ring's atoms beside
        the site, which the frame after the drag is read against: the spare is the bump that is new
        (the ring's atoms, 0.6-1.4 nm apart, show as separate bumps at 0.1 nm/px with a good tip).
        The target is the site, found in the fast frame (``aF`` and ``site`` are in it) and carried
        to the slow frame's places by the spare's own offset between the two."""
        attempts = self.log.setdefault("attempts", [])
        mid = 0.5 * (aF + site)
        pre = self.fine_frame(mid)
        me = min(pre, key=lambda p: float(np.hypot(*(p - aF))), default=None)
        if me is None or float(np.hypot(*(me - aF))) > 1.5:
            raise Lost("the spare is not in the slow frame beside its site")
        target = me + (site - aF)
        ring_b = [p for p in pre if float(np.hypot(*(p - me))) > 0.5]
        shift = np.zeros(2)
        prev_off = None
        for k in range(MAX_FINE):
            if self.ops.now() > self.ops.budget_s - RESERVE_S:
                raise RuntimeError("out of time for the repair")
            if r_ohm < LADDER_MIN_KOHM * 1e3:
                raise Lost("the spare did not follow down to the lowest resistance")
            vec = target + shift - me
            u = vec / max(float(np.hypot(*vec)), 1e-9)
            hops = self.drag(me, target + shift, r_ohm)
            post = self.fine_frame(target + shift)
            if ring_b and len(post) >= 2:
                try:
                    from ._common import find_shift
                    sx, sy, n = find_shift([tuple(p) for p in ring_b], [tuple(p) for p in post], within=0.35, max_shift=1.5)
                    if n >= 2:
                        shift = np.array([sx, sy])
                except RuntimeError:
                    pass
            new = [p for p in post if all(float(np.hypot(*(p - (q + shift)))) > 0.35 for q in ring_b)]
            tgt = target + shift
            atom = min(new, key=lambda p: float(np.hypot(*(p - tgt))), default=None)
            off = None if atom is None else float(np.hypot(*(atom - tgt)))
            attempts.append({"r_kohm": round(r_ohm / 1e3, 1), "hops": hops, "in": True, "fine": k,
                             "off_nm": None if off is None else round(off, 2)})
            if hops == 0 and (atom is None or off > FINE_NM):
                r_ohm *= LADDER_FACTOR
                if atom is not None:
                    me = atom
                continue
            if atom is None:
                return r_ohm, tgt                  # merged into the ring's bumps: held, as far as it shows
            if off > 1.6:
                atom, off = None, None             # a new bump that is far from where the spare went: not it
                continue
            if off <= FINE_NM:
                return r_ohm, tgt
            if prev_off is not None and off > prev_off - 0.2:       # hopped, but not nearer: a rung down
                r_ohm *= LADDER_FACTOR
            prev_off = off
            me = atom
        return r_ohm, target + shift                         # as near as it came: the ring shows whether it holds

    def place(self, guess, theta: float, r_ohm: float, others):
        """One spare from ``guess`` to the site at angle ``theta``: ``(resistance that moved it,
        where it stands)``. ``others``: where the other spares are (they are not this one).
        Raises :class:`Lost`, :class:`BadTip`.

        Each round reads one fast frame, in which the spare, the ring and the site are all
        found (so the loop's lag drops out of the vector from the spare to the site), and a slow
        frame centred on the spare for where the tip has to be put: the offset between the two
        places of the same atom is the lag the ring carries as well. The spare is the bump
        nearest the path the last drag gave it that is no other spare and no part of the ring."""
        attempts = self.log.setdefault("attempts", [])
        staged, partial = False, 0
        last, expect = np.array(guess, float), np.array(guess, float)
        prev_left = None
        again = None                      # (aS, vec) of a drag that moved nothing: repeated one rung down, no new survey
        for _ in range(MAX_STEPS):
            if self.ops.now() > self.ops.budget_s - RESERVE_S:
                raise RuntimeError("out of time for the repair")
            if r_ohm < LADDER_MIN_KOHM * 1e3:
                raise Lost("the spare did not follow down to the lowest resistance")
            if again is not None:
                aS, vec = again
                u = vec / max(float(np.hypot(*vec)), 1e-9)
                hops = self.drag(aS, aS + vec, r_ohm)
                attempts.append({"r_kohm": round(r_ohm / 1e3, 1), "hops": hops, "in": False, "again": True})
                if hops == 0:
                    r_ohm *= LADDER_FACTOR
                else:
                    again = None
                    last, expect = aS, aS + vec
                continue
            f, r, sp, _, d = self.survey()
            if d["verdict"] != "ok" or self.ops.unstable():
                raise BadTip(d["verdict"])
            c, R = np.array(r["centre"]), r["radius_nm"]
            seg = expect - last
            L2 = float(seg @ seg) or 1e-9
            best = None
            for a in locate(f, compact_frac=0.9):
                p = np.array(a[:2], float)
                if abs(float(np.hypot(*(p - c))) - R) < 0.8 and float(np.hypot(*(p - c))) < R + 1.0:
                    continue                                        # a bump of the ring
                if any(float(np.hypot(*(p - o))) < 0.9 for o in others):
                    continue                                        # another spare
                tt = min(max(float((p - last) @ seg) / L2, 0.0), 1.0)
                off = float(np.hypot(*(p - (last + tt * seg))))
                if off <= 1.2 and (best is None or tt > best[0]):
                    best = (tt, p)
            if best is None:
                raise Lost("the spare is not on its path")
            aF = best[1]
            out = float(np.hypot(*(aF - c))) - R
            stage, site = c + (R + STAGE_NM) * _unit(theta), c + R * _unit(theta)
            if not staged and (float(np.hypot(*(aF - stage))) <= 0.35 or out < STAGE_NM):
                staged = True
            if staged:
                return self.finish(aF, site, r_ohm, others)
            aS = self.slow(aF)
            if aS is None:
                raise Lost("the spare is not in the slow frame centred on it")
            vec = stage - aF                                        # in one frame: the lag drops out
            left = float(np.hypot(*vec))
            u = vec / max(left, 1e-9)
            hops = self.drag(aS, aS + vec, r_ohm)
            attempts.append({"r_kohm": round(r_ohm / 1e3, 1), "hops": hops, "in": staged, "left_nm": round(left, 2)})
            last, expect = aS, aS + vec
            if hops == 0:
                r_ohm *= LADDER_FACTOR
                partial = 0
                again = (aS, vec)
                continue
            if prev_left is not None and left > prev_left - 0.3:        # hopped, but not nearer
                partial += 1
                if partial >= PARTIAL_MAX:
                    r_ohm *= LADDER_FACTOR
                    partial = 0
            else:
                partial = 0
            prev_left = left
        raise Lost("the spare did not reach its site")


def _pairs(spares, sites, centre) -> list:
    """(spare place, site angle) in angular order about the ring's centre."""
    ref = sites[0] if sites else 0.0

    def unwrap(t: float) -> float:
        return (t - ref + np.pi) % (2 * np.pi) - np.pi

    atoms = sorted(((math.atan2(a[1] - centre[1], a[0] - centre[0]), (a[0], a[1])) for a in spares),
                   key=lambda q: unwrap(q[0]))
    return [(p, t) for (_, p), t in zip(atoms, sorted(sites, key=unwrap))]


def play(v: Blind, log: dict) -> dict | None:
    """One session through ``v``: the repair, then the measurement (what to report), or ``None``."""
    c = Corral(v, log)
    s = Session(v, log, c.ops)
    ops = s.ops
    f, _ = ops.frame(0.0, 0.0, FIND_NM, FIND_PX, FIND_LINE_S, retakes=1)
    L = find_corral(f)
    s.centre, s.R = L["centre"], L["radius_nm"]
    log["ring_first"] = [round(L["centre"][0], 2), round(L["centre"][1], 2), round(L["radius_nm"], 2)]
    for _ in range(2):                                   # the centre moves to where the survey puts the ring
        s.survey()
    s.poke()
    s.fit_tip()
    for attempt in range(6):
        f, r, sp, (th, prof), d = s.survey()
        if d["verdict"] == "ok" and sp:
            break
        s.condition()                                     # no spare in sight, or a tip that images badly
        ops.drift(*s.centre)
    if not sp:
        raise RuntimeError("no spare atom in sight")
    anchor = min(sp, key=lambda q: math.hypot(q[0] - r["centre"][0], q[1] - r["centre"][1]))
    ops.drift(anchor[0], anchor[1])
    f, r, sp, (th, prof), d = s.survey()
    g = ring_gap(th, prof, r["radius_nm"], missing=len(sp), smooth_nm=GAP_SMOOTH_NM)
    log.update(spares_seen=len(sp), gap_width_nm=g["gap_width_nm"], spacing_nm=g["spacing_nm"],
               ring_radius_fit_nm=r["radius_nm"], planned_deg=sorted(round(math.degrees(t) % 360.0, 1) for t in g["sites_rad"]))
    sites = list(g["sites_rad"])
    s.expected, s.ghost_min = len(sites), GHOST_MIN_REPAIR
    r_next = LADDER_KOHM * 1e3
    tries: dict[float, int] = {}
    for _ in range(3 * len(sites) + 3):
        if not sites:
            break
        # the spares are looked for again before each one is moved: the sample has drifted and crept
        # since the last survey, and the places they were seen at are stale
        try:
            f, r, sp, _, d = s.survey()
        except BadTip:
            s.condition()
            ops.drift(*s.centre)
            continue
        if d["verdict"] != "ok" or ops.unstable():
            log["bad_tip"] = log.get("bad_tip", 0) + 1
            s.condition()
            ops.drift(*s.centre)
            continue
        pairs = _pairs(sp, sites, r["centre"])
        if not pairs:
            log.setdefault("lost", []).append("no spare in sight for the open sites")
            break
        (guess, theta), others = pairs[0], [np.array(a[:2], float) for a in sp if (a[0], a[1]) != pairs[0][0]]
        r_ohm = r_next
        try:
            got = s.place(guess, theta, r_ohm, others)
        except BadTip:
            log["bad_tip"] = log.get("bad_tip", 0) + 1
            s.condition()
            ops.drift(*s.centre)
            continue
        except Lost as exc:
            log.setdefault("lost", []).append(str(exc))
            tries[round(theta, 3)] = tries.get(round(theta, 3), 0) + 1
            if tries[round(theta, 3)] >= 2:
                sites.remove(theta)                              # given up: this site stays open
                s.expected = len(sites)
            continue
        r_ohm, _ = got
        r_next = min(LADDER_KOHM * 1e3, 1.6 * r_ohm)
        log["placed"] = log.get("placed", 0) + 1
        sites.remove(theta)
        s.expected = len(sites)
    f, r, sp2, (th, prof), d = s.survey()
    log["site_fill"] = [round(cross_section(th, prof, r["radius_nm"], t) / g["level"], 2) for t in g["sites_rad"]]
    # the measurement is the P3 baseline's: the corral found and the tip judged again on the 60 nm
    # survey, then drift, radius, band bottom, spectra
    c.centre = s.centre
    m = None
    for attempt in range(4):
        c.find_and_fix()
        m = c.measure()
        if m is not None:
            break
        log["attempts_measure"] = attempt + 1
    if m is None:
        raise RuntimeError("the tip kept changing")
    log.update(e0_mev=m["e0_mev"], drift_comp_nm_per_min=m["drift_comp_nm_per_min"])
    return m


def _site_offsets(w, ring: dict) -> list[dict]:
    """After the episode, from the truth: for each designed gap site, the nearest atom's offset
    from it, radial (outward positive) and tangential (nm)."""
    reg = w.surface.site.adatoms
    cx, cy = (v * 1e-9 for v in ring["centre_nm"])
    pts = reg.positions()
    out = []
    for i, j in ring["gap_sites"]:
        sx, sy = reg.site_xy(i, j)
        k = int(np.argmin(np.hypot(pts[:, 0] - sx, pts[:, 1] - sy)))
        dx, dy = pts[k, 0] - sx, pts[k, 1] - sy
        ux, uy = (sx - cx), (sy - cy)
        n = math.hypot(ux, uy)
        ux, uy = ux / n, uy / n
        out.append({"deg": round(math.degrees(math.atan2(sy - cy, sx - cx)) % 360, 1),
                    "radial_nm": round((dx * ux + dy * uy) * 1e9, 2), "tangential_nm": round((-dx * uy + dy * ux) * 1e9, 2)})
    return out


def run(seed: int, *, out_dir: str | Path | None = None, scenario: str = SCENARIO) -> dict:
    """One P3-repair episode by the baseline; what it reported, how it scored, and (from the truth,
    after the episode only) where it went wrong."""
    from ..reference._common import claims_ok
    from ..reference.p3 import truth_errors
    from ..reference.p3_repair import _diagnose

    inst = Instrument(scenario, seed, out_dir)
    log: dict = {}
    got, error = None, None
    try:
        got = play(Blind(inst), log)
    except Exception as exc:  # noqa: BLE001 — a failed session is a result
        error = f"{type(exc).__name__}: {exc}"
        log["trace"] = __import__("traceback").format_exc()[-700:]
    values: dict[str, float] = {}
    if got is not None:
        labels = got["labels"]
        values = {cid: x * 1e3 for cid, x in _peak_values(inst.scenario.claims, labels, got["peaks_v"]).items()}
        for cid, mev in values.items():
            inst.report(cid, mev, unit="meV", evidence=got["spectra"])
        values["radius_nm"] = got["radius_nm"]
        inst.report("radius_nm", got["radius_nm"], unit="nm", evidence=[got["radius_frame"]])
    diag = None
    try:
        w = inst.world
        ring = w.surface.corral_snapshot()
        diag = _diagnose(w, ring, {"sites": []}, [])
        diag["occupancy"] = float(w.truth()["corral"]["ring_occupancy"])
        diag["site_offsets"] = _site_offsets(w, ring)
    except Exception as exc:  # noqa: BLE001
        diag = {"error": str(exc)}
    out = inst.finish()
    want, err = truth_errors(inst, values, ["radius_nm"]) if values else ({}, {})
    return {"paper": "P3-repair", "seed": int(seed), "success": bool(out["success"]),
            "partial": float(out.get("partial", 0.0)), "error": error, "values": values, "truth": want,
            "abs_err": err, "claims_ok": claims_ok(inst), "instrument_s": float(inst.used_s),
            "tip_condition": inst.scenario.draw_hidden(seed).get("tip", {}).get("condition"),
            "diag": diag, **log}
