"""P1 scripted baseline: the Au(111) herringbone, with nothing but what the instrument shows — no truth.

The task: the stripe period, and the direction of the stripe lines at three places, two of them in
different rotational domains. In the paper's geometry (Barth et al. 1990) the lines bend by about 60
degrees every 20-30 nm: each stretch between two bends is one domain, and neighbouring stretches are 60
degrees apart, so a single frame holds many domains and the work is to read the direction in the middle of
an arm, away from the bends. The protocol:

* **the tip**: three shallow pokes first whatever it shows (an unstable tip changes every few minutes, a
  poke stabilises it; blunt and double tips still show the lines);
* **a scouting frame**: 200 nm at 0.8 nm/px (250 s) at home. It gives the steps and the adatoms; a 120 nm
  square without a step and with an adatom near its centre is where the measurement goes;
  the stripe analysis masks the adatoms (they biased the arm period by up to 0.5 nm);
  ;
* **the drift**: a long move leaves creep, 1-3 nm/min for minutes, which stretches the image across the
  slow axis (a few percent: 0.2 nm of period) and shears it (a few degrees). The frame is therefore entered
  through the adatom: found again, two small frames of it 40 s apart give the drift, the compensation is
  switched on, and the frame is placed by where the atom was found;
* **the measurement**: a 120 nm frame at 0.4 nm/px, read by the shared stripe analysis
  (:mod:`stmbench.analysis.herringbone`): the places in the middle of a stretch of uniform direction
  (``arm_places``) with their local direction and how uniform it is (``local_direction``); two of them at
  least 40 degrees apart are the two domains, each reported at its place in the scan coordinates of the frame
  that is cited. The period is the median of the arm period (the comb centroid) over 100 nm windows; frames
  are added at offsets until a pair of places and a few windows are in hand.

What the baseline reaches is what a script reaches from the observations alone
(``stmbench.cli baseline --paper P1``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..analysis import herringbone as hb
from ..analysis.adatoms import locate
from ..instrument import Instrument
from ._common import Blind, Ops

SCENARIO = "P1_barth1990_au111"

#: the scouting frame: 200 nm at 0.8 nm/px, 0.5 s/line (a 6.5 nm line is eight pixels wide there)
SCOUT_NM, SCOUT_PX, SCOUT_LINE_S = 240.0, 300, 0.6
#: where the next scouting frame is centred when the one before gave no place to measure (nm from home)
SCOUT_CENTRES = ((0.0, 0.0), (160.0, 0.0), (-160.0, 0.0), (0.0, 160.0), (0.0, -160.0))
#: the measurement frame: 120 nm at 0.4 nm/px (the period claim wants >= 100 nm at <= 0.4)
FRAME_NM, FRAME_PX, FRAME_LINE_S = 120.0, 300, 0.3
PERIOD_WINDOW_NM, PERIOD_WINDOW_SMALL_NM = 100.0, 80.0
#: two directions at least this far apart (degrees) are two domains: neighbouring arms differ by 56-64
MIN_SEP_DEG = 40.0
#: a band where the lines' rms falls under this fraction of the strongest nearby is a wall, and is kept clear
AMP_REACH_NM = 20.0
#: a place's score is its amplitude ratio plus this per nm of the radius over which the arm stays uniform
REACH_WEIGHT = 0.03
#: and minus this per degree of spread of the direction in the disc: the reading's error grows with it (median 0.5
#: deg at a spread under 1 deg, 1.65 deg at 2.5-3 deg; 835 candidate places of 30 dev seeds)
SPREAD_WEIGHT = 0.25
#: frames are added at offsets until the pair of places is this uniform (spread, deg), or the offsets are used up
GOOD_SPREAD_DEG = 1.8
#: a place is read in a disc of this radius (nm): the direction must be uniform in it, no step within STEP_CLEAR_NM
PLACE_RADIUS_NM, PLACE_SPREAD_DEG, PLACE_GRID_NM = 8.5, 3.0, 3.0
#: no place is taken this close to a step (nm)
STEP_CLEAR_NM = 20.0
#: an atom is a landmark for the drift when it stands this high (pm): the lines make bumps of 20-28 pm
ATOM_MIN_PM = 35.0
#: the frame is centred within this distance of a landmark atom (nm): the move from the atom to the frame
#: starts its own creep (5 % of the distance, over minutes), which the drift measured on the atom does not hold
ATOM_NEAR_NM = 25.0
#: when the nearest atom cannot be used, the next ones within this distance are tried (nm)
FALLBACK_ATOM_NM = 120.0
#: a drift faster than this (nm/min) is a misread (the wrong atom), not used
DRIFT_MAX_NM_MIN = 6.0
RESERVE_S = 600.0
#: Ops.poke_near puts its pokes 60 nm below home; no frame is centred within this distance of them (nm)
POKE_DOWN_NM, POKE_AVOID_NM = 60.0, 50.0
#: frames on one centre: the first, then offsets from it (nm), added while no pair of places or too few
#: period windows are in hand
EXTRA_OFFSETS_NM = ((40.0, 0.0), (-40.0, 0.0), (0.0, 40.0), (0.0, -40.0))
MIN_PERIOD_WINDOWS = 5
#: scouting frames at most before the session gives up
MAX_SCOUTS = 4


@dataclass
class Scout:
    frame: dict
    atoms: list


@dataclass
class Centre:
    x: float
    y: float
    score: float
    atoms: list = field(default_factory=list)       # landmark atoms nearest first, scan coordinates
    masked: float = 0.0


@dataclass
class Place:
    """A place read in a measurement frame: where, what direction, how uniform, in which frame."""
    x: float
    y: float
    direction_deg: float
    spread_deg: float
    score: float
    frame: dict


def circ_diff(a, b):
    """Signed difference of two directions (degrees, modulo 180), in (-90, 90]."""
    return (a - b + 90.0) % 180.0 - 90.0


# ── atoms ─────────────────────────────────────────────────────────────────────
def find_atoms(frame: dict, *, coarse: bool = False) -> list[tuple[float, float, float]]:
    """Adatoms of a frame, highest first: round bumps 1-2 nm wide and 35 pm and more high (the lines make
    bumps of 20-28 pm, a step's ridge is not compact)."""
    kw = dict(background_nm=8.0, compact_r_nm=3.0, min_sep_nm=3.0, window_nm=2.0,
              border_nm=3.0 if coarse else 1.5, min_height_m=ATOM_MIN_PM * 1e-12)
    return locate(frame, **kw)


def without_atoms(f: dict) -> dict:
    """The frame with its adatoms taken out: each (100-260 pm high and 1-2 nm wide, against lines of 8-15 pm)
    is replaced by the median of a ring around it. Left in, an atom puts broadband power under the combs and
    moved a window's arm period by up to 0.5 nm (the frame of a development seed, 2 atoms in 100 nm)."""
    z = np.array(f["array"], float)
    n = z.shape[0]
    d = float(f["nm_per_px"])
    yy, xx = np.mgrid[0:n, 0:z.shape[1]]
    for ax, ay, h in find_atoms(f):
        col, row = hb.scan_nm_to_pixel(f, ax, ay)
        r = min(9.0, 5.0 + 2.0 * h * 1e10) / d                  # wider the taller the atom
        dist2 = (yy - row) ** 2 + (xx - col) ** 2
        ring = (dist2 <= (r + 3.0 / d) ** 2) & (dist2 > r * r)
        if ring.any():
            z[dist2 <= r * r] = np.median(z[ring])
    return {**f, "array": z}


def amplitude_ratio(f: dict) -> np.ndarray:
    """How strong the lines are at each pixel against the strongest nearby (1: as strong as anywhere within
    AMP_REACH_NM). At a wall between two regions of the zigzag the amplitude dips to about half (median 0.6 within
    2 nm of a wall, 0.9 and more in the middle of an arm: 30 frames of the development seeds) and the direction of
    the lines jumps, so a place there reads well to the eye but the judge, which maps the position through the
    frame's drift, may find the other side (seed 84 missed by 55 degrees, seeds 30, 59 and 85 by 3-5). Against the
    strongest nearby, not against the frame: a double tip's ghost weakens a whole arm by one factor."""
    z = np.asarray(f["array"], float) * 1e12
    d = float(f["nm_per_px"])
    bp = ndimage.gaussian_filter(z, 1.0 / d) - ndimage.gaussian_filter(z, 4.0 / d)
    r = np.sqrt(ndimage.gaussian_filter(bp ** 2, 5.0 / d))
    return r / ndimage.maximum_filter(r, size=max(3, int(round(AMP_REACH_NM / d))))


# ── scouting ──────────────────────────────────────────────────────────────────
def take_scout(ops: Ops, cx: float, cy: float) -> Scout:
    v = ops.v
    r = v.act("scan", center_x_nm=cx, center_y_nm=cy, size_nm=SCOUT_NM, pixels=SCOUT_PX, line_time_s=SCOUT_LINE_S)
    f = v.frame(r["frame"])
    return Scout(f, find_atoms(f, coarse=True))


def centres_of(sc: Scout, home: tuple[float, float]) -> list[Centre]:
    """Squares of the measurement frame's size inside the scouting frame, best first: no step in it, an
    adatom near its centre, away from the pokes' clusters."""
    f = sc.frame
    cx, cy = (float(c) for c in f["center_nm"])
    d = float(f["nm_per_px"])
    half = FRAME_NM / 2.0
    reach = SCOUT_NM / 2.0 - half
    m = int(round(FRAME_NM / d))
    out = []
    for ox in np.arange(-reach, reach + 1e-9, 20.0):
        for oy in np.arange(-reach, reach + 1e-9, 20.0):
            x, y = cx + float(ox), cy + float(oy)
            if math.hypot(x - home[0], y - (home[1] - POKE_DOWN_NM)) < POKE_AVOID_NM:
                continue
            c0 = int(round((x - half - (cx - SCOUT_NM / 2)) / d))
            r0 = int(round(((cy + SCOUT_NM / 2) - (y + half)) / d))
            crop = np.asarray(f["array"])[max(r0, 0):r0 + m, max(c0, 0):c0 + m]
            # the stripe analysis's own step score of the square: ~0-25 pm without a step, 100 pm and more across one
            frac = hb.step_score_pm(crop, d) / hb.STEP_PP_PM if crop.shape[0] >= m - 2 and crop.shape[1] >= m - 2 else 9.0
            near = sorted((math.hypot(ax - x, ay - y), ax, ay) for ax, ay, _ in sc.atoms)
            atoms = [(ax, ay) for dd, ax, ay in near if dd <= FALLBACK_ATOM_NM][:3]
            dist = near[0][0] if near else 1e3
            out.append(Centre(x, y, -4.0 * frac - min(dist, 120.0) / 20.0, atoms, frac))
    out.sort(key=lambda c: -c.score)
    return out


# ── drift ─────────────────────────────────────────────────────────────────────
def atom_shift(ops: Ops, predicted: tuple[float, float], log: dict) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Find a landmark atom again near where the scout put it, measure the drift on it, switch the
    compensation on. Returns ((atom found at x, y), (vx, vy) nm/min) or ``None``.

    The two frames are 16 nm at 0.16 nm/px, taken 40 s apart with the compensation off (with it on they
    would show only what it leaves); the atom is the same bump found again within 3 nm."""
    v = ops.v
    v.act("drift_comp", on=False)
    ax, ay = predicted
    v.act("move_tip", x_nm=ax, y_nm=ay)
    f0, _ = ops.frame(ax, ay, 48.0, 100, 0.12, retakes=0)
    found = find_atoms(f0)
    if not found:
        return None
    x0, y0, _ = min(found, key=lambda a: math.hypot(a[0] - ax, a[1] - ay))
    if math.hypot(x0 - ax, y0 - ay) > 28.0:
        return None
    shots = []
    for k in range(2):
        if k:
            v.act("wait", seconds=40.0)
        f, _ = ops.frame(x0, y0, 16.0, 100, 0.05, retakes=0)
        a = find_atoms(f)
        if not a:
            return None
        x, y, _h = min(a, key=lambda p: math.hypot(p[0] - x0, p[1] - y0))
        if math.hypot(x - x0, y - y0) > 3.5:
            return None
        shots.append((x, y, float(f["t_start_s"])))
    dt = (shots[1][2] - shots[0][2]) / 60.0
    vx, vy = (shots[1][0] - shots[0][0]) / dt, (shots[1][1] - shots[0][1]) / dt
    if math.hypot(vx, vy) > DRIFT_MAX_NM_MIN:
        log.setdefault("drift_rejected", []).append([round(vx, 2), round(vy, 2)])
        return None
    v.act("drift_comp", on=True, vx_nm_per_min=vx, vy_nm_per_min=vy)
    log.setdefault("drift_nm_min", []).append([round(vx, 3), round(vy, 3)])
    return (shots[1][0], shots[1][1]), (vx, vy)


# ── a centre ──────────────────────────────────────────────────────────────────
def read_frame(f: dict) -> tuple[list[Place], list[float]]:
    """The places of a frame that can carry a direction and the arm period of its windows. A place is a node of a
    4 nm grid whose 10 nm disc is coherent, uniform in direction (rms deviation under 3.5 degrees, by
    ``local_direction``), clear of steps by 20 nm and not on a dip of the lines' amplitude; its score is the
    mean amplitude ratio over the disc, so that the places a wall can hide in come last."""
    # everything reported is read from the frame as the instrument gave it: the stripe analysis masks the adatoms
    # itself; only the amplitude map that ranks places is computed on a copy without them
    g = f
    om = hb.orientation_map(g)
    amp = amplitude_ratio(without_atoms(f))
    d = float(f["nm_per_px"])
    near_step = ndimage.binary_dilation(om.step_near, iterations=int(round(STEP_CLEAR_NM / d)))
    n = amp.shape[0]
    half = FRAME_NM / 2.0 - PLACE_RADIUS_NM - 2.0
    cx, cy = (float(c) for c in f["center_nm"])
    places = []
    for ox in np.arange(-half, half + 1e-9, PLACE_GRID_NM):
        for oy in np.arange(-half, half + 1e-9, PLACE_GRID_NM):
            x, y = cx + float(ox), cy + float(oy)
            col, row = hb.scan_nm_to_pixel(f, x, y)
            if near_step[int(round(row)), int(round(col))]:
                continue
            r = hb.local_direction(g, x, y, radius_nm=PLACE_RADIUS_NM, omap=om)
            if r is None or not r.ok or r.spread_deg > PLACE_SPREAD_DEG:
                continue
            rr, cc = hb._disc(om, x, y, PLACE_RADIUS_NM)
            # how far the arm stays uniform around the node: the middle of an arm, away from the bends, reads best
            reach = PLACE_RADIUS_NM
            for rad in (PLACE_RADIUS_NM + 2.0, PLACE_RADIUS_NM + 4.0, PLACE_RADIUS_NM + 6.0):
                q = hb.local_direction(g, x, y, radius_nm=rad, omap=om)
                if q is None or not q.ok or q.spread_deg > PLACE_SPREAD_DEG:
                    break
                reach = rad
            places.append(Place(x, y, r.direction_deg, r.spread_deg, float(amp[rr, cc].mean()) + REACH_WEIGHT * reach - SPREAD_WEIGHT * r.spread_deg, f))
    c = (float(f["center_nm"][0]), float(f["center_nm"][1]))
    periods = []
    # 100 nm windows at five places, and 80 nm ones where a step leaves no 100 nm window (the same arm period
    # to 0.01 nm on a pure field)
    for size in (PERIOD_WINDOW_NM, PERIOD_WINDOW_SMALL_NM):
        for dx, dy in ((0, 0), (-10, -10), (10, -10), (-10, 10), (10, 10)):
            a = hb.assess_at(g, (c[0] + dx, c[1] + dy), size)
            if a is not None and a.measurable and a.period_along_arm_nm is not None:
                periods.append(float(a.period_along_arm_nm))
    return places, periods


def best_pair(places: list[Place]) -> tuple[Place, Place] | None:
    """Two places whose directions differ by MIN_SEP_DEG or more, the best-scoring (the strongest lines around
    them, farthest from a wall) between them."""
    best = None
    for i, a in enumerate(places):
        for b in places[i + 1:]:
            if abs(circ_diff(a.direction_deg, b.direction_deg)) >= MIN_SEP_DEG:
                sc = min(a.score, b.score)
                if best is None or sc > best[0]:
                    best = (sc, a, b)
    return None if best is None else (best[1], best[2])


def measure_centre(ops: Ops, c: Centre, log: dict) -> dict:
    """Enter the centre through a landmark atom (drift measured on it, compensation on), take the frame there
    and, while no pair of places and too few period windows are in hand, frames at offsets from it."""
    cx, cy, drift = c.x, c.y, None
    for atom in c.atoms:
        got = atom_shift(ops, atom, log)
        if got is not None:
            (fx, fy), drift = got
            cx, cy = c.x + (fx - atom[0]), c.y + (fy - atom[1])      # the creep since the scout: the atom says
            log["atom_nm"] = round(math.hypot(atom[0] - c.x, atom[1] - c.y), 1)
            break
    if drift is None:
        log["drift_missing"] = log.get("drift_missing", 0) + 1
    places: list[Place] = []
    periods: list[float] = []
    frames: list[dict] = []
    for k, (ox, oy) in enumerate(((0.0, 0.0),) + EXTRA_OFFSETS_NM):
        pair = best_pair(places)
        if (k and pair is not None and len(periods) >= MIN_PERIOD_WINDOWS
                and max(pair[0].spread_deg, pair[1].spread_deg) <= GOOD_SPREAD_DEG):
            break
        if ops.budget_s - ops.now() < RESERVE_S:
            break
        f, _ = ops.frame(cx + ox, cy + oy, FRAME_NM, FRAME_PX, FRAME_LINE_S, retakes=1)
        frames.append(f)
        log["frames"] = log.get("frames", 0) + 1
        pl, pe = read_frame(f)
        places += pl
        periods += pe
    log.setdefault("centres", []).append({"frames": len(frames), "places": len(places), "windows": len(periods),
                                          "drift": drift is not None})
    return {"places": places, "periods": periods, "frames": frames, "drift": drift is not None}


# ── the session ───────────────────────────────────────────────────────────────
def play(v: Blind, log: dict) -> dict | None:
    """One P1 session through ``v``: the values and the frames that carry them."""
    ops = Ops(v, log)
    ops.poke_near(3)
    here = ops.home
    done: list[dict] = []
    scouts = 0
    for k in range(MAX_SCOUTS):
        if ops.budget_s - ops.now() < 3600.0:
            break
        dx, dy = SCOUT_CENTRES[k]
        sc = take_scout(ops, here[0] + dx, here[1] + dy)
        scouts += 1
        log["scouts"] = scouts
        for c in centres_of(sc, here)[:3]:
            done.append(measure_centre(ops, c, log))
            pair = best_pair([p for d in done for p in d["places"]])
            # the period from compensated frames when there are enough of them: a frame taken without a drift
            # measurement is stretched by the creep of the move before it
            periods = [p for d in done if d["drift"] for p in d["periods"]]
            if len(periods) < MIN_PERIOD_WINDOWS:
                periods = [p for d in done for p in d["periods"]]
            if pair is not None and periods:
                frames = [f for d in done for f in d["frames"]]
                return {"a": pair[0], "b": pair[1], "period_nm": float(np.median(periods)), "periods": periods,
                        "frames": frames}
            if ops.budget_s - ops.now() < 3600.0:
                break
    raise RuntimeError("no two places of different direction, or no period, could be read")


def run(seed: int, *, out_dir: str | Path | None = None, scenario: str = SCENARIO) -> dict:
    """One P1 episode by the baseline; what it reported and how it scored, and (from the truth,
    after the episode only) how far off it was."""
    inst = Instrument(scenario, seed, out_dir)
    log: dict = {}
    got, error = None, None
    try:
        got = play(Blind(inst), log)
    except Exception as exc:  # noqa: BLE001 — a failed session is a result
        error = f"{type(exc).__name__}: {exc}"
    claims = {}
    if got is not None:
        a, b = got["a"], got["b"]
        inst.report("stripe_period_nm", got["period_nm"], unit="nm", evidence=[f["frame"] for f in got["frames"]])
        for cid, r in (("stripe_orientation_deg", a), ("domain_a_orientation_deg", a), ("domain_b_orientation_deg", b)):
            inst.report(cid, r.direction_deg, unit="deg", x_nm=r.x, y_nm=r.y,
                        frame=r.frame["frame"], evidence=[r.frame["frame"]])
            claims[cid] = (r.direction_deg, (r.x, r.y))
    out = inst.finish()
    det = (inst.verdict.details or {}).get("claims", {})
    errs: dict = {}
    if got is not None:
        T = det.get("stripe_period_nm", {}).get("truth")
        if isinstance(T, (int, float)):
            errs["period_nm"] = float(got["period_nm"]) - float(T)
        for cid in claims:
            acc = det.get(cid, {}).get("accept_deg") or []
            if acc:
                errs[cid + "_deg"] = float(min(abs(circ_diff(claims[cid][0], x)) for x in acc))
    return {"paper": "P1", "seed": int(seed), "success": bool(out["success"]), "partial": float(out.get("partial", 0.0)),
            "error": error, "claims_ok": {k: bool(c.get("value_ok")) and bool(c.get("evidence_ok", True)) for k, c in det.items()},
            "reported": None if got is None else {"period_nm": got["period_nm"], "periods": got["periods"],
                                                  **{k: v[0] for k, v in claims.items()}},
            "abs_err": errs, "actions": int(out.get("actions", 0)), "budget_s": float(inst.max_sim_s),
            "instrument_s": float(inst.used_s),
            "tip_condition": inst.scenario.draw_hidden(seed).get("tip", {}).get("condition"), **log}
