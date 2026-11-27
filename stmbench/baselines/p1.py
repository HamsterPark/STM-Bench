"""P1 scripted baseline: the Au(111) herringbone, with nothing but what the instrument shows — no truth.

The task: the stripe period, the stripe direction at one place, and two rotational domains whose
directions differ. Reproducing it from observations means finding two domains without knowing where
they are. The protocol:

* **the tip**: three shallow pokes first whatever it shows (an unstable tip changes every few
  minutes, a poke stabilises it; blunt and double tips still show the stripes, so a tip that cannot
  show them is only conditioned when a survey finds no stripe at all);
* **a survey**: 400 nm at 1.0 nm/px (800 s). A 6.3 nm stripe is six pixels wide there. The survey's
  windows give each place's mean stripe direction (:mod:`stmbench.analysis.au111`); steps are
  masked, a tip change in the frame only costs the rows it touches. Where a block of neighbouring
  windows agrees on one direction the place is well inside a rotational domain. Two such places whose
  directions differ by 40 degrees or more are two domains (the true means differ by 60). When the
  survey holds only one, the next survey is taken 300 nm away;
* **the drift**: a long move leaves creep, 1-3 nm/min for minutes, which stretches the image across
  the slow axis (a few percent: 0.2 nm of period) and shears it (a few degrees). Each site is therefore
  entered through an atom near it: found again (the survey's atoms are the landmarks, 150/um^2), two small
  frames of it 40 s apart give the drift, the compensation is switched on, and the site's window
  is placed by where the atom was found;
* **the measurement**: a 120 nm frame at 0.4 nm/px on each site, read by the shared stripe analysis
  (:mod:`stmbench.analysis.herringbone`): 100 nm windows for the period (the median over all
  windows of all frames), 64 nm windows for the direction (the mean of the two arms, the best window,
  reported at its centre in the scan coordinates of the frame that is cited).

What the baseline reaches is what a script reaches from the observations alone
(``stmbench.cli baseline --paper P1``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..analysis import herringbone as hb
from ..analysis.adatoms import locate, row_jumps, to_scan_nm
from ..analysis.au111 import circ_diff, clean_windows, direction_map
from ..instrument import Instrument
from ._common import Blind, Ops

SCENARIO = "P1_barth1990_au111"

#: the survey: 400 nm at 1.0 nm/px, 1 s/line (400 nm/s is the fastest line the baselines use)
SURVEY_NM, SURVEY_PX, SURVEY_LINE_S = 400.0, 400, 1.0
#: where the next survey is centred when the one before did not hold two domains (nm from home)
SURVEY_CENTRES = ((0.0, 0.0), (300.0, 0.0), (-300.0, 0.0), (0.0, 300.0), (0.0, -300.0),
                  (300.0, 300.0), (-300.0, -300.0))
#: the measurement frame on a site: 120 nm at 0.4 nm/px (the period claim wants >= 100 nm at <= 0.4)
FRAME_NM, FRAME_PX, FRAME_LINE_S = 120.0, 300, 0.3
PERIOD_WINDOW_NM = 100.0
DIRECTION_WINDOW_NM = 64.0
#: two directions at least this far apart (degrees) are two domains: the true means differ by 60, a
#: survey window's mean is good to ~10
MIN_SEP_DEG = 40.0
#: an atom is a landmark for the drift when it stands this high (pm): the stripes make bumps of 20-28 pm
ATOM_MIN_PM = 35.0
#: landmark atoms are taken within this distance of a site (nm): the move from the atom to the site
#: starts its own creep (5 % of the distance, over minutes), which the drift measured on the atom does not hold
ATOM_NEAR_NM = 70.0
#: when the nearest atom cannot be used, the next ones within this distance are tried (nm)
FALLBACK_ATOM_NM = 140.0
#: a drift faster than this (nm/min) is a misread (a lattice mismatch, the wrong atom), not used
DRIFT_MAX_NM_MIN = 6.0
RESERVE_S = 600.0
#: frames are added on a site (offsets from its first, nm) until this many two-arm windows give a period
MIN_PERIODS = 8
EXTRA_OFFSETS_NM = ((45.0, 0.0), (-45.0, 0.0), (0.0, 45.0), (0.0, -45.0))


@dataclass
class Survey:
    frame: dict
    atoms: list
    dm: object
    radius: np.ndarray
    t_s: float


@dataclass
class Site:
    """A place in a survey's scan coordinates that sits well inside one domain."""
    x: float
    y: float
    psi: float
    radius: int
    masked: float
    atom: tuple[float, float] | None = None       # the nearest landmark atom, survey coordinates
    atom_nm: float = 1e9
    atoms: list = field(default_factory=list)       # landmark atoms within FALLBACK_ATOM_NM, nearest first
    survey: int = 0


@dataclass
class Reading:
    """What a site's frame gave: the frame, the direction window, the period windows."""
    frame: dict
    direction_deg: float
    arms_deg: list[float]
    centre_nm: tuple[float, float]
    quality: float
    periods: list[float]
    site: Site
    drift: tuple[float, float] | None = None
    notes: dict = field(default_factory=dict)
    frames: list = field(default_factory=list)      # every frame taken on the site (the period's evidence)


# ── atoms ─────────────────────────────────────────────────────────────────────
def find_atoms(frame: dict, *, coarse: bool = False) -> list[tuple[float, float, float]]:
    """Adatoms of a frame, highest first: round bumps 1-2 nm wide and 35 pm and more high (the stripes make
    bumps of 20-28 pm, a step's ridge is not compact)."""
    kw = dict(background_nm=8.0, compact_r_nm=3.0, min_sep_nm=3.0, window_nm=2.0,
              border_nm=3.0 if coarse else 1.5, min_height_m=ATOM_MIN_PM * 1e-12)
    return locate(frame, **kw)


# ── survey ────────────────────────────────────────────────────────────────────
def take_survey(ops: Ops, cx: float, cy: float, log: dict) -> Survey:
    v = ops.v
    r = v.act("scan", center_x_nm=cx, center_y_nm=cy, size_nm=SURVEY_NM, pixels=SURVEY_PX, line_time_s=SURVEY_LINE_S)
    f = v.frame(r["frame"])
    rows = tuple(row_jumps(f["array"]))
    if rows:
        log.setdefault("survey_jumps", []).append(len(rows))
        ops.jumps.append(float(f["t_start_s"]) + 0.5 * float(r["duration_s"]))
    dm = direction_map(f["array"], float(f["nm_per_px"]), bad_rows=rows)
    cw = clean_windows(dm)
    return Survey(f, find_atoms(f, coarse=True), dm, cw, float(f["t_start_s"]))


def sites_of(sv: Survey, idx: int) -> list[Site]:
    """Windows of a survey that sit well inside one domain, with the landmark atom nearest each."""
    dm, cw = sv.dm, sv.radius
    out = []
    for i in range(cw.shape[0]):
        for j in range(cw.shape[1]):
            if cw[i, j] < 1:
                continue
            x, y = to_scan_nm(sv.frame, float(dm.cols[j]) - 0.5, float(dm.rows[i]) - 0.5)
            s = Site(x, y, float(dm.psi_deg[i, j]), int(cw[i, j]), float(dm.masked[i, j]), survey=idx)
            near = sorted((math.hypot(ax - x, ay - y), ax, ay) for ax, ay, _ in sv.atoms)
            if near:
                d, ax, ay = near[0]
                s.atom, s.atom_nm = (ax, ay), d
                s.atoms = [(ax_, ay_) for d_, ax_, ay_ in near if d_ <= FALLBACK_ATOM_NM][:3]
            out.append(s)
    return out


def _score(s: Site, here: tuple[float, float]) -> float:
    """Deep inside a domain, a landmark atom near, little step in the window, near where the tip is."""
    return (min(s.radius, 4) + 3.0 * max(0.0, 1.0 - s.atom_nm / ATOM_NEAR_NM) - 2.0 * s.masked
            - math.hypot(s.x - here[0], s.y - here[1]) / 400.0)


def choose_pair(sites: list[Site], here: tuple[float, float]) -> tuple[Site, Site] | None:
    """The best site, and the best one whose direction differs from it by MIN_SEP_DEG or more."""
    good = [s for s in sites if s.atom is not None and s.atom_nm <= ATOM_NEAR_NM] or sites
    best = None
    for a in sorted(good, key=lambda s: -_score(s, here))[:60]:
        for b in sorted(good, key=lambda s: -_score(s, (a.x, a.y))):
            if abs(circ_diff(a.psi, b.psi)) >= MIN_SEP_DEG and a.survey == b.survey:
                sc = _score(a, here) + _score(b, (a.x, a.y))
                if best is None or sc > best[0]:
                    best = (sc, a, b)
                break
    return None if best is None else (best[1], best[2])


# ── drift ─────────────────────────────────────────────────────────────────────
def atom_shift(ops: Ops, predicted: tuple[float, float], log: dict) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Find a landmark atom again near where the survey put it, measure the drift on it, switch the
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


# ── a site ────────────────────────────────────────────────────────────────────
def read_site(ops: Ops, site: Site, log: dict, *, expect: float | None = None) -> Reading | None:
    """Take the site's frame (compensated for the drift measured on a landmark atom near it), read
    the direction at the best window and the period in 100 nm windows. ``None`` when nothing could be read."""
    notes: dict = {}
    cx, cy = site.x, site.y
    drift = None
    for atom in site.atoms:
        got = atom_shift(ops, atom, log)
        if got is not None:
            (fx, fy), drift = got
            # the creep since the survey moved the sample under the scan frame: the atom says by how much
            cx, cy = site.x + (fx - atom[0]), site.y + (fy - atom[1])
            notes["offset_nm"] = [round(fx - atom[0], 2), round(fy - atom[1], 2)]
            notes["atom_nm"] = round(math.hypot(atom[0] - site.x, atom[1] - site.y), 1)
            break
    if drift is None:
        log["drift_missing"] = log.get("drift_missing", 0) + 1
    frames: list[dict] = []
    periods: list[float] = []
    best = None                  # (rank, window, frame) of the direction window
    for k, (ox, oy) in enumerate(((0.0, 0.0),) + EXTRA_OFFSETS_NM):
        if k and (len(periods) >= MIN_PERIODS or ops.budget_s - ops.now() < RESERVE_S):
            break
        f, _ = ops.frame(cx + ox, cy + oy, FRAME_NM, FRAME_PX, FRAME_LINE_S, retakes=1)
        frames.append(f)
        log["frames"] = log.get("frames", 0) + 1
        # direction windows: 64 nm on a 16 nm grid. A window that reads both arms gives the mean direction;
        # one that reads a single peak (a blunt tip merges the arms) gives that arm, which the judge accepts too
        every = hb.survey(f, DIRECTION_WINDOW_NM, 16.0)
        notes.setdefault("verdicts", []).append({k_: sum(1 for a in every if a.verdict == k_)
                                                 for k_ in sorted({a.verdict for a in every})})
        c = (float(f["center_nm"][0]), float(f["center_nm"][1]))
        for a in every:
            if not a.measurable or (expect is not None and abs(circ_diff(a.direction_deg, expect))
                                    > (20.0 if a.verdict == "ok" else 26.0)):
                continue
            rank = (a.verdict != "ok", k, math.hypot(a.centre_nm[0] - c[0], a.centre_nm[1] - c[1]) - 0.05 * a.quality)
            if best is None or rank < best[0]:
                best = (rank, a, f)
        # the period: two-arm windows only (a single peak may be the mean-direction peak, whose period is the
        # longer T, not the arm's T cos alpha): the 64 nm windows above and the 100 nm window at five places
        periods += [float(a.period_along_arm_nm) for a in every if a.verdict == "ok" and a.period_along_arm_nm is not None]
        for dx, dy in ((0, 0), (-10, -10), (10, -10), (-10, 10), (10, 10)):
            a = hb.assess_at(f, (c[0] + dx, c[1] + dy), PERIOD_WINDOW_NM)
            if a is not None and a.verdict == "ok" and a.period_along_arm_nm is not None:
                periods.append(float(a.period_along_arm_nm))
    if best is None:
        notes["no_window"] = True
        log.setdefault("site_failures", []).append({"psi": round(site.psi), "why": "no window reads a stripe", **notes})
        return None
    _, w, fd = best
    log.setdefault("sites", []).append({"psi": round(site.psi, 1), "radius": site.radius, "dir": round(w.direction_deg, 1),
                                        "verdict": w.verdict, "n_frames": len(frames), "n_periods": len(periods),
                                        "notes": notes})
    # The direction reported is that of the window's strongest peak. Each of the zigzag's two arms and the mean
    # direction is a direction the stripes run in, so the strongest peak is one of them whichever way the
    # window's peaks are sorted into arms (a pair of an arm and the mean peak, read as two arms, has a
    # bisector that is neither: seed 12 of the development runs missed by 6.9 degrees that way).
    direction = float(w.peaks[0].stripe_deg) if w.peaks else float(w.direction_deg)
    return Reading(fd, direction, [float(x) for x in w.arm_deg], (float(w.centre_nm[0]), float(w.centre_nm[1])),
                   float(w.quality), periods, site, drift, notes, frames)


# ── the session ───────────────────────────────────────────────────────────────
def play(v: Blind, log: dict) -> dict | None:
    """One P1 session through ``v``: the values and the frames that carry them."""
    ops = Ops(v, log)
    ops.poke_near(3)
    surveys: list[Survey] = []
    pair = None
    here = ops.home
    for k, (dx, dy) in enumerate(SURVEY_CENTRES[:5]):
        if ops.budget_s - ops.now() < 3600.0:
            break
        surveys.append(take_survey(ops, ops.home[0] + dx, ops.home[1] + dy, log))
        pair = choose_pair(sites_of(surveys[-1], len(surveys) - 1), (dx, dy))
        log["surveys"] = len(surveys)
        if pair is not None:
            break
    if pair is None:
        raise RuntimeError("no two domains found")
    sv = surveys[pair[0].survey]
    log["survey_s"] = ops.now()
    readings: list[Reading] = []
    for site in sorted(pair, key=lambda s: math.hypot(s.x - here[0], s.y - here[1])):
        r = read_site(ops, site, log, expect=site.psi)
        if r is None:
            raise RuntimeError("a site could not be read")
        readings.append(r)
    a, b = readings
    # the sites were chosen 40 degrees or more apart in the survey (domain means); what the frames read can be
    # an arm on either side of the mean, so the survey's means are what must still differ
    if abs(circ_diff(a.site.psi, b.site.psi)) < MIN_SEP_DEG:
        raise RuntimeError(f"the two sites read {a.direction_deg:.0f} and {b.direction_deg:.0f} deg")
    # a frame taken without a drift measurement (no landmark atom could be used) is stretched by the creep of
    # the long move before it, up to several percent of the period: its windows only count when the
    # compensated ones are few
    periods = [p for r in readings if r.drift is not None for p in r.periods]
    if len(periods) < 5:
        periods = [p for r in readings for p in r.periods]
    if not periods:
        raise RuntimeError("no period could be read")
    return {"readings": readings, "period_nm": float(np.median(periods)), "periods": periods}


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
        a, b = got["readings"]
        frames = [f["frame"] for r in got["readings"] for f in r.frames]
        inst.report("stripe_period_nm", got["period_nm"], unit="nm", evidence=frames)
        for cid, r in (("stripe_orientation_deg", a), ("domain_a_orientation_deg", a), ("domain_b_orientation_deg", b)):
            inst.report(cid, r.direction_deg, unit="deg", x_nm=r.centre_nm[0], y_nm=r.centre_nm[1],
                        frame=r.frame["frame"], evidence=[r.frame["frame"]])
            claims[cid] = (r.direction_deg, r.centre_nm)
    out = inst.finish()
    det = (inst.verdict.details or {}).get("claims", {})
    errs: dict = {}
    if got is not None:
        T = det.get("stripe_period_nm", {}).get("truth")
        if isinstance(T, (int, float)):
            errs["period_nm"] = float(got["period_nm"]) - float(T)
        for cid in claims:
            c = det.get(cid, {})
            acc = c.get("accept_deg") or []
            if acc:
                errs[cid + "_deg"] = float(min(abs(circ_diff(claims[cid][0], x)) for x in acc))
    return {"paper": "P1", "seed": int(seed), "success": bool(out["success"]), "partial": float(out.get("partial", 0.0)),
            "error": error, "claims_ok": {k: bool(c.get("value_ok")) and bool(c.get("evidence_ok", True)) for k, c in det.items()},
            "reported": None if got is None else {"period_nm": got["period_nm"], "periods": got["periods"],
                                                  **{k: v[0] for k, v in claims.items()}},
            "abs_err": errs, "actions": int(out.get("actions", 0)), "budget_s": float(inst.max_sim_s),
            "instrument_s": float(inst.used_s),
            "tip_condition": inst.scenario.draw_hidden(seed).get("tip", {}).get("condition"), **log}
