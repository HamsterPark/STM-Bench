"""P1 privileged reference solution: the Au(111) 22x sqrt(3) herringbone, as Barth, Brune, Ertl and
Behm (1990) saw it: the period between the lines of an arm, and the direction of the lines in two
rotational domains, which in the paper are the neighbouring arms of the zigzag.

The truth navigates, never measures. It says where a step-free area lies whose footprint is wholly
inside one Voronoi region of the zigzag orientation (away from its walls), where adsorbates sit to
register the drift on, and whether the tip needs conditioning. Every reported number is read from
a frame the instrument acquired, through :mod:`stmbench.analysis.herringbone`: the period from the
combs of the 80 nm windows of the frame (median over windows), and each direction as the direction
of the lines at a place in the middle of an arm, found in the image itself (``arm_places``: the
centre of the widest stretch of uniform direction, one per direction), reported at that place in
the scan coordinates of the cited frame.

Protocol:

1. one area of ``FRAME_NM`` (a square, a few nm of margin) that lies wholly in one region, step
   free, at least ``WALL_CLEAR_NM`` from a wall, the nearest to the origin: it holds ~5 arms of
   the zigzag (an elbow every 20-30 nm), so two neighbouring arms, 56-64 degrees apart;
2. a tip that is blunt, multiple, unstable or carries something is conditioned on bare surface at
   least ``CONDITION_CLEAR_NM`` from the area (a pulse sprays clusters ~40 nm around), five
   minutes for the creep of the long move;
3. the tip goes to an adsorbate near the area's centre, two 12 nm frames 30 s apart give the drift
   (bare herringbone cannot: a grating constrains only the component across its lines, modulo a
   period), the compensation is programmed and checked once more by the same two frames;
4. the area is scanned at ``NM_PER_PX`` nm/px and at ``SPEED_NM_S`` or slower: a faster scan with
   a tip of radius above 1.5 nm scratches the surface (``World.maybe_scratch``). A frame in which
   the tip changed (a jump of whole rows) or needed conditioning again is taken again;
5. the period is the median over the ``WINDOW_NM`` windows of the frame of the arm period (the
   power-weighted centroid of each arm's comb); the two places are the two best ``arm_places``
   whose directions differ by at least ``MIN_SEP_DEG``, each read with ``local_direction``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..analysis import herringbone as hb
from ._common import Session, bare_spot, claims_ok, lateral_drift, tip_needs_poke

SCENARIO = "P1_barth1990_au111"
#: the area scanned (nm): the period claim wants a frame of at least 100 nm; the larger frames
#: hold more windows and cost 4-5 minutes at SPEED_NM_S
FRAME_LADDER_NM = (140.0, 120.0, 110.0, 102.0)
#: 0.39 instead of the claim's 0.4 nm/px so that rounding the pixel count never exceeds the limit
NM_PER_PX = 0.39
#: under the 450 nm/s above which a tip of radius > 1.5 nm scratches the surface (world.maybe_scratch)
SPEED_NM_S = 400.0
WINDOW_NM, STRIDE_NM = 80.0, 20.0
#: the two places must differ in direction by at least this (neighbouring arms: 56-64 degrees)
MIN_SEP_DEG = 40.0
#: what the footprint of a frame has to clear: the wall's own width is 2-5 nm and the drift moves
#: the frame by a few nm during it
WALL_CLEAR_NM = 12.0
CONDITION_CLEAR_NM = 170.0
#: how far from the origin to look for the area: the near range first (a short move creeps less)
SEARCH_NM, SEARCH_FAR_NM, SEARCH_STEP_NM = 420.0, 700.0, 20.0
DRIFT_FRAME_NM = 12.0
#: one extra look at the drift after programming it; stop when the residual is below this (nm/min)
RESIDUAL_OK_NM_MIN = 0.15
CREEP_WAIT_S = 300.0
#: takes of the area (a tip change during a frame costs a retake)
MAX_TAKES = 4


def _domain_grid(w, cx: float, cy: float, size_nm: float, n: int = 9):
    """The sample-frame points of a square footprint (nm), for truth checks."""
    g = np.linspace(-size_nm / 2, size_nm / 2, n)
    return [(cx + a, cy + b) for a in g for b in g]


def _area_ok(w, cx: float, cy: float, size_nm: float):
    """Domain label of a square footprint (sample frame, nm) if it lies in one domain, clear of
    walls and of steps; else None. Privileged: reads the truth."""
    surf = w.surface
    margin = 4.0
    half = size_nm / 2 + margin
    if not surf.step_free_window(cx * 1e-9, cy * 1e-9, 2 * half * 1e-9, n=31):
        return None
    label = None
    for px, py in _domain_grid(w, cx, cy, 2 * half, 7):
        o = surf.orientation_at(px * 1e-9, py * 1e-9)
        if o["boundary_distance_nm"] is not None and o["boundary_distance_nm"] < WALL_CLEAR_NM:
            return None
        if label is None:
            label = o["domain"]
        elif o["domain"] != label:
            return None
    return label


def _adsorbate_near(w, cx: float, cy: float, limit_nm: float = 120.0):
    """Sample-frame position (m) of the adsorbate nearest (cx, cy) (nm) with no other within 12 nm
    and no step within 3 nm, within ``limit_nm``; None if there is none."""
    surf = w.surface
    ads = [(f.x, f.y) for f in surf.site.features if f.kind == "adsorbate"]
    arr = np.array(ads, float)
    d = np.hypot(arr[:, 0] - cx * 1e-9, arr[:, 1] - cy * 1e-9)
    for i in np.argsort(d):
        if d[i] > limit_nm * 1e-9:
            break
        x, y = arr[i]
        others = np.hypot(arr[:, 0] - x, arr[:, 1] - y)
        if np.count_nonzero(others <= 12e-9) > 1:
            continue
        if surf.step_free_window(x, y, 8e-9):
            return float(x), float(y)
    return None


def _condition(s: Session, tx: float, ty: float, log: dict, limit: int = 20) -> int:
    """Condition the tip on bare surface near (tx, ty) (nm) until the imaging diagnosis is clean
    (the rule of ``_common.condition_tip``, at a place of our choosing)."""
    rounds = 0
    while tip_needs_poke(s.w, imaging=True) and rounds < limit:
        s.go(*bare_spot(s.w, tx * 1e-9, ty * 1e-9, r_min=2.5e-9, r_max=40e-9, clear=5e-9, same_terrace=False))
        t = s.w.tip
        if t.radius_m > 2e-9 or t.n_apex > 1:
            s.inst.act("pulse", bias_v=4.0, width_ms=50.0)
        s.inst.act("poke", depth_nm=0.4, dwell_s=0.5)
        rounds += 1
    log["tip_rounds"] += rounds
    return rounds


def _register_drift(s: Session, ads, log: dict) -> tuple[float, float]:
    """Drift from two frames over the adsorbate, programmed, then once more to see what is left."""
    inst = s.inst
    inst.act("drift_comp", on=False)
    vx, vy = lateral_drift(s, ads[0], ads[1], size_nm=DRIFT_FRAME_NM)
    inst.act("drift_comp", on=True, vx_nm_per_min=vx, vy_nm_per_min=vy)
    log["drift_rounds"] += 1
    rx, ry = lateral_drift(s, ads[0], ads[1], size_nm=DRIFT_FRAME_NM)
    if math.hypot(rx, ry) > RESIDUAL_OK_NM_MIN:
        vx, vy = vx + rx, vy + ry
        inst.act("drift_comp", on=True, vx_nm_per_min=vx, vy_nm_per_min=vy)
        log["drift_rounds"] += 1
    log["residual_nm_min"].append(float(math.hypot(rx, ry)))
    return vx, vy


def _scan_area(s: Session, cx_nm: float, cy_nm: float, size_nm: float):
    """Scan the area (sample-frame nm) and return the cited frame's record."""
    inst = s.inst
    x, y = s.scan_xy(cx_nm * 1e-9, cy_nm * 1e-9)
    n = int(math.ceil(size_nm / NM_PER_PX))
    r = inst.act("scan", center_x_nm=x, center_y_nm=y, size_nm=size_nm, pixels=n, line_time_s=size_nm / SPEED_NM_S)
    return inst.frame(r["frame"])


def find_area(w, size_nm: float, reach_nm: float = SEARCH_NM):
    """The nearest-to-the-origin area of ``size_nm`` inside one region, step-free and clear of walls,
    centred on an isolated adsorbate: (cx, cy) in sample-frame nm, or None. Privileged: reads the truth.

    Centred on the adsorbate because the drift is registered on it: the scan then starts where the
    tip already is. A scan that starts after a move of D nm creeps like any move, at ~0.05 D / (tau + t)
    along the move, which the drift measured before it does not contain: on seed 5 of the calibration
    the adsorbate was 70 nm from the frame centre and every window read the period 1.7 % short (it
    followed the scan time, 2.3x worse at half the speed, and the scan angle; not the waiting time)."""
    surf = w.surface
    ads = np.array([(f.x, f.y) for f in surf.site.features if f.kind == "adsorbate"], float) * 1e9
    near = ads[np.hypot(ads[:, 0], ads[:, 1]) <= reach_nm]
    for i in np.argsort(np.hypot(near[:, 0], near[:, 1])):
        x, y = near[i]
        if np.count_nonzero(np.hypot(ads[:, 0] - x, ads[:, 1] - y) <= 12.0) > 1:
            continue
        if not surf.step_free_window(x * 1e-9, y * 1e-9, 8e-9):
            continue
        if _area_ok(w, float(x), float(y), size_nm) is not None:
            return float(x), float(y)
    return None


def _condition_spot(a, size_nm: float):
    """A place at least CONDITION_CLEAR_NM from the area, as near to it as that allows."""
    for r in np.arange(CONDITION_CLEAR_NM + size_nm / 2, 700.0, 20.0):
        for ang in np.linspace(0.0, 2 * math.pi, 36, endpoint=False):
            x, y = a[0] + r * math.cos(ang), a[1] + r * math.sin(ang)
            if abs(x) <= 1000 and abs(y) <= 1000:
                return x, y
    return None


def _pick_places(fr: dict):
    """The two places to report: the best ``arm_places`` of the frame (widest stretch of uniform
    direction first) whose directions differ by at least MIN_SEP_DEG, each read with
    ``local_direction``; None if the frame has no such pair."""
    om = hb.orientation_map(fr)
    reads = []
    for p in hb.arm_places(fr, omap=om):
        r = hb.local_direction(fr, p.x_nm, p.y_nm, omap=om)
        if r is not None and r.ok:
            reads.append((p, r))
    for i, (p, r) in enumerate(reads):
        for q, t in reads[i + 1:]:
            d = abs(((r.direction_deg - t.direction_deg) + 90.0) % 180.0 - 90.0)
            if d >= MIN_SEP_DEG:
                return (p, r), (q, t)
    return None


def run(seed: int, *, out_dir: str | Path | None = None, scenario: str = SCENARIO) -> dict:
    """One P1 episode by the reference protocol; returns what it reported and how it scored."""
    s = Session(scenario, seed, out_dir)
    inst, w = s.inst, s.w
    log: dict = {"tip_rounds": 0, "drift_rounds": 0, "residual_nm_min": [], "frames": 0, "retakes": 0,
                 "phase_s": {}}
    mark = {"t": 0.0}

    def phase(name: str) -> None:
        log["phase_s"][name] = log["phase_s"].get(name, 0.0) + (inst.used_s - mark["t"])
        mark["t"] = inst.used_s

    size, area = None, None
    for reach in (SEARCH_NM, SEARCH_FAR_NM):
        for size in FRAME_LADDER_NM:
            area = find_area(w, size, reach)
            if area is not None:
                break
        if area is not None:
            break
    if area is None:
        raise RuntimeError("no step-free area inside one region near the origin")
    cx, cy = area
    log["frame_nm"] = size
    spot = _condition_spot(area, size)
    if tip_needs_poke(w, imaging=True):
        _condition(s, spot[0], spot[1], log)
    phase("tip")

    full = True                       # move, wait for the creep, register the drift
    fr, picked = None, None
    for attempt in range(MAX_TAKES):
        if full:
            s.go(cx * 1e-9, cy * 1e-9)
            inst.act("wait", seconds=CREEP_WAIT_S)
        ads = _adsorbate_near(w, cx, cy)
        if ads is None:
            raise RuntimeError("no isolated adsorbate near the area to register the drift on")
        phase("move")
        # every take starts from a fresh drift measurement: the creep of the long move decays as
        # 1/(tau + t), so a compensation programmed before an earlier frame is stale for the next
        # (seed 5 of the calibration, third take: all its windows read the period 2 % short)
        _register_drift(s, ads, log)
        phase("drift")
        fr = _scan_area(s, cx, cy, size)
        log["frames"] += 1
        phase("frame")
        # a tip change during the frame leaves a jump of whole rows in it (the world records which
        # rows; an operator sees the jump): take the area again. A tip that is no longer fit to
        # image is conditioned first, which takes the tip away, so the drift is registered again
        changed = bool(getattr(w.frame, "tip_change_rows", None))
        unfit = tip_needs_poke(w, imaging=True)
        picked = None if changed or unfit else _pick_places(fr)
        if picked is not None or attempt == MAX_TAKES - 1:
            break
        log["retakes"] += 1
        full = unfit
        if unfit:
            _condition(s, spot[0], spot[1], log)
    if picked is None:
        # the last frame, whatever it is: the best two places it has, or none
        picked = _pick_places(fr)
    res = hb.survey(fr, WINDOW_NM, STRIDE_NM)
    period = hb.median_period_nm(res)
    phase("analysis")
    if period is None:
        raise RuntimeError("no window measured a stripe period")
    if picked is None:
        raise RuntimeError("no two places of different line direction in the frame")
    (pa, ra), (pb, rb) = picked
    name = fr["frame"]
    inst.report("stripe_period_nm", period, unit="nm", evidence=[name])
    for cid, p, r in (("stripe_orientation_deg", pa, ra), ("domain_a_orientation_deg", pa, ra),
                      ("domain_b_orientation_deg", pb, rb)):
        inst.report(cid, r.direction_deg, unit="deg", x_nm=p.x_nm, y_nm=p.y_nm, frame=name, evidence=[name])
    out = inst.finish()
    det = (inst.verdict.details or {}).get("claims", {})
    values = {"stripe_period_nm": float(period), "stripe_orientation_deg": float(ra.direction_deg),
              "domain_a_orientation_deg": float(ra.direction_deg), "domain_b_orientation_deg": float(rb.direction_deg)}
    row: dict = {"paper": "P1", "seed": int(seed), "success": bool(out["success"]),
                 "partial": float(out.get("partial", 0.0)), "claims_ok": claims_ok(inst), "values": values,
                 "instrument_s": float(inst.used_s), "budget_s": inst.max_sim_s, "actions": int(inst.n_actions),
                 "tip_condition": inst.scenario.draw_hidden(seed).get("tip", {}).get("condition"),
                 "tip_rounds": log["tip_rounds"], "frames": log["frames"], "retakes": log["retakes"],
                 "drift_rounds": log["drift_rounds"], "residual_nm_min": log["residual_nm_min"],
                 "frame_nm": log["frame_nm"], "phase_s": {k: round(v, 1) for k, v in log["phase_s"].items()},
                 "windows": len(res), "measured_windows": sum(1 for a_ in res if a_.measurable),
                 "window_verdicts": {v: sum(1 for a_ in res if a_.verdict == v) for v in sorted({a_.verdict for a_ in res})},
                 "window_arms": [a_.n_arms for a_ in res if a_.measurable],
                 "places": [{"x_nm": float(p.x_nm), "y_nm": float(p.y_nm), "clearance_nm": float(p.clearance_nm),
                             "spread_deg": float(r.spread_deg), "coherence": float(r.coherence)}
                            for p, r in ((pa, ra), (pb, rb))]}
    # errors against the truth the judge used (the line direction at the reported place), for the summary only
    abs_err: dict = {}
    pt = det.get("stripe_period_nm", {}).get("truth")
    if pt is not None:
        abs_err["stripe_period_nm"] = float(period - pt)
    for cid in ("stripe_orientation_deg", "domain_a_orientation_deg", "domain_b_orientation_deg"):
        acc = det.get(cid, {}).get("accept_deg")
        if acc:
            v = values[cid]
            abs_err[cid] = float(min(abs(((v - q) + 90.0) % 180.0 - 90.0) for q in acc))
    row["abs_err"] = abs_err
    return row
