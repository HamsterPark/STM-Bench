"""P1 privileged reference solution: the Au(111) 22x sqrt(3) herringbone, as Barth, Brune, Ertl and
Behm (1990) saw it: the stripe period, and the stripe direction in two rotational domains.

The truth navigates, never measures. It says where two step-free areas of two different rotational
domains lie (a footprint wholly inside one domain, away from its walls, on one terrace), where
adsorbates sit to register the drift on, and whether the tip needs conditioning. Every reported
number is read from frames the instrument acquired, through :mod:`stmbench.analysis.herringbone`
(period, arm and mean directions of 80 nm windows of those frames); the position of each
orientation is the centre of the window it was read in, in the scan coordinates of the frame
that is cited.

Protocol:

1. two areas of ``FRAME_NM`` (a square, a few nm of margin) that lie wholly in one domain, step
   free, at least ``WALL_CLEAR_NM`` from a wall, in two domains of different stripe direction
   (the truth's labels differ, so the stripe directions are 60 degrees apart), the nearest pair
   to the origin;
2. a tip that is blunt, multiple, unstable or carries something is conditioned on bare surface at
   least ``CONDITION_CLEAR_NM`` from both areas (a pulse sprays clusters ~40 nm around), five
   minutes for the creep of the long move;
3. per area: the tip goes to an adsorbate near its centre, two 12 nm frames 30 s apart give the
   drift (bare herringbone cannot: a grating constrains only the component across its stripes,
   modulo a period), the compensation is programmed and checked once more by the same two frames;
4. the area is scanned at ``NM_PER_PX`` nm/px and at ``SPEED_NM_S`` or slower: a faster scan with
   a tip of radius above 1.5 nm scratches the surface (``World.maybe_scratch``). A frame in which
   the tip needed conditioning again is taken again;
5. windows of ``WINDOW_NM`` on a ``STRIDE_NM`` grid over each frame are assessed. The period is the
   median of the along-arm period over every window that measured one, both frames together. The
   direction of each domain is that of the window with two resolved arms and the highest peak
   quality (the mean of the two arms; failing that, a single arm's), reported at the window's
   centre.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..analysis import herringbone as hb
from ._common import Session, bare_spot, claims_ok, lateral_drift, tip_needs_poke

SCENARIO = "P1_barth1990_au111"
#: the area scanned in each domain (nm): the period claim wants a frame of at least 100 nm; the
#: larger frames hold more windows and cost 4-5 minutes each at SPEED_NM_S
FRAME_NM = 140.0
FRAME_LADDER_NM = (140.0, 120.0, 110.0, 102.0)
#: 0.39 instead of the claim's 0.4 nm/px so that rounding the pixel count never exceeds the limit
NM_PER_PX = 0.39
#: under the 450 nm/s above which a tip of radius > 1.5 nm scratches the surface (world.maybe_scratch)
SPEED_NM_S = 400.0
WINDOW_NM, STRIDE_NM = 80.0, 20.0
#: what the footprint of a frame has to clear: the wall's own width is 2-5 nm and the drift moves
#: the frame by a few nm during it
WALL_CLEAR_NM = 12.0
CONDITION_CLEAR_NM = 170.0
#: how far from the origin to look for the areas: the near range first (a short move creeps less),
#: then the far one (within +-600 nm of the origin there are always two domains of different direction)
SEARCH_NM, SEARCH_FAR_NM, SEARCH_STEP_NM = 420.0, 700.0, 20.0
DRIFT_FRAME_NM = 12.0
#: one extra look at the drift after programming it; stop when the residual is below this (nm/min)
RESIDUAL_OK_NM_MIN = 0.15
CREEP_WAIT_S = 300.0


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


def find_areas(w, size_nm: float, reach_nm: float = SEARCH_NM):
    """The two nearest-to-the-origin areas of ``size_nm`` in different domains: ((cx, cy, label), (cx, cy, label))
    in sample-frame nm, or None."""
    found = []
    g = np.arange(-reach_nm, reach_nm + 1e-9, SEARCH_STEP_NM)
    for cx in g:
        for cy in g:
            lab = _area_ok(w, float(cx), float(cy), size_nm)
            if lab is not None:
                found.append((math.hypot(cx, cy), float(cx), float(cy), int(lab)))
    found.sort()
    if not found:
        return None
    a = found[0]
    # the second: another domain, the one nearest to the first area (a short move: less creep)
    others = [f for f in found if f[3] != a[3]]
    if not others:
        return None
    b = min(others, key=lambda f: math.hypot(f[1] - a[1], f[2] - a[2]))
    return (a[1], a[2], a[3]), (b[1], b[2], b[3])


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


def _condition_spot(a, b, size_nm: float):
    """A place at least CONDITION_CLEAR_NM from both areas, as near to them as that allows."""
    best = None
    for r in np.arange(CONDITION_CLEAR_NM, 700.0, 20.0):
        for ang in np.linspace(0.0, 2 * math.pi, 36, endpoint=False):
            mx, my = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
            x, y = mx + r * math.cos(ang), my + r * math.sin(ang)
            if abs(x) > 1000 or abs(y) > 1000:
                continue
            if min(math.hypot(x - a[0], y - a[1]), math.hypot(x - b[0], y - b[1])) < CONDITION_CLEAR_NM + size_nm / 2:
                continue
            return x, y
    return best


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


def _best_window(res):
    """The window whose direction to report: two resolved arms and the highest peak quality; else
    a single arm's."""
    ok = [a for a in res if a.verdict == "ok"]
    pool = ok or [a for a in res if a.verdict == "one_arm"]
    return max(pool, key=lambda a: a.quality) if pool else None


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

    size, areas = None, None
    for reach in (SEARCH_NM, SEARCH_FAR_NM):
        for size in FRAME_LADDER_NM:
            areas = find_areas(w, size, reach)
            if areas is not None:
                break
        if areas is not None:
            break
    if areas is None:
        raise RuntimeError("no two step-free areas of different domains near the origin")
    a, b = areas
    log["frame_nm"] = size
    spot = _condition_spot(a, b, size)
    if tip_needs_poke(w, imaging=True):
        _condition(s, spot[0], spot[1], log)
    phase("tip")

    frames, windows = [], []
    for area in (a, b):
        cx, cy, _label = area
        for attempt in range(3):
            s.go(cx * 1e-9, cy * 1e-9)
            inst.act("wait", seconds=CREEP_WAIT_S)
            ads = _adsorbate_near(w, cx, cy)
            if ads is None:
                raise RuntimeError("no isolated adsorbate near the area to register the drift on")
            phase("move")
            _register_drift(s, ads, log)
            phase("drift")
            fr = _scan_area(s, cx, cy, size)
            log["frames"] += 1
            phase("frame")
            if tip_needs_poke(w, imaging=True):
                # the tip changed during the frame: condition it and take the area again
                log["retakes"] += 1
                _condition(s, spot[0], spot[1], log)
                continue
            break
        res = hb.survey(fr, WINDOW_NM, STRIDE_NM)
        frames.append(fr)
        windows.append(res)
    phase("analysis")
    all_windows = [a_ for res in windows for a_ in res]
    period = hb.median_period_nm(all_windows)
    if period is None:
        raise RuntimeError("no window measured a stripe period")
    picks = [_best_window(res) for res in windows]
    if any(p is None for p in picks):
        raise RuntimeError("no window of a frame measured a stripe direction")
    names = [f["frame"] for f in frames]
    inst.report("stripe_period_nm", period, unit="nm", evidence=names)
    inst.report("stripe_orientation_deg", picks[0].direction_deg, unit="deg", x_nm=picks[0].centre_nm[0],
                y_nm=picks[0].centre_nm[1], frame=names[0], evidence=[names[0]])
    inst.report("domain_a_orientation_deg", picks[0].direction_deg, unit="deg", x_nm=picks[0].centre_nm[0],
                y_nm=picks[0].centre_nm[1], frame=names[0], evidence=[names[0]])
    inst.report("domain_b_orientation_deg", picks[1].direction_deg, unit="deg", x_nm=picks[1].centre_nm[0],
                y_nm=picks[1].centre_nm[1], frame=names[1], evidence=[names[1]])
    out = inst.finish()
    det = (inst.verdict.details or {}).get("claims", {})
    row: dict = {"paper": "P1", "seed": int(seed), "success": bool(out["success"]),
                 "partial": float(out.get("partial", 0.0)), "claims_ok": claims_ok(inst),
                 "values": {"stripe_period_nm": float(period),
                            "stripe_orientation_deg": float(picks[0].direction_deg),
                            "domain_a_orientation_deg": float(picks[0].direction_deg),
                            "domain_b_orientation_deg": float(picks[1].direction_deg)},
                 "instrument_s": float(inst.used_s), "budget_s": inst.max_sim_s, "actions": int(inst.n_actions),
                 "tip_condition": inst.scenario.draw_hidden(seed).get("tip", {}).get("condition"),
                 "tip_rounds": log["tip_rounds"], "frames": log["frames"], "retakes": log["retakes"],
                 "drift_rounds": log["drift_rounds"], "residual_nm_min": log["residual_nm_min"],
                 "frame_nm": log["frame_nm"], "phase_s": {k: round(v, 1) for k, v in log["phase_s"].items()},
                 "windows": [len(r) for r in windows],
                 "measured_windows": sum(1 for a_ in all_windows if a_.measurable),
                 "window_verdicts": {v: sum(1 for a_ in all_windows if a_.verdict == v)
                                     for v in sorted({a_.verdict for a_ in all_windows})},
                 "picked_verdicts": [p.verdict for p in picks]}
    # errors against the truth the judge used, for the summary only
    abs_err: dict = {}
    pt = det.get("stripe_period_nm", {}).get("truth")
    if pt is not None:
        abs_err["stripe_period_nm"] = float(period - pt)
    for cid in ("stripe_orientation_deg", "domain_a_orientation_deg", "domain_b_orientation_deg"):
        d = det.get(cid, {})
        acc = d.get("accept_deg")
        if acc:
            v = row["values"][cid]
            abs_err[cid] = float(min(abs(((v - q) + 90.0) % 180.0 - 90.0) for q in acc))
    row["abs_err"] = abs_err
    return row
