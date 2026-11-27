"""P5 scripted baseline: Sader–Jarvis force spectroscopy over one Fe atom with nothing but what the
instrument shows — no truth.

The measurement is the reference's (:mod:`stmbench.reference.p5`): drift compensated laterally and
in Z, the qPlus oscillation on, Δf(z) curves alternating bare / atom / bare … ending on bare, the
bare ones on two spots opposite each other across the atom, medians of the repeats, background
subtracted at equal loop Z, inverted (:func:`stmbench.analysis.force.p5_numbers`). What the
reference reads from the truth, this reads from frames and curves:

* **the tip**: a 40 nm survey at home judges it (:func:`stmbench.analysis.tip.diagnose`); three
  shallow pokes 60 nm away come first whatever it shows (an unstable or dirty tip does not show in
  one frame, and a shallow contact only stabilises and cleans an apex); a blunt or double tip is
  pulsed 150 nm away and judged again, as P4's and P7's baselines do;
* **the drift**: two frames 30 s apart over the survey's atoms, the compensation off while they are
  taken (:meth:`stmbench.baselines._common.Ops.drift`); the Z drift is measured over bare surface;
* **which atom**: a bump of a fresh survey with no other bump within 3 nm, the terrace flat around it
  (no step: the smoothed topography does not step by more than 100 pm), 5 nm inside the frame, and
  room for two bare spots on opposite sides of it, each 2.5 nm from every bump on the same terrace;
* **where the tip is**: a small fresh frame over the atom before every curve over it (the evidence
  rule wants the tip within 0.15 nm of the atom, the bond fades as a Gaussian of the lateral
  distance), the tip then moved to the bump it shows, and the bare spots moved by the same shift;
  a curve over the atom that shows no short-range dip against the bare curves is taken again after a
  fresh frame.

Constants of the curves (offsets, sweeps, 100 pm amplitude, 512 points, four sweeps) are the
reference's: they were calibrated on the reference's runs, not read from any world.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..analysis.adatoms import locate
from ..analysis.force import Curve, combine, p5_numbers
from ..analysis.tip import diagnose
from ..instrument import Instrument
from ._common import JUDGE_NM, JUDGE_PX, Blind, Ops

SCENARIO = "P5_sader_jarvis_qplus"
LINE_S = 0.1
#: survey for the site: 36 nm at 0.1 nm/px (the scale the tip is judged at) around home or a neighbour of it
SITE_NM, SITE_PX = 36.0, 360
#: places to look for a site: home, then four neighbours 24 nm away (the field is uniform)
SEARCH = ((0.0, 0.0), (24.0, 0.0), (-24.0, 0.0), (0.0, 24.0), (0.0, -24.0))
#: no other bump within this of the atom; bare spots this far from every bump (the judge wants more
#: than 2 nm from anything that scatters, steps aside)
ISOLATION_NM, BARE_CLEAR_NM = 3.0, 2.5
#: the smoothed topography over the atom and its spots may not range by more than this: a step on
#: Cu(111) is 208 pm, a bare terrace is flat to ~10 pm after the rows are levelled
STEP_PM = 100.0
#: curves over the atom: start this far above the feedback height, sweep this far (nm); over bare
#: surface wider, so that they span the atom's curves in absolute Z (the feedback height over the
#: atom is higher by the atom's height). The reference's, calibrated on its runs
ATOM_OFFSET_NM, ATOM_SWEEP_NM = 0.75, 0.95
BG_OFFSET_NM, BG_SWEEP_NM = 0.95, 1.10
POINTS, SWEEPS, REPEATS, AMPLITUDE_PM = 512, 4, 3, 100.0
#: a curve over the atom must run below the bare curves, at equal height, by more than this (Hz) and
#: by more than this many times the scatter of the difference far from the surface (see _dip_hz): a
#: physisorbed bond is a Hz deep or less
MIN_DIP_HZ, MIN_DIP_SIGMAS = 0.3, 6.0
#: frame that places the tip on the atom
FINE_NM, FINE_PX, FINE_LINE_S = 5.0, 50, 0.05
#: the grid (nm, both axes) of coarse curves that finds the bond's centre, and their points
CENTRE_GRID, CENTRE_POINTS = tuple(np.arange(-0.24, 0.241, 0.08)), 96
#: the fitted centre is used when the fit puts it within this (nm, one standard error from the fit's
#: Jacobian and residuals) in each direction, else the image's place stands: a bond too weak to
#: see in the grid (a physisorbed one, ~0.5 Hz deep) is fitted to the bounds
CENTRE_MAX_SE_NM = 0.03
#: sweeps averaged per point of the coarse grid: one, then eight for a bond too shallow for one
CENTRE_SWEEPS = (1, 8)
CENTRE_FINE = tuple(np.arange(-0.12, 0.121, 0.06))
SPOT_SHIFT_MAX_NM = 1.5
#: how long the atom is watched to find the drift the survey's measurement left
RETUNE_WAIT_S = 60.0


# ── analysis helpers; the two Z-drift routines are copied from stmbench/reference/_common.py ──
# (instrument actions only; the reference package is privileged and is not imported here)

def z_rate_pm_per_min(v: Blind, seconds: float = 60.0) -> float:
    """The loop's Z drift with the feedback holding the setpoint."""
    z0 = v.status()["z_nm"]
    v.act("wait", seconds=seconds)
    return (v.status()["z_nm"] - z0) * 1e3 / (seconds / 60.0)


def compensate_z_drift(v: Blind, vx: float, vy: float) -> float:
    """With the lateral compensation running, measure the Z drift, program it, measure again; keep
    the sign that cancels (the documented "measure again" step). Returns the programmed rate
    (pm/min). The lateral drift has to be stopped first: a tip sliding over a tilted surface reads
    as Z drift."""
    v.act("wait", seconds=5.0)
    rate = z_rate_pm_per_min(v)
    best = (abs(rate), 0.0)
    for sign in (-1.0, 1.0):
        vz = sign * rate
        v.act("drift_comp", on=True, vx_nm_per_min=vx, vy_nm_per_min=vy, vz_pm_per_min=vz)
        left = abs(z_rate_pm_per_min(v))
        if left < best[0]:
            best = (left, vz)
        if left < 0.2 * abs(rate):
            break
    v.act("drift_comp", on=True, vx_nm_per_min=vx, vy_nm_per_min=vy, vz_pm_per_min=best[1])
    return best[1]


def terrace_map(frame: dict, *, size_nm: float = 2.0) -> np.ndarray:
    """The topography with the rows' offsets taken out (a tip change steps every row after it) and
    a running median over ``size_nm``: atoms (40–100 pm over 0.4 nm) drop out, a step (208 pm) stays."""
    from scipy.ndimage import median_filter

    z = np.asarray(frame["array"], float)
    z = np.where(np.isfinite(z), z, np.nanmedian(z))
    z = z - np.concatenate([[0.0], np.cumsum(np.median(np.diff(z, axis=0), axis=1))])[:, None]
    n = max(5, int(round(size_nm / float(frame["nm_per_px"]))) | 1)
    return median_filter(z, size=n, mode="nearest")


def _px(frame: dict, x: float, y: float) -> tuple[float, float]:
    """(col, row) of scan coordinates (nm) in an unrotated frame (:func:`to_scan_nm`'s inverse)."""
    cx, cy = (float(t) for t in frame["center_nm"])
    size, n = float(frame["size_nm"]), frame["array"].shape
    return ((x - cx) / size + 0.5) * n[1] - 0.5, (0.5 - (y - cy) / size) * n[0] - 0.5


def flat_around(frame: dict, tmap: np.ndarray, x: float, y: float, r_nm: float) -> bool:
    """Is the terrace map within ``STEP_PM`` over the disc of radius ``r_nm`` around (x, y), and the
    disc inside the frame?"""
    npx = float(frame["nm_per_px"])
    col, row = _px(frame, x, y)
    r = r_nm / npx
    ny, nx = tmap.shape
    if col - r < 0 or row - r < 0 or col + r > nx - 1 or row + r > ny - 1:
        return False
    yy, xx = np.mgrid[int(row - r):int(row + r) + 1, int(col - r):int(col + r) + 1]
    m = (xx - col) ** 2 + (yy - row) ** 2 <= r * r
    sel = tmap[yy[m], xx[m]]
    return bool(float(sel.max() - sel.min()) * 1e12 < STEP_PM)


def pick_site(frame: dict, *, edge_nm: float = 4.0) -> list[dict]:
    """The sites of a survey for a measurement, nearest the frame's centre first: ``{atom, spots}``
    (scan coordinates, nm). An atom with no other bump (compact or not) within ``ISOLATION_NM``,
    ``edge_nm`` inside the frame, a flat terrace around it and its spots, and a pair of spots on
    opposite sides of it at the same distance, each at least ``BARE_CLEAR_NM`` from every bump."""
    atoms = locate(frame)
    bumps_all = np.array([(x, y) for x, y, _ in locate(frame, compact_frac=1.0)], float).reshape(-1, 2)
    tmap = terrace_map(frame)
    cx, cy = (float(t) for t in frame["center_nm"])
    half = float(frame["size_nm"]) / 2.0
    out = []
    for x, y, _h in atoms:
        if abs(x - cx) > half - edge_nm or abs(y - cy) > half - edge_nm:
            continue
        d = np.hypot(bumps_all[:, 0] - x, bumps_all[:, 1] - y)
        if np.any((d > 0.5) & (d < ISOLATION_NM)):            # the atom itself is within 0.5 nm of its own bump
            continue
        pair = None
        for r in np.arange(2.5, 9.01, 0.5):
            if not flat_around(frame, tmap, x, y, r + 2.0):
                break
            for ang in np.radians(np.arange(0.0, 180.0, 15.0)):
                c, s = r * math.cos(ang), r * math.sin(ang)
                a, b = (x + c, y + s), (x - c, y - s)
                if all(abs(p[0] - cx) <= half - 1.0 and abs(p[1] - cy) <= half - 1.0 for p in (a, b)) and all(
                        float(np.min(np.hypot(bumps_all[:, 0] - p[0], bumps_all[:, 1] - p[1]))) >= BARE_CLEAR_NM
                        for p in (a, b)):
                    pair = (a, b)
                    break
            if pair:
                break
        if pair:
            out.append({"atom": (float(x), float(y)), "spots": pair})
    out.sort(key=lambda s: math.hypot(s["atom"][0] - cx, s["atom"][1] - cy))
    return out


class Rig:
    """The qPlus measurement at one site."""

    def __init__(self, v: Blind, ops: Ops, log: dict):
        self.v, self.ops, self.log = v, ops, log

    def atom_here(self, guess: tuple[float, float], *, size: float = FINE_NM) -> tuple[float, float, float] | None:
        """Where a fresh small frame puts the atom nearest ``guess`` (scan nm) and the instrument time
        of its middle row; None if it shows none within 1.5 nm of it."""
        px = int(round(size / 0.1))
        f, t_mid = self.ops.frame(guess[0], guess[1], size, px, FINE_LINE_S)
        # the Z loop lags the surface along the fast axis, so an atom shows displaced along it, the
        # other way in the backward image (0.05 nm in the first runs, all in one direction, which
        # costs a few % of the bond's strength); the mean of the two is where the atom is
        b = dict(f)
        b["array"] = self.v.frame(f["frame"], direction="backward")["array"]
        seen = []
        for img in (f, b):
            near = [(math.hypot(x - guess[0], y - guess[1]), x, y) for x, y, _h in locate(img)]
            near = [t for t in near if t[0] <= 1.5]
            if not near:
                return None
            _, x, y = min(near)
            seen.append((x, y))
        if math.hypot(seen[0][0] - seen[1][0], seen[0][1] - seen[1][1]) > 0.5:
            return None                                  # not the same bump: a tip change or a neighbour
        return float(np.mean([p[0] for p in seen])), float(np.mean([p[1] for p in seen])), float(t_mid)

    def _depths(self, here, offsets, sweeps: int, bare: Curve) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Coarse Δf(z) at every point of a grid around ``here``: ``(tip positions, depth, times)``, the
        depth being how far the curve runs below the bare curve ``bare`` at equal loop Z (the analysis's
        own background subtraction, which takes the van der Waals part and the atom's height out of
        the comparison), at its deepest, lightly smoothed."""
        from scipy.signal import savgol_filter

        pts, depth, times = [], [], []
        for gx in offsets:
            for gy in offsets:
                self.ops.go(here[0] + gx, here[1] + gy)
                r = self.v.act("zspec", sweep_nm=ATOM_SWEEP_NM, z_offset_nm=ATOM_OFFSET_NM, points=CENTRE_POINTS,
                               sweeps=sweeps, backward=False)
                sp = self.v.spectrum(r["spectrum"])
                h, df = Curve.from_spectrum(sp).height()
                hb, _ = bare.height()
                lo, hi = max(float(h[0]), float(hb[0])), min(float(h[-1]), float(hb[-1]))
                if hi - lo > 0.2e-9 and h.size >= 40:
                    grid = np.linspace(lo, hi, 150)
                    diff = np.interp(grid, h, df) - combine([bare], grid)
                    ok = np.isfinite(diff)
                    if ok.sum() >= 40:
                        pts.append((here[0] + gx, here[1] + gy))
                        depth.append(float(np.min(savgol_filter(diff[ok], 9, 2))))
                        times.append(float(sp["t_s"]))
        return np.asarray(pts), np.asarray(depth), np.asarray(times)

    def centre(self, first: tuple[float, float, float], bare: Curve) -> tuple[float, float] | None:
        """How far the bond's centre is from where an image puts the atom (scan nm), found with the
        force itself. ``first`` is the atom from a frame (x, y, time), ``bare`` a curve over bare surface.
        A coarse Δf(z) at every point of a grid, its depth against the bare curve compared across the
        grid, a Gaussian fitted to it; then again on a
        finer grid around the first estimate with the width held. The sample drifts a little while
        the grid is taken (the compensation leaves some), so the atom is modelled as moving at the
        velocity two frames around the grid give. The image places the atom for the tip's leading
        apex; the bond is the tip body's, and a double tip with its second apex a fraction of a
        nanometre away (it images as one atom) is off by that much. None when no fit finds a peak
        inside its grid."""
        from scipy.optimize import least_squares

        def fit(p, d, t, ref, vel, sigma):
            """(c, A, bx, by[, sigma]): d = c + A exp(-|p - (ref + vel (t - t_ref) + b)|^2 / 2 sigma^2)."""
            def model(q):
                sg = q[4] if sigma is None else sigma
                cx = ref[0] + vel[0] * (t - ref[2]) + q[2]
                cy = ref[1] + vel[1] * (t - ref[2]) + q[3]
                return q[0] + q[1] * np.exp(-0.5 * ((p[:, 0] - cx) ** 2 + (p[:, 1] - cy) ** 2) / sg ** 2)
            best = None
            for x0, y0 in ((0.0, 0.0), (0.1, 0.1), (-0.1, -0.1), (0.1, -0.1), (-0.1, 0.1)):
                q0 = [float(np.max(d)), float(np.min(d) - np.max(d)), x0, y0, 0.15 if sigma is None else sigma]
                lo = [-np.inf, -np.inf, -0.35, -0.35, 0.06 if sigma is None else sigma - 1e-9]
                hi = [np.inf, 0.0, 0.35, 0.35, 0.4 if sigma is None else sigma + 1e-9]
                f = least_squares(lambda q: model(q) - d, q0, bounds=(lo, hi))
                if best is None or f.cost < best.cost:
                    best = f
            return best

        def se_xy(f, free):
            """Standard errors (nm) of the centre's two coordinates in a fit with ``free`` parameters."""
            j = f.jac[:, free]
            dof = max(len(f.fun) - len(free), 1)
            cov = np.linalg.pinv(j.T @ j) * (2.0 * f.cost / dof)
            return float(np.sqrt(cov[2, 2])), float(np.sqrt(cov[3, 3]))

        def moved(a, b):
            dt = max(b[2] - a[2], 1.0)
            return ((b[0] - a[0]) / dt, (b[1] - a[1]) / dt)

        for sweeps in CENTRE_SWEEPS:
            p, d, t = self._depths(first[:2], CENTRE_GRID, sweeps, bare)
            if len(p) < 20:
                return None
            second = self.atom_here(first[:2])
            if second is None:
                return None
            f1 = fit(p, d, t, first, moved(first, second), None)
            rms1, q1 = float(np.sqrt(2 * f1.cost / len(d))), f1.x
            se1 = se_xy(f1, [0, 1, 2, 3, 4])
            rec = {"sweeps": sweeps, "bx": round(float(q1[2]), 3), "by": round(float(q1[3]), 3), "sigma": round(float(q1[4]), 3),
                   "span_hz": round(float(np.max(d) - np.min(d)), 2), "rms_hz": round(rms1, 3),
                   "se_nm": [round(se1[0], 3), round(se1[1], 3)],
                   "v_nm_min": [round(60.0 * c, 3) for c in moved(first, second)]}
            self.log.setdefault("centre", []).append(rec)
            if max(se1) <= CENTRE_MAX_SE_NM:
                break
            # a bond that is there (a well, deeper than the scatter) but too shallow for one sweep a
            # point: average more before giving up on the force
            if not (q1[1] < 0.0 and float(np.max(d) - np.min(d)) > 2.5 * rms1):
                return None
        if abs(q1[2]) > 0.3 or abs(q1[3]) > 0.3 or q1[1] >= 0.0 or max(se1) > CENTRE_MAX_SE_NM:
            return None
        mid = (second[0] + float(q1[2]), second[1] + float(q1[3]))
        p2, d2, t2 = self._depths(mid, CENTRE_FINE, 2 * sweeps, bare)
        third = self.atom_here(second[:2])
        if len(p2) < 15 or third is None:
            return float(q1[2]), float(q1[3])
        f2 = fit(p2, d2, t2, second, moved(second, third), float(q1[4]))
        q2 = f2.x
        rec.update(fine_bx=round(float(q2[2]), 3), fine_by=round(float(q2[3]), 3),
                   fine_rms_hz=round(float(np.sqrt(2 * f2.cost / len(d2))), 3))
        if abs(q2[2] - q1[2]) > 0.2 or abs(q2[3] - q1[3]) > 0.2 or max(se_xy(f2, [0, 1, 2, 3])) > CENTRE_MAX_SE_NM:
            return float(q1[2]), float(q1[3])
        return float(q2[2]), float(q2[3])

    def curve(self, offset_nm: float, sweep_nm: float) -> tuple[str, dict]:
        r = self.v.act("zspec", sweep_nm=sweep_nm, z_offset_nm=offset_nm, points=POINTS,
                       sweeps=SWEEPS, backward=False)
        return r["spectrum"], self.v.spectrum(r["spectrum"])


def _dip_hz(atom: Curve, bare: list[Curve]) -> tuple[float, float]:
    """``(dip, noise)`` in Hz: how far the atom's curve runs below the bare curves, in Δf at equal loop Z
    (the short-range attraction the tip over an atom adds; a tip beside it adds less), and the scatter
    of that difference over the farthest quarter of the heights, where the bond is negligible."""
    if not bare:
        return float("nan"), float("nan")
    h, df = atom.height()
    grid = np.linspace(h[0], h[-1], 200)
    diff = np.interp(grid, h, df) - combine(bare, grid)
    ok = np.isfinite(diff)
    if not ok.any():
        return float("nan"), float("nan")
    far = diff[ok][-50:]
    noise = float(np.std(far - np.polyval(np.polyfit(np.arange(far.size), far, 1), np.arange(far.size))))
    return float(np.min(diff[ok])), noise


def play(v: Blind, log: dict) -> dict:
    """One P5 session through ``v``: ``{"atom": [Curve], "bare": [Curve], "names": [...], "amp_m": float}``."""
    ops = Ops(v, log)
    # the tip: judged on a survey, three shallow pokes far away, a blunt or double tip reshaped
    f, _ = ops.frame(*ops.home, JUDGE_NM, JUDGE_PX, LINE_S)
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
    site = None
    for attempt in range(3):
        # the drift, measured over the atoms the survey shows
        f, _ = ops.frame(*ops.home, JUDGE_NM, JUDGE_PX, LINE_S)
        atoms = locate(f)
        ax0, ay0 = (float(np.median([a[0] for a in atoms])), float(np.median([a[1] for a in atoms]))) if atoms else ops.home
        vx, vy = ops.drift(ax0, ay0, size_nm=20.0)
        log["drift_nm_per_min"] = [round(vx, 3), round(vy, 3)]
        # the site, from a fresh survey with the compensation running; a survey that shows a blunt
        # or double tip sends the tip back to conditioning
        bad = False
        for k, (dx, dy) in enumerate(SEARCH):
            f, _ = ops.frame(ops.home[0] + dx, ops.home[1] + dy, SITE_NM, SITE_PX, LINE_S)
            if diagnose(f)["verdict"] in ("blunt", "double"):
                bad = True
                break
            sites = pick_site(f)
            if sites:
                site = sites[0]
                log["site_search"] = k
                break
        if site is not None:
            break
        if bad:
            log["reconditioned"] = log.get("reconditioned", 0) + 1
            diag, f = ops.condition()
            if diag["verdict"] != "ok":
                raise RuntimeError("the tip could not be conditioned")
            v.act("wait", seconds=300.0)
        else:
            break
    if site is None:
        raise RuntimeError("no isolated atom with bare surface on both sides was found")
    rig = Rig(v, ops, log)
    atom, spots = site["atom"], [tuple(p) for p in site["spots"]]
    # the drift measured from the survey's atoms (often a single one) leaves some: two small frames
    # of the atom 60 s apart under the compensation give what is left, which is added to it
    a0 = rig.atom_here(atom)
    if a0 is None:
        raise RuntimeError("the atom is not where the survey put it")
    v.act("wait", seconds=RETUNE_WAIT_S)
    a1 = rig.atom_here(a0[:2])
    if a1 is None:
        raise RuntimeError("the atom is not where the first frame put it")
    dt_min = (a1[2] - a0[2]) / 60.0
    rx, ry = (a1[0] - a0[0]) / dt_min, (a1[1] - a0[1]) / dt_min
    vx, vy = vx + rx, vy + ry
    v.act("drift_comp", on=True, vx_nm_per_min=vx, vy_nm_per_min=vy)
    log["drift_residual_nm_per_min"] = [round(rx, 3), round(ry, 3)]
    spots = [(p[0] + a1[0] - atom[0], p[1] + a1[1] - atom[1]) for p in spots]
    atom = (a1[0], a1[1])
    # the Z drift over bare surface, then the oscillation
    ops.go(*spots[0])
    vz = compensate_z_drift(v, vx, vy)
    log["vz_pm_per_min"] = round(vz, 3)
    v.act("pll", on=True, amplitude_pm=AMPLITUDE_PM)
    atom_c, bare_c, names = [], [], []
    amp_m = AMPLITUDE_PM * 1e-12
    shift = (0.0, 0.0)
    delta = None                                         # bond centre minus the atom's image, found once
    for i in range(REPEATS + 1):
        sx, sy = spots[i % 2]
        ops.go(sx + shift[0], sy + shift[1])
        n, sp = rig.curve(BG_OFFSET_NM, BG_SWEEP_NM)
        bare_c.append(Curve.from_spectrum(sp))
        names.append(n)
        if i == REPEATS:
            break
        for attempt in range(3):
            here = rig.atom_here((atom[0] + shift[0], atom[1] + shift[1]))
            if here is None:
                log["atom_not_seen"] = log.get("atom_not_seen", 0) + 1
                raise RuntimeError("the atom is not in the frame centred on it")
            new = (here[0] - atom[0], here[1] - atom[1])
            if math.hypot(*new) > SPOT_SHIFT_MAX_NM:
                raise RuntimeError("the atom moved more than the spots can follow")
            shift = new
            if delta is None:
                delta = rig.centre(here, bare_c[0]) or (0.0, 0.0)
                log["centre_delta_nm"] = [round(delta[0], 3), round(delta[1], 3)]
                here = rig.atom_here(here[:2])
                if here is None:
                    raise RuntimeError("the atom is not in the frame centred on it")
                shift = (here[0] - atom[0], here[1] - atom[1])
            ops.go(here[0] + delta[0], here[1] + delta[1])
            n, sp = rig.curve(ATOM_OFFSET_NM, ATOM_SWEEP_NM)
            c = Curve.from_spectrum(sp)
            names.append(n)
            amp_m = float(np.nanmedian(sp["amp"]))
            dip, noise = _dip_hz(c, bare_c)
            if dip <= -max(MIN_DIP_HZ, MIN_DIP_SIGMAS * noise):
                atom_c.append(c)
                break
            log["retries"] = log.get("retries", 0) + 1
        else:
            raise RuntimeError("no short-range dip over the atom in three tries")
    log["curves"] = len(names)
    return {"atom": atom_c, "bare": bare_c, "names": names, "amp_m": amp_m}


UNITS = {"f_min_pn": "pN", "f_decay_pm": "pm", "e_bind_mev": "meV"}


def run(seed: int, *, out_dir: str | Path | None = None, scenario: str = SCENARIO) -> dict:
    """One P5 episode by the baseline; what it reported and how the judge scored it (the truth is
    read after the episode only, for the summary row: ``draw_hidden`` (the tip category), the judge's record
    and the world's zspec records are read after ``finish()`` and never feed back into a decision)."""
    inst = Instrument(scenario, seed, out_dir)
    log: dict = {}
    got, values, error = None, None, None
    try:
        got = play(Blind(inst), log)
        q = inst.info()["qplus"]
        values = p5_numbers(got["atom"], got["bare"], f0=q["f0_hz"], k=q["k_n_per_m"], amplitude=got["amp_m"])
    except Exception as exc:  # noqa: BLE001 — a failed session is a result
        error = f"{type(exc).__name__}: {exc}"
    if values is not None:
        for cid, unit in UNITS.items():
            inst.report(cid, float(values[cid]), unit=unit, evidence=got["names"])
    out = inst.finish()
    det = (inst.verdict.details or {}).get("claims", {})
    row = {"paper": "P5", "seed": int(seed), "success": bool(out["success"]), "partial": float(out.get("partial", 0.0)),
           "error": error,
           "claims_ok": {k: bool(c.get("value_ok")) and bool(c.get("evidence_ok", True)) for k, c in det.items()},
           "instrument_s": float(inst.used_s), "budget_s": float(inst.max_sim_s),
           "actions": int(out.get("actions", inst.n_actions)),
           "tip_condition": inst.scenario.draw_hidden(seed).get("tip", {}).get("condition"), **log}
    if values is not None:
        truth = {cid: float(det[cid]["truth"]) for cid in UNITS if cid in det and det[cid].get("truth") is not None}
        row["values"] = {k: float(values[k]) for k in UNITS}
        row["truth"] = truth
        row["rel_err"] = {k: float(values[k] / truth[k] - 1.0) for k in UNITS if k in truth}
        cited = set(got["names"])
        row["near_nm"] = [round(float(r["near_nm"]), 3) for r in inst.world.zspec_records
                          if r.get("path") and Path(r["path"]).name in cited]
    return row
