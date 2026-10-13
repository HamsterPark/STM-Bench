"""The tip's state, read off a topograph of adatoms: what an operator sees before trusting a tip
with a manipulation.

* A **blunt** tip widens every atom and lowers it. On P4's rendered frames (48 seeds, a 30 nm
  frame each) good tips gave atoms σ 0.35-0.42 nm and 36-65 pm high (over a 2 nm running
  median); tips of 3.3-4.9 nm radius 0.47-0.54 nm and 18-27 pm, and blunter ones left no
  atom standing above the detector's 20 pm at all.
* A **double** tip images every atom twice, the copy at the same displacement and weaker: on
  6 of 8 double-tip seeds at least half the atoms had a partner at one displacement, on a
  seventh 0.3 of ten (good tips at most 0.17, chance pairs). A copy closer than ~0.8 nm merges
  into its atom and widens or stretches it.
* An **unstable** tip changes now and then; a change steps every row after it
  (:func:`stmbench.analysis.adatoms.row_jumps`), so it shows over several frames, not in one.
"""
from __future__ import annotations

import math

import numpy as np

from .adatoms import bumps, row_jumps

#: σ of an atom (nm) above which the tip is taken for blunt; good tips stay below 0.43
BLUNT_SIGMA_NM = 0.45
#: atoms (over a 2 nm running median, pm) lower than this: blunt as well
BLUNT_HEIGHT_PM = 30.0
#: a double tip: at least this fraction of the atoms has a weaker partner at one displacement
#: (0.28 once six atoms or more are seen, when a chance pair weighs less)
DOUBLE_FRACTION, DOUBLE_FRACTION_MANY = 0.4, 0.28
#: atoms stretched beyond this axis ratio, all the same way: a copy too close to separate
STRETCH_RATIO = 1.4


def atom_shapes(frame: dict, *, isolation_nm: float = 2.0) -> list[dict]:
    """Elliptical Gaussians fitted to the isolated bumps of an instrument frame:
    ``[{x_nm, y_nm (frame coordinates), height_m, sigma_nm (geometric mean of the axes),
    ratio (long/short), angle_rad}]``."""
    from scipy.ndimage import map_coordinates
    from scipy.optimize import curve_fit

    z = np.asarray(frame["array"], float)
    npx = float(frame["nm_per_px"])
    found, hp = bumps(z, npx)
    ny = hp.shape[0]
    pts = [((c + 0.5) * npx, (ny - r - 0.5) * npx, h) for c, r, h in found]
    t = np.arange(-1.0, 1.0 + 1e-9, npx)
    out = []
    for x0, y0, h in pts:
        if any(math.hypot(x0 - q[0], y0 - q[1]) < isolation_nm for q in pts if q[:2] != (x0, y0)):
            continue
        X, Y = np.meshgrid(x0 + t, y0 + t)
        v = map_coordinates(hp, [(ny - Y / npx - 0.5).ravel(), (X / npx - 0.5).ravel()], order=1, mode="nearest")

        def g(xy, a, xc, yc, s1, s2, th, b):
            c, s = math.cos(th), math.sin(th)
            u = (xy[0] - xc) * c + (xy[1] - yc) * s
            w = -(xy[0] - xc) * s + (xy[1] - yc) * c
            return a * np.exp(-u * u / (2 * s1 * s1) - w * w / (2 * s2 * s2)) + b

        try:
            p, _ = curve_fit(g, (X.ravel(), Y.ravel()), v, p0=(h, x0, y0, 0.4, 0.4, 0.0, 0.0), maxfev=4000)
        except (RuntimeError, ValueError):
            continue
        s1, s2 = abs(p[3]), abs(p[4])
        if not (0.1 < s1 < 2.0 and 0.1 < s2 < 2.0):
            continue
        out.append({"x_nm": float(p[1]), "y_nm": float(p[2]), "height_m": float(p[0]),
                    "sigma_nm": float(math.sqrt(s1 * s2)), "ratio": float(max(s1, s2) / min(s1, s2)),
                    "angle_rad": float(p[5] if s1 >= s2 else p[5] + math.pi / 2) % math.pi})
    return out


def pair_vector(frame: dict, *, min_nm: float = 0.6, max_nm: float = 4.5, within_nm: float = 0.25):
    """The most common displacement from an atom to a weaker bump near it, and the fraction of
    atoms that have one there: ``((dx, dy, height ratio), fraction, atoms)``. Weak bumps are
    searched down to 4 pm, as a double tip's copy can be faint."""
    z = np.asarray(frame["array"], float)
    npx = float(frame["nm_per_px"])
    strong, _ = bumps(z, npx)
    weak, hp = bumps(z, npx, min_height_m=4e-12, compact_frac=0.95)
    ny = hp.shape[0]
    a = [((c + 0.5) * npx, (ny - r - 0.5) * npx, h) for c, r, h in strong]
    b = [((c + 0.5) * npx, (ny - r - 0.5) * npx, h) for c, r, h in weak]
    if len(a) < 3:
        return None, 0.0, len(a)
    vecs = []                                  # (atom index, dx, dy, ratio)
    for i, (x0, y0, h0) in enumerate(a):
        for x1, y1, h1 in b:
            d = math.hypot(x1 - x0, y1 - y0)
            if min_nm < d < max_nm and h1 < h0:
                vecs.append((i, x1 - x0, y1 - y0, h1 / h0))
    best, best_n = None, 0
    for _, dx, dy, r in vecs:
        n = len({i for i, ex, ey, _ in vecs if math.hypot(ex - dx, ey - dy) < within_nm})
        if n > best_n:
            best, best_n = (dx, dy, r), n
    return best, best_n / len(a), len(a)


def diagnose(frame: dict) -> dict:
    """The tip as one frame shows it: ``{"verdict": "ok" | "blunt" | "double" | "unknown",
    ...measurements}``. ``unknown``: too few atoms to judge — on a frame where a few atoms are
    expected, itself the sign of a tip blunt enough to smear them below detection."""
    shapes = atom_shapes(frame)
    vec, frac, n = pair_vector(frame)
    out = {"atoms": n, "isolated": len(shapes), "ghost_fraction": float(frac),
           "ghost": None if vec is None else [round(float(v), 3) for v in vec],
           "jumps": len(row_jumps(frame["array"]))}
    if shapes:
        out["sigma_nm"] = float(np.median([s["sigma_nm"] for s in shapes]))
        out["height_pm"] = float(np.median([s["height_m"] for s in shapes])) * 1e12
        out["ratio"] = float(np.median([s["ratio"] for s in shapes]))
    if (n >= 3 and frac >= DOUBLE_FRACTION) or (n >= 6 and frac >= DOUBLE_FRACTION_MANY):
        out["verdict"] = "double"
    elif shapes and (out["sigma_nm"] > BLUNT_SIGMA_NM or out["height_pm"] < BLUNT_HEIGHT_PM):
        out["verdict"] = "blunt"
    elif len(shapes) >= 3 and out["ratio"] > STRETCH_RATIO:
        out["verdict"] = "double"
    elif not shapes:
        out["verdict"] = "unknown"
    else:
        out["verdict"] = "ok"
    return out
