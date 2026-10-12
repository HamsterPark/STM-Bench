"""A quantum corral's geometry from a topograph: the circle through its ring of atoms.

The ring's atoms sit ~1 nm apart, so along the ring they read as a ridge, and a detector of
isolated adatoms (``stmbench.papers._clusters``, which rejects anything that stays high 0.7 nm
away in some direction) finds none of them. The fit here takes the bright pixels instead.
"""
from __future__ import annotations

import numpy as np


def fit_circle(points: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float, float]:
    """Least-squares circle (Kåsa's algebraic fit): centre and radius in the points' units."""
    p = np.asarray(points, float)
    w = np.ones(len(p)) if weights is None else np.sqrt(np.asarray(weights, float))
    a = np.column_stack([2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))]) * w[:, None]
    b = (p ** 2).sum(axis=1) * w
    (cx, cy, c), *_ = np.linalg.lstsq(a, b, rcond=None)
    return float(cx), float(cy), float(np.sqrt(c + cx * cx + cy * cy))


def bright_pixels(array: np.ndarray, nm_per_px: float, *, frac: float = 0.4,
                  background_nm: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) in nm within the frame and the height above the local background of the pixels
    standing more than ``frac`` of the way to the top (99.5th percentile) above it. The
    background is a running median over ``background_nm``: it follows a step edge (208 pm on
    Cu, three times an atom) where a levelling plane would leave a whole terrace above the
    threshold, and it stays under the ~0.8 nm atoms. x runs right from the left edge, y up from
    the bottom edge (row 0 of an instrument frame is the top)."""
    from scipy.ndimage import gaussian_filter, median_filter

    z = np.asarray(array, float)
    z = np.where(np.isfinite(z), z, np.nanmedian(z))
    ny, nx = z.shape
    size = max(5, int(round(background_nm / nm_per_px)) | 1)
    z = gaussian_filter(z - median_filter(z, size=size, mode="nearest"), 0.6)
    yy, xx = np.mgrid[0:ny, 0:nx]
    base, top = float(np.median(z)), float(np.percentile(z, 99.5))
    sel = z > base + frac * (top - base)
    pts = np.column_stack([(xx[sel] + 0.5) * nm_per_px, (ny - yy[sel] - 0.5) * nm_per_px])
    return pts, z[sel] - base


def ring(points: np.ndarray, weights: np.ndarray | None = None, *, keep_nm: float = 0.5,
         rounds: int = 6) -> tuple[float, float, float, int]:
    """The circle through a ring among other bright things: fit, drop the points farther than
    ``keep_nm`` from it, refit. Returns (cx, cy, radius, points on it)."""
    p = np.asarray(points, float)
    w = np.ones(len(p)) if weights is None else np.asarray(weights, float)
    if len(p) < 5:
        raise ValueError("too few points for a ring")
    sel = np.ones(len(p), bool)
    for _ in range(rounds):
        cx, cy, r = fit_circle(p[sel], w[sel])
        d = np.abs(np.hypot(p[:, 0] - cx, p[:, 1] - cy) - r)
        new = d <= keep_nm
        if new.sum() < 5 or np.array_equal(new, sel):
            break
        sel = new
    cx, cy, r = fit_circle(p[sel], w[sel])
    return cx, cy, r, int(sel.sum())


def corral_radius(array: np.ndarray, nm_per_px: float) -> dict:
    """Centre (nm within the frame) and radius of the ring in a topograph: a circle through
    the bright pixels, outliers dropped."""
    pts, h = bright_pixels(array, nm_per_px)
    cx, cy, r, n = ring(pts, h)
    return {"centre_nm": [cx, cy], "radius_nm": r, "pixels": n}


#: ħ²/2mₑ (eV·nm²)
HBAR2_OVER_2ME_EV_NM2 = 0.0380998


def hard_wall_levels(e0_v: float, m_star: float, radius_nm: float, n: int = 6) -> list[float]:
    """The textbook circular box, m = 0 series (the states with weight at the centre):
    E_n = E0 + (ħ²/2m*)(j₀ₙ/R)², in V."""
    from scipy.special import jn_zeros

    return [float(e0_v + HBAR2_OVER_2ME_EV_NM2 / m_star * (z / radius_nm) ** 2) for z in jn_zeros(0, n)]


def label_levels(peaks_v: list[float], e0_v: float, radius_nm: float, *, m_star: float = 0.38,
                 shift_v: float = 0.010) -> dict[int, float]:
    """Quantum numbers for the resonances at a corral's centre, as Crommie et al. assigned
    them: each peak goes to the nearest box level (n = 1, 2, …), shifted up by ``shift_v`` —
    a ring of point scatterers is leaky, and its resonances sit 5–20 meV above the hard-wall
    levels. ``m_star`` is Cu(111)'s textbook 0.38 mₑ; the radius is the measured one."""
    levels = hard_wall_levels(e0_v, m_star, radius_nm)
    out: dict[int, float] = {}
    for p in sorted(peaks_v):
        n = int(np.argmin([abs(p - (e + shift_v)) for e in levels])) + 1
        out.setdefault(n, float(p))
    return out

