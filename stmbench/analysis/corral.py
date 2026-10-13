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


def ring(points: np.ndarray, weights: np.ndarray | None = None, *, keep_nm=0.5,
         rounds: int = 6) -> tuple[float, float, float, int]:
    """The circle through a ring among other bright things: fit, drop the points farther than
    ``keep_nm`` from it, refit. ``keep_nm`` may be a sequence, narrowing round by round: in a
    frame wide enough to hold the spare atoms outside a ring, a first fit through everything is
    off by a nanometre, and a 0.5 nm cut around it can keep the wrong arc. Returns (cx, cy,
    radius, points on it)."""
    p = np.asarray(points, float)
    w = np.ones(len(p)) if weights is None else np.asarray(weights, float)
    if len(p) < 5:
        raise ValueError("too few points for a ring")
    keeps = list(keep_nm) if np.ndim(keep_nm) else [float(keep_nm)] * rounds
    sel = np.ones(len(p), bool)
    for keep in keeps:
        cx, cy, r = fit_circle(p[sel], w[sel])
        d = np.abs(np.hypot(p[:, 0] - cx, p[:, 1] - cy) - r)
        new = d <= keep
        if new.sum() < 5 or (np.array_equal(new, sel) and keep == keeps[-1]):
            break
        sel = new
    cx, cy, r = fit_circle(p[sel], w[sel])
    return cx, cy, r, int(sel.sum())


def corral_radius(array: np.ndarray, nm_per_px: float, *, keep_nm=0.5) -> dict:
    """Centre (nm within the frame) and radius of the ring in a topograph: a circle through
    the bright pixels, outliers dropped (:func:`ring`)."""
    pts, h = bright_pixels(array, nm_per_px)
    cx, cy, r, n = ring(pts, h, keep_nm=keep_nm)
    return {"centre_nm": [cx, cy], "radius_nm": r, "pixels": n}


def _background_free(array: np.ndarray, nm_per_px: float, background_nm: float) -> np.ndarray:
    """The image over a running median ``background_nm`` wide, after the rows are levelled
    against each other. The median is taken on a copy binned to ~0.5 nm pixels and interpolated
    back: the background varies over nanometres, and a 6 nm median at 0.1 nm/px costs seconds
    per frame."""
    from scipy.ndimage import median_filter, zoom

    z = np.asarray(array, float)
    z = np.where(np.isfinite(z), z, np.nanmedian(z))
    # rows levelled against each other first: a tip change mid-frame steps every row after it
    z = z - np.concatenate([[0.0], np.cumsum(np.median(np.diff(z, axis=0), axis=1))])[:, None]
    f = max(1, int(round(0.5 / nm_per_px)))
    ny, nx = z.shape
    my, mx = ny // f * f, nx // f * f
    coarse = np.median(z[:my, :mx].reshape(my // f, f, mx // f, f).transpose(0, 2, 1, 3).reshape(my // f, mx // f, -1), axis=2)
    size = max(3, int(round(background_nm / (nm_per_px * f))) | 1)
    bg = median_filter(coarse, size=size, mode="nearest")
    bg = zoom(bg, (ny / bg.shape[0], nx / bg.shape[1]), order=1, mode="nearest", grid_mode=True)
    return z - bg[:ny, :nx]


def _sample(z: np.ndarray, nm_per_px: float, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear samples at frame coordinates (x from the left edge, y up from the bottom)."""
    from scipy.ndimage import map_coordinates

    ny = z.shape[0]
    cols, rows = xs / nm_per_px - 0.5, ny - ys / nm_per_px - 0.5
    return map_coordinates(z, [rows.ravel(), cols.ravel()], order=1, mode="nearest").reshape(np.shape(xs))


def find_ring(array: np.ndarray, nm_per_px: float, *, search_nm: float = 3.0,
              radius_range_nm: tuple[float, float] = (2.5, 13.0)) -> dict:
    """A ring of atoms in a topograph that also holds other atoms (spares outside it, a sparse
    deposit), centred within ``search_nm`` of the frame's centre. The bright pixels
    (:func:`bright_pixels`) vote for circles; the circle with the most height within 0.1 nm of
    it wins, and :func:`ring` refines it from the pixels within 0.6 nm. A least-squares circle
    through everything, as :func:`corral_radius` starts, lands 1-2 nm off in such a frame.
    Returns like :func:`corral_radius` (centre in frame coordinates)."""
    pts, h = bright_pixels(array, nm_per_px)
    ny, nx = np.shape(array)
    c0 = np.array([nx * nm_per_px / 2.0, ny * nm_per_px / 2.0])
    offs = np.arange(-search_nm, search_nm + 1e-9, 0.25)
    bins = np.arange(radius_range_nm[0], radius_range_nm[1] + 1e-9, 0.1)
    best = (-1.0, c0, 0.0)
    for dx in offs:
        for dy in offs:
            c = c0 + (dx, dy)
            hist = np.histogram(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]), bins=bins, weights=h)[0]
            pair = hist[:-1] + hist[1:]
            k = int(np.argmax(pair))
            if pair[k] > best[0]:
                best = (float(pair[k]), c, float(bins[k + 1]))
    _, c, r = best
    near = np.abs(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) - r) <= 0.6
    cx, cy, rr, n = ring(pts[near], h[near])
    return {"centre_nm": [cx, cy], "radius_nm": rr, "pixels": n}


#: the running median under a ring has to be wider than a 2 nm one: on a ring of atoms 0.6-0.9 nm
#: apart, a 2 nm window is mostly ring and takes the ring's own level for the background
RING_BACKGROUND_NM = 6.0


def ring_profile(array: np.ndarray, nm_per_px: float, centre_nm, radius_nm: float, *,
                 half_width_nm: float = 1.0, step_deg: float = 0.25,
                 background_nm: float = RING_BACKGROUND_NM) -> tuple[np.ndarray, np.ndarray]:
    """The ring's cross-section along it: the image over a running-median background, integrated
    across the ring from ``radius - half_width`` to ``radius + half_width``, at every
    ``step_deg``. Integrating across, rather than reading the ring's crest, makes the level
    indifferent to where exactly the crest runs (the atoms sit on lattice sites up to 0.15 nm
    off the circle, and the fit has its own error). ``centre_nm`` is in frame coordinates (x
    from the left edge, y up from the bottom edge). Returns ``(angles in rad from +x,
    counter-clockwise, m·nm)``."""
    z = _background_free(array, nm_per_px, background_nm)
    th = np.radians(np.arange(0.0, 360.0, step_deg))
    radii = radius_nm + np.arange(-half_width_nm, half_width_nm + 1e-9, 0.05)
    xs = centre_nm[0] + radii[:, None] * np.cos(th)[None, :]
    ys = centre_nm[1] + radii[:, None] * np.sin(th)[None, :]
    return th, np.trapezoid(_sample(z, nm_per_px, xs, ys), radii, axis=0)


def atom_volume(array: np.ndarray, nm_per_px: float, centre_nm, *, radius_nm: float = 1.2,
                background_nm: float = RING_BACKGROUND_NM) -> float:
    """The volume (m·nm²) of one isolated atom over the same background :func:`ring_profile`
    uses, within ``radius_nm`` of its centre (frame coordinates). A ring of such atoms ``s`` apart
    has, on average, ``atom_volume / s`` of cross-section: that is how far apart atoms too close
    to resolve one by one are."""
    z = _background_free(array, nm_per_px, background_nm)
    ny, nx = z.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    x, y = (xx + 0.5) * nm_per_px, (ny - yy - 0.5) * nm_per_px
    inside = np.hypot(x - centre_nm[0], y - centre_nm[1]) <= radius_nm
    return float(z[inside].sum() * nm_per_px * nm_per_px)


def ring_gap(angles: np.ndarray, profile: np.ndarray, radius_nm: float, *,
             missing: int | None = None, volume: float | None = None) -> dict:
    """The gap in a ring of atoms and the sites that would close it, from :func:`ring_profile`.

    The gap is where the ring's cross-section falls to half its level. A chain of atoms ``s``
    apart falls to half half a spacing past its last atom, so the gap is ``missing × s`` wide at
    half level, and the missing sites sit evenly in it, half a spacing in from either edge.

    How many are missing is ``missing`` when it is known (P3's repair puts as many spare atoms
    by the gap). Otherwise it comes from the volume of one atom (:func:`atom_volume`): the
    atoms of a ring 0.6-0.9 nm apart are hardly resolved one by one, but they give it a
    cross-section ``volume / spacing``. That estimate is rough: on rendered corrals the ring's
    cross-section runs ~13 % above the sum of its atoms (the surface state's density of states
    rides on the topograph, and interferes along the ring), so the count can be one too many."""
    th = np.asarray(angles, float)
    y = np.asarray(profile, float)
    n = y.size
    step = float(th[1] - th[0])
    k = max(1, int(round(0.15 / (radius_nm * step))))            # smooth over ~0.3 nm of arc
    ys = np.convolve(np.concatenate([y[-k:], y, y[:k]]), np.ones(2 * k + 1) / (2 * k + 1), "same")[k:-k]
    level0 = float(np.median(ys))
    low = ys < 0.5 * level0
    if not low.any() or low.all():
        return {"gap_rad": None, "gap_width_nm": 0.0, "spacing_nm": None, "missing": 0, "sites_rad": [],
                "level": level0}
    # the longest circular run of low samples, as [a, b): low from a to b - 1
    first_high = int(np.argmin(low))
    order = (np.arange(n) + first_high) % n
    runs, start = [], None
    for pos, i in enumerate(order):
        if low[i] and start is None:
            start = pos
        if not low[i] and start is not None:
            runs.append((start, pos))
            start = None
    if start is not None:
        runs.append((start, n))
    s0, s1 = max(runs, key=lambda r: r[1] - r[0])
    a, b = s0 + first_high, s1 + first_high                     # sample indices, unwrapped
    level = float(np.median(ys[~low]))

    def crossing(i0: int, i1: int) -> float:
        v0, v1 = ys[i0 % n], ys[i1 % n]
        f = (0.5 * level - v0) / (v1 - v0) if v1 != v0 else 0.5
        return (i0 + (i1 - i0) * f) * step

    t_a, t_b = crossing(a - 1, a), crossing(b - 1, b)
    width = (t_b - t_a) * radius_nm
    if missing is None:
        spacing = volume / level if volume and level > 0 else float("nan")
        missing = int(round(width / spacing)) if np.isfinite(spacing) else 0
    else:
        spacing = width / missing if missing else float("nan")
    sites = [float((t_a + (j + 0.5) * (t_b - t_a) / missing) % (2 * np.pi)) for j in range(missing)]
    return {"gap_rad": [float(t_a % (2 * np.pi)), float(t_b % (2 * np.pi))], "gap_width_nm": float(width),
            "spacing_nm": float(spacing), "missing": missing, "sites_rad": sites, "level": level}


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

