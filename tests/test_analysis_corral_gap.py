"""A corral's ring among spare atoms, its gap and the sites that close it
(``stmbench.analysis.corral``: find_ring, ring_profile, ring_gap)."""
from __future__ import annotations

import math

import numpy as np

from stmbench.analysis.corral import corral_radius, find_ring, ring_gap, ring_profile

NPX = 0.1


def _frame(seed: int, *, radius: float = 7.0, n: int = 48, missing: int = 4, size: float = 36.0):
    """A topograph (row 0 at the top) of a ring of ``n`` atoms, ``missing`` adjacent ones left
    out, as many spares 3-6 nm outside the gap, and the missing sites (frame coordinates)."""
    rng = np.random.default_rng(seed)
    px = int(round(size / NPX))
    yy, xx = np.mgrid[0:px, 0:px]
    x, y = (xx + 0.5) * NPX, (px - yy - 0.5) * NPX
    c = np.array([size / 2 + rng.uniform(-1, 1), size / 2 + rng.uniform(-1, 1)])
    phi0 = rng.uniform(0, 2 * math.pi)
    start = int(rng.integers(0, n))
    gone = {(start + k) % n for k in range(missing)}
    atoms, sites = [], []
    for k in range(n):
        p = c + radius * np.array([math.cos(phi0 + 2 * math.pi * k / n), math.sin(phi0 + 2 * math.pi * k / n)])
        (sites if k in gone else atoms).append(p)
    gap_mid = phi0 + 2 * math.pi * (start + (missing - 1) / 2) / n
    for _ in range(missing):
        a = gap_mid + rng.uniform(-0.6, 0.6)
        atoms.append(c + (radius + rng.uniform(3, 6)) * np.array([math.cos(a), math.sin(a)]))
    z = rng.normal(0.0, 1e-12, x.shape)
    for p in atoms:
        z += 50e-12 * np.exp(-((x - p[0]) ** 2 + (y - p[1]) ** 2) / (2 * 0.38 ** 2))
    return z, c, sites


def test_the_ring_is_found_among_the_spares():
    """A least-squares circle through every bright pixel is pulled off by the spares outside
    the gap; the vote of the pixels finds the ring."""
    worst_plain, worst = 0.0, 0.0
    for seed in range(3):
        z, c, _ = _frame(seed)
        got = find_ring(z, NPX)
        worst = max(worst, math.hypot(*(np.array(got["centre_nm"]) - c)), abs(got["radius_nm"] - 7.0))
        plain = corral_radius(z, NPX)
        worst_plain = max(worst_plain, abs(plain["radius_nm"] - 7.0))
    assert worst < 0.1, worst
    assert worst_plain > worst


def test_the_missing_sites_sit_where_the_ring_left_them():
    """Told how many are missing (P3's repair puts as many spares by the gap), the sites go
    evenly into the stretch where the ring's cross-section falls to half: within a quarter of
    the 0.92 nm spacing of where the ring left them."""
    for seed in range(3):
        z, c, sites = _frame(seed)
        ring = find_ring(z, NPX)
        th, prof = ring_profile(z, NPX, ring["centre_nm"], ring["radius_nm"])
        g = ring_gap(th, prof, ring["radius_nm"], missing=len(sites))
        got = [np.array(ring["centre_nm"]) + ring["radius_nm"] * np.array([math.cos(t), math.sin(t)])
               for t in g["sites_rad"]]
        err = max(min(math.hypot(*(p - q)) for p in got) for q in sites)
        assert g["missing"] == 4 and err < 0.23, (seed, err)
        assert abs(g["spacing_nm"] - 2 * math.pi * 7.0 / 48) < 0.15
