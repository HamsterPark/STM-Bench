"""Band edge and confined-state resonances, and the corral's geometry, on known inputs."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import erf

from stmbench.analysis.corral import corral_radius, fit_circle, hard_wall_levels, label_levels
from stmbench.analysis.spectra import band_edge, resonance_peaks


def test_the_band_edge_is_the_step_even_on_a_noisy_spectrum():
    rng = np.random.default_rng(0)
    v = np.linspace(-0.55, 0.15, 351)
    for e0 in (-0.465, -0.44, -0.415):
        y = 0.2 + 1.0 * 0.5 * (1 + erf((v - e0) / 0.012)) + 0.3 * (v + 0.55)
        y = y + rng.normal(0, 0.04, v.size)
        assert band_edge(v, y) == pytest.approx(e0, abs=0.004)


def test_resonances_are_the_prominent_maxima_above_the_edge():
    v = np.linspace(-0.55, 0.15, 351)
    e0 = -0.44
    step = 0.5 * (1 + erf((v - e0) / 0.01))
    lor = lambda c, w: (w / 2) ** 2 / ((v - c) ** 2 + (w / 2) ** 2)    # noqa: E731
    y = step * (1 + 0.6 * lor(-0.38, 0.03) + 0.9 * lor(-0.29, 0.03) + 0.02 * lor(-0.2, 0.01))
    got = [p["v"] for p in resonance_peaks(v, y, e0)]
    assert got == pytest.approx([-0.38, -0.29], abs=0.004)          # the 2 % bump is not a resonance


def test_a_circle_comes_back_and_the_box_levels_label_shifted_peaks():
    t = np.linspace(0, 2 * math.pi, 50, endpoint=False)
    pts = np.column_stack([3.0 + 7.2 * np.cos(t), -1.0 + 7.2 * np.sin(t)])
    assert fit_circle(pts) == pytest.approx((3.0, -1.0, 7.2), abs=1e-9)
    hw = hard_wall_levels(-0.44, 0.38, 7.2)
    assert hw[1] - hw[0] > 0.04 and hw[0] > -0.44
    # leaky-ring resonances sit 5-20 meV above the box levels; n = 1 may or may not be seen
    seen = [hw[1] + 0.012, hw[2] + 0.015, hw[3] + 0.008]
    assert label_levels(seen, -0.44, 7.2) == {2: seen[0], 3: seen[1], 4: seen[2]}
    assert label_levels([hw[0] + 0.006] + seen, -0.44, 7.2)[2] == seen[0]


def test_the_ring_radius_survives_a_step_in_the_frame():
    """A Cu step (208 pm, three atoms high) across a corner of the frame: a levelling plane
    leaves the upper terrace above any threshold; the running-median background follows it."""
    n, npx = 280, 0.1
    yy, xx = np.mgrid[0:n, 0:n]
    x, y = (xx + 0.5) * npx, (n - yy - 0.5) * npx
    z = np.where(x + y > 44.0, 208e-12, 0.0)
    for a in np.linspace(0, 2 * math.pi, 48, endpoint=False):
        ax, ay = 14.0 + 6.5 * math.cos(a), 14.0 + 6.5 * math.sin(a)
        z = z + 70e-12 * np.exp(-((x - ax) ** 2 + (y - ay) ** 2) / (2 * 0.35 ** 2))
    got = corral_radius(z, npx)
    assert got["radius_nm"] == pytest.approx(6.5, abs=0.05)
    assert got["centre_nm"] == pytest.approx([14.0, 14.0], abs=0.1)
