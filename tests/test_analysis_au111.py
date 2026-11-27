"""The survey analysis of the Au(111) herringbone (:mod:`stmbench.analysis.au111`): where steps are and
which way the stripes of each window run, on synthetic frames of known direction."""
from __future__ import annotations

import math

import numpy as np

from stmbench.analysis.au111 import circ_diff, clean_windows, direction_map, step_mask


def _stripes(angle_deg: float, *, n=300, npx=1.0, period=6.3, amp=10e-12, noise=2e-12, seed=0, step_at=None):
    """Straight stripes running along ``angle_deg`` (counter-clockwise from +x, row 0 at the top), with an
    optional 235 pm step up along column ``step_at``."""
    rng = np.random.default_rng(seed)
    c = (np.arange(n) + 0.5) * npx
    X, Y = np.meshgrid(c, -c)
    th = math.radians(angle_deg)
    u = -X * math.sin(th) + Y * math.cos(th)           # across the stripes
    z = amp * np.cos(2 * math.pi * u / period) + rng.normal(0.0, noise, (n, n))
    if step_at is not None:
        z += np.where(X > step_at, 235e-12, 0.0)
    return z


def test_a_window_reads_the_direction_of_plain_stripes():
    for angle in (10.0, 45.0, 100.0, 170.0):
        dm = direction_map(_stripes(angle), 1.0)
        got = dm.psi_deg[~np.isnan(dm.psi_deg)]
        assert np.abs(circ_diff(got, angle)).max() < 3.0, (angle, got.min(), got.max())


def test_two_zones_of_different_direction_give_clean_windows_only_inside_each():
    z = np.hstack([_stripes(20.0, n=300)[:, :150], _stripes(80.0, n=300)[:, 150:]])
    dm = direction_map(z, 1.0)
    cw = clean_windows(dm)
    assert (cw >= 2).any()
    # a window on the seam is not clean: its block of neighbours disagrees
    seam = np.argmin(np.abs(dm.cols - 150))
    assert cw[:, seam].max() < cw.max()
    left = dm.psi_deg[:, dm.cols < 100]
    right = dm.psi_deg[:, dm.cols > 200]
    assert np.abs(circ_diff(left, 20.0)).max() < 4.0 and np.abs(circ_diff(right, 80.0)).max() < 4.0


def test_a_step_is_masked_and_does_not_set_the_direction():
    """A 235 pm step along a column is 20 times the stripes: it is masked, and the windows beside it still
    read the stripes' direction rather than the step's."""
    z = _stripes(60.0, step_at=150.0)
    m = step_mask(z, 1.0)
    assert m[:, 146:154].all() and not m[:, :100].any() and not m[:, 200:].any()
    dm = direction_map(z, 1.0)
    ok = ~np.isnan(dm.psi_deg) & (dm.masked < 0.3)
    assert ok.any() and np.abs(circ_diff(dm.psi_deg[ok], 60.0)).max() < 6.0


def test_the_locator_and_the_stripe_analysis_agree_on_a_windows_direction():
    """The locator only finds places; what is reported comes from stmbench.analysis.herringbone. On the
    same window of plain stripes the two read the same direction (the locator within a few degrees)."""
    from stmbench.analysis import herringbone as hb

    for angle in (15.0, 70.0, 130.0):
        z = _stripes(angle, n=128, npx=0.5, period=6.3)
        a = hb.assess_window(z, 0.5)
        dm = direction_map(z, 0.5, win_nm=64.0, stride_nm=64.0)
        assert a.measurable and abs(circ_diff(a.direction_deg, angle)) < 1.0
        assert abs(circ_diff(dm.psi_deg[0, 0], a.direction_deg)) < 4.0
