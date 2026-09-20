"""Physics sanity: the P1 gate items from docs/DESIGN.md §8."""
from __future__ import annotations

import math

import numpy as np
import pytest

from stmsim.physics.feedback import Loop, LoopParams
from stmsim.physics.junction import (apparent_barrier_from_iz, current_metal, gap_for_current,
                                     iv_curve, kappa_per_m, LDOSTemplate)
from stmsim.physics.surface import Surface
from stmsim.physics.tip import Apex, Tip


@pytest.mark.parametrize("phi", [0.9, 2.0, 4.0])
def test_iz_curve_returns_the_barrier_it_was_built_with(phi):
    gap0 = gap_for_current(100e-12, 1.0, phi)
    z = np.linspace(0, 0.55e-9, 12)
    i = current_metal(gap0 + z, 1.0, phi)
    phi_fit, r2, n = apparent_barrier_from_iz(z, i, floor_a=0.05e-12)
    assert n >= 5 and r2 > 0.999
    assert phi_fit == pytest.approx(phi, rel=0.05)


def test_gap_for_current_is_physical():
    g = gap_for_current(100e-12, 1.0, 4.0)
    assert 0.4e-9 < g < 0.8e-9
    g2 = gap_for_current(50e-12, 0.02, 4.0)
    assert g2 < g  # lower bias & current → closer


def test_iv_curve_has_shockley_onset():
    vs = np.linspace(-1.0, 1.0, 201)
    au = LDOSTemplate("Au", onset_ev=-0.49)
    i = iv_curve(vs, 0.55e-9, 4.0, au)
    didv = np.gradient(i, vs)
    # dI/dV is larger just above the onset than just below it
    above = didv[(vs > -0.40) & (vs < -0.30)].mean()
    below = didv[(vs > -0.70) & (vs < -0.60)].mean()
    assert above > 1.2 * below


def test_multi_apex_ghost_offset_equals_apex_offset():
    s = Surface("Au(111)", seed=3, adsorbate_density_per_um2=0.0)
    # a single sharp bump on the surface
    from stmsim.physics.surface import Feature
    s.add_feature(Feature(0.0, 0.0, 0.5e-9, 0.6e-9, kind="cluster"))
    k = kappa_per_m(4.0)
    x = np.linspace(-10e-9, 10e-9, 801)
    y = np.zeros_like(x)
    t1 = Tip()
    h1 = t1.effective_height(s, x, y, k) - s.terrace_height(x, y)
    t2 = Tip(apexes=[Apex(0, 0, 0, 1.0), Apex(-4e-9, 0, 0.0, 0.8)])
    h2 = t2.effective_height(s, x, y, k) - s.terrace_height(x, y)
    peaks = x[np.argsort(h2)[-2:]] if False else None
    # ghost: a second bump appears at x = +4 nm (surface sampled at x + dx with dx = -4 nm)
    i_ghost = np.argmin(np.abs(x - 4e-9))
    i_main = np.argmin(np.abs(x - 0.0))
    assert h2[i_ghost] > 0.6 * h2[i_main]
    assert h1[i_ghost] < 0.05 * h1[i_main]


def test_blunt_tip_widens_step_edge_monotonically():
    from scipy import ndimage
    s = Surface("Au(111)", seed=5, adsorbate_density_per_um2=0.0)
    k = kappa_per_m(4.0)
    x = np.linspace(-30e-9, 30e-9, 600)
    y = np.zeros_like(x)
    prof = s.terrace_height(x, y)
    dx = x[1] - x[0]
    widths = []
    for r in [0.5e-9, 2e-9, 8e-9]:
        t = Tip(radius_m=r)
        sig = t.sigma_eff_m(4.0) / dx
        h = ndimage.gaussian_filter1d(prof, sig) if sig > 0.3 else prof
        g = np.abs(np.gradient(h, dx))
        widths.append(1.0 / (g.max() + 1e-30))   # 1/slope ~ edge width
    assert widths[0] < widths[1] < widths[2]


def test_loop_ringing_and_instability_with_gain():
    lp = LoopParams(p_m=3e-12, i_m_per_s=50e-9, kappa_m=1e10)
    assert lp.stable and lp.damping > 0.7
    hot = LoopParams(p_m=3e-12, i_m_per_s=3000e-9, kappa_m=1e10)
    assert hot.natural_hz > 300
    assert hot.damping < 0.5           # rings
    crazy = LoopParams(p_m=3e-12, i_m_per_s=60000e-9, kappa_m=1e10)
    assert not crazy.stable
    # unstable loop produces a bounded oscillation at ~natural frequency
    loop = Loop(crazy)
    dt = 0.586 / 256
    z = loop.track(np.zeros(256), dt, z0=0.0)
    assert np.isfinite(z).all() and 0 < np.abs(z).max() <= crazy.z_clamp_m * 1.01
    assert np.abs(z[64:]).max() > 0.2 * crazy.z_clamp_m


def test_loop_tracks_step_with_second_order_response():
    lp = LoopParams(p_m=3e-12, i_m_per_s=200e-9, kappa_m=1e10)
    loop = Loop(lp)
    t = np.arange(0, 0.05, 1 / 20000)
    z = loop.transient(np.full(t.size, 1e-9), t, 0.0)
    assert z[-1] == pytest.approx(1e-9, rel=0.02)
    assert z[1] < 0.9e-9  # not instantaneous
    assert np.all(np.diff(z[:3]) > 0)


def test_tip_pulse_and_poke_change_state_reproducibly():
    t1 = Tip(rng=np.random.default_rng(7), radius_m=6e-9)
    t2 = Tip(rng=np.random.default_rng(7), radius_m=6e-9)
    o1 = t1.pulse(0.0, 10.0, 0.5)
    o2 = t2.pulse(0.0, 10.0, 0.5)
    assert o1 == o2 and abs(o1["dz_m"]) >= 20e-9
    weak = Tip(rng=np.random.default_rng(1))
    assert weak.pulse(0.0, 1.0, 0.5)["outcome"] == "no_effect"
    q = Tip(form="qplus", rng=np.random.default_rng(2))
    assert q.poke(0.0, 0.5e-9, 1.0)["outcome"] == "ring_up"
    q2 = Tip(form="qplus", rng=np.random.default_rng(2))
    out = q2.poke(0.0, 0.5e-9, 0.02)
    assert out["outcome"] in ("cluster", "pit", "no_change")


class _StepSurface:
    """A single down-step of height H at x = 0 on a flat terrace; no lattice."""

    class material:
        first_order_period_m = 0.0

    def __init__(self, h_m: float):
        self.h = h_m

    def height_smooth(self, x, y):
        return np.where(np.asarray(x) < 0, self.h, 0.0)

    def atomic_height(self, x, y):
        return np.zeros_like(np.asarray(x, float))


def test_two_apex_softmax_splits_a_step_into_the_analytic_pair():
    """The physics MAST's ``detect_step_splitting`` assumes (double_tip.py): apexes at
    separation d and weights a₁, a₂ turn a step of height H into an intermediate plateau
    of width |d| whose height above the lower level is (1/2κ)·ln[(a_hi·e^{2κH} + a_lo)/
    (a₁ + a₂)] with ``a_hi`` the apex over the high terrace — every step in the frame
    split the same way. Equal-height apexes at 0.7 weight on a 235 pm step: a 25 + 210 pm
    pair when the strong apex leads, 42 + 194 pm when the weak one does. An apex one step
    LOWER is invisible (a 0.2 pm ledge) — the reason the fidelity probe uses dz = 0."""
    H = 0.2354e-9
    k = kappa_per_m(4.1)
    x = np.linspace(-6e-9, 6e-9, 1201)
    y = np.zeros_like(x)
    s = _StepSurface(H)
    d, w2 = 3e-9, 0.7
    lo = np.log(1.0 + w2) / (2 * k)                    # both apexes on one terrace: the soft-max's DC
    assert lo == pytest.approx(0.0255e-9, abs=0.001e-9)
    for sign, a_hi, a_lo in ((+1, 1.0, w2), (-1, w2, 1.0)):
        # apex 2 at +d leads on a scan to the right: over (−d, 0) the strong apex 1 is the one
        # still on the high terrace; at −d it trails and the weak apex 2 holds the high terrace
        t = Tip(apexes=[Apex(0.0, 0.0, 0.0, 1.0), Apex(sign * d, 0.0, 0.0, w2)])
        h = t.effective_height(s, x, y, k)
        span = (x > -d + 0.3e-9) & (x < -0.3e-9) if sign > 0 else (x > 0.3e-9) & (x < d - 0.3e-9)
        plateau = h[span] - lo
        expect = np.log((a_hi * np.exp(2 * k * H) + a_lo) / (1.0 + w2)) / (2 * k)
        assert np.allclose(plateau, expect, atol=1e-13)                   # one value frame-wide
        assert expect == pytest.approx((0.210e-9 if sign > 0 else 0.194e-9), abs=0.003e-9)
        both_hi = h[x < -max(sign * d, 0) - 0.3e-9] - lo
        both_lo = h[x > max(-sign * d, 0) + 0.3e-9] - lo
        assert np.allclose(both_hi, H, atol=1e-13) and np.abs(both_lo).max() < 1e-13
        edges = x[np.flatnonzero(np.abs(np.diff(h)) > 0.01e-9)]
        assert edges.max() - edges.min() == pytest.approx(d, abs=0.02e-9)   # the pair is |d| apart
    # one step lower: over (0, d) the trailing ghost on the high terrace is exactly level with
    # the leading apex on the low one → a ledge of ln(1 + w2)/2κ ≈ 25 pm, a tenth of the step
    t2 = Tip(apexes=[Apex(0.0, 0.0, 0.0, 1.0), Apex(-d, 0.0, -H, w2)])
    h2 = t2.effective_height(s, x, y, k)
    ledge = h2[(x > 0.3e-9) & (x < d - 0.3e-9)]
    assert np.allclose(ledge, lo, atol=1e-13) and ledge.max() < 0.03e-9
    assert (h2[x < -0.3e-9] - H).max() < 0.001e-9 and np.abs(h2[x > d + 0.3e-9]).max() < 0.001e-9


def test_crash_leaves_an_unstable_quantised_multi_apex_that_a_shallow_poke_pins():
    """``Tip.crash``: metastable, λ ≥ 0.05/s, 1–3 extra apexes 2–6 nm off with dz at whole
    steps, a per-pass apex re-draw and mid-line length hops; the base apexes never move
    under the re-draw; a shallow poke (or a reshaping pulse) clears the flicker."""
    step = 0.2354e-9
    for seed in range(12):
        t = Tip(rng=np.random.default_rng(seed), radius_m=1e-9)
        out = t.crash(0.0, 1.0)
        assert out["outcome"] == "crashed" and t.metastable and t.lambda_per_s >= 0.05
        assert 0.12e-9 <= t.flicker_dz_m <= 0.20e-9 and 1.0 <= t.flicker_rate_hz <= 4.0
        assert 2 <= t.n_apex <= 4 and t.radius_m >= 2e-9 and t.phi_ev <= 3.0
        for a in t.apexes[1:]:
            assert 2e-9 <= math.hypot(a.dx, a.dy) <= 6e-9 and 0.5 <= a.w <= 1.0
            assert abs(a.dz / step - round(a.dz / step)) < 1e-9 and -2 <= round(a.dz / step) <= 0
        base = [a.as_tuple() for a in t.apexes]
        p1, p2 = t.pass_apexes(), t.pass_apexes()
        assert [a.as_tuple() for a in t.apexes] == base                 # base untouched
        assert p1[0].as_tuple() == base[0] and p2[0].as_tuple() == base[0]
        assert any(abs(p1[i].dz - p2[i].dz) > 1e-12 for i in range(1, len(base)))   # passes differ
        assert all(p1[i].dx == base[i][0] and p1[i].dy == base[i][1] for i in range(len(base)))
    # mid-line hops: segments of constant offset with the configured rms
    t = Tip(rng=np.random.default_rng(3))
    t.crash(0.0, 1.0)
    t.flicker_dz_m, t.flicker_rate_hz = 0.15e-9, 40.0
    fl = t.flicker_series(256, 0.586 / 256)
    hops = np.flatnonzero(np.abs(np.diff(fl)) > 0)
    assert 5 <= hops.size <= 60 and 0.05e-9 < fl.std() < 0.3e-9        # ~23 hops expected at 40 Hz
    assert t.flicker_series(256, 0.586 / 256) is not fl                  # a fresh draw per pass
    t.flicker_rate_hz = 0.0
    one = t.flicker_series(256, 0.586 / 256)
    assert np.ptp(one) == 0.0                                            # one configuration per pass
    t0 = Tip()
    assert t0.flicker_series(256, 1e-3) is None and t0.pass_apexes() is t0.apexes
    t.poke(1.0, 0.4e-9, 0.02)                                            # shallow: stabilises
    assert not t.metastable and t.flicker_dz_m == 0.0 and t.flicker_rate_hz == 0.0
    assert t.flicker_series(256, 1e-3) is None and t.pass_apexes() is t.apexes
    t2 = Tip(rng=np.random.default_rng(5))
    t2.crash(0.0, 1.0)
    t2.pulse(2.0, 10.0, 0.5)                                             # inside the envelope: re-formed
    assert t2.flicker_dz_m == 0.0 and t2.flicker_rate_hz == 0.0
    t3 = Tip(rng=np.random.default_rng(6))
    assert t3.pulse(3.0, 14.0, 0.5)["outcome"] == "destroyed" and t3.flicker_dz_m > 0 and t3.metastable


def test_surface_step_height_and_lattice_period():
    s = Surface("Cu(111)", seed=11, adsorbate_density_per_um2=0.0)   # no herringbone
    x = np.linspace(-200e-9, 200e-9, 4001)
    y = np.zeros_like(x)
    h = s.terrace_height(x, y)
    d = np.diff(h)
    steps = np.abs(d[np.abs(d) > 1e-10])
    assert steps.size > 0 and np.allclose(steps, 0.208e-9, atol=1e-11)
    xl = np.linspace(0, 5e-9, 5001)
    lat = s.lattice_height(xl, np.zeros_like(xl))
    f = np.fft.rfftfreq(xl.size, xl[1] - xl[0])
    p = np.abs(np.fft.rfft(lat - lat.mean()))
    band = f > 1 / 1.0e-9
    peak = f[band][np.argmax(p[band])]
    period = 1 / peak
    # projection of one of the three row spacings along x: d = a*sqrt(3)/2 / cos(theta) ≥ 0.221 nm
    assert 0.2e-9 < period < 0.6e-9
