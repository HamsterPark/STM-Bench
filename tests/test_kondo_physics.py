"""The numerics behind a resolved bias sweep (stmsim.physics.sts_resolution): the instrument's
kernels, the Fano parts broadened by them, and the lateral average over a tip's apexes."""
from __future__ import annotations

import math

import numpy as np
import pytest

from stmsim.physics import sts_resolution as sr
from stmsim.physics.kondo import KB_EV_PER_K

KT = KB_EV_PER_K * 4.36                      # V: the reference rig's 4.36 K, 0.376 mV


def _fwhm(x: np.ndarray, y: np.ndarray) -> float:
    above = x[y >= 0.5 * y.max()]
    return float(above.max() - above.min())


def test_the_kernels_have_unit_area_at_any_step():
    for h in (2e-6, 3e-5, 4e-4):
        assert sr.thermal_weights(h, KT).sum() == pytest.approx(1.0, abs=1e-12)
        assert sr.lockin_weights(h, 2e-3).sum() == pytest.approx(1.0, abs=1e-12)
        assert sr.instrument_weights(h, KT, 2e-3).sum() == pytest.approx(1.0, abs=1e-12)
    # and the densities integrate to one on a fine grid
    x = np.linspace(-0.02, 0.02, 400001)
    assert np.trapezoid(sr.thermal_kernel(x, KT), x) == pytest.approx(1.0, abs=1e-6)
    assert np.trapezoid(sr.lockin_kernel(x, 3e-3), x) == pytest.approx(1.0, abs=1e-5)


def test_the_lockin_kernel_is_a_semicircle_of_fwhm_sqrt3_times_the_amplitude():
    vp = 2.5e-3
    x = np.linspace(-vp, vp, 200001)
    assert _fwhm(x, sr.lockin_kernel(x, vp)) == pytest.approx(math.sqrt(3.0) * vp, rel=1e-3)
    assert sr.lockin_kernel(np.array([1.01 * vp, -2 * vp]), vp).tolist() == [0.0, 0.0]


def test_the_thermal_kernel_is_3p53_kt_wide_and_is_minus_the_derivative_of_the_fermi_function():
    x = np.linspace(-20 * KT, 20 * KT, 400001)
    k = sr.thermal_kernel(x, KT)
    assert _fwhm(x, k) == pytest.approx(2.0 * math.log(3.0 + 2.0 * math.sqrt(2.0)) * KT, rel=1e-3)
    assert _fwhm(x, k) == pytest.approx(3.53 * KT, rel=2e-3)
    fermi = 1.0 / (np.exp(x / KT) + 1.0)
    assert np.allclose(k[1:-1], -(fermi[2:] - fermi[:-2]) / (x[2] - x[0]), rtol=1e-4, atol=1e-9 * k.max())


def test_the_discrete_kernels_have_the_widths_of_the_densities():
    h = 5e-6
    w = sr.thermal_weights(h, KT)
    x = (np.arange(w.size) - w.size // 2) * h
    assert _fwhm(x, w / h) == pytest.approx(3.53 * KT, rel=0.02)
    vp = 4e-3
    w = sr.lockin_weights(h, vp)
    x = (np.arange(w.size) - w.size // 2) * h
    assert _fwhm(x, w / h) == pytest.approx(math.sqrt(3.0) * vp, rel=0.01)


def test_the_broadened_shapes_go_to_the_bare_ones_as_the_broadening_goes_to_zero():
    g, e_k = 5e-3, 1e-3
    v = np.linspace(-0.03, 0.03, 61)
    win = sr.Window.build(-0.03, 0.03, 2e-6, None, g)           # a kernel narrower than every step used
    sh = sr.kondo_shapes(win, v, e_k, g, 2e-6, None)
    l_b, d_b = sr.fano_parts(v, e_k, g)
    assert np.allclose(sh["L_k"], l_b, atol=2e-4) and np.allclose(sh["D_k"], d_b, atol=2e-4)
    # the integrals are the antiderivatives, Γ atan ε and (Γ/2) ln(1 + ε²), taken from E = 0
    prim_l, prim_d = sr._fano_primitives(v, e_k, g)
    p0_l, p0_d = sr._fano_primitives(np.array([0.0]), e_k, g)
    assert np.allclose(sh["lam_L"], prim_l - p0_l[0], atol=2e-6)
    assert np.allclose(sh["lam_D"], prim_d - p0_d[0], atol=2e-6)


def test_the_thermal_and_lockin_broadening_is_the_convolution_it_claims_to_be():
    """One point checked against a brute-force quadrature of the kernel and the line."""
    g, e_k, vp = 4e-3, 0.5e-3, 1.5e-3
    v0 = -2.0e-3
    win = sr.Window.build(-0.02, 0.02, KT, vp, g)
    sh = sr.kondo_shapes(win, np.array([v0]), e_k, g, KT, vp)
    ker = np.convolve(sr.thermal_kernel(np.linspace(-0.012, 0.012, 24001), KT) * (0.024 / 24000),
                      sr.lockin_kernel(np.linspace(-vp, vp, 3001), vp) * (2 * vp / 3000), mode="full")
    centre = np.linspace(-0.012 - vp, 0.012 + vp, ker.size)
    l_b, d_b = sr.fano_parts(v0 - centre, e_k, g)               # reading at v0 sees the line at v0 - u
    assert np.sum(ker * l_b) == pytest.approx(sh["L_k"][0], abs=2e-4)
    assert np.sum(ker * d_b) == pytest.approx(sh["D_k"][0], abs=2e-4)


def test_a_wide_window_keeps_the_shapes_continuous_where_the_grid_ends():
    g, e_k = 5e-3, 0.0
    win = sr.Window.build(-1.5, 1.5, KT, 5e-3, g)
    assert win.lo == -sr.W_CAP_V and win.hi == sr.W_CAP_V and win.grid.size <= sr.MAX_GRID + 2
    for edge in (-sr.W_CAP_V, sr.W_CAP_V):
        v = np.array([edge - 1e-6, edge + 1e-6])           # either side of the last convolved point
        sh = sr.kondo_shapes(win, v, e_k, g, KT, 5e-3)
        for key in ("L_k", "D_k", "L_th", "D_th"):
            assert abs(sh[key][1] - sh[key][0]) < 1e-5, key
        for key in ("lam_L", "lam_D"):
            assert abs(sh[key][1] - sh[key][0]) < 1e-6, key
    # far out the integrals keep growing as the antiderivatives do (Γ atan ε saturates at ±πΓ/2)
    sh = sr.kondo_shapes(win, np.array([-1.5, 1.5]), e_k, g, KT, 5e-3)
    assert sh["lam_L"] == pytest.approx([-math.pi * g / 2, math.pi * g / 2], rel=0.02)
    assert sh["lam_D"][0] == pytest.approx(sh["lam_D"][1], rel=0.02) and sh["lam_D"][1] > 0.0


def _tip_average(r: float, s: float, sigma_t: float) -> float:
    """exp(-r²/2s²) averaged over a normalised 2-D Gaussian of width sigma_t, by quadrature."""
    u = np.linspace(-6 * sigma_t, 6 * sigma_t, 241)
    xx, yy = np.meshgrid(u, u)
    w = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_t ** 2))
    w /= w.sum()
    return float(np.sum(w * np.exp(-((r + xx) ** 2 + yy ** 2) / (2 * s ** 2))))


@pytest.mark.parametrize("r", [0.0, 0.3e-9, 0.8e-9])
def test_the_closed_form_blur_is_the_average_over_the_apex(r):
    s, st = 0.35e-9, 0.08e-9
    assert sr._blur(np.array([r * r]), s, st)[0] == pytest.approx(_tip_average(r, s, st), rel=2e-3)


def test_a_point_apex_on_an_atom_gives_the_fano_line_of_the_parameters():
    a0, q0 = 0.25, 0.3
    c0, c1, c2 = sr.kondo_coefficients(np.zeros((1, 2)), np.zeros((1, 2)), np.ones((1, 1)), np.zeros((1, 2)),
                                       a0=a0, q0=q0, sigma_a=0.35e-9, sigma_q=0.55e-9, sigma_t=1e-15)
    assert (c0[0], c1[0], c2[0]) == pytest.approx((a0, a0 * q0, a0 * q0 * q0), rel=1e-9)
    # 1 + (C2 - C0) L + 2 C1 D is 1 + a0 [(q0 + ε)²/(1 + ε²) - 1]
    eps = np.linspace(-6, 6, 25)
    l, d = 1.0 / (1 + eps ** 2), eps / (1 + eps ** 2)
    assert np.allclose(1 + (c2 - c0) * l + 2 * c1 * d, 1 + a0 * ((q0 + eps) ** 2 / (1 + eps ** 2) - 1))


def test_the_coefficients_sum_over_atoms_and_weight_the_apexes():
    pts = np.zeros((1, 2))
    atoms = np.array([[0.0, 0.0], [0.2e-9, 0.0]])
    kw = dict(a0=0.2, q0=0.5, sigma_a=0.35e-9, sigma_q=0.55e-9, sigma_t=0.08e-9)
    both = sr.kondo_coefficients(pts, np.zeros((1, 2)), np.ones((1, 1)), atoms, **kw)
    one = sr.kondo_coefficients(pts, np.zeros((1, 2)), np.ones((1, 1)), atoms[:1], **kw)
    two = sr.kondo_coefficients(pts, np.zeros((1, 2)), np.ones((1, 1)), atoms[1:], **kw)
    assert np.allclose(both, np.add(one, two))
    # two apexes at the same place with weights 0.7 and 0.3 are one apex
    split = sr.kondo_coefficients(pts, np.zeros((2, 2)), np.array([[0.7], [0.3]]), atoms, **kw)
    assert np.allclose(split, both)
    # an apex 5 nm away sees nothing
    far = sr.kondo_coefficients(pts, np.array([[5e-9, 0.0]]), np.ones((1, 1)), atoms, **kw)
    assert max(abs(float(c[0])) for c in far) < 1e-12


def test_the_background_is_convolved_with_the_lockin_kernel_only():
    """A step in the background much sharper than k_B T: the lock-in's semicircle smooths it, the
    thermal kernel must not (the sample LDOS may already carry its temperature)."""
    vp = 3e-3
    win = sr.Window.build(-0.02, 0.02, KT, vp, 5e-3)
    nodes = np.linspace(-0.03, 0.03, 6001)
    g = np.where(nodes < 0.0, 1.0, 2.0)
    v = np.array([-0.002, 0.0, 0.002, 0.01])
    off = sr.smoothed_background(win, v, nodes, g, None)
    assert np.allclose(off, np.interp(v, nodes, g))                  # lock-in off: the background itself
    on = sr.smoothed_background(win, v, nodes, g, vp)
    # the semicircle's cumulative distribution gives the smoothed step in closed form
    want = 1.0 + sr._lockin_cdf(v, vp)
    assert np.allclose(on, want, atol=2e-3)
    # with the thermal kernel as well the step at +-2 mV would be visibly different from the CDF
    both = np.convolve(np.interp(win.grid, nodes, g), win.weights, mode="same")
    assert abs(np.interp(0.002, win.grid, both) - want[2]) > 3e-3
