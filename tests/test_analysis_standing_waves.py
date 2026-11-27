"""Step-edge standing waves on synthetic maps: edge finding, the phase-carrying fit, dispersion."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import special
from scipy.optimize import curve_fit
from scipy.special import erf

from stmbench.analysis.standing_waves import (HBAR2_OVER_2M_EV_NM2, edge_positions, fit_dispersion,
                                              fit_step_wave, k_guess, step_profile)
from stmsim.physics.surface_state import SurfaceState, SurfaceStateParams

E0, MSTAR = -0.44, 0.38
PX = 0.2


def _ss(e0=E0, m=MSTAR):
    return SurfaceState(SurfaceStateParams.for_material("Cu(111)", e0_ev=e0, m_star=m))


def _wave(ss, e, d_nm, r=0.45, phi=2.6):
    return 1.0 + ss.line_modulation(np.asarray(d_nm) * 1e-9, e, r, phi)


def _k_true(ss, e):
    return float(ss.k_of_e(e))


def test_the_fit_finds_k_within_half_a_percent_on_a_noisy_profile():
    rng = np.random.default_rng(1)
    d = np.arange(0.6, 18.0, PX)
    ss = _ss()
    for e in (-0.30, -0.18, -0.06, 0.05, 0.15):
        for phi in (1.2, 2.6, 3.9):
            y = _wave(ss, e, d, phi=phi) * (1 + rng.normal(0, 0.003, d.size))
            fit = fit_step_wave(d, y, k_guess=k_guess(e), sem=np.full(d.size, 0.003))
            assert fit["ok"]
            assert fit["k"] == pytest.approx(_k_true(ss, e), rel=0.005)
            assert fit["k_err"] > 0


def test_k_guess_is_the_literature_dispersion():
    assert k_guess(0.0) == pytest.approx(np.sqrt(0.38 * 0.44 / HBAR2_OVER_2M_EV_NM2))


def _step_map(n_rows=60, n_cols=120, edge0=14.0, slope=0.02, jumps=(), noise=0.002, seed=0):
    rng = np.random.default_rng(seed)
    x = (np.arange(n_cols) + 0.5) * PX
    z = np.empty((n_rows, n_cols))
    truth = edge0 + slope * np.arange(n_rows)
    for r in range(n_rows):
        p = truth[r] + (jumps[r] if r < len(jumps) else 0.0)
        z[r] = 0.2 * erf((x - p) / 0.4) + 0.001 * x + rng.normal(0, noise, n_cols)
    return z, truth


def test_edge_positions_are_found_to_a_tenth_of_a_nanometre_and_a_row_jump_is_dropped():
    jumps = np.zeros(60)
    jumps[[10, 11, 40]] = 1.6                      # the tip changed mid-frame: those rows moved 8 px
    z, truth = _step_map(jumps=jumps)
    e = edge_positions(z, PX)
    assert e["sign"] == 1
    ok = np.ones(60, bool)
    ok[[10, 11, 40]] = False
    assert not e["good"][[10, 11, 40]].any()
    assert e["good"][ok].mean() > 0.95
    assert np.abs(e["pos_nm"][e["good"]] - truth[e["good"]]).max() < 0.1
    assert e["slope_nm_per_row"] == pytest.approx(0.02, abs=0.004)
    assert e["n_good"] == int(e["good"].sum())


def test_a_descending_step_has_negative_sign():
    z, _ = _step_map()
    assert edge_positions(-z, PX)["sign"] == -1


def test_the_profile_of_aligned_rows_beats_the_unaligned_mean():
    """Rows aligned at their own edge keep the ripples that a common position would smear."""
    ss = _ss()
    e = -0.12
    rng = np.random.default_rng(3)
    n_rows, n_cols = 80, 120
    x = (np.arange(n_cols) + 0.5) * PX
    edge = 3.5 + 0.03 * np.arange(n_rows)           # edge drifts 2.4 nm over the frame
    z = np.empty((n_rows, n_cols))
    a = np.empty((n_rows, n_cols))
    for r in range(n_rows):
        d = np.clip(x - edge[r], 0, None)
        z[r] = 0.2 * erf((x - edge[r]) / 0.4) + rng.normal(0, 0.002, n_cols)
        a[r] = np.where(x > edge[r], _wave(ss, e, d), 1.0) * (1 + rng.normal(0, 0.03, n_cols))
    edges = edge_positions(z, PX)
    prof = step_profile(a, edges, PX, side=+1)
    assert prof["n_rows"] > 70
    fit = fit_step_wave(prof["d_nm"], prof["y"], k_guess=k_guess(e), sem=prof["sem"])
    assert fit["ok"] and fit["k"] == pytest.approx(_k_true(ss, e), rel=0.01)
    # without alignment (one edge position for all rows) the wave is washed out
    flat = dict(edges)
    flat["pos_nm"] = np.full(n_rows, np.median(edges["pos_nm"]))
    p2 = step_profile(a, flat, PX, side=+1)
    assert np.ptp(p2["y"][20:]) < 0.6 * np.ptp(prof["y"][20:])


def test_the_profile_on_the_other_side_runs_towards_lower_columns():
    a = np.tile(np.arange(100, dtype=float) + 1.0, (6, 1))
    edges = {"pos_nm": np.full(6, 15.0), "good": np.ones(6, bool)}
    p = step_profile(a, edges, 0.2, side=-1, d_min_nm=1.0, d_max_nm=10.0)
    assert p["y"][0] > p["y"][-1]                  # smaller columns as d grows


def test_the_dispersion_fit_recovers_the_band_bottom_and_mass():
    rng = np.random.default_rng(5)
    for e0, m in ((-0.47, 0.36), (-0.44, 0.38), (-0.41, 0.42)):
        e = np.array([-0.30, -0.24, -0.18, -0.12, -0.06, 0.0, 0.06, 0.12])
        k = np.sqrt(m * (e - e0) / HBAR2_OVER_2M_EV_NM2) * (1 + rng.normal(0, 0.003, e.size))
        fit = fit_dispersion(e, k, 0.003 * k)
        assert fit["e0_ev"] == pytest.approx(e0, abs=0.005)
        assert fit["m_star"] == pytest.approx(m, rel=0.01)
        assert fit["dropped"] is None and fit["used"].all()
        assert np.abs(fit["resid_mev"]).max() < 10


def test_the_dispersion_fit_drops_one_wild_point_and_says_which():
    e = np.array([-0.30, -0.24, -0.18, -0.12, -0.06, 0.0, 0.06, 0.12])
    k = np.sqrt(0.38 * (e + 0.44) / HBAR2_OVER_2M_EV_NM2)
    k[3] *= 1.15
    fit = fit_dispersion(e, k, 0.003 * k)
    assert fit["dropped"] == 3 and not fit["used"][3]
    assert fit["m_star"] == pytest.approx(0.38, rel=0.005)
    assert fit_dispersion(e, k, 0.003 * k, drop_outliers=False)["used"].all()


def test_a_phase_free_fit_survives_an_edge_position_error_that_a_j0_fit_does_not():
    """Why the model carries a phase: the profile is aligned 0.15 nm off the true edge. Far from
    the edge that is a phase 2 k delta; J0 alone reads it as a different k, a free phase does not."""
    ss = _ss()
    e = -0.06
    d_true = np.arange(0.0, 20.0, PX)
    y = _wave(ss, e, d_true, r=0.45, phi=np.pi)     # a hard wall: 1 - J0 type
    d = d_true[3:] - 0.15                           # edge placed 0.15 nm wrongly
    y = y[3:]
    d_keep = d > 0.4
    d, y = d[d_keep], y[d_keep]
    kt = _k_true(ss, e)

    def j0_only(dd, c, a, k, L):
        return c + a * special.j0(2 * k * dd) * np.exp(-2 * dd / L)

    best = None
    for k0 in np.linspace(0.92, 1.08, 9) * kt:
        try:
            p, _ = curve_fit(j0_only, d, y, p0=(1.0, 0.4, k0, 60.0), maxfev=5000)
        except RuntimeError:
            continue
        r = np.sum((j0_only(d, *p) - y) ** 2)
        if best is None or r < best[0]:
            best = (r, p[2])
    j0_err = abs(best[1] / kt - 1)
    fit = fit_step_wave(d, y, k_guess=k_guess(e))
    assert abs(fit["k"] / kt - 1) < 0.005
    assert j0_err > 3 * abs(fit["k"] / kt - 1) and j0_err > 0.01


def test_a_fit_whose_grid_crosses_scipys_struve_hole_still_fits():
    """scipy 1.18.1's struve(0, z) is NaN on ~2.5e-5 at z = 25.765; a fit's grid in k landing on
    it once stopped the least squares (P2 reference, seeds 0, 18 and 21, 2026-11-27). A profile
    with a point exactly at 2kd = 25.76537118 for the guessed k fits, and the basis is finite."""
    from stmbench.analysis.standing_waves import _basis

    k = 1.9
    d = np.sort(np.append(np.arange(0.6, 17.0, 0.2), 25.76537118 / (2 * k)))
    assert np.isfinite(_basis(d, k, 40.0)).all()
    y = 1.0 + 0.3 * special.j0(2 * k * d) * np.exp(-2 * d / 40.0)
    fit = fit_step_wave(d, y, k_guess=k)
    assert fit["ok"] and abs(fit["k"] / k - 1) < 0.002
