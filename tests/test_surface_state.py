"""Surface-state standing waves: the dispersion, the interference, and whether the task
the P2 scenario asks for can actually be done."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import optimize, special

from stmsim.physics.surface import Surface
from stmsim.physics.surface_state import (HBAR2_OVER_2M_EV_NM2, PointScatterer, SurfaceState,
                                          SurfaceStateParams)

CU = SurfaceStateParams.for_material("Cu(111)")


def test_dispersion_matches_the_free_electron_relation():
    ss = SurfaceState(CU)
    k_f = float(ss.k_of_e(0.0))
    assert k_f == pytest.approx(5.123 * math.sqrt(0.38 * 0.44), rel=1e-3)
    assert k_f == pytest.approx(2.09, abs=0.02)
    assert math.pi / k_f == pytest.approx(1.5, abs=0.05)     # the 1.5 nm ripple of the 1993 papers
    assert float(ss.k_of_e(CU.e0_ev - 0.05)) == 0.0          # nothing below the band bottom
    # E = E0 + hbar^2 k^2 / 2m*, inverted
    for e in (-0.3, 0.0, 0.4):
        k = float(ss.k_of_e(e))
        assert CU.e0_ev + HBAR2_OVER_2M_EV_NM2 * k ** 2 / CU.m_star == pytest.approx(e, abs=1e-9)


def test_phase_coherence_length_matches_the_measured_value():
    """Bürgi's 66 nm at the Fermi level on Cu(111) is what Γ_F was chosen to reproduce."""
    assert float(SurfaceState(CU).l_phi_nm(0.0)) == pytest.approx(66.0, rel=0.25)


def test_hard_wall_step_is_one_minus_j0():
    p = SurfaceStateParams.for_material("Cu(111)", gamma_f_ev=1e-9, gamma_0_ev=1e-9,
                                        temperature_k=1e-6)
    ss = SurfaceState(p)
    d = np.linspace(0.5e-9, 20e-9, 400)
    s = ss.line_modulation(d, 0.0, 1.0, math.pi)
    k = float(ss.k_of_e(0.0))
    assert np.allclose(s, -special.j0(2 * k * d * 1e9), atol=2e-3)


def test_a_transparent_scatterer_scatters_nothing():
    ss = SurfaceState(CU)
    pt = [PointScatterer(0.0, 0.0, delta=0.0, alpha=1.0)]
    s = ss.point_modulation(np.linspace(1e-9, 10e-9, 50), np.zeros(50), 0.0, pt)
    assert np.allclose(s, 0.0, atol=1e-12)


def test_interference_dies_away_from_the_scatterer():
    ss = SurfaceState(CU)
    pt = [PointScatterer(0.0, 0.0)]
    near = float(np.ravel(ss.point_modulation(np.array([1e-9]), np.array([0.0]), 0.0, pt))[0])
    far = float(np.ravel(ss.point_modulation(np.array([120e-9]), np.array([0.0]), 0.0, pt))[0])
    assert abs(near) > 20 * abs(far)


@pytest.mark.parametrize("e_ev", [-0.3, 0.0, 0.3])
def test_ripple_period_is_pi_over_k(e_ev):
    """The wavelength of the standing wave at a fixed energy is π/k(E) — the measurement the
    whole paper rests on."""
    p = SurfaceStateParams.for_material("Cu(111)", gamma_f_ev=1e-4, gamma_0_ev=1e-4)
    ss = SurfaceState(p)
    k = float(ss.k_of_e(e_ev))
    d = np.arange(2e-9, 30e-9, 0.02e-9)
    s = ss.line_modulation(d, e_ev, 0.6, math.pi)
    s = s - s.mean()
    freq = np.fft.rfftfreq(d.size, d[1] - d[0])
    got = 1.0 / freq[np.argmax(np.abs(np.fft.rfft(s * np.hanning(d.size))))]
    assert got == pytest.approx(math.pi / k * 1e-9, rel=0.06)


def test_multiple_scattering_reduces_to_single_for_one_scatterer():
    ss = SurfaceState(CU)
    pt = [PointScatterer(0.0, 0.0, delta=1.1, alpha=0.5)]
    x = np.linspace(1e-9, 12e-9, 40)
    y = np.zeros_like(x)
    a = ss.point_modulation(x, y, 0.0, pt, multiple=False)
    b = ss.point_modulation(x, y, 0.0, pt, multiple=True)
    assert np.allclose(a, b, atol=1e-12)


def test_corral_resonances_land_near_the_hard_wall_levels():
    """A ring of 48 scatterers confines the surface state; with a nearly reflecting wall the
    resonances sit close to the circular-box levels, and absorption damps them."""
    from stmsim.physics.surface import _hard_wall_levels

    ss_hard = SurfaceState(SurfaceStateParams.for_material("Cu(111)", gamma_f_ev=2e-3,
                                                           gamma_0_ev=6e-3))
    radius, n_atoms = 7.13e-9, 48
    ang = np.arange(n_atoms) * 2 * np.pi / n_atoms
    ring = [PointScatterer(radius * math.cos(a), radius * math.sin(a), math.pi / 2, 1.0)
            for a in ang]
    grid = np.arange(CU.e0_ev + 0.02, 0.3, 0.002)
    s = ss_hard.point_modulation(np.array(0.0), np.array(0.0), grid, ring, multiple=True)
    s = np.asarray(s).ravel()
    peaks, _ = __import__("scipy.signal", fromlist=["find_peaks"]).find_peaks(
        s, prominence=0.08 * float(np.ptp(s)))
    got = grid[peaks]
    hard = _hard_wall_levels(CU.e0_ev, CU.m_star, radius, n=6)
    assert len(got) >= 3
    for e in got[:3]:
        assert min(abs(e - h) for h in hard) < 0.03, (e, hard)
    # an absorbing ring gives the same resonances, weaker
    lossy = [PointScatterer(p.x, p.y, 1.2, 0.4) for p in ring]
    s2 = np.asarray(ss_hard.point_modulation(np.array(0.0), np.array(0.0), grid, lossy,
                                             multiple=True)).ravel()
    assert float(np.ptp(s2)) < float(np.ptp(s))


def test_a_gap_in_the_ring_barely_changes_the_centre_spectrum():
    """Pinning a finding: the l = 0 modes do not care about a few missing atoms, so the P3
    repair task is judged on occupancy and images, not on the spectrum."""
    ss = SurfaceState(CU)
    radius, n_atoms = 7.13e-9, 48
    ang = np.arange(n_atoms) * 2 * np.pi / n_atoms
    full = [PointScatterer(radius * math.cos(a), radius * math.sin(a)) for a in ang]
    gapped = [p for i, p in enumerate(full) if i not in (10, 11, 12, 13)]
    grid = np.arange(CU.e0_ev + 0.02, 0.2, 0.002)
    a = np.asarray(ss.point_modulation(np.array(0.0), np.array(0.0), grid, full, multiple=True)).ravel()
    b = np.asarray(ss.point_modulation(np.array(0.0), np.array(0.0), grid, gapped, multiple=True)).ravel()
    assert float(np.max(np.abs(a - b))) < 0.35 * float(np.ptp(a))


def test_topograph_and_spectrum_come_from_the_same_ldos(tmp_path):
    from stmsim.physics.rig import RigProfile
    from stmsim.physics.world import World

    w = World(rig=RigProfile.load("reference-stm"), seed=2, material="Cu(111)",
              session_dir=tmp_path / "s", adsorbate_density_per_um2=0.0,
              surface_state=CU)
    w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
    w.withdrawn = False
    w.zctrl_set(True)
    w.bias_v = 0.05
    # the same step that modulates the image also modulates the spectrum taken over it
    left, right = w.surface._step_distances(0.0, 0.0)
    d = min(v for v in (left, right) if v is not None)
    assert d is not None
    near = w.surface.ldos_at(0.0, 0.0)
    far = w.surface.ldos_at(0.0, 0.0)
    assert near.modulation(0.0) == pytest.approx(far.modulation(0.0))     # cached, same place
    e = np.array([-0.2, 0.0, 0.2])
    assert np.any(np.abs(near.rho(e) - 1.0) > 0.05)


def test_no_surface_state_leaves_everything_untouched(tmp_path):
    """B6 and every earlier scenario must see the material template and a zero contribution."""
    from stmsim.physics.junction import LDOSTemplate

    s = Surface("Au(111)", seed=3)
    assert s.surface_state is None
    assert isinstance(s.ldos_at(0.0, 0.0), LDOSTemplate)
    assert not isinstance(s.ldos_at(0.0, 0.0), object.__class__)
    x = np.linspace(-5e-9, 5e-9, 32)
    assert np.array_equal(s.electronic_height(x, x, 0.1, 1e10, 0.08e-9), np.zeros_like(x))


def test_e0_and_m_star_are_recoverable_from_the_full_generator():
    """The task itself: fit the ripples the generator produces — lifetime, complex reflection
    phase, damping and all — with the simple J0 model an experimenter uses, and see whether
    E0 and m* come back inside the scenario's tolerance."""
    p = SurfaceStateParams.for_material("Cu(111)", e0_ev=-0.437, m_star=0.402,
                                        step_r_up=0.55, step_phi_up_rad=math.pi - 0.3)
    ss = SurfaceState(p)
    energies = np.linspace(-0.35, 0.25, 12)
    d = np.linspace(1.0e-9, 21.0e-9, 96)
    rng = np.random.default_rng(0)
    ks = []
    for e in energies:
        y = 1.0 + p.step_height * float(ss.band_edge(e)) * (
            1.0 + ss.line_modulation(d, e, p.step_r_up, p.step_phi_up_rad))
        y = y + rng.normal(0, 0.02 * float(np.ptp(y)), d.size)      # 2 % measurement noise

        def model(x, amp, k, lam, c):
            return c + amp * special.j0(2 * k * x * 1e9) * np.exp(-2 * x * 1e9 / lam)

        k0 = float(ss.k_of_e(e)) or 1.0
        try:
            popt, _ = optimize.curve_fit(model, d, y, p0=[0.2, k0, 60.0, float(np.mean(y))],
                                         bounds=([-2, 0.2, 5, -5], [2, 12, 500, 5]), maxfev=20000)
        except RuntimeError:
            continue
        ks.append((e, popt[1]))
    assert len(ks) >= 8
    e_arr = np.array([a for a, _ in ks])
    k_arr = np.array([b for _, b in ks])
    # E = E0 + (hbar^2/2m*) k^2 — a straight line in k^2
    slope, intercept = np.polyfit(k_arr ** 2, e_arr, 1)
    e0_fit = intercept
    m_fit = HBAR2_OVER_2M_EV_NM2 / slope
    assert abs(e0_fit - p.e0_ev) * 1e3 < 25.0, (e0_fit, p.e0_ev)     # the scenario's 25 meV
    assert abs(m_fit - p.m_star) / p.m_star < 0.12, (m_fit, p.m_star)


def test_nothing_propagates_below_the_band_bottom_so_no_nan_marks_e0():
    """Below E0 the interference term is zero, not NaN. Evaluating Y0(0) there put a NaN
    edge at exactly E0 into every spectrum near a scatterer, which handed the hidden band
bottom to a reader of the resulting `.dat` file."""
    from stmsim.physics.surface_state import local_ldos

    ss = SurfaceState(CU)
    pts = [PointScatterer(0.0, 0.0), PointScatterer(3e-9, 1e-9)]
    grid = np.arange(-1.0, 0.6, 0.001)
    below = grid < CU.e0_ev - 1e-9
    for multiple in (False, True):
        s = np.asarray(ss.point_modulation(np.array(1.5e-9), np.array(0.4e-9), grid, pts,
                                           multiple=multiple), float).ravel()
        assert np.isfinite(s).all(), multiple
        assert np.all(s[below] == 0.0) and np.any(s[~below] != 0.0), multiple
    rho = local_ldos(ss, 1.5e-9, 0.4e-9, points=pts).rho(grid)
    assert np.isfinite(rho).all() and (rho > 0).all()
    # the edge is the Lorentzian-broadened step, not a cliff: the largest jump between
    # neighbouring 1 meV samples is a small fraction of the step height
    assert np.max(np.abs(np.diff(rho))) < 0.1 * CU.step_height
    # the bias-integrated closed form at a bias window that starts below the band bottom
    m = ss.point_modulation_integrated(np.array([1.5e-9, 4e-9]), np.array([0.4e-9, 0.0]),
                                       CU.e0_ev - 0.1, pts)
    assert np.isfinite(m).all()
