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


def test_a_small_corrals_lowest_resonance_is_counted():
    """The truth's resonances at a corral's centre are counted from 5 meV above the band
    bottom. From 20 meV (until 2026-10-13) the n = 1 resonance of small rings, which leaks
    5-20 meV above its hard-wall level, was cut: a measured spectrum showed it, the truth did
    not, and every rank was off by one."""
    import tempfile
    from pathlib import Path

    from stmsim.scenario import Scenario

    sc = Scenario.load(Path(__file__).resolve().parent.parent / "stmbench" / "trackB" / "scenarios"
                       / "P3_corral_cu111.yaml")
    sc.hidden["adatoms"] = {**sc.hidden["adatoms"], "ring_radius_nm": [4.5, 5.0]}     # small rings
    for seed in range(3):
        w = sc.build_world(seed, session_dir=Path(tempfile.mkdtemp()) / "s", virtual_clock=True)
        c = w.surface.corral_snapshot()
        hw1 = c["hard_wall_ev"][0] * 1e3
        assert c["peaks_mev"][0] - hw1 < 25.0, (seed, c["radius_nm"], c["peaks_mev"][:2], hw1)


def test_the_struve_hole_leaves_a_finite_wave_and_every_other_value_as_it_was():
    """scipy 1.18.1's struve(0, z) is NaN on ~2.5e-5 around z = 25.765, and one pixel of a dI/dV
    map whose 2kd fell there blanked the whole map (2026-11-27). A step's wave is finite there
    now, continuous with its neighbours, and wherever scipy's value is finite it is the value it
    was, bit for bit."""
    ss = SurfaceState(CU)
    e, r, phi = 0.0, 0.5, 3.0
    k = float(ss.k_of_e(e))
    z = np.concatenate([np.linspace(25.7650, 25.7657, 7001), np.linspace(0.05, 60.0, 4001)])
    d_m = z / (2.0 * k) * 1e-9
    got = ss.line_modulation(d_m, e, r, phi)
    assert np.isfinite(got).all()
    # the expression as it was, evaluated the same way
    d_nm = np.abs(np.asarray(d_m, float)) * 1e9
    zz = 2 * ss.k_of_e(e) * d_nm
    damp = np.exp(-2 * d_nm / np.maximum(ss.l_phi_nm(e), 1e-6))
    raw = special.struve(0, zz)
    before = r * (math.cos(phi) * special.j0(zz) + math.sin(phi) * raw) * damp
    fin = np.isfinite(raw)
    assert np.array_equal(got[fin], before[fin])
    hole = np.flatnonzero(~fin)
    if hole.size:                                   # the hole this scipy has; a fixed one has none
        assert np.max(np.abs(got[hole] - got[hole.min() - 1])) < 1e-4
        one = ss.line_modulation(float(d_m[hole[0]]), e, r, phi)      # a scalar argument too
        assert np.isfinite(one) and abs(float(one) - float(got[hole[0]])) < 1e-12


def _edge_k(d_nm: np.ndarray, y: np.ndarray, k0: float, span: float = 0.08) -> float:
    """k of one edge's wave, c + A[cos φ J0(2kd) + sin φ H0(2kd)]·exp(-2d/L), on a grid in k and L
    with (c, A cos φ, A sin φ) solved linearly at every node (H0 through the simulator's own
    hole-free struve, since a k grid this fine lands on scipy's NaN at z = 25.765)."""
    from stmsim.physics.surface_state import _struve0

    best = (np.inf, k0)
    for lam in (20.0, 40.0, 80.0, 160.0, 400.0):
        damp = np.exp(-2.0 * d_nm / lam)
        for k in k0 * np.linspace(1.0 - span, 1.0 + span, 2001):
            z = 2.0 * k * d_nm
            B = np.column_stack([np.ones_like(d_nm), special.j0(z) * damp, _struve0(z) * damp])
            coef, *_ = np.linalg.lstsq(B, y, rcond=None)
            r = float(np.sum((B @ coef - y) ** 2))
            if r < best[0]:
                best = (r, float(k))
    return best[1]


def _wander_slope(site, v: float) -> float:
    a, p = site.wander_amp, site.wander_period
    return float(a * 2 * math.pi / p * math.cos(2 * math.pi * v / p)
                 + 0.4 * a * 2 * math.pi / (0.37 * p) * math.cos(2 * math.pi * v / (0.37 * p) + 1.3))


def test_a_bent_step_shows_its_true_wavelength_across_the_local_edge():
    """Where a step bends, its ripples follow it: a profile across the local edge reads the band's
    k, and so does the line term a spectrum uses (2026-11-27). Before, the waves were laid out along
    the staircase's mean normal and such a profile read k high by sqrt(1 + w'^2) - 1."""
    e = -0.10
    for seed in range(60):
        surf = Surface("Cu(111)", seed=seed, adsorbate_density_per_um2=0.0, surface_state=CU,
                       terrace_median_m=150e-9)
        site = surf.site
        b = np.asarray(site.boundaries)
        j = int(np.searchsorted(b, 0.0))
        if j + 1 >= b.size or b[j + 1] - b[j] < 60e-9:
            continue
        vs = np.linspace(-400e-9, 400e-9, 1601)
        slopes = np.array([_wander_slope(site, v) for v in vs])
        i = int(np.argmax(np.abs(slopes)))
        if abs(slopes[i]) >= 0.2:
            break
    else:
        raise AssertionError("no seed with a bend of 11 degrees or more")
    v, wp = float(vs[i]), float(slopes[i])
    assert math.sqrt(1 + wp ** 2) - 1 > 0.019            # what the old layout would have read high
    a = site.step_angle
    n = np.array([math.cos(a), math.sin(a)])
    t = np.array([-math.sin(a), math.cos(a)])
    wander = float(surf._staircase_u(np.array([v * t[0]]), np.array([v * t[1]]))[0])   # t is normal to n
    p0 = (b[j] - wander) * n + v * t                       # on the edge, at the bend
    normal = (n + wp * t) / math.hypot(1.0, wp)            # across the local edge, into the terrace
    s = np.arange(0.6, 16.0, 0.2)
    pts = p0[None, :] + (s * 1e-9)[:, None] * normal[None, :]
    y = surf.ldos_map(pts[:, 0], pts[:, 1], e)
    y = y / np.mean(y)
    k_true = float(surf.surface_state.k_of_e(e))
    k = _edge_k(s, y, k_true)
    assert abs(k / k_true - 1.0) < 0.005, (seed, math.degrees(math.atan(wp)), k / k_true - 1.0)
    # the spectrum's line term: 5 nm across the edge is 5 nm away
    x5, y5 = p0 + 5e-9 * normal
    dists = [float(ln.distance_m) for ln in surf._line_terms(float(x5), float(y5))]
    assert min(abs(d - 5e-9) for d in dists) < 0.03e-9, dists


def test_scenarios_without_a_surface_state_never_reach_the_step_stretch(monkeypatch):
    """The distance across the local edge only exists where a surface state does: P4, P7 and
    every scenario without one never call it, so their worlds are what they were, bit for bit."""
    from stmbench.instrument import Instrument

    def refuse(self, *a, **k):
        raise AssertionError("_staircase_stretch called without a surface state")

    monkeypatch.setattr(Surface, "_staircase_stretch", refuse)
    for scenario in ("P4_atom_positioning_cu111", "P7_pull_threshold_cu111"):
        inst = Instrument(scenario, 0)
        assert inst.world.surface.surface_state is None
        inst.act("lockin", on=True, amplitude_mv=10.0)
        inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=20.0, pixels=48, line_time_s=0.1)
        inst.act("sts", v_start=-0.5, v_end=0.1, points=41, backward=False)
