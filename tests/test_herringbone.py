"""Au(111) herringbone: the legacy field, and rotational domains on top of it."""
from __future__ import annotations

import math

import numpy as np
import pytest

from stmsim.physics.herringbone import (HerringboneParams, height, orientation_at,
                                        orientation_from_snapshot, snapshot)
from stmsim.physics.rig import RigProfile
from stmsim.physics.scanner import SIG_Z, ScanSettings
from stmsim.physics.surface import Surface
from stmsim.physics.world import World


def _flat_spot(surface: Surface, window_m: float, reach_m: float = 500e-9) -> tuple[float, float]:
    """A sample point whose window crosses no step edge — terraces are ~17 nm on Au(111)."""
    g = np.arange(-reach_m, reach_m + 1e-12, 10e-9)
    for r in sorted({abs(v) for v in g}):
        for x in (r, -r):
            for y in g:
                if surface.step_free_window(x, y, window_m):
                    return float(x), float(y)
    raise AssertionError(f"no {window_m * 1e9:.0f} nm step-free window in reach")


def _legacy(m, s, x, y):
    """The single-orientation cosine the simulator used before domains existed."""
    th = s.lattice_angle + np.pi / 6
    u = x * math.cos(th) + y * math.sin(th)
    v = -x * math.sin(th) + y * math.cos(th)
    zig = np.sign(np.sin(2 * np.pi * v / (m.chevron_period_m or 30e-9)))
    return m.herringbone_amp_m * np.cos(2 * np.pi * (u + 0.15 * zig * v) / m.herringbone_period_m)


def test_single_domain_equals_legacy_formula():
    """The default field must be the OLD field, to the bit: every calibrated number in the
    repo (σ*, the fidelity bands, the hysteresis fit) was measured on it."""
    s = Surface("Au(111)", seed=3, adsorbate_density_per_um2=0.0)
    g = np.linspace(-50e-9, 50e-9, 301)
    x, y = np.meshgrid(g, g)
    assert np.array_equal(s.herringbone_height(x, y), _legacy(s.material, s.site, x, y))


def test_rng_streams_untouched_by_domains():
    """Domain layout draws from its own stream: the terrain and adsorbates cannot move."""
    plain = Surface("Au(111)", seed=3)
    before = (plain.site.step_angle, plain.site.lattice_angle, plain.site.tilt,
              plain.site.features[0].x, len(plain.site.features))
    params = HerringboneParams.from_nm({"domain_spacing_nm": 150.0})
    domained = Surface("Au(111)", seed=3, herringbone=params)
    g = np.linspace(-200e-9, 200e-9, 51)
    domained.herringbone_height(*np.meshgrid(g, g))       # force the layout to be built
    after = (domained.site.step_angle, domained.site.lattice_angle, domained.site.tilt,
             domained.site.features[0].x, len(domained.site.features))
    assert before == after


@pytest.mark.parametrize("period_nm,arm_deg", [(6.3, 8.0), (6.45, 15.0)])
def test_fft_period_and_arm_orientation(period_nm, arm_deg):
    """A 2D FFT of the field peaks at the drawn period, along one of the two arms."""
    params = HerringboneParams.from_nm({"period_nm": period_nm, "arm_half_angle_deg": arm_deg,
                                        "chevron_period_nm": 30.0, "amp_pm": 12.0})
    s = Surface("Au(111)", seed=7, adsorbate_density_per_um2=0.0, herringbone=params)
    span = 100e-9
    n = 512
    g = (np.arange(n) - n / 2) * span / n
    x, y = np.meshgrid(g, g)
    z = height(x, y, s.site, params)
    z = z - z.mean()
    p = np.abs(np.fft.fftshift(np.fft.fft2(z * np.hanning(n)[:, None] * np.hanning(n)[None, :])))
    freq = np.fft.fftshift(np.fft.fftfreq(n, span / n))
    fx, fy = np.meshgrid(freq, freq)
    r = np.hypot(fx, fy)
    p[r < 1.0 / 40e-9] = 0.0                       # ignore DC and the chevron sidebands
    i, j = np.unravel_index(np.argmax(p), p.shape)
    got_period = 1.0 / r[i, j]
    assert got_period == pytest.approx(period_nm * 1e-9, rel=0.05)
    # the peak is a stripe wavevector: its direction is the mean orientation ± the arm angle
    k_deg = math.degrees(math.atan2(fy[i, j], fx[i, j])) % 180.0
    mean_k = (math.degrees(s.site.lattice_angle + math.pi / 6)) % 180.0
    off = min(abs(k_deg - mean_k) % 180.0, 180.0 - abs(k_deg - mean_k) % 180.0)
    assert off == pytest.approx(arm_deg, abs=4.0)


def test_layout_gives_three_orientations_and_boundaries():
    params = HerringboneParams.from_nm({"domain_spacing_nm": 150.0, "boundary_width_nm": 3.0})
    s = Surface("Au(111)", seed=11, adsorbate_density_per_um2=0.0, herringbone=params)
    snap = snapshot(s.site, params, half_m=1e-6)
    assert not snap["single_domain"]
    ks = {d["k"] for d in snap["domains"]}
    assert ks <= {0, 1, 2} and len(ks) >= 2
    # the three stripe directions are 60° apart, and every domain reports one of them
    ori = snap["orientations_deg"]
    assert min(abs((ori[1] - ori[0]) % 180.0 - 60.0), abs((ori[1] - ori[0]) % 180.0 - 120.0)) < 1e-6
    for d in snap["domains"]:
        assert d["stripe_deg"] == pytest.approx(ori[d["k"]])
    # a second orientation is reachable from the start position, or the task is unsolvable
    near = [d["k"] for d in snap["domains"]
            if abs(d["seed_xy_nm"][0]) < 600 and abs(d["seed_xy_nm"][1]) < 600]
    assert len(set(near)) >= 2
    # orientation_at agrees with the snapshot the judge reads, at any point
    for x_nm, y_nm in [(0, 0), (120, -80), (-250, 200)]:
        live = orientation_at(x_nm * 1e-9, y_nm * 1e-9, s.site, params)
        assert live["stripe_deg"] == pytest.approx(orientation_from_snapshot(snap, x_nm, y_nm))
    # the wall distance is the Voronoi half-gap
    layout = s.site.hb_layout
    d = np.hypot(layout.seeds[:, 0] - 30e-9, layout.seeds[:, 1] - 40e-9)
    d.sort()
    got = orientation_at(30e-9, 40e-9, s.site, params)["boundary_distance_nm"]
    assert got == pytest.approx((d[1] - d[0]) / 2 * 1e9, rel=1e-9)


def test_rendered_frame_matches_the_orientation_truth(tmp_path):
    """The whole chain — renderer, drift, pixel geometry — has to land on the same angle the
    judge reads out of the truth, or a right answer would be failed for a convention."""
    params = HerringboneParams.from_nm({"period_nm": 6.45, "arm_half_angle_deg": 12.0,
                                        "chevron_period_nm": 30.0, "amp_pm": 14.0})
    w = World(rig=RigProfile.load("reference-stm"), seed=5, session_dir=tmp_path / "s",
              adsorbate_density_per_um2=0.0, herringbone=params)
    w.tip.apex_sigma_m = 0.07e-9
    w.tip.lambda_per_s = 0.0
    w.drift_v_m_per_s = np.zeros(3)
    w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
    w.withdrawn = False
    w.zctrl_set(True)
    # a step is 235 pm and the reconstruction 14 pm, so this is measured on a flat terrace —
    # which is what an operator does too (FindFlatRegion before the reconstruction frame)
    cx, cy = _flat_spot(w.surface, 45e-9)
    # the operator's real line time: the Z loop low-passes along the fast axis only, and at
    # 12× that speed a 6.45 nm stripe (125 Hz) is smeared out of the forward direction while
    # the row-to-row component survives — a real artefact, and not what this test is about
    st = ScanSettings(cx=cx, cy=cy, w=40e-9, h=40e-9, nx=256, ny=256,
                      line_time_fwd_s=0.586, line_time_bwd_s=0.586)
    fr = w.renderer.render(st, 0.0)
    z = np.asarray(fr.data[SIG_Z][0], float)
    # the sample plane is 400 pm across this window and the reconstruction 14 pm, so the plane
    # comes off first — the same first step every real analysis of an STM frame takes
    ii, jj = np.mgrid[0:256, 0:256]
    A = np.c_[ii.ravel(), jj.ravel(), np.ones(ii.size)]
    coef, *_ = np.linalg.lstsq(A, z.ravel(), rcond=None)
    z = (z.ravel() - A @ coef).reshape(z.shape)
    p = np.abs(np.fft.fftshift(np.fft.fft2(z * np.hanning(256)[:, None] * np.hanning(256)[None, :])))
    freq = np.fft.fftshift(np.fft.fftfreq(256, st.w / st.nx))
    fx, fy = np.meshgrid(freq, freq)
    r = np.hypot(fx, fy)
    band = (r > 1 / 10e-9) & (r < 1 / 4.5e-9)         # the stripe band only
    p = np.where(band, p, 0.0)
    i, j = np.unravel_index(np.argmax(p), p.shape)
    assert 1.0 / r[i, j] == pytest.approx(6.45e-9, rel=0.10)
    # frame row 0 is the TOP of the scan window (pixel_xy: v decreases with the row index), so
    # the array's slow axis runs along −y in the sample frame. Pinning that here is the point
    # of this test: a right orientation reported in the scan frame must not fail on a sign.
    k_deg = math.degrees(math.atan2(-fy[i, j], fx[i, j])) % 180.0
    stripe_deg = (k_deg + 90.0) % 180.0
    truth = w.surface.orientation_at(cx, cy)["stripe_deg"]
    off = abs(stripe_deg - truth) % 180.0
    assert min(off, 180.0 - off) <= 15.0             # within the arm angle of the mean direction
