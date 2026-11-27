"""Stripe analysis (stmbench.analysis.herringbone): synthetic gratings of known period and
direction, then frames of a P1 world against its truth."""
from __future__ import annotations

import math

import numpy as np
import pytest

from stmbench import render
from stmbench.analysis import herringbone as hb


def grating(n=300, d=0.4, T=6.3, stripe_deg=30.0, alpha_deg=12.0, chev=28.0, amp=10e-12, noise=0.0,
            h2=0.2, step=0.0, seed=0, phase=0.3):
    """The simulator's own height expression (stmsim.physics.herringbone.height) on an n x n
    window, row 0 on top, plus a faint tilt: stripes along ``stripe_deg`` (mean), zigzag
    +-alpha with chevron period ``chev``; ``step`` puts a step edge through the window."""
    rng = np.random.default_rng(seed)
    x = (np.arange(n) - n / 2) * d
    y = (n / 2 - np.arange(n)) * d
    X, Y = np.meshgrid(x, y)
    th = math.radians(stripe_deg - 90.0)               # u runs along the stripe wavevector
    u = X * math.cos(th) + Y * math.sin(th)
    v = -X * math.sin(th) + Y * math.cos(th)
    zig = np.sign(np.sin(2 * np.pi * v / chev + phase))
    ph = 2 * np.pi * (u + math.tan(math.radians(alpha_deg)) * zig * v) / T
    z = amp * (np.cos(ph) + h2 * np.cos(2 * ph)) + noise * rng.standard_normal((n, n))
    z = z + 0.3e-12 * X + 0.2e-12 * Y
    if step:
        z = z + step * (X > 5.0)
    return z


def angle_err(value, accepted):
    return min(abs(((value - a) + 90.0) % 180.0 - 90.0) for a in accepted)


@pytest.mark.parametrize("i", range(8))
def test_two_arm_gratings_give_the_along_arm_period_and_the_directions(i):
    rng = np.random.default_rng(i)
    T, alpha = rng.uniform(6.0, 6.6), rng.uniform(8.0, 18.0)
    sd = rng.uniform(0.0, 180.0)
    z = grating(T=T, stripe_deg=sd, alpha_deg=alpha, chev=rng.uniform(22, 34), amp=rng.uniform(8, 15) * 1e-12,
                noise=3e-12, seed=i, phase=rng.uniform(0, 6))
    a = hb.assess_window(z, 0.4)
    assert a.measurable, a.reason
    assert abs(a.period_along_arm_nm - T * math.cos(math.radians(alpha))) < 0.03
    accepted = [sd, (sd + alpha) % 180.0, (sd - alpha) % 180.0]
    assert angle_err(a.direction_deg, accepted) < 3.0
    assert all(angle_err(d, accepted) < 3.0 for d in a.arm_deg)
    if a.verdict == "ok":
        assert abs(a.half_angle_deg - alpha) < 2.0
        assert abs(a.period_mean_nm - T) < 0.1


def test_the_mean_direction_peak_is_not_taken_for_an_arm():
    """Many two-arm windows also show a peak on the mean direction (at the longer period T,
    as strong as an arm): the arms are the outer two of three, never the middle one."""
    triples = 0
    for i in range(24):
        rng = np.random.default_rng(100 + i)
        T, alpha, sd = rng.uniform(6.0, 6.6), rng.uniform(8.0, 18.0), rng.uniform(0.0, 180.0)
        z = grating(T=T, stripe_deg=sd, alpha_deg=alpha, chev=rng.uniform(22, 34), noise=3e-12, seed=i,
                    phase=rng.uniform(0, 6.28))
        a = hb.assess_window(z, 0.4)
        if a.n_peaks == 3 and a.verdict == "ok":
            triples += 1
            assert abs(a.period_along_arm_nm - T * math.cos(math.radians(alpha))) < 0.03
            assert angle_err(a.mean_deg, [sd]) < 1.5 and abs(a.half_angle_deg - alpha) < 1.5
    assert triples >= 3


def test_a_step_is_reported_and_stripes_alone_are_not_a_step():
    flat = hb.assess_window(grating(noise=3e-12), 0.4)
    stepped = hb.assess_window(grating(noise=3e-12, step=235e-12), 0.4)
    assert flat.verdict != "step" and flat.step_pp_pm < hb.STEP_PP_PM
    assert stepped.verdict == "step" and stepped.step_pp_pm > 2 * hb.STEP_PP_PM
    assert not stepped.measurable and "step" in stepped.reason


def test_noise_alone_has_no_stripe_peak():
    rng = np.random.default_rng(1)
    a = hb.assess_window(5e-12 * rng.standard_normal((300, 300)), 0.4)
    assert a.verdict == "no_stripe" and not a.measurable


def test_two_domains_in_one_window_are_called_mixed():
    left = grating(stripe_deg=20.0, alpha_deg=10.0, noise=2e-12)
    right = grating(stripe_deg=80.0, alpha_deg=10.0, noise=2e-12, seed=1)
    z = np.hstack([left[:, :150], right[:, 150:]])
    a = hb.assess_window(z, 0.4)
    assert a.verdict == "mixed" and not a.measurable


def test_an_image_direction_is_turned_into_the_scan_frame_by_the_scan_angle():
    z = grating(stripe_deg=60.0, alpha_deg=10.0, noise=2e-12)
    a0 = hb.assess_window(z, 0.4, angle_deg=0.0)
    a30 = hb.assess_window(z, 0.4, angle_deg=30.0)       # image turned clockwise: scan = image - 30
    assert angle_err(a30.direction_deg, [(a0.direction_deg - 30.0) % 180.0]) < 0.1
    assert angle_err(a0.direction_deg, [60.0]) < 3.0


@pytest.mark.parametrize("angle", [0.0, 25.0, -40.0])
def test_pixel_mapping_is_the_render_modules_and_invertible(angle):
    geom = {"center_nm": [12.5, -7.0], "size_nm": 80.0, "pixels": 200, "angle_deg": angle}
    rg = {"cx_nm": 12.5, "cy_nm": -7.0, "w_nm": 80.0, "h_nm": 80.0, "nx": 200, "ny": 200, "angle_deg": angle}
    for col, row in [(0.5, 0.5), (100.0, 37.3), (199.5, 150.0)]:
        got = hb.pixel_to_scan_nm(geom, col, row)
        assert np.allclose(got, render.pixel_to_scan_nm(rg, col, row))
        assert np.allclose(hb.scan_nm_to_pixel(geom, *got), (col, row))


def test_survey_tiles_a_frame_and_reports_window_centres():
    z = grating(n=300, noise=2e-12)
    frame = {"array": z, "nm_per_px": 0.4, "pixels": 300, "size_nm": 120.0, "center_nm": [10.0, 20.0],
             "angle_deg": 0.0}
    res = hb.survey(frame, 60.0)
    assert len(res) == 4
    assert np.allclose(res[0].centre_nm, (10.0 - 30.0 + 0.2 * 0, 20.0 + 30.0), atol=0.3)
    assert hb.median_period_nm(res) is not None
    one = hb.assess_at(frame, (10.0, 20.0), 60.0)
    assert one is not None and one.measurable
    assert hb.assess_at(frame, (500.0, 0.0), 60.0) is None


# ── frames of a P1 world ──────────────────────────────────────────────────────
def test_frames_of_a_p1_world_agree_with_its_truth():
    """Seed 13 (arrives with a good tip): a 200 nm frame scanned slowly enough not to scratch
    (<= 450 nm/s). The step detector agrees with the world's own step-free test on every
    window, and every measured window is within tolerance of the truth there."""
    from stmbench.instrument import Instrument

    inst = Instrument("P1_barth1990_au111", 13)
    w = inst.world
    truth_p = w.truth()["herringbone"]["period_along_arm_nm"]
    r = inst.act("scan", center_x_nm=0, center_y_nm=0, size_nm=200, pixels=500, line_time_s=200 / 430.0)
    fr = inst.frame(r["frame"])
    ox, oy, _ = w.drift_offset(w.clock.sim())
    sf = w.surface
    measured = 0
    for a in hb.survey(fr, 80.0, 60.0):
        sx, sy = (a.centre_nm[0] + ox * 1e9) * 1e-9, (a.centre_nm[1] + oy * 1e9) * 1e-9
        stepped = not sf.step_free_window(sx, sy, 80e-9)
        assert (a.verdict == "step") == stepped, (a.centre_nm, a.verdict, a.step_pp_pm)
        if not a.measurable:
            continue
        measured += 1
        t = sf.orientation_at(sx, sy)
        accepted = [t["stripe_deg"], *t["arms_deg"]]
        assert angle_err(a.direction_deg, accepted) < 6.0, (a.centre_nm, a.arm_deg, accepted)
        # the frame is uncompensated: the drift (~1 nm/min here) smears it, hence 0.4 nm
        assert abs(a.period_along_arm_nm - truth_p) < 0.4
    assert measured >= 3


def test_strong_chevron_sidebands_are_not_another_domain():
    """A short chevron with wide arms (alpha 16, chevron 23 nm) puts sidebands at 55-75 % of an arm
    10-16 degrees outside it: the window is still one domain."""
    for phase in (0.3, 1.7, 3.1, 4.4):
        z = grating(T=6.51, stripe_deg=47.0, alpha_deg=16.0, chev=23.3, amp=11e-12, noise=3e-12, phase=phase)
        a = hb.assess_window(z, 0.4)
        assert a.verdict == "ok", (phase, a.reason)
        assert abs(a.period_along_arm_nm - 6.51 * math.cos(math.radians(16.0))) < 0.03
        assert angle_err(a.mean_deg, [47.0]) < 1.5


def test_a_small_arm_angle_does_not_make_an_arm_and_the_mean_peak_a_pair():
    """At alpha = 8.6 degrees the mean-direction peak's period is only 1.1 % longer than the arms'
    (seed 24 of the P1 calibration read the mean 3.7 degrees off when the tolerance was 1.2 %)."""
    for i, phase in enumerate((0.3, 1.1, 2.0, 2.9, 3.8, 4.7, 5.6)):
        z = grating(T=6.06, stripe_deg=159.2, alpha_deg=8.6, chev=33.4, amp=13e-12, noise=3e-12, seed=i, phase=phase)
        a = hb.assess_window(z, 0.4)
        assert a.measurable, (phase, a.reason)
        assert abs(a.period_along_arm_nm - 6.06 * math.cos(math.radians(8.6))) < 0.03, (phase, a.verdict)
        assert angle_err(a.direction_deg, [159.2, 159.2 + 8.6, 159.2 - 8.6]) < 2.0, (phase, a.verdict, a.arm_deg)


def test_a_bow_is_not_a_step_and_a_jump_of_whole_rows_is():
    """The Z loop's creep curves a window by tens of pm (seed 55 of the P1 calibration scored 50 pm
    after a plane fit and was called stepped); a tip change leaves a jump of whole rows, which
    looks like a step edge and is called one."""
    n = 300
    yy = (np.arange(n) - n / 2) / (n / 2)
    bow = 60e-12 * np.outer(yy ** 2, np.ones(n))
    assert hb.assess_window(grating(noise=3e-12) + bow, 0.4).verdict != "step"
    jump = np.zeros((n, n))
    jump[150:] = 100e-12
    a = hb.assess_window(grating(noise=3e-12) + jump, 0.4)
    assert a.verdict == "step" and a.step_pp_pm > hb.STEP_PP_PM
