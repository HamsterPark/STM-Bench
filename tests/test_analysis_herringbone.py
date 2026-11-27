"""Stripe analysis (stmbench.analysis.herringbone) on P1's continuous herringbone: synthetic
zigzags of known period and direction at the corners of the scenario's ranges (alpha 28-32 deg,
chevron period 40-60 nm, T 6.9-7.6 nm), then a frame of a P1 world against its truth."""
from __future__ import annotations

import math

import numpy as np
import pytest

from stmbench import render
from stmbench.analysis import herringbone as hb


def zigzag(n=300, d=0.4, T=7.3, stripe_deg=30.0, alpha_deg=30.0, L=50.0, amp=10e-12, noise=0.0, h2=0.2,
           step=0.0, seed=0, pv=0.3):
    """The simulator's continuous-elbow height expression (stmsim.physics.herringbone.height) on an
    n x n window, row 0 on top, plus a faint tilt. Returns (z, local line direction per pixel in
    degrees, the arm period T cos(alpha), the grids X and Y in nm)."""
    rng = np.random.default_rng(seed)
    x = (np.arange(n) - n / 2 + 0.5) * d
    y = (n / 2 - 0.5 - np.arange(n)) * d
    X, Y = np.meshgrid(x, y)
    th = math.radians(stripe_deg - 90.0)
    u = X * math.cos(th) + Y * math.sin(th)
    v = -X * math.sin(th) + Y * math.cos(th)
    xc = 2 * np.pi * v / L + 2 * np.pi * pv
    g = -(L / (2 * np.pi)) * np.arcsin(np.clip(np.cos(xc), -1, 1))
    ph = 2 * np.pi * (u + math.tan(math.radians(alpha_deg)) * g) / T
    z = amp * (np.cos(ph) + h2 * np.cos(2 * ph)) + noise * rng.standard_normal((n, n))
    z = z + 0.3e-12 * X + 0.2e-12 * Y
    if step:
        z = z + step * (X > 5.0)
    loc = (stripe_deg + np.where(np.sin(xc) >= 0, 1.0, -1.0) * alpha_deg) % 180.0
    return z, loc, T * math.cos(math.radians(alpha_deg)), X, Y


def frame_of(z, d=0.4, angle=0.0, centre=(0.0, 0.0)):
    n = z.shape[0]
    return {"array": z, "nm_per_px": d, "pixels": n, "size_nm": n * d, "center_nm": list(centre),
            "angle_deg": angle}


def angle_err(value, truth):
    return abs(((value - truth) + 90.0) % 180.0 - 90.0)


CORNERS = [(T, a, L) for T in (6.9, 7.6) for a in (28.0, 32.0) for L in (40.0, 60.0)]


@pytest.mark.parametrize("T,alpha,L", CORNERS)
def test_a_window_reads_both_arms_with_the_arm_period_at_the_corners_of_the_ranges(T, alpha, L):
    sd = 70.0
    z, _, pa, _, _ = zigzag(T=T, alpha_deg=alpha, L=L, stripe_deg=sd, noise=3e-12, amp=10e-12, pv=0.4)
    a = hb.assess_window(z, 0.4)
    assert a.verdict == "ok" and a.n_arms == 2, a.reason
    assert abs(a.period_along_arm_nm - pa) < 0.04                 # the claim allows 0.15
    arms = sorted(arm.direction_deg for arm in a.arms)
    want = sorted([(sd + alpha) % 180.0, (sd - alpha) % 180.0])
    assert all(angle_err(x, y) < 2.5 for x, y in zip(arms, want))
    # nothing is read off the mean direction: no arm sits near it
    assert all(angle_err(arm.direction_deg, sd) > 20.0 for arm in a.arms)


def test_random_draws_of_the_ranges_give_period_and_directions():
    errs, derrs = [], []
    for i in range(20):
        r = np.random.default_rng(i)
        T, al, L, sd = r.uniform(6.9, 7.6), r.uniform(28, 32), r.uniform(40, 60), r.uniform(0, 180)
        z, _, pa, _, _ = zigzag(T=T, alpha_deg=al, L=L, stripe_deg=sd, noise=3e-12, pv=r.uniform(),
                                amp=r.uniform(8, 15) * 1e-12, seed=i)
        a = hb.assess_window(z, 0.4)
        assert a.n_arms == 2
        errs.append(abs(a.period_along_arm_nm - pa))
        want = [(sd + al) % 180.0, (sd - al) % 180.0]
        derrs += [min(angle_err(arm.direction_deg, t) for t in want) for arm in a.arms]
    assert max(errs) < 0.04 and max(derrs) < 2.5


def test_a_step_a_row_jump_and_noise():
    z = zigzag(noise=3e-12)[0]
    assert hb.assess_window(z, 0.4).verdict == "ok"
    assert hb.assess_window(zigzag(noise=3e-12, step=235e-12)[0], 0.4).verdict == "step"
    jump = np.zeros(z.shape)
    jump[150:] = 100e-12
    assert hb.assess_window(z + jump, 0.4).verdict == "step"
    yy = (np.arange(300) - 150) / 150
    bow = 60e-12 * np.outer(yy ** 2, np.ones(300))
    assert hb.assess_window(z + bow, 0.4).verdict == "ok"          # a bow is not a step
    rng = np.random.default_rng(1)
    assert hb.assess_window(5e-12 * rng.standard_normal((300, 300)), 0.4).verdict == "no_stripe"


def test_two_regions_in_one_window_give_their_arms():
    """A window across a region wall holds the arms of both: three or four combs, never mistaken
    for the two of one zigzag."""
    left = zigzag(stripe_deg=20.0, noise=2e-12)[0]
    right = zigzag(stripe_deg=80.0, noise=2e-12, seed=1)[0]
    a = hb.assess_window(np.hstack([left[:, :150], right[:, 150:]]), 0.4)
    assert a.verdict == "ok" and a.n_arms >= 3


def test_the_scan_angle_turns_an_image_direction_into_the_scan_frame():
    z = zigzag(stripe_deg=60.0, noise=2e-12)[0]
    a0 = hb.assess_window(z, 0.4, angle_deg=0.0)
    a30 = hb.assess_window(z, 0.4, angle_deg=30.0)        # image turned clockwise: scan = image - 30
    for arm0 in a0.arms:
        assert min(angle_err(arm.direction_deg, (arm0.direction_deg - 30.0) % 180.0) for arm in a30.arms) < 0.1


@pytest.mark.parametrize("angle", [0.0, 25.0, -40.0])
def test_pixel_mapping_is_the_render_modules_and_invertible(angle):
    geom = {"center_nm": [12.5, -7.0], "size_nm": 80.0, "pixels": 200, "angle_deg": angle}
    rg = {"cx_nm": 12.5, "cy_nm": -7.0, "w_nm": 80.0, "h_nm": 80.0, "nx": 200, "ny": 200, "angle_deg": angle}
    for col, row in [(0.5, 0.5), (100.0, 37.3), (199.5, 150.0)]:
        got = hb.pixel_to_scan_nm(geom, col, row)
        assert np.allclose(got, render.pixel_to_scan_nm(rg, col, row))
        assert np.allclose(hb.scan_nm_to_pixel(geom, *got), (col, row))


def test_survey_tiles_a_frame_and_the_median_period_is_the_arm_period():
    z, _, pa, _, _ = zigzag(n=300, noise=2e-12)
    frame = frame_of(z, centre=(10.0, 20.0))
    res = hb.survey(frame, 60.0)
    assert len(res) == 4 and np.allclose(res[0].centre_nm, (10.0 - 30.0, 20.0 + 30.0), atol=0.3)
    assert abs(hb.median_period_nm(res) - pa) < 0.1
    one = hb.assess_at(frame, (10.0, 20.0), 60.0)
    assert one is not None and one.measurable
    assert hb.assess_at(frame, (500.0, 0.0), 60.0) is None


def test_the_local_direction_in_the_middle_of_an_arm_is_the_arms_and_uniform():
    z, loc, _, X, Y = zigzag(n=350, noise=3e-12, stripe_deg=40.0, L=50.0, pv=0.2)
    fr = frame_of(z)
    om = hb.orientation_map(fr)
    found = 0
    for r in range(40, 310, 6):
        for c in range(40, 310, 6):
            rd = hb.local_direction(fr, *hb.pixel_to_scan_nm(fr, c, r), omap=om)
            if rd.ok:
                found += 1
                assert angle_err(rd.direction_deg, loc[r, c]) < 4.0, (r, c, rd)
    assert found > 100
    # across an elbow line the field is not uniform: a reading there says so
    rows = [r for r in range(40, 310) if abs(loc[r, 175] - loc[r - 1, 175]) > 20]
    assert rows
    rd = hb.local_direction(fr, *hb.pixel_to_scan_nm(fr, 175, rows[0]), omap=om)
    assert not rd.ok and rd.spread_deg > hb.MAX_SPREAD_DEG


def test_arm_places_finds_the_middle_of_two_neighbouring_arms():
    for i in range(6):
        r = np.random.default_rng(50 + i)
        al, L, sd = r.uniform(28, 32), r.uniform(40, 60), r.uniform(0, 180)
        z, loc, _, _, _ = zigzag(n=350, alpha_deg=al, L=L, stripe_deg=sd, T=r.uniform(6.9, 7.6), noise=3e-12,
                                 pv=r.uniform(), amp=r.uniform(8, 15) * 1e-12, seed=i)
        fr = frame_of(z)
        places = hb.arm_places(fr)
        assert len(places) >= 2
        a, b = places[:2]
        assert 50.0 < angle_err(a.direction_deg, b.direction_deg) * 1.0 + 0 < 70.0 or \
            angle_err(a.direction_deg, b.direction_deg) > 40.0
        for p in (a, b):
            col, row = hb.scan_nm_to_pixel(fr, p.x_nm, p.y_nm)
            assert angle_err(p.direction_deg, loc[int(round(row)), int(round(col))]) < 4.0
            assert p.clearance_nm > 5.0 and p.spread_deg < hb.MAX_SPREAD_DEG


def test_a_step_makes_a_reading_near_it_not_ok():
    z = zigzag(n=350, noise=3e-12, step=235e-12)[0]
    fr = frame_of(z)
    om = hb.orientation_map(fr)
    rd_near = hb.local_direction(fr, 5.0, 0.0, omap=om)
    rd_far = hb.local_direction(fr, -50.0, 0.0, omap=om)
    assert rd_near.step_nearby and not rd_near.ok
    assert not rd_far.step_nearby


def test_a_frame_at_a_scan_angle_gives_directions_in_the_scan_frame():
    z, loc, _, _, _ = zigzag(n=350, noise=2e-12, stripe_deg=40.0)
    plain = hb.arm_places(frame_of(z))
    turned = hb.arm_places(frame_of(z, angle=30.0))
    for p in plain:
        assert min(angle_err(q.direction_deg, (p.direction_deg - 30.0) % 180.0) for q in turned) < 2.0
