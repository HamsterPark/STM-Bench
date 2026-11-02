"""A positive scan angle turns the frame clockwise on the stage (x right, y up).

The expected positions come from multiplying the frame offset ``u + iv`` by ``exp(-i*theta)``,
which does not share a sign convention with the matrix in ``ScanSettings.pixel_xy``.
"""
from __future__ import annotations

import cmath
import math

import numpy as np
import pytest

from stmbench.analysis.adatoms import to_scan_nm
from stmbench.render import pixel_to_scan_nm
from stmbench.replay.truth_view import AtomsAt, TruthView, frame_extent_nm
from stmbench.trackB.claims import frame_covering, to_sample_frame
from stmsim.physics.scanner import SIG_Z, ScanSettings
from stmsim.physics.surface import Feature
from tests.test_drift_mapping import FRAME
from tests.test_tilt import _tilted_world


def test_a_positive_scan_angle_turns_the_frame_clockwise():
    """Pixel (column 3, row 1) of a 4 by 4 frame, 40 nm on a side, centred at the origin.

    Frame coordinates of that pixel are u = 15 nm, v = 5 nm. A positive angle must land
    it where a clockwise turn of the rightward axis puts it.
    """
    expected = {
        90.0: (5e-9, -15e-9),
        30.0: (15.490381e-9, -3.169873e-9),
        -75.0: (-0.947343e-9, 15.782983e-9),
    }
    for angle, want in expected.items():
        st = ScanSettings(cx=0.0, cy=0.0, w=40e-9, h=40e-9, nx=4, ny=4, angle_deg=angle)
        assert st.pixel_xy(3, 1) == pytest.approx(want, abs=1e-15)

    st = ScanSettings(cx=0.0, cy=0.0, w=40e-9, h=40e-9, nx=4, ny=4, angle_deg=90.0)
    x0, y0 = st.pixel_xy(1, 1)
    x_right, y_right = st.pixel_xy(2, 1)
    x_down, y_down = st.pixel_xy(1, 2)
    # one column to the right has turned to stage -y; one row down has turned to stage -x
    assert (x_right - x0, y_right - y0) == pytest.approx((0.0, -10e-9), abs=1e-15)
    assert (x_down - x0, y_down - y0) == pytest.approx((-10e-9, 0.0), abs=1e-15)


@pytest.mark.parametrize("angle", [-137.0, -75.0, -30.0, 0.0, 15.0, 33.0, 90.0, 180.0])
def test_pixel_xy_is_a_clockwise_rotation_for_any_angle(angle):
    st = ScanSettings(cx=30e-9, cy=-12e-9, w=80e-9, h=20e-9, nx=8, ny=3, angle_deg=angle)
    cols, rows = np.meshgrid(np.arange(st.nx), np.arange(st.ny))
    got = np.stack(st.pixel_xy(cols, rows))
    u = (cols + 0.5) / st.nx * st.w - st.w / 2
    v = st.h / 2 - (rows + 0.5) / st.ny * st.h
    turned = (st.cx + 1j * st.cy) + (u + 1j * v) * cmath.exp(-1j * math.radians(angle))
    expect = np.stack((np.real(turned), np.imag(turned)))
    assert np.allclose(got, expect, rtol=0, atol=1e-15)


def test_a_zero_angle_frame_is_unchanged_to_the_last_bit():
    """At angle 0 the sine term is +0, so either sign leaves every finite value unchanged."""
    st = ScanSettings(cx=12.5e-9, cy=-7.5e-9, w=80e-9, h=30e-9, nx=5, ny=4, angle_deg=0.0)
    cols = np.array([-1.0, 0.0, 1.5, 4.0])
    rows = np.array([-0.5, 0.0, 2.0, 3.5])
    got = np.stack(st.pixel_xy(cols, rows))
    u = (cols + 0.5) / st.nx * st.w - st.w / 2
    v = st.h / 2 - (rows + 0.5) / st.ny * st.h
    th = math.radians(0.0)
    c, s = math.cos(th), math.sin(th)
    old = np.stack((st.cx + u * c - v * s, st.cy + u * s + v * c))
    assert np.array_equal(got, old)


def _clockwise_nm(cx_m, cy_m, u_m, v_m, angle_deg) -> tuple[float, float]:
    """Stage nm of a frame offset, from ``(u + iv) * exp(-i*theta)`` — not from ``pixel_xy``."""
    turned = complex(u_m, v_m) * cmath.exp(-1j * math.radians(angle_deg))
    return (cx_m + turned.real) * 1e9, (cy_m + turned.imag) * 1e9


def _uv_m(w_m, h_m, nx, ny, col, row) -> tuple[float, float]:
    u = (col + 0.5) / nx * w_m - w_m / 2
    v = h_m / 2 - (row + 0.5) / ny * h_m
    return u, v


@pytest.mark.parametrize("angle", [-75.0, 33.0, 90.0])
@pytest.mark.parametrize("col,row", [(0.0, 0.0), (3.5, 1.25), (-1.0, 2.0), (10.0, -2.0)])
def test_render_pixel_to_scan_nm_agrees_with_the_scanner(angle, col, row):
    st = ScanSettings(cx=30e-9, cy=-12e-9, w=80e-9, h=20e-9, nx=8, ny=5, angle_deg=angle)
    geom = {"cx_nm": st.cx * 1e9, "cy_nm": st.cy * 1e9, "w_nm": st.w * 1e9, "h_nm": st.h * 1e9,
            "angle_deg": angle, "nx": st.nx, "ny": st.ny}
    u, v = _uv_m(st.w, st.h, st.nx, st.ny, col, row)
    want = _clockwise_nm(st.cx, st.cy, u, v, angle)
    assert pixel_to_scan_nm(geom, col, row) == pytest.approx(want, abs=1e-9)
    x, y = st.pixel_xy(col, row)
    assert pixel_to_scan_nm(geom, col, row) == pytest.approx((float(x) * 1e9, float(y) * 1e9), abs=1e-9)


@pytest.mark.parametrize("angle", [-75.0, 33.0, 90.0])
@pytest.mark.parametrize("col,row", [(0.0, 0.0), (2.5, 4.0), (5.0, 1.0)])
def test_adatom_analysis_pixel_to_scan_nm_agrees_with_the_scanner(angle, col, row):
    n = 6
    size_nm = 40.0
    frame = {"center_nm": [30.0, -12.0], "size_nm": size_nm, "array": np.zeros((n, n)), "angle_deg": angle}
    st = ScanSettings(cx=30e-9, cy=-12e-9, w=size_nm * 1e-9, h=size_nm * 1e-9, nx=n, ny=n, angle_deg=angle)
    u, v = _uv_m(st.w, st.h, st.nx, st.ny, col, row)
    want = _clockwise_nm(st.cx, st.cy, u, v, angle)
    assert to_scan_nm(frame, col, row) == pytest.approx(want, abs=1e-9)
    x, y = st.pixel_xy(col, row)
    assert to_scan_nm(frame, col, row) == pytest.approx((float(x) * 1e9, float(y) * 1e9), abs=1e-9)


@pytest.mark.parametrize("angle", [-75.0, 33.0, 90.0])
def test_the_replay_frame_outline_agrees_with_the_scanner(angle):
    st = ScanSettings(cx=30e-9, cy=-12e-9, w=80e-9, h=20e-9, nx=8, ny=5, angle_deg=angle)
    geom = {"cx_nm": 30.0, "cy_nm": -12.0, "w_nm": 80.0, "h_nm": 20.0, "angle_deg": angle, "nx": 8, "ny": 5}
    corners = frame_extent_nm({}, geom, None)
    pixels = ((-0.5, -0.5), (st.nx - 0.5, -0.5), (st.nx - 0.5, st.ny - 0.5), (-0.5, st.ny - 0.5))
    for corner, (col, row) in zip(corners, pixels):
        u, v = _uv_m(st.w, st.h, st.nx, st.ny, col, row)
        assert corner == pytest.approx(list(_clockwise_nm(st.cx, st.cy, u, v, angle)), abs=1e-9)
        x, y = st.pixel_xy(col, row)
        assert corner == pytest.approx([float(x) * 1e9, float(y) * 1e9], abs=1e-9)


def test_the_judges_footprint_test_agrees_with_the_scanner():
    """A point near the top-right of a 30 degree, non-square frame.

    Its clockwise stage position lies inside the frame; the anticlockwise position does not,
    because there |v| is 38.3 nm on a frame only 20 nm tall.
    """
    fr = {"kind": "scan_saved", "complete": True, "cx_m": 0.0, "cy_m": 0.0, "w_m": 80e-9, "h_m": 20e-9,
          "angle_deg": 30.0, "nx": 64, "ny": 16, "idx": 1}
    rule = {"min_fov_nm": 10, "max_nm_per_px": 5}
    assert frame_covering([fr], 38.274991, -11.705771, rule) is fr
    assert frame_covering([fr], 29.274991, 27.294229, rule) is None


def test_the_judges_row_mapping_agrees_with_the_scanner():
    fr = {**FRAME, "w_m": 60e-9, "h_m": 60e-9, "nx": 256, "ny": 256, "angle_deg": 90.0,
          "scan_dir": "down", "row_offsets_nm": [[0.02 * k, 0.0] for k in range(256)]}
    # clockwise, (25 nm, 0) is in the upper part of the frame (v = +25 nm), acquisition row 20.83
    assert to_sample_frame(fr, 25.0, 0.0)[0] == pytest.approx(25.416667, abs=1e-4)


def _peak(image: np.ndarray) -> tuple[int, int]:
    flat = int(np.argmax(np.abs(image - np.median(image))))
    row, col = np.unravel_index(flat, image.shape)
    return int(row), int(col)


def test_a_feature_is_imaged_where_the_clockwise_map_puts_it(tmp_path):
    """A bump 10 nm to the right of the frame centre, imaged at +90 degrees.

    The prediction is the clockwise map, not ``pixel_xy``: column 31.5, row 15.5.
    The opposite convention would put it on row 47.5.
    """
    w = _tilted_world(tmp_path, seed=3, window_m=48e-9)
    x0, y0 = w.tip_x, w.tip_y
    w.surface.add_feature(Feature(x0 + 10e-9, y0, 1.0e-9, 2.0e-9, kind="cluster"))
    st = ScanSettings(cx=x0, cy=y0, w=40e-9, h=40e-9, nx=64, ny=64, angle_deg=90.0,
                      line_time_fwd_s=0.2, line_time_bwd_s=0.2)
    z = w.renderer.render(st, w.clock.sim()).data[SIG_Z][0]
    row, col = _peak(z)
    assert math.hypot(col - 31.5, row - 15.5) <= 2.0
    assert math.hypot(col - 31.5, row - 47.5) >= 20.0

    view = TruthView(w, [])
    truth = view.frame(
        {"sim0": 0.0, "per_row_s": 0.0, "scan_dir": "down", "bias_v": None},
        {"nx": 64, "ny": 64, "w_nm": 40.0, "h_nm": 40.0, "cx_nm": x0 * 1e9, "cy_nm": y0 * 1e9,
         "angle_deg": 90.0},
        AtomsAt(None),
    )
    trow, tcol = _peak(truth)
    assert math.hypot(tcol - 31.5, trow - 15.5) <= 2.0
    assert math.hypot(tcol - 31.5, trow - 47.5) >= 20.0


def test_a_scratch_runs_along_the_fast_axis_of_a_rotated_frame(tmp_path):
    """The groove a too-fast scan leaves follows the fast axis, (cos θ, −sin θ) at +30 degrees."""
    w = _tilted_world(tmp_path, seed=1, window_m=48e-9)
    w.tip.radius_m = 3e-9

    class _Always:
        def random(self):
            return 0.0

    w.rng = _Always()
    st = ScanSettings(w=40e-9, h=40e-9, nx=32, ny=8, angle_deg=30.0, line_time_fwd_s=0.08)
    w.maybe_scratch(st, 0, 500e-9, w.clock.sim())
    groove = w.surface.site.features[-1]
    assert groove.kind == "scratch"
    step = 10e-9
    th = math.radians(30.0)
    along = groove.eval(np.array(groove.x + step * math.cos(th)), np.array(groove.y - step * math.sin(th)))
    perp = groove.eval(np.array(groove.x + step * math.sin(th)), np.array(groove.y + step * math.cos(th)))
    # the groove is a negative height; compare its depth with |height|
    assert abs(float(along)) > 0.5 * abs(groove.height)
    assert abs(float(perp)) < 1e-3 * abs(groove.height)
