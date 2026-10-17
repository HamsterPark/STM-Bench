"""Frame and spectrum rendering (:mod:`stmbench.render`): pixel → scan-frame nm, plane
flattening, controller-header geometry, and the pictures of saved frames and spectra."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest


# ── geometry and flattening ────────────────────────────────────────────────
def test_pixel_to_scan_nm_is_the_inverse_of_the_judges_footprint_test():
    """A position the person reads off a frame must be a position the judge finds covered by
    that frame — same centre, same axes, same rotation sense, row 0 at the high-v edge."""
    from stmbench.render import pixel_to_scan_nm
    from stmbench.trackB.claims import frame_covering

    geom = {"cx_nm": 120.0, "cy_nm": -40.0, "w_nm": 80.0, "h_nm": 50.0, "angle_deg": 33.0, "nx": 64, "ny": 40}
    fr = {"kind": "scan_saved", "complete": True, "cx_m": 120e-9, "cy_m": -40e-9, "w_m": 80e-9, "h_m": 50e-9,
          "angle_deg": 33.0, "nx": 64, "ny": 40, "idx": 1}
    rule = {"min_fov_nm": 10, "max_nm_per_px": 5}
    for col, row in [(0, 0), (63, 0), (0, 39), (63, 39), (31.5, 19.5)]:
        x, y = pixel_to_scan_nm(geom, col, row)
        assert frame_covering([fr], x, y, rule) is fr, (col, row, x, y)
    x, y = pixel_to_scan_nm(geom, -1.2, 0)          # a pixel past the edge is outside
    assert frame_covering([fr], x, y, rule) is None
    g0 = {**geom, "angle_deg": 0.0}
    assert pixel_to_scan_nm(g0, 31.5, 19.5) == pytest.approx((120.0, -40.0))   # centre pixel = scan offset
    assert pixel_to_scan_nm(g0, 31.5, 0)[1] > pixel_to_scan_nm(g0, 31.5, 39)[1]  # row 0 is the high-y edge
    assert pixel_to_scan_nm(g0, 0, 19.5)[0] < pixel_to_scan_nm(g0, 63, 19.5)[0]


def test_flatten_removes_a_plane_and_keeps_unacquired_rows_nan():
    from stmbench.render import flatten

    yy, xx = np.mgrid[0:16, 0:16]
    z = 3.0 * xx + 2.0 * yy + 5.0
    z[-1] = np.nan
    f = flatten(z, "plane")
    assert np.nanmax(np.abs(f[:-1])) < 1e-9 and np.isnan(f[-1]).all()
    ln = flatten(z, "line")
    assert np.allclose(ln[0], ln[5]) and np.isnan(ln[-1]).all()
    assert np.array_equal(flatten(z, "none")[:-1], z[:-1])


def test_frame_geometry_reads_a_controller_header():
    from stmbench.render import frame_geometry

    h = {"scan_range": "1.200000E-07       6.000000E-08", "scan_offset": "5.000000E-08      -2.000000E-08",
         "scan_angle": "3.000000E+01", "scan_pixels": [384, 192]}
    g = frame_geometry(h)
    assert g == {"cx_nm": pytest.approx(50.0), "cy_nm": pytest.approx(-20.0), "w_nm": pytest.approx(120.0),
                 "h_nm": pytest.approx(60.0), "angle_deg": 30.0, "nx": 384, "ny": 192}
    assert frame_geometry({}) is None


# ── pictures of saved frames and spectra ───────────────────────────────────
def _tunnelling_world(tmp_path, seed: int, time_scale: float):
    from stmsim.physics.rig import RigProfile
    from stmsim.physics.world import World

    w = World(rig=RigProfile.load("reference-stm"), seed=seed, session_dir=tmp_path / "s", time_scale=time_scale)
    w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
    w.withdrawn = False
    w.zctrl_set(True)
    w.transients.clear()
    w.achievable_z_tip()
    return w


def test_frames_and_spectra_render_for_the_person(tmp_path):
    pytest.importorskip("matplotlib")
    from stmsim.io.dat_writer import write_dat

    from stmbench import render

    w = _tunnelling_world(tmp_path, seed=5, time_scale=20.0)
    w.scan.nx = w.scan.ny = 32
    w.scan.w = w.scan.h = 20e-9
    w.scan.cx, w.scan.cy, w.scan.angle_deg = 30e-9, -10e-9, 15.0
    w.scan.line_time_fwd_s = w.scan.line_time_bwd_s = 0.02
    w.scan_start()
    time.sleep(0.6)
    w.scan_stop()
    path = w.save_frame()
    assert path and Path(path).exists()
    png, meta = render.render_frame(path, channel="Z", direction="forward", flatten_mode="plane", max_px=64)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and meta["nx"] == 32 and meta["ny"] == 32 and meta["unit"] in ("pm", "nm")
    assert meta["hi"] > meta["lo"]
    bwd, _ = render.render_frame(path, direction="backward", flatten_mode="line")
    assert bwd[:8] == b"\x89PNG\r\n\x1a\n"
    fm = render.frame_meta(path)
    assert fm["cx_nm"] == pytest.approx(30.0) and fm["cy_nm"] == pytest.approx(-10.0)
    assert fm["w_nm"] == pytest.approx(20.0) and fm["angle_deg"] == pytest.approx(15.0) and fm["nx"] == 32
    assert "Z" in fm["channels"]
    # the geometry the page maps mouse positions through is the geometry the judge recorded
    ev = [e for e in w.events if e["kind"] == "scan_saved"][-1]
    assert (ev["cx_m"] * 1e9, ev["cy_m"] * 1e9, ev["angle_deg"]) == pytest.approx((fm["cx_nm"], fm["cy_nm"], fm["angle_deg"]))
    # a spectrum
    dat = tmp_path / "s" / "spec001.dat"
    v = np.linspace(-1.0, 1.0, 50)
    write_dat(dat, header={"Experiment": "bias spectroscopy", "X (m)": 1e-9, "Y (m)": 2e-9, "Bias>Bias (V)": 0.1},
              names=["Bias (V)", "Current (A)"], columns=[v, v ** 2 * 1e-10])
    assert render.render_dat(dat)[:8] == b"\x89PNG\r\n\x1a\n"
    dm = render.dat_meta(dat)
    assert dm["x_nm"] == pytest.approx(1.0) and dm["columns"] == ["Bias (V)", "Current (A)"] and dm["n_points"] == 50
    files = render.list_files(tmp_path / "s")
    assert {f["kind"] for f in files} == {"sxm", "dat"} and files[0]["mtime"] >= files[-1]["mtime"]
