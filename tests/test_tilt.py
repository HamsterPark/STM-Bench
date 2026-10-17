"""Piezo tilt correction: Piezo.TiltSet flattens a tilted terrace and refuses absurd angles,
and a tilt step under the closed loop is a Z transient."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from stmsim.modules import build_dispatcher
from stmsim.physics.rig import RigProfile
from stmsim.physics.scanner import SIG_Z, ScanSettings
from stmsim.physics.surface import Surface
from stmsim.physics.world import World
from stmsim.wire.errors import BadArguments


SAMPLE_TILT = (4e-3, -3e-3)      # rad ≈ slope; 0.23° / −0.17° — a typical un-levelled sample


def _flat_spot(surface: Surface, window_m: float, reach_m: float = 400e-9) -> tuple[float, float]:
    """A sample-frame point whose window_m square crosses no step edge (the search is the
    test's business: the terrain is random per seed and terraces are ~17 nm on Au(111))."""
    g = np.arange(-reach_m, reach_m + 1e-12, 20e-9)
    for r in sorted({abs(v) for v in g}):
        for x in (r, -r):
            for y in g:
                if surface.step_free_window(x, y, window_m):
                    return float(x), float(y)
    raise AssertionError(f"no {window_m * 1e9:.0f} nm step-free window within ±{reach_m * 1e9:.0f} nm")


def _tilted_world(tmp_path: Path, seed: int, window_m: float, **kw) -> World:
    w = World(rig=RigProfile.load("reference-stm"), seed=seed, session_dir=tmp_path / "s",
              adsorbate_density_per_um2=0.0, **kw)
    w.surface.site.tilt = SAMPLE_TILT
    w.tip.lambda_per_s = 0.0          # a stable tip: no spontaneous apex changes mid-frame
    # no thermal drift: over a 64-row frame the rig's ~1 pm/s Z drift is 20 % of a 0.2°
    # tilt span — exactly the frame-method flaw a circle probe exists to avoid
    w.drift_v_m_per_s = np.zeros(3)
    x, y = _flat_spot(w.surface, window_m)
    w.tip_x, w.tip_y = x, y
    w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
    w.withdrawn = False
    w.zctrl_set(True)
    w.transients.clear()
    w.achievable_z_tip()
    return w


def _plane_slope(z: np.ndarray, st: ScanSettings) -> tuple[float, float]:
    """Least-squares plane through a frame → (dz/dx, dz/dy) in m/m over stage coordinates.

    Rows are stored in acquisition order (a "down" scan starts at +y), so the coordinates
    come from the settings' own pixel map rather than from the array index."""
    ny, nx = z.shape
    cols = np.arange(nx)
    gx = np.empty_like(z)
    gy = np.empty_like(z)
    for k in range(ny):
        xs, ys = st.pixel_xy(cols, st.row_index_for_acq(k))
        gx[k], gy[k] = xs - st.cx, ys - st.cy
    a = np.c_[gx.ravel(), gy.ravel(), np.ones(gx.size)]
    coef, *_ = np.linalg.lstsq(a, z.ravel(), rcond=None)
    return float(coef[0]), float(coef[1])


def _call(d, command: str, *args):
    """Invoke the bound controller handler the way the wire layer does (decoded positional
    args in, raw return out)."""
    return d._handlers[d._resolve(command)](*args)


# ── surface layer ─────────────────────────────────────────────────────────────

def test_compensation_removes_exactly_the_sample_plane():
    s = Surface("Au(111)", seed=3, adsorbate_density_per_um2=0.0)
    s.site.tilt = SAMPLE_TILT
    x = np.linspace(-30e-9, 30e-9, 301)
    y = np.linspace(-20e-9, 25e-9, 301)
    before = s.terrace_height(x, y)
    s.tilt_comp = SAMPLE_TILT
    after = s.terrace_height(x, y)
    assert s.residual_tilt() == (0.0, 0.0)
    assert np.allclose(before - after, SAMPLE_TILT[0] * x + SAMPLE_TILT[1] * y, atol=1e-15)
    # steps are untouched: the staircase is not what levelling changes
    assert np.allclose(np.diff(after)[np.abs(np.diff(after)) > 1e-10].__abs__(), s.material.step_m, atol=1e-11)
    assert s.site.tilt == SAMPLE_TILT


# ── world + controller module ────────────────────────────────────────────────────

def test_piezo_tilt_flattens_a_rendered_frame(tmp_path):
    w = _tilted_world(tmp_path, seed=21, window_m=48e-9)
    d = build_dispatcher(w)
    # 200 nm/s: under the 450 nm/s scratch hazard, so the surface is the same for every render
    st = ScanSettings(cx=w.tip_x, cy=w.tip_y, w=40e-9, h=40e-9, nx=64, ny=64,
                      line_time_fwd_s=0.2, line_time_bwd_s=0.2)
    z0 = w.renderer.render(st, w.clock.sim()).data[SIG_Z][0]
    sx0, sy0 = _plane_slope(z0, st)
    # reported-Z slope = −sample_slope/extend_sign (−1 on this rig) = +sample slope
    assert sx0 == pytest.approx(SAMPLE_TILT[0], rel=0.15)
    assert sy0 == pytest.approx(SAMPLE_TILT[1], rel=0.15)
    assert _call(d, "Piezo_TiltGet") == [0.0, 0.0]
    t0 = w.truth()
    assert t0["tilt_residual_mrad"] == pytest.approx((4.0, -3.0), abs=1e-3)   # atan(4e-3) rad, in mrad

    tx = math.degrees(math.atan(SAMPLE_TILT[0]))
    ty = math.degrees(math.atan(SAMPLE_TILT[1]))
    _call(d, "Piezo_TiltSet", tx, ty)
    assert _call(d, "Piezo_TiltGet") == pytest.approx([tx, ty], abs=1e-6)
    z1 = w.renderer.render(st, w.clock.sim()).data[SIG_Z][0]
    sx1, sy1 = _plane_slope(z1, st)
    assert math.hypot(sx1, sy1) < 0.05 * math.hypot(sx0, sy0), (sx1, sy1)
    t1 = w.truth()
    assert max(abs(v) for v in t1["tilt_residual_mrad"]) < 1e-6
    assert t1["piezo_tilt_deg"] == pytest.approx((tx, ty))
    assert t1["sample_tilt_mrad"] == pytest.approx((4.0, -3.0), abs=1e-3)
    # physical junction untouched: with the loop settled the gap is still the setpoint gap
    w.transients.clear()
    gap = w.achievable_z_tip() - w.surface_height_here()
    assert gap == pytest.approx(w.equilibrium_gap(), rel=1e-6)

    # over-compensating by the same amount tilts the frame the other way (sign is linear)
    _call(d, "Piezo_TiltSet", 2 * tx, 2 * ty)
    sx2, sy2 = _plane_slope(w.renderer.render(st, w.clock.sim()).data[SIG_Z][0], st)
    assert sx2 == pytest.approx(-SAMPLE_TILT[0], rel=0.15)
    assert sy2 == pytest.approx(-SAMPLE_TILT[1], rel=0.15)
    assert w.truth()["tilt_residual_mrad"] == pytest.approx((-4.0, 3.0), abs=1e-3)


def test_piezo_tilt_rejects_absurd_angles(tmp_path):
    w = World(rig=RigProfile.load("reference-stm"), seed=1, session_dir=tmp_path / "s")
    d = build_dispatcher(w)
    with pytest.raises(BadArguments):
        _call(d, "Piezo_TiltSet", 60.0, 0.0)
    with pytest.raises(BadArguments):
        _call(d, "Piezo_TiltSet", 0.0, float("nan"))
    assert w.piezo_tilt_deg == (0.0, 0.0)
    assert w.surface.tilt_comp == (0.0, 0.0)


def test_tilt_step_under_closed_loop_is_a_z_transient(tmp_path):
    w = _tilted_world(tmp_path, seed=21, window_m=48e-9)
    assert abs(w.tilt_plane_z(w.tip_x, w.tip_y)) < 1e-15
    # away from the x axis the correction plane has a value at the tip → the surface the
    # loop follows steps by that much → a real Z transient (why tilt is applied in
    # ≤1° sub-steps with feedback on), then the loop settles at the same gap as before
    w.move_xy(w.tip_x + 50e-9, w.tip_y)
    w.transients.clear()
    w.set_piezo_tilt(1.0, 0.0)
    assert abs(w.tilt_plane_z(w.tip_x, w.tip_y)) == pytest.approx(math.tan(math.radians(1.0)) * 50e-9, rel=1e-9)
    assert w.transients and w.transients[-1].label == "tilt"
    assert w.events[-1]["kind"] == "tilt_set"
    w.transients.clear()
    assert w.achievable_z_tip() - w.surface_height_here() == pytest.approx(w.equilibrium_gap(), rel=1e-6)
