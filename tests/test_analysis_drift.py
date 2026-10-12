"""Drift from two frames: the shift of the features, on frames the instrument took."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.ndimage import shift as nd_shift

from stmbench.analysis.drift import frame_shift
from stmbench.instrument import Instrument


def _still(seed: int) -> Instrument:
    inst = Instrument("P5_sader_jarvis_qplus", seed)
    w = inst.world
    w.drift_v_m_per_s = np.zeros(3)
    w._creep.clear()
    w.note_move = lambda *a, **k: None
    return inst


def _atom_nm(inst: Instrument) -> tuple[float, float]:
    reg = inst.world.surface.site.adatoms
    a = min(reg.atoms.values(), key=lambda a: math.hypot(*reg.site_xy(a.i, a.j, a.sub)))
    return tuple(v * 1e9 for v in reg.site_xy(a.i, a.j, a.sub))


def test_a_known_shift_comes_back():
    inst = _still(1)
    r = inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=12.0, pixels=128, line_time_s=0.1)
    a = inst.frame(r["frame"])["array"]
    npx = r["nm_per_px"]
    for dx, dy in ((0.37, -0.81), (-1.2, 0.5)):
        b = nd_shift(a, (-dy / npx, dx / npx), mode="nearest", order=3)     # +y is up: rows decrease
        got = frame_shift(a, b, npx)
        assert got == pytest.approx((dx, dy), abs=0.03)


@pytest.mark.parametrize("seed", [1, 2])
def test_moving_the_scan_window_moves_the_features_the_other_way(seed):
    """Two real frames, the second with its window 0.5 nm right and 0.3 nm down: the features
    appear 0.5 nm left and 0.3 nm up. Plain cross-correlation; phase correlation locks onto the
    Cu lattice here and misses by 0.1-0.2 nm."""
    inst = _still(seed)
    ax, ay = _atom_nm(inst)
    r1 = inst.act("scan", center_x_nm=ax, center_y_nm=ay, size_nm=12.0, pixels=128, line_time_s=0.1)
    r2 = inst.act("scan", center_x_nm=ax + 0.5, center_y_nm=ay - 0.3, size_nm=12.0, pixels=128, line_time_s=0.1)
    f1, f2 = inst.frame(r1["frame"]), inst.frame(r2["frame"])
    assert f2["t_start_s"] > f1["t_start_s"]
    assert frame_shift(f1["array"], f2["array"], r1["nm_per_px"]) == pytest.approx((-0.5, 0.3), abs=0.06)
