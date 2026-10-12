"""The lock-in dI/dV map: what it shows, and that it disturbs nothing else."""
from __future__ import annotations

import math

import numpy as np
import pytest

from stmbench.instrument import ActionError, Instrument

P2 = "P2_standing_waves_cu111"


def _near_step(inst: Instrument, want_nm: float = 6.0) -> tuple[float, float]:
    """A scan-frame point about ``want_nm`` from a step edge (privileged: reads the surface)."""
    w = inst.world
    surf = w.surface
    ox, oy, _ = w.drift_offset(w.clock.sim())
    for x in np.arange(-150.0, 150.0, 0.5):
        left, right = surf._step_distances(x * 1e-9 + ox, oy)
        d = min(v for v in (left, right) if v is not None) if (left or right) else None
        if d is not None and abs(d * 1e9 - want_nm) < 0.5:
            return float(x), 0.0
    raise AssertionError("no step edge within ±150 nm")


def _period_nm(a: np.ndarray, nm_per_px: float) -> float:
    """Dominant spatial period of a map (2-D FFT peak away from the origin)."""
    x = a - a.mean()
    f = np.abs(np.fft.fftshift(np.fft.fft2(x * np.outer(np.hanning(x.shape[0]), np.hanning(x.shape[1])))))
    cy, cx = np.array(f.shape) // 2
    yy, xx = np.indices(f.shape)
    r = np.hypot(yy - cy, xx - cx)
    f[r < 3] = 0.0
    i = np.unravel_index(int(np.argmax(f)), f.shape)
    k_cyc = np.hypot((i[0] - cy) / (f.shape[0] * nm_per_px), (i[1] - cx) / (f.shape[1] * nm_per_px))
    return float(1.0 / k_cyc)


def test_the_lockin_adds_a_map_and_changes_nothing_else():
    """The map's noise has its own random stream: topography with and without the lock-in
    is the same to the last bit."""
    runs = []
    for on in (False, True):
        inst = Instrument(P2, 0)
        inst.act("lockin", on=on, amplitude_mv=10.0)        # same instrument time either way
        r = inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=20.0, pixels=64)
        runs.append((inst, r))
    (off, r_off), (on, r_on) = runs
    assert r_off["channels"] == ["z", "current"] and r_on["channels"] == ["z", "current", "didv"]
    assert np.array_equal(off.frame(r_off["frame"])["array"], on.frame(r_on["frame"])["array"])
    with pytest.raises(ActionError, match="channels"):
        off.frame(r_off["frame"], "didv")
    assert on.frame(r_on["frame"], "didv")["array"].shape == (64, 64)


def test_a_map_near_a_step_shows_the_standing_wave_of_its_energy():
    inst = Instrument(P2, 0)
    x, y = _near_step(inst, 7.0)
    inst.act("lockin", on=True, amplitude_mv=10.0)
    ss = inst.world.surface.surface_state
    periods = {}
    for bias in (-0.2, 0.3):
        inst.act("set_bias", bias_v=bias)
        r = inst.act("scan", center_x_nm=x, center_y_nm=y, size_nm=16.0, pixels=160, line_time_s=0.3)
        periods[bias] = _period_nm(inst.frame(r["frame"], "didv")["array"], r["nm_per_px"])
        expect = math.pi / float(ss.k_of_e(bias))           # LDOS ~ cos(2kd): period pi/k
        assert periods[bias] == pytest.approx(expect, rel=0.15), (bias, periods[bias], expect)
    assert periods[0.3] < periods[-0.2]                      # higher energy, shorter wave

