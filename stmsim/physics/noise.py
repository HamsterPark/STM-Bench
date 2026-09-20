"""Noise streams for current and Z.

Everything is generated from one seeded ``numpy`` generator so a scenario replays exactly.
Two access patterns:

* :meth:`NoiseModel.current_sample` — one O(1) sample at a given wall time (for
  ``Current.Get`` polled at kHz rates by MAST's read-back skills);
* :meth:`NoiseModel.current_series` — a vector for a scope screen / a scan line.

Components: white (preamp floor + shot), 1/f, mains pickup (50 Hz + harmonics, plus
optional spectral lines such as a 787.6 Hz pump line),
random telegraph noise when the tip is unstable, and coarse-motor crosstalk spikes.
"""
from __future__ import annotations

import math

import numpy as np

_E = 1.602176634e-19


class NoiseModel:
    def __init__(self, rng: np.random.Generator, *, floor_a: float = 0.05e-12,
                 bandwidth_hz: float = 1000.0, mains_hz: float = 50.0,
                 mains_amp_a: float = 0.2e-12, pink_amp_a: float = 0.1e-12,
                 spectral_lines: dict[float, float] | None = None,
                 z_floor_m: float = 2e-12):
        self.rng = rng
        self.floor_a = floor_a
        self.bandwidth_hz = bandwidth_hz
        self.mains_hz = mains_hz
        self.mains_amp_a = mains_amp_a
        self.pink_amp_a = pink_amp_a
        self.spectral_lines = dict(spectral_lines or {})   # Hz → amplitude (A)
        self.z_floor_m = z_floor_m
        # random telegraph noise state
        self.rtn_rate_hz = 0.0          # 0 → off; set by the tip model when unstable
        self.rtn_amp_a = 0.0
        self._rtn_level = 1.0
        self._rtn_last_t = 0.0
        # transient spike list (t_wall, amplitude, tau)
        self.spikes: list[tuple[float, float, float]] = []

    # ── scalar ──
    def _rtn(self, t: float) -> float:
        if self.rtn_rate_hz <= 0 or self.rtn_amp_a <= 0:
            return 0.0
        dt = max(0.0, t - self._rtn_last_t)
        self._rtn_last_t = t
        p_flip = 1.0 - math.exp(-self.rtn_rate_hz * dt)
        if self.rng.random() < p_flip:
            self._rtn_level = -self._rtn_level
        return self.rtn_amp_a * self._rtn_level

    def _spikes(self, t: float) -> float:
        out = 0.0
        keep = []
        for t0, a, tau in self.spikes:
            if t < t0:
                keep.append((t0, a, tau))
                continue
            if t - t0 < 6 * tau:
                out += a * math.exp(-(t - t0) / tau)
                keep.append((t0, a, tau))
        self.spikes = keep
        return out

    def add_spike(self, t_wall: float, amp_a: float, tau_s: float = 0.02) -> None:
        self.spikes.append((t_wall, amp_a, tau_s))

    def current_sample(self, i_dc: float, t: float) -> float:
        """One sample around ``i_dc`` at wall time ``t`` (bandwidth-limited)."""
        shot = math.sqrt(2 * _E * abs(i_dc) * self.bandwidth_hz)
        white = self.rng.normal(0.0, math.hypot(self.floor_a, shot))
        pink = self.pink_amp_a * self.rng.normal(0.0, 0.5)
        mains = self.mains_amp_a * math.sin(2 * math.pi * self.mains_hz * t)
        lines = sum(a * math.sin(2 * math.pi * f * t) for f, a in self.spectral_lines.items())
        return i_dc + white + pink + mains + lines + self._rtn(t) + self._spikes(t)

    def z_sample(self) -> float:
        return self.rng.normal(0.0, self.z_floor_m)

    # ── vectors ──
    def current_series(self, i_dc: np.ndarray | float, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, float)
        i_dc = np.broadcast_to(np.asarray(i_dc, float), t.shape)
        n = t.size
        shot = np.sqrt(2 * _E * np.abs(i_dc) * self.bandwidth_hz)
        white = self.rng.normal(0.0, 1.0, n) * np.hypot(self.floor_a, shot)
        pink = self.pink_amp_a * _pink(n, self.rng)
        mains = self.mains_amp_a * np.sin(2 * np.pi * self.mains_hz * t)
        lines = np.zeros(n)
        for f, a in self.spectral_lines.items():
            lines += a * np.sin(2 * np.pi * f * t)
        rtn = np.zeros(n)
        if self.rtn_rate_hz > 0 and self.rtn_amp_a > 0 and n > 1:
            dt = float(t[1] - t[0]) if n > 1 else 0.0
            flips = self.rng.random(n) < (1 - np.exp(-self.rtn_rate_hz * dt))
            level = self._rtn_level * np.cumprod(np.where(flips, -1.0, 1.0))
            self._rtn_level = float(level[-1])
            rtn = self.rtn_amp_a * level
        spikes = np.zeros(n)
        for t0, a, tau in list(self.spikes):
            m = t >= t0
            spikes[m] += a * np.exp(-(t[m] - t0) / tau)
        return i_dc + white + pink + mains + lines + rtn + spikes

    def z_series(self, n: int) -> np.ndarray:
        return self.rng.normal(0.0, self.z_floor_m, n)


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """Approximate 1/f noise (unit rms) via spectral shaping."""
    if n < 4:
        return rng.normal(0.0, 1.0, n)
    white = rng.normal(0.0, 1.0, n)
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    spec = np.fft.rfft(white) / np.sqrt(f)
    out = np.fft.irfft(spec, n)
    s = out.std()
    return out / s if s > 0 else out
