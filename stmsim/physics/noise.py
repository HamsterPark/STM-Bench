"""Noise streams for current and Z.

Everything is generated from one seeded ``numpy`` generator so a scenario replays exactly.
Two access patterns:

* :meth:`NoiseModel.current_sample` — one O(1) sample at a given wall time (for
  ``Current.Get`` polled at kHz rates by read-back routines);
* :meth:`NoiseModel.current_series` — a vector for a scope screen / a scan line.

Components: white (preamp floor + shot), 1/f, mains pickup (50 Hz + harmonics, plus
optional spectral lines such as a 787.6 Hz pump line),
random telegraph noise when the tip is unstable, and coarse-motor crosstalk spikes.

A scan pixel carries these as they are: the white figures (``floor_a``, the shot noise at
``bandwidth_hz``, ``z_floor_m``) are those of a reading at :data:`REF_DWELL_S`, whatever the
pixel's own dwell. A sample that is the mean over a longer dwell (the trace of a move) is
:meth:`NoiseModel.averaged_over` that dwell.
"""
from __future__ import annotations

import math

import numpy as np

_E = 1.602176634e-19

#: The dwell (s) that the white figures of a reading stand for: the pixel dwell of the default
#: scan, 0.586 s per line of 256 pixels (:class:`~stmsim.physics.scanner.ScanSettings`; pinned by
#: a test). A scan pixel keeps ``z_floor_m`` (2 pm) and the preamplifier's floor whatever its
#: dwell, so the default scan is the one dwell the figures are tied to; nothing in this repository
#: measures them at another.
REF_DWELL_S = 0.586 / 256


class NoiseModel:
    def __init__(self, rng: np.random.Generator, *, floor_a: float = 0.05e-12,
                 bandwidth_hz: float = 1000.0, mains_hz: float = 50.0,
                 mains_amp_a: float = 0.2e-12, pink_amp_a: float = 0.1e-12,
                 spectral_lines: dict[float, float] | None = None,
                 z_floor_m: float = 2e-12, z_slow_m: float = 0.0,
                 ref_dwell_s: float = REF_DWELL_S):
        self.rng = rng
        self.floor_a = floor_a
        self.bandwidth_hz = bandwidth_hz
        self.mains_hz = mains_hz
        self.mains_amp_a = mains_amp_a
        self.pink_amp_a = pink_amp_a
        self.spectral_lines = dict(spectral_lines or {})   # Hz → amplitude (A)
        self.z_floor_m = z_floor_m
        # slow Z wander (rms, m) that no mean over a dwell removes; 0 → none. Only samples that
        # are means over a dwell carry it (:meth:`z_slow_series`); a scan pixel does not
        self.z_slow_m = z_slow_m
        self.ref_dwell_s = ref_dwell_s     # the dwell the white figures above stand for
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

    def z_slow_series(self, n: int) -> np.ndarray:
        """The slow Z wander over ``n`` samples: 1/f, ``z_slow_m`` rms over the whole record."""
        if self.z_slow_m <= 0.0:
            return np.zeros(n)
        return self.z_slow_m * _pink(n, self.rng)

    # ── a sample that is a mean over a dwell ──
    def averaged_over(self, dwell_s: float, rng: np.random.Generator | None = None) -> "NoiseModel":
        """The noise of a sample that is the mean over ``dwell_s``, as a controller records one
        while the tip crosses its step, in place of a reading at :attr:`ref_dwell_s`.

        The white parts (the preamplifier floor, the shot noise, Z's floor) average: their rms
        falls as ``1 / sqrt(dwell / ref_dwell_s)``, and is left as it is for a dwell at or below
        the reference (a shorter one is not made noisier: the reference figures are all the model
        has). A mains or pump line is a sinusoid and averages as one, its amplitude times
        ``sinc(f dwell)``. What a mean over the dwell does not remove stays: the 1/f current noise
        (``pink_amp_a``), the slow Z wander (``z_slow_m``) and a tip's telegraph noise.

        A new model on ``rng`` (the same generator if none is given); this one is not changed.
        Its reference dwell is the longer of the two, so averaging it again composes."""
        ref = self.ref_dwell_s
        longer = dwell_s > ref
        a = math.sqrt(ref / dwell_s) if longer else 1.0
        att = (lambda f: float(np.sinc(f * dwell_s))) if longer else (lambda f: 1.0)
        out = NoiseModel(self.rng if rng is None else rng, floor_a=self.floor_a * a,
                         bandwidth_hz=self.bandwidth_hz * a * a, mains_hz=self.mains_hz,
                         mains_amp_a=self.mains_amp_a * att(self.mains_hz), pink_amp_a=self.pink_amp_a,
                         spectral_lines={f: amp * att(f) for f, amp in self.spectral_lines.items()},
                         z_floor_m=self.z_floor_m * a, z_slow_m=self.z_slow_m,
                         ref_dwell_s=max(ref, dwell_s))
        out.rtn_rate_hz, out.rtn_amp_a = self.rtn_rate_hz, self.rtn_amp_a
        return out


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
