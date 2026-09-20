"""Z feedback loop as a ≥2nd-order closed-loop linear system (docs/DESIGN.md §4.2).

Controller: controller PI on the log-current error ``e = 2κ (z − z_eq)`` (dimensionless),
``u = P e + I ∫e``, ``z = −u`` — P in metres, I in metres/second (the panel numbers).
Plant: piezo first-order lag ``τ_p`` plus a small pure delay ``τ_d`` (Padé-1), which is
what lets a too-high gain go unstable rather than merely ring.

Closed loop (z tracks z_eq)::

    T(s) = C(s) G(s) / (1 + C(s) G(s)),   C(s) = 2κ (P + I/s),   G(s) = (1 − sτ_d/2) / ((1 + sτ_p)(1 + sτ_d/2))

Two uses:

* :meth:`Loop.track` — a scan line: ``z_eq`` samples at the pixel rate → ``z`` samples
  (oversampled ×8 internally so kHz dynamics survive, then decimated as the ADC would);
* :meth:`Loop.transient` — an explicit time series at ``fs`` for read-back windows
  (poke / pulse / setpoint changes).

The unstable branch is integrated sample by sample with an amplitude clamp so it
produces a bounded limit cycle at ≈ω_n instead of an exponential blow-up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from functools import lru_cache

import numpy as np
from scipy import signal


@dataclass
class LoopParams:
    p_m: float = 3e-12
    i_m_per_s: float = 50e-9
    kappa_m: float = 1.0e10           # 2κ ≈ 2e10 /m for φ≈4 eV
    tau_piezo_s: float = 1.0 / (2 * math.pi * 1500.0)
    tau_delay_s: float = 60e-6
    z_clamp_m: float = 0.3e-9         # limit-cycle amplitude when unstable

    def _key(self) -> tuple:
        return (self.p_m, self.i_m_per_s, self.kappa_m, self.tau_piezo_s, self.tau_delay_s)

    def tf(self) -> tuple[np.ndarray, np.ndarray]:
        """Continuous transfer function num/den (z / z_eq)."""
        num, den = _tf_cached(self._key())
        return num.copy(), den.copy()

    def poles(self) -> np.ndarray:
        return _poles_cached(self._key()).copy()

    @property
    def stable(self) -> bool:
        return _stable_cached(self._key())

    @property
    def natural_hz(self) -> float:
        p = self.poles()
        osc = p[np.abs(np.imag(p)) > 1e-6]
        if osc.size == 0:
            return 0.0
        return float(np.max(np.abs(np.imag(osc))) / (2 * math.pi))

    @property
    def damping(self) -> float:
        p = self.poles()
        osc = p[np.abs(np.imag(p)) > 1e-6]
        if osc.size == 0:
            return 1.0
        k = np.argmax(np.abs(np.imag(osc)))
        w = abs(osc[k])
        return float(-np.real(osc[k]) / w) if w > 0 else 1.0

    def settle_time_s(self) -> float:
        p = self.poles()
        slowest = np.max(np.real(p))
        return float(4.0 / max(-slowest, 1e-3))


class Loop:
    def __init__(self, params: LoopParams):
        self.params = params
        self._cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def discrete(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        key = round(dt, 12)
        if key not in self._cache:
            num, den = self.params.tf()
            (bd, ad, _) = signal.cont2discrete((num, den), dt, method="bilinear")
            self._cache[key] = (np.asarray(bd).ravel(), np.asarray(ad).ravel())
        return self._cache[key]

    def track(self, z_eq: np.ndarray, dt_pixel: float, z0: float | None = None,
              oversample: int = 8) -> np.ndarray:
        """Feedback-tracked z for a line of equilibrium heights sampled every ``dt_pixel``."""
        z_eq = np.asarray(z_eq, float)
        n = z_eq.size
        if n == 0:
            return z_eq
        if z0 is None:
            z0 = float(z_eq[0])
        dt = dt_pixel / oversample
        x = np.repeat(z_eq, oversample)
        if self.params.stable:
            b, a = self.discrete(dt)
            zi = signal.lfiltic(b, a, [z0] * (len(a) - 1), [z0] * (len(b) - 1))
            y, _ = signal.lfilter(b, a, x, zi=zi)
        else:
            y = self._unstable(x, dt, z0)
        return y[oversample - 1::oversample]

    def transient(self, z_eq_fn, t: np.ndarray, z0: float) -> np.ndarray:
        """z(t) for a time-varying equilibrium (callable or array) starting from ``z0``."""
        t = np.asarray(t, float)
        if callable(z_eq_fn):
            x = np.asarray([z_eq_fn(tt) for tt in t], float)
        else:
            x = np.asarray(z_eq_fn, float)
        dt = float(t[1] - t[0]) if t.size > 1 else 1e-4
        if self.params.stable:
            b, a = self.discrete(dt)
            zi = signal.lfiltic(b, a, [z0] * (len(a) - 1), [z0] * (len(b) - 1))
            y, _ = signal.lfilter(b, a, x, zi=zi)
            return y
        return self._unstable(x, dt, z0)

    def _unstable(self, x: np.ndarray, dt: float, z0: float) -> np.ndarray:
        b, a = self.discrete(dt)
        a = a / a[0]
        b = b / a[0] if False else b
        nb, na = len(b), len(a)
        y = np.empty_like(x)
        xs = np.full(nb, x[0])
        ys = np.full(na - 1, z0)
        clamp = self.params.z_clamp_m
        # loop noise seeds the instability (a perfectly quiet unstable loop stays at rest)
        kick = np.random.default_rng(12345).normal(0.0, 1e-12, x.size)
        for i in range(x.size):
            xs = np.roll(xs, 1)
            xs[0] = x[i] + kick[i]
            val = float(np.dot(b, xs) - np.dot(a[1:], ys))
            err = val - x[i]
            if abs(err) > clamp:
                val = x[i] + math.copysign(clamp, err)
            y[i] = val
            ys = np.roll(ys, 1)
            ys[0] = val
        return y


def line_time_for_speed(width_m: float, v_tip_m_per_s: float) -> float:
    return width_m / max(v_tip_m_per_s, 1e-12)


# ── the coefficients depend only on the five loop numbers, and a scan line asks for them
# 1024 times a frame. Memoised by value, so a gain change still recomputes them exactly once.
@lru_cache(maxsize=64)
def _tf_cached(key: tuple) -> tuple[np.ndarray, np.ndarray]:
    p_m, i_m_per_s, kappa_m, tau_piezo_s, tau_delay_s = key
    g = 2.0 * kappa_m
    tp, td = tau_piezo_s, tau_delay_s / 2.0
    # C G = g (P s + I) (1 - td s) / ( s (1 + tp s)(1 + td s) )
    num_cg = np.polymul([g * p_m, g * i_m_per_s], [-td, 1.0])
    den_cg = np.polymul([1.0, 0.0], np.polymul([tp, 1.0], [td, 1.0]))
    return num_cg, np.polyadd(den_cg, num_cg)


@lru_cache(maxsize=64)
def _poles_cached(key: tuple) -> np.ndarray:
    return np.roots(_tf_cached(key)[1])


@lru_cache(maxsize=64)
def _stable_cached(key: tuple) -> bool:
    return bool(np.all(np.real(_poles_cached(key)) < 0))
