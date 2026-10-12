"""The short-range force from qPlus Δf(z) curves, and the three numbers P5 reports.

The procedure is the one force spectroscopy papers follow (Lantz et al. 2001; Ternes et al.
2008; Huber et al. 2019):

1. record Δf(z) over the atom and over bare surface nearby, and put both on the loop's absolute
   Z so that equal Z means equal tip height (``Curve``; repeated curves are combined point by
   point with the median, so a frequency jump in one of them does not survive);
2. subtract the background in Δf at equal tip height: the long-range van der Waals part is the
   same over both and cancels, and so does any constant offset of the frequency zero;
3. invert the difference to a force with Sader & Jarvis (2004, eq. 9) — a linear operation, so
   inverting the difference equals subtracting the inverted curves;
4. read off the force minimum, the decay length (a Morse fit: the tail of a single bond decays
   as ``exp(-a z)``) and the binding energy (the minimum of ``U(z) = ∫_z^∞ F dz'``).

The heights ``z`` returned here increase away from the surface and are the oscillation's mean
position; Sader–Jarvis gives the force at the lower turning point, a fixed shift by the
amplitude that changes none of the three numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EV = 1.602176634e-19


@dataclass
class Curve:
    """One Δf(z) sweep: the loop's Z at every point (m), Δf (Hz), and the sweep coordinate
    ``z_rel`` (m; it decreases toward the surface), which fixes which way is "away"."""
    z_abs: np.ndarray
    df: np.ndarray
    z_rel: np.ndarray

    @classmethod
    def from_spectrum(cls, s: dict) -> "Curve":
        return cls(np.asarray(s["z_abs"], float), np.asarray(s["df"], float), np.asarray(s["z_rel"], float))

    def height(self) -> tuple[np.ndarray, np.ndarray]:
        """(height increasing away from the surface, Δf), sorted by height, NaNs dropped."""
        away = np.sign((self.z_abs[-1] - self.z_abs[0]) * (self.z_rel[-1] - self.z_rel[0])) or 1.0
        h = away * self.z_abs
        ok = np.isfinite(h) & np.isfinite(self.df)
        o = np.argsort(h[ok])
        return h[ok][o], self.df[ok][o]


def combine(curves: list[Curve], grid: np.ndarray) -> np.ndarray:
    """Point-by-point median of repeated curves resampled onto ``grid``."""
    rows = []
    for c in curves:
        h, df = c.height()
        rows.append(np.interp(grid, h, df, left=np.nan, right=np.nan))
    return np.nanmedian(np.vstack(rows), axis=0)


def short_range_df(atom: list[Curve], background: list[Curve]) -> tuple[np.ndarray, np.ndarray]:
    """(height, Δf over the atom − Δf over bare surface) on the heights every curve covers."""
    spans = [c.height()[0] for c in atom + background]
    lo = max(float(h[0]) for h in spans)
    hi = min(float(h[-1]) for h in spans)
    if not hi > lo:
        raise ValueError("the atom and background curves do not overlap in height")
    step = float(np.median(np.diff(atom[0].height()[0])))
    grid = np.arange(lo, hi, max(step, 1e-13))
    return grid, combine(atom, grid) - combine(background, grid)


def sader_jarvis(z, df, *, f0: float, k: float, amplitude: float) -> np.ndarray:
    """Force (N) from Δf(z) (Hz) at heights ``z`` (m, increasing away from the surface).

    Sader & Jarvis 2004, eq. 9. The correction terms diverge as t → z: the integral runs
    by trapezoid from the next sample on, and the first interval [z, z + Δz] is added in closed
    form with Ω and Ω' held at z (∫ dt/√(t − z) = 2√Δz). Leaving that interval out loses
    ~10 % of the force at 2 pm sampling and a 50 pm amplitude."""
    z = np.asarray(z, float)
    omega = np.asarray(df, float) / float(f0)
    d_omega = np.gradient(omega, z)
    a = float(amplitude)
    c1 = math.sqrt(a) / (8.0 * math.sqrt(math.pi))
    c2 = a ** 1.5 / math.sqrt(2.0)
    out = np.zeros_like(z)
    for i in range(z.size - 1):
        t = z[i + 1:]
        dt = t - z[i]
        integrand = (1.0 + c1 / np.sqrt(dt)) * omega[i + 1:] - c2 * d_omega[i + 1:] / np.sqrt(dt)
        h = float(z[i + 1] - z[i])
        first = omega[i] * h + 2.0 * c1 * omega[i] * math.sqrt(h) - 2.0 * c2 * d_omega[i] * math.sqrt(h)
        out[i] = 2.0 * float(k) * (float(np.trapezoid(integrand, t)) + first)
    if z.size > 1:
        out[-1] = out[-2]
    return out


def smooth(y: np.ndarray, z: np.ndarray, width_m: float = 20e-12) -> np.ndarray:
    """Savitzky–Golay (cubic) over about ``width_m`` of height."""
    from scipy.signal import savgol_filter

    step = float(np.median(np.diff(z)))
    n = max(5, int(round(width_m / max(step, 1e-15))) | 1)          # odd window, at least 5 points
    return savgol_filter(y, n, 3) if y.size > n else y


def _morse_force(z, a, d, c):
    u = np.exp(-a * (z - c))
    return -2.0 * a * d * u * (1.0 - u)


def force_numbers(z: np.ndarray, force: np.ndarray, *, tail_trim_m: float = 30e-12) -> dict:
    """F_min (pN), the decay length (pm) and the binding energy (meV) of a short-range force.

    The last ``tail_trim_m`` of height is dropped: the inversion there lacks the curve beyond
    the sweep."""
    from scipy.optimize import curve_fit

    keep = z <= z[-1] - tail_trim_m
    z, f = z[keep], force[keep]
    i_min = int(np.argmin(f))
    f_min = float(f[i_min])
    best = None
    sel = z >= z[max(i_min - 10, 0)]
    for a0 in (8e9, 12e9, 18e9):
        p0 = (a0, 2.0 * abs(f_min) / a0, float(z[i_min]) - math.log(2.0) / a0)
        try:
            popt, _ = curve_fit(_morse_force, z[sel], f[sel], p0=p0, maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        res = float(np.sum((_morse_force(z[sel], *popt) - f[sel]) ** 2))
        if popt[0] > 0 and (best is None or res < best[0]):
            best = (res, popt)
    decay_pm = 1e12 / float(best[1][0]) if best is not None else float("nan")
    u = np.array([np.trapezoid(f[i:], z[i:]) for i in range(z.size)])
    return {"f_min_pn": f_min * 1e12, "f_decay_pm": decay_pm,
            "e_bind_mev": float(-np.min(u)) / EV * 1e3, "z_fmin_m": float(z[i_min])}


def p5_numbers(atom: list[Curve], background: list[Curve], *, f0: float, k: float,
               amplitude: float) -> dict:
    """The whole chain: subtract, smooth, invert, read off."""
    z, dfs = short_range_df(atom, background)
    ok = np.isfinite(dfs)
    z, dfs = z[ok], dfs[ok]
    force = sader_jarvis(z, smooth(dfs, z), f0=f0, k=k, amplitude=amplitude)
    return force_numbers(z, force)
