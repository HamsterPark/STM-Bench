"""Bias spectra: the surface-state band edge, and confined-state resonances above it.

A 2-D surface-state band switches on as a step in dI/dV at its bottom E0. On bare terrace the
step is the steepest rise of the spectrum; inside a quantum corral the rising flank of a
resonance can be steeper than the band edge, so E0 is read where there are no resonances.
Resonances are then the local maxima above the edge, kept when their prominence clears a
fraction of the spectrum's range above it — the definition P3's truth uses.
"""
from __future__ import annotations

import numpy as np


def smooth(y: np.ndarray, points: int = 9) -> np.ndarray:
    """Savitzky–Golay (cubic) over ``points`` samples."""
    from scipy.signal import savgol_filter

    y = np.asarray(y, float)
    n = max(5, int(points) | 1)
    return savgol_filter(y, n, 3) if y.size > n else y


def band_edge(v: np.ndarray, didv: np.ndarray, *, window: tuple[float, float] = (-0.55, -0.30),
              guess_v: float = -0.44) -> float:
    """E0 (V) of a bare-terrace spectrum: a smoothed step (error function on a sloping
    background) fitted inside ``window``. The steepest point of a noisy derivative is not
    reliable — on noisy spectra it lands on the window's edge."""
    from scipy.optimize import curve_fit
    from scipy.special import erf

    v = np.asarray(v, float)
    y = np.asarray(didv, float)
    sel = (v >= window[0]) & (v <= window[1]) & np.isfinite(y)
    if sel.sum() < 10:
        raise ValueError("the spectrum does not cover the band-edge window")
    vv, yy = v[sel], y[sel]
    lo, hi = float(np.median(yy[: max(3, yy.size // 8)])), float(np.median(yy[-max(3, yy.size // 8):]))

    def step(x, e0, w, a, b, c):
        return a + b * 0.5 * (1.0 + erf((x - e0) / w)) + c * (x - window[0])

    best = None
    for e_try in (guess_v, guess_v - 0.03, guess_v + 0.03):
        try:
            popt, _ = curve_fit(step, vv, yy, p0=(e_try, 0.01, lo, hi - lo, 0.0), maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        res = float(np.sum((step(vv, *popt) - yy) ** 2))
        if window[0] < popt[0] < window[1] and (best is None or res < best[0]):
            best = (res, popt)
    if best is None:
        raise ValueError("no band edge in the window")
    return float(best[1][0])


def resonance_peaks(v: np.ndarray, didv: np.ndarray, e0_v: float, *, above_edge_v: float = 0.005,
                    prominence_frac: float = 0.08, points: int = 9) -> list[dict]:
    """Local maxima at least ``above_edge_v`` above the band edge whose prominence is at least
    ``prominence_frac`` of the spectrum's range there, lowest first: ``[{"v", "prominence"}]``."""
    from scipy.signal import find_peaks

    v = np.asarray(v, float)
    y = smooth(didv, points)
    sel = v >= e0_v + above_edge_v
    vv, yy = v[sel], y[sel]
    span = float(np.ptp(yy)) if yy.size else 0.0
    if span <= 0:
        return []
    idx, props = find_peaks(yy, prominence=prominence_frac * span)
    return [{"v": float(vv[i]), "prominence": float(p) / span} for i, p in zip(idx, props["prominences"])]
