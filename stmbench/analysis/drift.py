"""Drift from two frames of the same area: what an operator measures before compensating.

``frame_shift`` cross-correlates two topographs (sub-pixel by a parabola through the peak) and
returns how far the features moved in the scan frame; divided by the time between the frames it
is the velocity the instrument's ``drift_comp`` action takes.

Plain cross-correlation, not phase correlation: whitening the spectrum hands the peak to the
atomic lattice, which is periodic, and on Cu(111) frames the shift then comes out wrong by a
fraction of a lattice vector (0.1–0.2 nm on P5 frames shifted by a known 0.58 nm, against
0.02–0.08 nm without whitening).
"""
from __future__ import annotations

import numpy as np


def _flatten(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    a = np.where(np.isfinite(a), a, np.nanmedian(a))
    a = a - np.median(a, axis=1, keepdims=True)          # row offsets of a creeping, drifting scan
    return a - a.mean()


def _subpixel(line: np.ndarray, i: int) -> float:
    """Offset of a correlation peak from sample ``i``: the vertex of a parabola through three points."""
    n = line.size
    y0, y1, y2 = float(line[(i - 1) % n]), float(line[i]), float(line[(i + 1) % n])
    den = y0 - 2.0 * y1 + y2
    return 0.0 if den == 0 else 0.5 * (y0 - y2) / den


def frame_shift(a: np.ndarray, b: np.ndarray, nm_per_px: float) -> tuple[float, float]:
    """(dx, dy) in nm by which the features of frame ``b`` sit displaced from frame ``a``.

    Arrays are instrument frames: row 0 is the top (largest y), columns run with x."""
    fa, fb = _flatten(a), _flatten(b)
    win = np.outer(np.hanning(fa.shape[0]), np.hanning(fa.shape[1]))
    A, B = np.fft.fft2(fa * win), np.fft.fft2(fb * win)
    c = np.real(np.fft.ifft2(B * np.conj(A)))
    i, j = np.unravel_index(int(np.argmax(c)), c.shape)
    di = i + _subpixel(c[:, j], i)
    dj = j + _subpixel(c[i, :], j)
    ny, nx = c.shape
    if di > ny / 2:
        di -= ny
    if dj > nx / 2:
        dj -= nx
    return float(dj * nm_per_px), float(-di * nm_per_px)
