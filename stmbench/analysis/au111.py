"""Au(111) herringbone from a survey frame: where the steps are and which way the stripes run.

A LOCATOR ONLY: no number reported in a claim comes from here. The stripe FFT that measures a window
to the accuracy of a claim is :mod:`stmbench.analysis.herringbone`, and every reported period and direction
is read from a measurement frame with it. What is here is coarser and is for finding places to
measure in a large frame taken at ~1 nm/px (a 6.3 nm stripe is still six pixels wide):

* :func:`step_mask`: where a step edge (235 pm against stripes of 8-15 pm) runs, so that a
  window is not put on it;
* :func:`direction_map`: the mean stripe direction in a grid of windows. Within one domain the
  stripes zigzag between two arms at +-alpha about the mean, and the spectrum of a window shows a
  peak on each arm; the power-weighted mean of the doubled peak angles is the mean direction
  whichever arm is stronger in the window. Two domains differ by 60 degrees in their means.

Conventions are those of the instrument's frames: row 0 is the top (the high-y edge), columns run
with x, directions are stripe directions in degrees counter-clockwise from +x, modulo 180.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter

#: stripe wavenumbers searched (1/nm): periods of 4.5-9 nm. The drawn period is 6.0-6.6 nm and an
#: arm's is that times cos(alpha) (alpha up to 18 deg): 5.7-6.6 nm.
K_MIN, K_MAX = 0.11, 0.22
#: height scales (m) of a step edge's gradient after smoothing: the reconstruction itself gives
#: well under 1 pm/nm there, a step ~40 pm/nm (235 pm over its width in the image)
STEP_GRAD_PM_PER_NM = 12.0


def _filled(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    return np.where(np.isfinite(z), z, np.nanmedian(z))


def step_mask(z: np.ndarray, nm_per_px: float, *, grow_nm: float = 5.0) -> np.ndarray:
    """Pixels on or near a step edge or any other large topographic feature (True).

    The image is smoothed over 2.5 nm, which removes the stripes (and atoms) but leaves a step,
    and its gradient, less the frame's tilt, is compared with :data:`STEP_GRAD_PM_PER_NM`."""
    z = _filled(z) * 1e12
    s = max(0.8, 2.5 / nm_per_px)
    zs = gaussian_filter(z, s)
    gy, gx = np.gradient(zs, nm_per_px)
    g = np.hypot(gx - np.median(gx), gy - np.median(gy))
    m = g > STEP_GRAD_PM_PER_NM
    r = int(round(grow_nm / nm_per_px))
    return binary_dilation(m, iterations=r) if r > 0 else m


def highpass(z: np.ndarray, nm_per_px: float, *, sigma_nm: float = 4.0) -> np.ndarray:
    """The frame less its smooth part (terraces, tilt): what is left is the stripes and the noise."""
    z = _filled(z) * 1e12
    return z - gaussian_filter(z, max(1.0, sigma_nm / nm_per_px))


@dataclass
class DirectionMap:
    """Window grid of a survey: window centres (frame pixels, ``rows`` x ``cols``), and per window
    the mean stripe direction ``psi_deg``, how sharply the ring's power is concentrated (``coh``,
    0-1), the ring's power over the background (``snr``) and the fraction of the window
    on a step or masked row (``masked``)."""
    rows: np.ndarray
    cols: np.ndarray
    psi_deg: np.ndarray
    coh: np.ndarray
    snr: np.ndarray
    masked: np.ndarray
    win_px: int
    nm_per_px: float


def direction_map(z: np.ndarray, nm_per_px: float, *, win_nm: float = 64.0, stride_nm: float = 16.0,
                  bad_rows: tuple[int, ...] = ()) -> DirectionMap:
    z = _filled(z)
    hp = highpass(z, nm_per_px)
    mask = step_mask(z, nm_per_px)
    for r in bad_rows:                                      # rows a tip change stepped
        mask[max(r - 2, 0):r + 3, :] = True
    # clip what is left of atoms and edge ridges so that they do not set the spectrum
    scale = 1.4826 * float(np.median(np.abs(hp - np.median(hp)))) or 1.0
    hp = np.clip(hp, -4 * scale, 4 * scale)
    hp = np.where(mask, 0.0, hp)
    n = int(round(win_nm / nm_per_px))
    n += n % 2
    st = max(1, int(round(stride_nm / nm_per_px)))
    ny, nx = hp.shape
    rows = np.arange(n // 2, ny - n // 2 + 1, st)
    cols = np.arange(n // 2, nx - n // 2 + 1, st)
    pad = 4
    k1 = np.fft.fftshift(np.fft.fftfreq(n * pad, nm_per_px))
    KX, KY = np.meshgrid(k1, -k1)                           # image rows run down, y up
    K = np.hypot(KX, KY)
    ring = (K > K_MIN) & (K < K_MAX)
    bg = (K > 0.06) & (K < 0.3) & ~ring
    phi = np.arctan2(KY, KX)
    win = np.outer(np.hanning(n), np.hanning(n))
    psi = np.full((rows.size, cols.size), np.nan)
    coh = np.zeros_like(psi)
    snr = np.zeros_like(psi)
    frac = np.zeros_like(psi)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            sl = (slice(r - n // 2, r + n // 2), slice(c - n // 2, c + n // 2))
            frac[i, j] = float(mask[sl].mean())
            w = hp[sl]
            if frac[i, j] > 0.5:
                continue
            P = np.abs(np.fft.fftshift(np.fft.fft2((w - w.mean()) * win, s=(n * pad, n * pad)))) ** 2
            pr = P[ring]
            tot = float(pr.sum())
            b = float(np.median(P[bg])) * ring.sum() / pad ** 2 if bg.any() else 0.0
            if tot <= 0:
                continue
            R = (pr * np.exp(2j * phi[ring])).sum() / tot
            psi[i, j] = (0.5 * math.degrees(math.atan2(R.imag, R.real)) - 90.0) % 180.0
            coh[i, j] = abs(R)
            snr[i, j] = tot / pad ** 2 / max(b / ring.sum() * ring.sum(), 1e-30) if b else 0.0
    return DirectionMap(rows, cols, psi, coh, snr, frac, n, nm_per_px)


def circ_diff(a, b):
    """Signed difference of two directions (degrees, modulo 180), in (-90, 90]."""
    return (np.asarray(a, float) - np.asarray(b, float) + 90.0) % 180.0 - 90.0


def clean_windows(dm: DirectionMap, *, min_snr: float = 8.0, max_masked: float = 0.3,
                  agree_deg: float = 14.0, max_radius: int = 4) -> np.ndarray:
    """For each window of the map, the radius (in strides) of the largest block of windows around it
    that are all measurable (no step, a stripe peak) and have the same mean direction as it (within
    ``agree_deg``) — a window with a large radius sits well inside one domain, at least that many strides
    from a wall. -1 where the window itself is not measurable."""
    ok = (~np.isnan(dm.psi_deg)) & (dm.snr >= min_snr) & (dm.masked <= max_masked)
    n_i, n_j = dm.psi_deg.shape
    out = np.full((n_i, n_j), -1, int)
    for i in range(n_i):
        for j in range(n_j):
            if not ok[i, j]:
                continue
            r = 0
            while r < max_radius:
                i0, i1, j0, j1 = i - r - 1, i + r + 2, j - r - 1, j + r + 2
                if i0 < 0 or j0 < 0 or i1 > n_i or j1 > n_j:
                    break
                blk = dm.psi_deg[i0:i1, j0:j1]
                if not ok[i0:i1, j0:j1].all() or np.abs(circ_diff(blk, dm.psi_deg[i, j])).max() > agree_deg:
                    break
                r += 1
            out[i, j] = r
    return out
