"""Bright, compact features in a saved frame, detected from local image contrast.

``ExtractClusters`` thresholds a plane-levelled frame at k·MAD of the whole frame. A frame
taken right after a move is bowed by piezo creep, and a survey wide enough to hold a corral
usually crosses a step: a plane through either leaves residuals of ~100 pm at the corners,
more than an adatom (50–90 pm). The threshold then sits above the atoms and the corners pass
— three candidates at the frame edges, none on an atom; in the 2026-09-13 calibration gate,
P3 / P3-repair / P4 / P5 produced 0/20 usable detections. The implementation instead
subtracts the *local* background and looks for compact bumps above it.

Only the frame's own pixels are read; nothing here touches the simulator's truth.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

#: scale of the background that is subtracted (nm). Adatoms are ~0.7 nm across; the bow of a
#: creeping frame and the rise of a step are tens of nanometres.
BACKGROUND_NM = 3.0
#: a feature has to stand this far above its surroundings — half an adatom, twice the lattice
MIN_HEIGHT_M = 35e-12
#: … and this many noise sigmas (the residual after the background is taken out)
MIN_SIGMA = 4.0
#: two features closer than this are one
MIN_SEP_NM = 0.6
#: a compact bump falls to below this fraction of its height one feature-width away in
#: EVERY direction; a step's ridge does not
COMPACT_FRAC = 0.45
COMPACT_R_NM = 0.7


def _plane(z: np.ndarray) -> np.ndarray:
    ny, nx = z.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    ok = np.isfinite(z)
    if ok.sum() < 3:
        return z
    a = np.column_stack([xx[ok], yy[ok], np.ones(int(ok.sum()))])
    coef, *_ = np.linalg.lstsq(a, z[ok], rcond=None)
    return z - (coef[0] * xx + coef[1] * yy + coef[2])


def blobs_in_array(z: np.ndarray, nm_per_px: float, *, background_nm: float = BACKGROUND_NM,
                   min_height_m: float = MIN_HEIGHT_M, min_sigma: float = MIN_SIGMA,
                   min_sep_nm: float = MIN_SEP_NM) -> list[tuple[float, float, float]]:
    """``[(col, row, height_m), …]`` of compact bright bumps, highest first. ``col``/``row``
    index the array as given (row 0 = the array's first row); NaN rows are ignored."""
    from scipy.ndimage import gaussian_filter, maximum_filter

    z = np.array(z, dtype=float)
    if z.ndim != 2 or not (nm_per_px and nm_per_px > 0):
        return []
    finite = np.isfinite(z)
    if finite.sum() < 64:
        return []
    fill = float(np.nanmedian(z))
    z = np.where(finite, z, fill)
    z = _plane(z)
    sig_bg = max(1.0, background_nm / nm_per_px)
    hp = z - gaussian_filter(z, sig_bg)
    hp = gaussian_filter(hp, max(0.5, 0.15 / nm_per_px))
    hp[~finite] = -np.inf
    med = float(np.median(hp[finite]))
    noise = 1.4826 * float(np.median(np.abs(hp[finite] - med)))
    thresh = max(float(min_height_m), min_sigma * noise) + med
    sep = max(3, int(round(min_sep_nm / nm_per_px)) | 1)
    peaks = (hp == maximum_filter(hp, size=sep)) & (hp > thresh)
    border = max(2, int(round(background_nm / nm_per_px)) // 2)
    peaks[:border, :] = peaks[-border:, :] = False
    peaks[:, :border] = peaks[:, -border:] = False
    r = max(2, int(round(COMPACT_R_NM / nm_per_px)))
    ny, nx = hp.shape
    out = []
    for row, col in zip(*np.nonzero(peaks)):
        h = float(hp[row, col] - med)
        ring = []
        for dr, dc in ((r, 0), (-r, 0), (0, r), (0, -r)):
            rr, cc = row + dr, col + dc
            if 0 <= rr < ny and 0 <= cc < nx and np.isfinite(hp[rr, cc]):
                ring.append(float(hp[rr, cc] - med))
        if len(ring) < 4 or max(ring) > COMPACT_FRAC * h:
            continue                      # a ridge, a step, a frame edge: not a compact bump
        out.append((float(col), float(row), h))
    out.sort(key=lambda t: -t[2])
    return out


def bright_blobs(path: str | Path, *, channel: str = "Z", **kw) -> list[tuple[float, float]]:
    """Scan-frame positions (metres) of the compact bright features in a saved frame,
    highest first — a drop-in for what the baselines used to read out of ``ExtractClusters``."""
    from mast.skills.builtins._sxm_frame import load_frame

    from ..human.render import frame_meta, pixel_to_scan_nm

    fr = load_frame(str(path), channel)
    if fr.error or fr.forward is None:
        return []
    geom = frame_meta(path)
    if not geom.get("w_nm") or not geom.get("nx"):
        return []
    blobs = blobs_in_array(np.asarray(fr.forward, float), float(fr.nm_per_px), **kw)
    out = []
    for col, row, _h in blobs:
        x_nm, y_nm = pixel_to_scan_nm(geom, col, row)
        out.append((x_nm * 1e-9, y_nm * 1e-9))
    return out


def merge_close(points: list[tuple[float, float]], min_sep_m: float = MIN_SEP_NM * 1e-9) -> list[tuple[float, float]]:
    """Drop later points within ``min_sep_m`` of an earlier (higher) one — tiles overlap."""
    kept: list[tuple[float, float]] = []
    for p in points:
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) >= min_sep_m for q in kept):
            kept.append(p)
    return kept
