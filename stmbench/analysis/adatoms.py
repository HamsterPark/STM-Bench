"""Adatoms in a topograph, and the lattice they sit on.

Positions are read from the frame alone (:func:`locate`): bumps above a running median over
2 nm, after the rows are levelled against each other, each placed at the centroid of its upper
half. At 0.1 nm/px that places an atom to
~0.02 nm, at 0.2 nm/px to ~0.05 nm (p90). The running median follows what a levelling plane does
not: a step edge (208 pm on Cu, three times an atom), and the row-wise jump a tip change leaves
in the middle of a frame. Two things a fast frame does to an atom are allowed for: the Z loop
lags the surface along the fast axis, so the atom trails a shoulder in the scan direction (0.7 nm
out it can stand at 0.45 of its height; a step's high side stays at its full height), and the
lowest drawn atoms under a slightly blunt tip stand ~35 pm above the background.

Atoms of one species on an fcc(111) surface sit on hollow sites of one sublattice, so the vector
between any two of them is a lattice vector of the substrate. A handful of atoms in one frame
therefore fixes the lattice's orientation without atomic resolution — an adatom ruler — and with
it the lattice site an atom has to be moved to.
"""
from __future__ import annotations

import math

import numpy as np

#: Cu(111) nearest-neighbour distance (nm), the spacing of the hollow sites: a = 3.603 Å at
#: 4 K (3.615 Å at room temperature), over √2
CU111_NN_NM = 0.2548


def to_scan_nm(frame: dict, col: float, row: float) -> tuple[float, float]:
    """Scan coordinates (nm) of a pixel of an instrument frame (row 0 is the top)."""
    cx, cy = (float(v) for v in frame["center_nm"])
    size = float(frame["size_nm"])
    n = frame["array"].shape
    u = (col + 0.5) / n[1] * size - size / 2
    v = size / 2 - (row + 0.5) / n[0] * size
    th = math.radians(float(frame.get("angle_deg", 0.0) or 0.0))
    c, s = math.cos(th), math.sin(th)
    return cx + u * c - v * s, cy + u * s + v * c


def row_jumps(z: np.ndarray, *, threshold_m: float = 15e-12) -> list[int]:
    """Rows at which the whole image steps up or down: the median difference from the row
    before exceeds ``threshold_m``. A tip that changes during a frame does that (its length
    jumps by 20-150 pm, and its apex moves sideways with it, so places read on either side of
    the row disagree). A step edge does not, unless it runs along the rows: it moves only the
    few pixels where it crosses."""
    z = np.asarray(z, float)
    z = np.where(np.isfinite(z), z, np.nanmedian(z))
    d = np.median(np.diff(z, axis=0), axis=1)
    return [int(k) + 1 for k in np.flatnonzero(np.abs(d) > threshold_m)]


def bumps(z: np.ndarray, nm_per_px: float, *, background_nm: float = 2.0, min_height_m: float = 20e-12,
          min_sigma: float = 4.0, min_sep_nm: float = 0.6, compact_frac: float = 0.6,
          compact_r_nm: float = 0.7, border_nm: float = 0.5) -> tuple[list[tuple[float, float, float]], np.ndarray]:
    """Compact bumps in a topograph: ``([(col, row, height_m), …] highest first, background-
    subtracted array)``. A bump is a local maximum standing ``min_height_m`` and ``min_sigma``
    noise sigmas above a running median over ``background_nm``, and falling below
    ``compact_frac`` of its height ``compact_r_nm`` out in all eight directions (a step's
    ridge does not, nor the corner where a step meets a row jump)."""
    from scipy.ndimage import gaussian_filter, maximum_filter, median_filter

    z = np.asarray(z, float)
    z = np.where(np.isfinite(z), z, np.nanmedian(z))
    # a tip change shifts every row after it: the median step between consecutive rows takes
    # that out and nothing else (a step edge or an atom moves only a few pixels of a row)
    z = z - np.concatenate([[0.0], np.cumsum(np.median(np.diff(z, axis=0), axis=1))])[:, None]
    size = max(5, int(round(background_nm / nm_per_px)) | 1)
    hp = gaussian_filter(z - median_filter(z, size=size, mode="nearest"), max(0.5, 0.06 / nm_per_px))
    med = float(np.median(hp))
    noise = 1.4826 * float(np.median(np.abs(hp - med)))
    hp = hp - med
    sep = max(3, int(round(min_sep_nm / nm_per_px)) | 1)
    peaks = (hp == maximum_filter(hp, size=sep)) & (hp > max(min_height_m, min_sigma * noise))
    b = max(1, int(round(border_nm / nm_per_px)))
    peaks[:b, :] = peaks[-b:, :] = False
    peaks[:, :b] = peaks[:, -b:] = False
    r = compact_r_nm / nm_per_px
    dirs = [(int(round(r * math.sin(a))), int(round(r * math.cos(a)))) for a in np.arange(8) * math.pi / 4]
    ny, nx = hp.shape
    out = []
    for row, col in zip(*np.nonzero(peaks)):
        h = float(hp[row, col])
        ring = [float(hp[row + dr, col + dc]) for dr, dc in dirs
                if 0 <= row + dr < ny and 0 <= col + dc < nx]
        if len(ring) >= 5 and max(ring) <= compact_frac * h:
            out.append((float(col), float(row), h))
    out.sort(key=lambda t: -t[2])
    return out, hp


def locate(frame: dict, *, window_nm: float = 0.6, **kw) -> list[tuple[float, float, float]]:
    """``[(x_nm, y_nm, height_m), …]`` in scan coordinates of the adatoms in an instrument frame
    (``Instrument.frame``), highest first: :func:`bumps`, each refined to the centroid of the
    part of it above half its height."""
    npx = float(frame["nm_per_px"])
    found, hp = bumps(frame["array"], npx, **kw)
    ny, nx = hp.shape
    r = max(2, int(round(window_nm / npx)))
    out = []
    for col, row, h in found:
        k, i = int(row), int(col)
        k0, i0 = max(k - r, 0), max(i - r, 0)
        win = np.clip(hp[k0:k + r + 1, i0:i + r + 1] - 0.5 * h, 0.0, None)
        if win.sum() <= 0:
            continue
        yy, xx = np.mgrid[k0:k0 + win.shape[0], i0:i0 + win.shape[1]]
        kc, ic = float((win * yy).sum() / win.sum()), float((win * xx).sum() / win.sum())
        x, y = to_scan_nm(frame, ic, kc)
        out.append((x, y, h))
    return out


def _basis(phi: float, a_nm: float) -> np.ndarray:
    return a_nm * np.array([[math.cos(phi), math.cos(phi + math.pi / 3)],
                            [math.sin(phi), math.sin(phi + math.pi / 3)]])


def _residuals(d: np.ndarray, phi: float, a_nm: float) -> np.ndarray:
    """Distance (nm) of each vector in ``d`` (N×2) from the nearest lattice vector."""
    b = _basis(phi, a_nm)
    uv = np.linalg.solve(b, d.T)                              # 2×N
    best = np.full(d.shape[0], np.inf)
    for du in (0, 1):
        for dv in (0, 1):
            n = np.floor(uv) + np.array([[du], [dv]])
            best = np.minimum(best, np.hypot(*(b @ n - d.T)))
    return best


def _residuals_grid(d: np.ndarray, phis: np.ndarray, a_nm: float) -> np.ndarray:
    """(M, N): the distance of each of the N vectors ``d`` from the nearest lattice vector, for
    each of the M orientations ``phis`` (vectorised :func:`_residuals`)."""
    c, s = np.cos(phis)[:, None], np.sin(phis)[:, None]
    c2, s2 = np.cos(phis + math.pi / 3)[:, None], np.sin(phis + math.pi / 3)[:, None]
    det = a_nm * (c * s2 - c2 * s)
    dx, dy = d[None, :, 0], d[None, :, 1]
    u, v = (s2 * dx - c2 * dy) / det, (-s * dx + c * dy) / det
    fu, fv = np.floor(u), np.floor(v)
    best = np.full(u.shape, np.inf)
    for du in (0.0, 1.0):
        for dv in (0.0, 1.0):
            px = a_nm * ((fu + du) * c + (fv + dv) * c2)
            py = a_nm * ((fu + du) * s + (fv + dv) * s2)
            best = np.minimum(best, np.hypot(px - dx, py - dy))
    return best


def lattice_angle(points_nm, *, a_nm: float = CU111_NN_NM, min_len_nm: float = 1.0,
                  cap_nm: float = 0.08, max_shear: float = 0.005) -> tuple[float, float, int]:
    """Orientation of the lattice the atoms sit on, from the vectors between them.

    Returns ``(phi, rms_nm, n_vectors)``: ``phi`` (rad, in [0, π/3)) is the direction of a
    nearest-neighbour row, ``rms_nm`` how far the vectors sit from lattice vectors at that
    orientation (~0.03 nm, the measurement error, when it is right).

    The points come from one frame, whose rows were taken one after the other: whatever drift
    the compensation left moves each row's atoms by an amount that grows with the row, a shear
    ``(sx, sy)·Δy`` of every vector (0.004 is 0.16 nm across a 40 nm frame, 0.12 nm/min left
    over 80 s). Along y it stretches every long vector past the fit's tolerance and the search
    lands on a wrong orientation, so the shear is searched with the orientation, within
    ``max_shear``.

    It takes enough atoms. Somewhere in 60° of orientations, a few vectors all land close to
    lattice points by chance and fit as well as the right orientation. Without shear, with the
    atoms placed to 0.015-0.02 nm on 60 random lattices: three atoms gave a wrong orientation
    47-77 % of the time, four 13-35 %, five 2-15 %, six 0-10 %, seven or eight 0-2 %. With a
    random shear up to 0.004 searched as well, on 40 lattices: six atoms 3/40 wrong at
    0.015 nm, eight 0/40 (5/40 at 0.025 nm), twelve 0/40 (2/40). How much better the best
    orientation fits than the next does not tell a wrong answer from a right one, so the count
    is the criterion: use eight atoms or more."""
    p = np.asarray([q[:2] for q in points_nm], float)
    if len(p) < 3:
        raise ValueError("an adatom ruler needs three atoms at least")
    d = np.array([p[j] - p[i] for i in range(len(p)) for j in range(i + 1, len(p))])
    d = d[np.hypot(d[:, 0], d[:, 1]) >= min_len_nm]
    if len(d) < 2:
        raise ValueError("the atoms are too close together to orient the lattice")
    dy = d[:, 1:2]

    def best_of(phis, sxs, sys_):
        out = (np.inf, 0.0, 0.0, 0.0)
        for sx in sxs:
            for sy in sys_:
                r = _residuals_grid(d - dy * np.array([[sx, sy]]), phis, a_nm)
                sc = np.sum(np.minimum(r, cap_nm) ** 2, axis=1)
                k = int(np.argmin(sc))
                if sc[k] < out[0]:
                    out = (float(sc[k]), float(phis[k]), float(sx), float(sy))
        return out

    shears = np.arange(-max_shear, max_shear + 1e-12, 0.001)
    _, phi, sx, sy = best_of(np.radians(np.arange(0.0, 60.0, 0.05)), shears, shears)
    for dphi, dsh in ((0.05, 0.0005), (0.01, 0.0001), (0.002, 0.00002)):
        steps = np.arange(-2, 3)
        _, phi, sx, sy = best_of(phi + np.radians(np.arange(-5, 6) * dphi), sx + steps * dsh, sy + steps * dsh)
    r = _residuals_grid(d - dy * np.array([[sx, sy]]), np.array([phi]), a_nm)[0]
    return phi % (math.pi / 3), float(np.sqrt(np.mean(r ** 2))), int(len(d))


def lattice_vector(d_nm, phi: float, *, a_nm: float = CU111_NN_NM) -> tuple[float, float]:
    """The lattice vector nearest ``d_nm`` (nm) for a lattice whose rows run along ``phi``."""
    d = np.asarray(d_nm, float).reshape(1, 2)
    b = _basis(phi, a_nm)
    uv = np.linalg.solve(b, d.T)
    best, best_v = math.inf, (0.0, 0.0)
    for du in (0, 1):
        for dv in (0, 1):
            n = np.floor(uv) + np.array([[du], [dv]])
            v = (b @ n).ravel()
            r = math.hypot(v[0] - d[0, 0], v[1] - d[0, 1])
            if r < best:
                best, best_v = r, (float(v[0]), float(v[1]))
    return best_v
