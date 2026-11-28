"""P2 scripted baseline: the standing waves of the Cu(111) surface state at a step edge, from
nothing but what the instrument shows — no truth, no hidden quantity, no seed.

The measurement is the 1993 one, done with dI/dV maps (the evidence route the rule card calls
(b)): one lock-in map per energy, each one frame, each row a line profile across one straight
step edge, so a frame is free of the drift that spoils a line of spectra taken over an hour.
Per map the wavevector k comes from a fit of the standing wave of the step; the dispersion
E(k) = E0 + (hbar^2 / 2 m*) k^2 of the maps gives E0 and m*. The protocol:

* **survey**: a 400 nm frame at 2 nm/px over the starting place (about 400 s). The mean step
  direction is the angle at which the steep pixels pile up into the sharpest lines (a long
  stretch of staircase fixes it to a degree or two: one step wanders by a few nm over a period
  of a few hundred nm, so a short piece would not). Each row of the survey crossing the staircase
  is shifted onto the mean profile, which gives the steps' positions, the terrace widths and
  the wander of the lines (:func:`staircase`), and the adatoms stand out of the straightened
  staircase (:func:`survey_adatoms`);
* **where**: the terrace on the side the profile is taken must be wide (the other edge's standing
  wave would otherwise be part of the profile: at 30-40 nm it biases m* by several percent), the
  edge locally parallel to the mean direction (the rows, along the mean normal, then cross it
  along its local normal too, the direction its standing wave runs in), and no adatom within
  25 nm (:func:`pick_site`). The direction measurement is one function (:func:`measure_axis`) so a
  different rule for it can replace it. A 100 nm frame at the place finds the staircase again
  (the sample has crept: the pattern of steps of the survey is matched, :func:`match_steps`) and
  looks for adatoms the survey missed; a place with one near is given up for the next;
* **drift**: the sample creeps and drifts by one to three nm a minute after a long move, and
  the rule measures the frame centre's distance to the edge at the start of a frame. The edge's
  position in each frame, against time, gives its velocity along the normal; that is programmed
  as the drift compensation (:meth:`Standing.set_comp`) and each map is centred on where the edge
  will be when it starts;
* **the tip**: a trial map in the middle of the energy range judges it by the chi-square of the
  fit of one straight edge's standing wave. A second apex draws the wave twice, a nanometre or
  three apart: the map fits badly (chi-square of tens, a sound tip 1-3) and k is wrong by a
  percent or two or worse. Then the tip is reshaped away from the place (a 4 V pulse, two shallow
  pokes; :meth:`Standing.repair_tip`) and the trial taken again. A blunt tip images the same LDOS
  and a dirty tip's states do not reach the map, so neither is repaired; an unstable tip steps
  the rows, which the row-by-row alignment takes out, and a poke settles it. The maps are judged
  once more when they are all in (:meth:`Standing.quality`: one tip and one straight edge give k on one
  parabola and a wave of one amplitude and phase at every energy; a second apex that the trial
  map did not show makes them turn with k), and the maps are taken again after a repair if not;
* **maps**: the fast axis along the mean step normal (``angle_deg``), 24 nm at 0.2 nm/px, lock-in
  on, eight energies from -0.30 V to +0.12 V, the edge 7 nm from the centre (the rule asks for
  10 nm) on the side away from the wide terrace;
* **analysis**: the forward rows only (the backward ones carry the scanner's hysteresis, up to
  ~0.3 nm along a row), each aligned on its own edge, normalised by its own mean and averaged, one
  profile per energy fitted with the standing wave of one straight edge with a complex reflection
  (:mod:`stmbench.analysis.standing_waves`); the maps' k against E gives the dispersion. The
  sample's drift along the steps carries the place through the twenty minutes of the series, now
  and then onto a stretch where the edge is bent by ten or twenty degrees: the wave is a function
  of the distance across the local edge, a row at alpha to the local normal fits k cos(alpha), and
  each map's k is divided by cos(alpha) of its own edge's slope (:meth:`Standing.fit_map`).

What the baseline reaches is what a script reaches from the observations alone
(``stmbench.cli baseline --paper P2``).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks

from ..analysis.adatoms import locate, row_jumps
from ..analysis.standing_waves import edge_positions, fit_dispersion, fit_step_wave, k_guess, step_profile
from ..analysis.tip import diagnose
from ..instrument import Instrument
from ._common import Blind, Ops

SCENARIO = "P2_standing_waves_cu111"
#: the survey: 400 nm at 2 nm/px, 1.0 s/line (400 nm/s, the fastest line speed used)
SURVEY_NM, SURVEY_PX, SURVEY_LINE_S = 400.0, 200, 1.0
#: a map: 24 nm at 0.2 nm/px, 0.6 s/line (5 ms per pixel, 144 s)
MAP_NM, MAP_PX, MAP_LINE_S = 24.0, 120, 0.6
#: where the edge is put at the start of a map, from its centre (nm): 3 nm inside the 10 nm of the rule
EDGE_FROM_CENTRE_NM = 7.0
#: columns within this of where the edge should be are searched for it in a map (nm)
EDGE_WINDOW_NM = 9.0
ENERGIES_V = (-0.30, -0.24, -0.18, -0.12, -0.06, 0.00, 0.06, 0.12)
#: a terrace is wide when the profile's side is at least this far across (nm)
MIN_TERRACE_NM = 60.0
#: no adatom within this distance of the frame centre (nm)
ADATOM_CLEAR_NM = 25.0
#: the edge may run this far off the rows' perpendicular where the map is taken (rms slope over the frame)
MAX_SLOPE = 0.035
#: over this far along the steps each way (nm). The sample also drifts along them, by a nanometre a
#: minute, which carries the frame along a wandering edge through the twenty minutes of the series
#: (one development run went from a straight stretch to one 19 degrees off): a longer
#: stretch that is straight (25 nm) left nine of seventy surveys without a place
SLOPE_HALF_NM = 12.0
#: a repair goes this far from the place along the steps (nm) and waits this long afterwards (s)
REPAIR_AWAY_NM, REPAIR_WAIT_S = 150.0, 450.0
#: whole measurements at most (each after a repair of the tip but the first)
MAX_ATTEMPTS = 3
#: no further attempt is begun after this instrument time (s). Set when the budget was 3 h (10800 s; an
#: attempt takes ~2000 s) and kept, a fixed number rather than a share of the budget, when the budget
#: became 5 h on 2026-11-28: every validated result stands as it was
BUDGET_STOP_S = 6500.0
#: the maps are suspect when their k scatters this much about one parabola (rms, relative) or a
#: profile fit has a reduced chi-square above this: sound tips and clean edges stay below 0.3 % and 3
RESID_MAX, CHI2_MAX = 0.004, 4.0
#: and when the waves' amplitudes spread more than this (relative) or their phases more than this (rad) about a line in k
AMP_CV_MAX, PHASE_RMS_MAX = 0.12, 0.15
#: the compensation is adjusted when the edge still moves faster than this (nm/s) through the scan frame
COMP_UPDATE_NM_S = 0.004
#: a map counts for the evidence when the edge began this near the centre (the rule: 10 nm), nm
START_MAX_NM = 9.3
#: the trial map that judges the tip before the series: its energy (V), the repairs at most, and the
#: reduced chi-square above which the tip is repaired
TRIAL_V, MAX_REPAIRS, CHI2_TRIAL = -0.12, 3, 4.0
#: the profile of a map runs from the edge out to this far, or as far as the frame goes (nm), and
#: a profile shorter than the second is not fitted
PROFILE_FROM_NM, PROFILE_MAX_NM, PROFILE_MIN_NM = 0.6, 18.0, 9.0
#: how far a check frame looks for the staircase of the survey from where it was planned (nm)
MATCH_SEARCH_NM = 40.0
#: surveys at most, before the place is given up
MAX_SURVEYS = 4
#: places tried at most (each one checked by a frame before the maps)
MAX_PLACES = 3
#: the frame that checks the place before the maps: 100 nm at 0.4 nm/px (200 s), wide enough to
#: hold an adatom 25 nm from the centre, to show the pattern of steps (one terrace over fits the
#: pattern of two) and to find the edge again after the long move
SITE_NM, SITE_PX, SITE_LINE_S = 100.0, 250, 0.4


# ── the staircase in a survey ────────────────────────────────────────────────
def stepness(z: np.ndarray, npx: float) -> np.ndarray:
    """|grad z| in nm/nm after a 1 px blur: the steep pixels are the step edges."""
    z = np.where(np.isfinite(z), z, np.nanmedian(z)) * 1e9
    gy, gx = np.gradient(ndi.gaussian_filter(z, 1.0))
    return np.hypot(gx, gy) / npx


def measure_axis(z: np.ndarray, npx: float, *, coarse: float = 2.0, fine: float = 0.25) -> float:
    """The mean direction of the step normal in the scan frame (degrees in [0, 180), from +x towards
    +y), from the angle at which the steep pixels pile up into the sharpest lines: the projection of
    the steepness on the normal, smoothed, is largest in sum of squares when the steps are straight
    across it. A wandering step (a few nm over a few hundred) does not move the maximum by more
    than a degree or two over a 400 nm survey. Kept apart so that another rule (the normal of the
    local step) can take its place."""
    gm = stepness(z, npx)
    ny, nx = gm.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    x, y = ((xx + 0.5) * npx - nx * npx / 2).ravel(), (ny * npx / 2 - (yy + 0.5) * npx).ravel()
    w = np.clip(gm - 0.5 * np.percentile(gm, 98), 0, None).ravel()
    sig = max(1.5, 0.8 * npx) / 0.5

    def score(th: float) -> float:
        c, s = math.cos(math.radians(th)), math.sin(math.radians(th))
        u = x * c + y * s
        h, _ = np.histogram(u, bins=np.arange(u.min(), u.max() + 0.5, 0.5), weights=w)
        return float((ndi.gaussian_filter1d(h, sig) ** 2).sum())

    best = max(np.arange(0.0, 180.0, coarse), key=score)
    return float(max(np.arange(best - coarse, best + coarse + 1e-9, fine), key=score)) % 180.0


def _to_uv(a: np.ndarray, npx: float, theta_deg: float, du: float = 1.0):
    """``a`` (a survey array, row 0 the high-y edge) resampled onto a grid of (u along the normal,
    v along the steps), 1 nm, centred on the frame centre: (grid[v, u], valid mask, axis values)."""
    ny, nx = a.shape
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    half = 0.5 * math.hypot(nx, ny) * npx * 0.98
    g = np.arange(-half, half + du, du)
    U, V = np.meshgrid(g, g, indexing="xy")
    X, Y = U * c - V * s, U * s + V * c
    col, row = X / npx + nx / 2 - 0.5, ny / 2 - 0.5 - Y / npx
    ok = (col >= 0) & (col <= nx - 1) & (row >= 0) & (row <= ny - 1)
    return np.where(ok, ndi.map_coordinates(a, [row, col], order=1, mode="nearest"), np.nan), ok, g


def staircase(z: np.ndarray, npx: float, theta_deg: float, *, iters: int = 4, max_shift_nm: float = 12.0,
              min_rows: int = 60) -> dict:
    """The staircase of a survey along the normal at ``theta_deg``: every row of the resampled
    steepness is shifted (within ±``max_shift_nm``) onto the mean profile, which is rebuilt from the
    shifted rows a few times. Returns ``u`` (the axis, nm from the frame centre), ``profile``,
    ``count`` (rows behind each profile point), ``edges`` (u of the steps in the mean profile),
    ``v`` (the rows' axis), ``shift`` (where each row's steps stand relative to the mean, nm, NaN for
    a row without enough pixels) and ``valid`` (the grid's validity mask)."""
    G, ok, g = _to_uv(stepness(z, npx), npx, theta_deg)
    nv, nu = G.shape
    rows = np.flatnonzero(ok.sum(axis=1) >= min_rows)
    shift = np.zeros(nv)
    cand = np.arange(-max_shift_nm, max_shift_nm + 0.01, 0.5)
    for _ in range(iters):
        S = np.full(G.shape, np.nan)
        for j in rows:
            S[j] = np.interp(g, g - shift[j], np.where(ok[j], G[j], np.nan), left=np.nan, right=np.nan)
        cnt = np.isfinite(S).sum(axis=0)
        P = np.where(cnt > 0, np.nansum(S, axis=0) / np.maximum(cnt, 1), 0.0)
        for j in rows:
            row = np.where(ok[j], G[j], np.nan)
            best, best_sc = shift[j], -np.inf
            for sh in cand:
                r = np.interp(g, g - sh, row, left=np.nan, right=np.nan)
                m = np.isfinite(r)
                if m.sum() < min_rows:
                    continue
                sc = float(np.sum(r[m] * P[m])) / m.sum()
                if sc > best_sc:
                    best, best_sc = sh, sc
            shift[j] = best
        shift[rows] -= np.median(shift[rows])
    Ps = ndi.gaussian_filter1d(P, 1.0)
    ref = np.percentile(Ps[cnt >= 0.25 * cnt.max()], 99)
    pk, _ = find_peaks(Ps, height=0.5 * ref, distance=6)
    pk = pk[cnt[pk] >= 0.25 * cnt.max()]
    sh = np.full(nv, np.nan)
    sh[rows] = shift[rows]
    return {"u": g, "profile": P, "count": cnt, "edges": g[pk], "v": g, "shift": sh, "valid": ok, "theta_deg": float(theta_deg)}


def survey_adatoms(z: np.ndarray, npx: float, st: dict, *, min_height_m: float = 30e-12,
                   max_height_m: float = 350e-12, margin_nm: float = 8.0) -> list[tuple[float, float, float]]:
    """Adatoms of a survey as (u, v, height_m) in the survey's normal/along-step frame (nm from its
    centre). The steps are straightened out (each row shifted back by its measured wander) and
    removed with the median over 21 nm along the steps; what stands out from that is a bump."""
    a, ok, g = _to_uv(np.where(np.isfinite(z), z, np.nanmedian(z)), npx, st["theta_deg"])
    nv = a.shape[0]
    sh = st["shift"]
    s = np.full(a.shape, np.nan)
    for j in range(nv):
        if np.isfinite(sh[j]):
            s[j] = np.interp(g, g - sh[j], np.where(ok[j], a[j], np.nan), left=np.nan, right=np.nan)
    hp = s - ndi.median_filter(np.where(np.isfinite(s), s, np.nanmedian(s)), size=(21, 1), mode="nearest")
    hp = np.where(np.isfinite(s), hp, 0.0)
    hp = ndi.gaussian_filter(hp, 1.0)
    mx = ndi.maximum_filter(hp, size=5)
    # an adatom is 50-300 pm high; taller is the edge of the grid or a step's residue. The grid's
    # margins (the median over 21 nm and the interpolation fall short there) are not looked at.
    inner = ndi.binary_erosion(np.isfinite(s), structure=np.ones((3, 3)), iterations=int(margin_nm))
    jj, ii = np.nonzero((hp == mx) & (hp > min_height_m) & (hp < max_height_m) & inner)
    near_step = np.zeros(len(g), bool)
    for e in st["edges"]:
        near_step |= np.abs(g - e) < 4.0
    out = []
    for j, i in zip(jj, ii):
        if near_step[i] or not np.isfinite(sh[j]):
            continue
        out.append((float(g[i] + sh[j]), float(g[j]), float(hp[j, i])))
    return out


def frame_adatoms(frame: dict, ed: dict | None = None, *, min_height_m: float = 40e-12, max_height_m: float = 350e-12) -> list[tuple[float, float, float]]:
    """Adatoms of a frame taken with its rows across the steps, as (x, y, height_m) in scan
    coordinates: bumps 40-350 pm high standing out of the median taken down the columns (the
    steps run along them, and so are taken out too; within 3 nm of the edge line ``ed`` they are
    not looked at). The compact bumps of :func:`stmbench.analysis.adatoms.locate` are the P4
    adatoms; the adsorbates of this sample are 2-4 nm wide blobs and are not found by it."""
    from ..analysis.adatoms import to_scan_nm

    z = np.asarray(frame["array"], float)
    npx = float(frame["nm_per_px"])
    z = np.where(np.isfinite(z), z, np.nanmedian(z))
    n = max(5, int(round(30.0 / npx)) | 1)
    hp = ndi.gaussian_filter(z - ndi.median_filter(z, size=(n, 1), mode="nearest"), max(0.7, 0.4 / npx))
    mx = ndi.maximum_filter(hp, size=max(3, int(round(2.0 / npx)) | 1))
    rr, cc = np.nonzero((hp == mx) & (hp > min_height_m) & (hp < max_height_m))
    out = []
    m = int(round(2.0 / npx))
    for r, c in zip(rr, cc):
        if r < m or c < m or r >= z.shape[0] - m or c >= z.shape[1] - m:
            continue
        if ed is not None and abs((c + 0.5) * npx - (ed["offset_nm"] + ed["slope_nm_per_row"] * r)) < 3.0:
            continue
        x, y = to_scan_nm(frame, float(c), float(r))
        out.append((x, y, float(hp[r, c])))
    return out


def crop_edges(z: np.ndarray, npx: float, x_nm: float, half_nm: float, slope_nm_per_row: float = 0.0) -> dict:
    """:func:`edge_positions` for the edge near column position ``x_nm`` (nm from the frame's left
    side) at the middle row: only the columns within ``half_nm`` of the expected line are looked at,
    so another step in the frame (the neighbouring terrace of a bunch, the far edge of a narrow
    one) is not taken for it. The sample drifts and creeps at a nanometre a minute while a frame is
    taken, so the edge runs obliquely through it: ``slope_nm_per_row`` is the slope expected (the
    last frame's), and each row is looked at around the line it gives. Positions are those of
    the whole frame."""
    z = np.asarray(z, float)
    n_rows, n_col = z.shape
    shift = slope_nm_per_row * (np.arange(n_rows) - n_rows / 2.0)
    if slope_nm_per_row:
        cols = np.arange(n_col) * npx
        zz = np.vstack([np.interp(cols + sh, cols, np.where(np.isfinite(row), row, np.nanmedian(z))) for row, sh in zip(z, shift)])
    else:
        zz = z
    c0 = min(max(0, int(round((x_nm - half_nm) / npx))), n_col - 8)
    c1 = max(min(n_col, int(round((x_nm + half_nm) / npx))), c0 + 8)
    ed = edge_positions(zz[:, c0:c1], npx)
    ed["pos_nm"] = ed["pos_nm"] + c0 * npx + shift
    ed["offset_nm"] = float(ed["offset_nm"] + c0 * npx - slope_nm_per_row * n_rows / 2.0)
    ed["slope_nm_per_row"] = float(ed["slope_nm_per_row"] + slope_nm_per_row)
    return ed


def match_steps(z: np.ndarray, npx: float, steps_x: list[float], *, search_nm: float = 28.0, margin_nm: float = 3.0,
                n: int = 4) -> list[tuple[float, float]]:
    """Where the steps of a survey are in a later frame whose fast axis runs along their normal:
    the shifts (nm, along the fast axis) of ``steps_x`` (their positions from the frame centre, as
    planned) that make them lie on the frame's steep columns, by the correlation of the column
    steepness with a template of the steps, best first: ``[(shift, correlation), ...]``, one per
    peak of the correlation. The pattern of steps, not one step, is what is matched, and a
    pattern of two steps one terrace apart fits again one terrace over, so the caller gets the
    alternatives and decides with what else it knows (:func:`side_is_clear`)."""
    P = stepness(z, npx).mean(axis=0)
    x = (np.arange(len(P)) + 0.5) * npx - len(P) * npx / 2.0
    P = P - P.mean()
    offs = np.arange(-search_nm, search_nm + 0.01, 0.25)
    rs = np.full(len(offs), -2.0)
    for k, off in enumerate(offs):
        xs = [xk + off for xk in steps_x if abs(xk + off) < len(P) * npx / 2.0 - margin_nm]
        if not xs:
            continue
        T = sum(np.exp(-0.5 * ((x - xk) / 1.5) ** 2) for xk in xs)
        den = float(np.linalg.norm(T - T.mean()) * np.linalg.norm(P))
        rs[k] = float(np.dot(T - T.mean(), P)) / den if den > 0 else -2.0
    peaks = [k for k in range(len(offs)) if rs[k] > 0.0 and rs[k] >= rs[max(0, k - 12):k + 13].max()]
    peaks.sort(key=lambda k: -rs[k])
    return [(float(offs[k]), float(rs[k])) for k in peaks[:n]] or [(0.0, -2.0)]


def side_is_clear(z: np.ndarray, npx: float, x_edge_nm: float, side_col: int, *, from_nm: float = 4.0, edge_nm: float = 1.5,
                  ratio: float = 0.35) -> bool:
    """Is the frame free of another step on the profile's side of the edge (at ``x_edge_nm`` from
    the frame's left side)? The steepness of the columns beyond ``from_nm`` from the edge, down
    to ``edge_nm`` from the frame's border, is looked at: a step there as steep as ``ratio`` of
    the edge's own means the terrace is narrower than the survey said (or the edge is not the
    planned one)."""
    P = stepness(z, npx).mean(axis=0)
    x = (np.arange(len(P)) + 0.5) * npx
    here = np.abs(x - x_edge_nm) < 2.0
    side = (side_col * (x - x_edge_nm) >= from_nm) & (x >= edge_nm) & (x <= len(P) * npx - edge_nm)
    if side.sum() * npx < 10.0 or not here.any():
        return True
    return bool(P[side].max() < ratio * P[here].max())


def ghost_steps(z: np.ndarray, npx: float, *, lo_nm: float = 1.2, hi_nm: float = 8.0, ratio_min: float = 0.3,
                same_nm: float = 0.8, same_ratio: float = 0.2) -> dict | None:
    """A second apex of the tip, as a frame whose rows cross the steps shows it: every step drawn
    twice, the copy weaker and a fixed distance to one side of it (the apex's offset along the
    normal, the part of it that moves the standing wave). In the steepness of the columns the
    steps are peaks, a copy a weaker peak ``lo_nm`` to ``hi_nm`` off; two or more steps with a copy
    at one distance (``same_nm``), one side and one relative height (``same_ratio``) are a second
    apex (one pair alone could be two steps a terrace of a few nm apart, and pairs of bunched
    steps differ in distance and in height). Returns ``{"offset_nm": signed distance of the copy, "pairs": n}``
    or None."""
    P = ndi.gaussian_filter1d(stepness(z, npx).mean(axis=0), 0.3 / npx)
    pk, _ = find_peaks(P, height=0.25 * P.max(), distance=max(1, int(round(0.5 / npx))))
    x = (pk + 0.5) * npx
    found = []
    for i in range(len(pk)):
        best = None
        for j in range(len(pk)):
            d = x[j] - x[i]
            r = P[pk[j]] / P[pk[i]]
            if lo_nm < abs(d) < hi_nm and ratio_min <= r < 0.98 and (best is None or r > best[1]):
                best = (d, r)
        if best is not None:
            found.append(best)
    if len(found) < 2:
        return None
    ds, rs = np.array([f[0] for f in found]), np.array([f[1] for f in found])
    for d0, r0 in found:
        close = (np.abs(ds - d0) <= same_nm) & (np.abs(rs - r0) <= same_ratio) & (np.sign(ds) == np.sign(d0))
        if close.sum() >= 2:
            return {"offset_nm": float(np.median(ds[close])), "pairs": int(close.sum())}
    return None


def edge_present(z: np.ndarray, npx: float, ed: dict, *, floor: float = 0.03, factor: float = 5.0) -> bool:
    """Is there a step where the rows' edge line ``ed`` runs? The steepness of the frame at the
    line, averaged over the rows, is compared with the frame's own (the median): an edge is
    several times steeper than a terrace and its noise. Where the expected edge is not in the
    frame at all (the sample has moved further than the window) the line follows noise and this
    says so, which the fit's own residuals do not."""
    gm = stepness(z, npx)
    cols = np.clip(np.round((np.asarray(ed["pos_nm"]) / npx) - 0.5).astype(int), 2, gm.shape[1] - 3)
    rows = np.flatnonzero(np.asarray(ed["good"], bool))
    if rows.size < 10:
        return False
    here = float(np.mean([gm[r, cols[r] - 2:cols[r] + 3].max() for r in rows]))
    return bool(here >= max(floor, factor * float(np.median(gm))))


def pick_site(st: dict, adatoms: list, *, width_nm: float = MIN_TERRACE_NM, clear_nm: float = ADATOM_CLEAR_NM,
              max_slope: float = MAX_SLOPE, frame_nm: float = MAP_NM, other_side_nm: float = 8.0,
              avoid: list | None = None) -> list[dict]:
    """The places of a surveyed staircase where a map can be taken, best first. A place is a step
    edge, the side of it the profile is taken on (``side`` = +1 towards larger u), and a position
    along it ``v``; it needs a terrace at least ``width_nm`` wide on that side (and not a sliver on
    the other), the edge locally parallel to the mean direction (rms slope of the wander over the
    frame at most ``max_slope``), and no adatom of the survey within ``clear_nm`` of the frame
    centre. ``avoid`` lists (u, v) of places already refused. Each candidate: u_edge, v, side,
    centre (u, v), width_nm, slope, clear_nm, key (lower is better)."""
    v_ax, sh = st["v"], np.asarray(st["shift"], float)
    fin = np.isfinite(sh)
    if fin.sum() < 40:
        return []
    filled = np.interp(v_ax, v_ax[fin], sh[fin])
    sm = ndi.gaussian_filter1d(filled, 6.0)
    slope = np.gradient(sm, v_ax)
    e = np.asarray(st["edges"], float)
    half = frame_nm / 2.0
    ok = st["valid"]
    g = st["u"]
    out = []
    for k in range(len(e)):
        for side in (+1, -1):
            kk = k + side
            if kk < 0 or kk >= len(e):
                continue
            width = abs(e[kk] - e[k])
            kb = k - side
            other = abs(e[k] - e[kb]) if 0 <= kb < len(e) else other_side_nm
            if width < width_nm or other < other_side_nm:
                continue
            for j in range(len(v_ax)):
                v0 = float(v_ax[j])
                lo, hi = max(0, j - int(half)), min(len(v_ax), j + int(half) + 1)
                lo2, hi2 = max(0, j - int(SLOPE_HALF_NM)), min(len(v_ax), j + int(SLOPE_HALF_NM) + 1)
                if hi2 - lo2 < 2 * int(SLOPE_HALF_NM) or not fin[lo2:hi2].all():
                    continue
                ue = float(e[k] + sm[j])
                cu = ue + side * EDGE_FROM_CENTRE_NM
                # the frame must lie inside the survey's valid region, and so must the profile's far
                # edge (its position is what makes the width real)
                iu0, iu1 = np.searchsorted(g, [cu - half, cu + half])
                if iu0 < 2 or iu1 > len(g) - 2 or not ok[lo:hi:4, iu0:iu1:4].all():
                    continue
                far = float(e[kk] + sm[j])
                if not 0 <= np.searchsorted(g, far) < len(g) or not ok[j, min(max(np.searchsorted(g, far), 0), len(g) - 1)]:
                    continue
                rms = float(np.sqrt(np.mean(slope[lo2:hi2] ** 2)))
                clear = min([math.hypot(a[0] - cu, a[1] - v0) for a in adatoms] + [1e3])
                if avoid and any(math.hypot(cu - p[0], v0 - p[1]) < 20.0 for p in avoid):
                    continue
                key = rms / max_slope + max(0.0, 40.0 - clear) / 30.0 + max(0.0, 90.0 - width) / 200.0
                out.append({"u_edge": ue, "v": v0, "side": side, "centre": (cu, v0), "steps_u": [float(x + sm[j]) for x in e],
                            "width_nm": float(width),
                            "other_nm": float(other), "slope": rms, "clear_nm": float(clear), "key": float(key),
                            "ok": bool(rms <= max_slope and clear >= clear_nm)})
    out.sort(key=lambda c: (not c["ok"], c["key"]))
    return out


# ── the session ──────────────────────────────────────────────────────────────
class NoSite(Exception):
    """No place with a wide terrace, a straight edge and no adatom near it was found."""


def _unit(deg: float) -> np.ndarray:
    return np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])


class Standing:
    """One session: the survey, the place, the maps, the fits."""

    def __init__(self, v: Blind, log: dict):
        self.v, self.log = v, log
        self.ops = Ops(v, log)
        self.c0 = np.zeros(2)                       # centre of the survey in use
        self.st: dict | None = None
        self.maps: list[dict] = []
        self.edge_track: list[tuple[float, float]] = []   # (instrument time, edge position along the fast axis, nm)
        self.cands: list[dict] = []                 # the places of the survey in use, best first
        self.slope_prior = 0.0                      # how the edge runs through a map (nm per row), from the last frame
        self.comp_nm_s = 0.0                        # the drift compensation programmed along the fast axis (nm/s)
        self.track_from = 0                         # edge_track entries before this were taken under another compensation

    # ── survey and place ──
    def survey(self, centre) -> tuple[dict, list]:
        cx, cy = float(centre[0]), float(centre[1])
        r = self.v.act("scan", center_x_nm=cx, center_y_nm=cy, size_nm=SURVEY_NM, pixels=SURVEY_PX, line_time_s=SURVEY_LINE_S)
        f = self.v.frame(r["frame"])
        z = f["array"]
        th = measure_axis(z, float(f["nm_per_px"]))
        st = staircase(z, float(f["nm_per_px"]), th)
        ad = survey_adatoms(z, float(f["nm_per_px"]), st)
        self.c0, self.st = np.array([cx, cy]), st
        self.log.setdefault("surveys", []).append({"centre": [cx, cy], "theta_deg": round(th, 2), "edges": len(st["edges"]),
                                                   "adatoms": len(ad), "jumps": len(row_jumps(z))})
        return st, ad

    def find_place(self) -> tuple[dict, dict]:
        """Survey until a candidate place is found; returns (candidate, survey)."""
        centre = np.zeros(2)
        for k in range(MAX_SURVEYS):
            st, ad = self.survey(centre)
            cands = pick_site(st, ad, avoid=self.log.get("refused_places"))
            good = [c for c in cands if c["ok"]]
            self.log["surveys"][-1]["candidates"] = len(good)
            if good:
                self.cands, self.st = good, st
                return good[0], st
            wide = any(abs(st["edges"][i + 1] - st["edges"][i]) >= MIN_TERRACE_NM for i in range(len(st["edges"]) - 1))
            n, t = _unit(st["theta_deg"]), _unit(st["theta_deg"] + 90.0)
            step = SURVEY_NM * 0.8 * (1 if k % 2 == 0 else -1) * (1 + k // 2)
            centre = self.c0 + (t if wide else n) * step
        raise NoSite("no wide terrace with a straight edge and no adatom near it")

    def plan(self, cand: dict, st: dict) -> dict:
        """The scan geometry of a map at a candidate place."""
        theta = st["theta_deg"]
        thp = theta if theta <= 90.0 else theta - 180.0
        e_u = _unit(thp)                                   # the fast axis (angle_deg = -thp, clockwise positive)
        n = _unit(theta)
        centre = self.c0 + cand["centre"][0] * n + cand["centre"][1] * _unit(theta + 90.0)
        sg = 1.0 if float(n @ e_u) > 0 else -1.0
        side_col = int(cand["side"] * sg)
        # the staircase as it should show along the fast axis, from the frame centre
        steps_x = [sg * (u - cand["centre"][0]) for u in cand["steps_u"]]
        return {"angle_deg": -thp, "e_u": e_u, "centre": centre, "side_col": side_col, "steps_x": steps_x,
                "origin": np.array(centre, float), "along": _unit(theta + 90.0)}

    # ── frames ──
    def scan_map(self, bias_v: float, centre, angle_deg: float, x_expect: float) -> dict:
        v = self.v
        v.act("set_bias", bias_v=float(bias_v))
        r = v.act("scan", center_x_nm=float(centre[0]), center_y_nm=float(centre[1]), size_nm=MAP_NM, pixels=MAP_PX,
                  angle_deg=float(angle_deg), line_time_s=MAP_LINE_S)
        name = r["frame"]
        zf, zb = v.frame(name, "z", "forward"), v.frame(name, "z", "backward")
        df, db = v.frame(name, "didv", "forward"), v.frame(name, "didv", "backward")
        return {"frame": name, "bias_v": float(bias_v), "centre": np.array(centre, float), "angle_deg": float(angle_deg),
                "t_mid": float(zf["t_start_s"]) + 0.5 * float(r["duration_s"]), "zf": zf["array"], "zb": zb["array"],
                "df": df["array"], "db": db["array"], "npx": float(zf["nm_per_px"]), "x_expect": float(x_expect)}

    def fit_map(self, m: dict, side_col: int, slope_prior: float = 0.0) -> dict:
        """The standing wave of a map: each forward row aligned on its own edge, normalised,
        averaged, and fitted, and k corrected for the tilt of the edge. The profile reaches as far
        from the edge as the frame allows (the edge drifts towards the middle of the frame while
        the series runs)."""
        npx = m["npx"]
        pairs = []
        # forward rows only: the backward ones carry the scanner's hysteresis, up to ~0.3 nm along the row,
        # which stretches distances by a few percent
        for z, d in ((m["zf"], m["df"]),):
            ed = crop_edges(z, npx, MAP_NM / 2.0 + m["x_expect"], EDGE_WINDOW_NM, slope_prior)
            if ed["n_good"] >= 20:
                pairs.append((z, d, ed))
        if not pairs:
            return {"ok": False, "n_rows": 0}
        room = []
        for z, _, ed in pairs:
            good = np.asarray(ed["pos_nm"])[np.asarray(ed["good"], bool)]
            room.append((z.shape[1] * npx - float(good.max())) if side_col > 0 else float(good.min()))
        d_max = min(PROFILE_MAX_NM, min(room) - 0.4)
        if d_max < PROFILE_MIN_NM:
            return {"ok": False, "n_rows": 0, "edge": pairs[0][2]}
        ys, ns, sems = [], [], []
        for _, d, ed in pairs:
            p = step_profile(d, ed, npx, side=side_col, d_min_nm=PROFILE_FROM_NM, d_max_nm=d_max)
            if p["n_rows"] >= 20:
                ys.append(p["y"]); sems.append(p["sem"]); ns.append(p["n_rows"]); grid = p["d_nm"]
        if not ys:
            return {"ok": False, "n_rows": 0, "edge": pairs[0][2]}
        w = 1.0 / np.maximum(np.array(sems), 1e-6) ** 2
        y = (np.array(ys) * w).sum(axis=0) / w.sum(axis=0)
        sem = 1.0 / np.sqrt(w.sum(axis=0))
        fit = fit_step_wave(grid, y, k_guess=k_guess(m["bias_v"]), sem=sem)
        # Tilt: the standing wave depends on the distance perpendicular to the local step, and a row
        # that crosses the edge at alpha to its normal measures s = d / cos(alpha) along itself, so
        # the fitted k is k * cos(alpha). alpha is the edge's slope in this frame, atan(nm per row /
        # row spacing). That slope is the step's own direction plus the shear of the sample's drift
        # along the normal during the frame (rows are 2 line times apart); with the drift
        # compensation on, as it is for the series, the shear is a degree or two and is left in:
        # taking it out would need the drift, which is only known from these same slopes.
        alpha = math.atan(float(pairs[0][2]["slope_nm_per_row"]) / npx)
        fit["tilt_deg"] = math.degrees(alpha)
        if math.isfinite(fit.get("k", float("nan"))):
            fit["k"] = float(fit["k"]) / math.cos(alpha)
            if math.isfinite(fit.get("k_err", float("nan"))):
                fit["k_err"] = float(fit["k_err"]) / math.cos(alpha)
        fit["n_rows"] = int(sum(ns))
        fit["edge"] = pairs[0][2]
        fit["d_nm"], fit["y"], fit["sem"] = grid, y, sem
        return fit

    def check_place(self, plan: dict) -> dict:
        """A 100 nm frame at the planned centre (the long move to it has crept the sample): where the
        edge really is, the adatoms near the centre, and what the tip makes of the frame."""
        v = self.v
        c = plan["centre"]
        r = v.act("scan", center_x_nm=float(c[0]), center_y_nm=float(c[1]), size_nm=SITE_NM, pixels=SITE_PX,
                  angle_deg=plan["angle_deg"], line_time_s=SITE_LINE_S)
        f = v.frame(r["frame"])
        npx = float(f["nm_per_px"])
        want = -plan["side_col"] * EDGE_FROM_CENTRE_NM
        moved = float((np.array(c, float) - plan["origin"]) @ plan["e_u"])
        ghost = ghost_steps(f["array"], npx)
        alts = match_steps(f["array"], npx, [x - moved for x in plan["steps_x"]], search_nm=MATCH_SEARCH_NM)
        # the edge is looked at where the maps will be, not along the whole 100 nm of it (it
        # wanders by ten nm over that): the band of rows a map will cover
        n_rows = f["array"].shape[0]
        half = int(round(0.5 * MAP_NM / npx)) + 2
        zb = f["array"][n_rows // 2 - half:n_rows // 2 + half]
        found, present = None, None
        for alt, (off, corr) in enumerate(alts):
            ed = crop_edges(zb, npx, SITE_NM / 2.0 + want + off, EDGE_WINDOW_NM)
            pos_mid = ed["offset_nm"] + ed["slope_nm_per_row"] * zb.shape[0] / 2.0
            if edge_present(zb, npx, ed):
                present = present or (alt, off, corr, ed, pos_mid)     # an edge, if not the one with a clear side
                if side_is_clear(zb, npx, pos_mid, plan["side_col"], from_nm=9.0 if ghost else 4.0):
                    found = (alt, off, corr, ed, pos_mid)
                    break
        found = found or present
        if found is None:
            # no edge where the pattern puts it: steps drawn twice do not fit it (the tip is repaired first), or
            # the sample has gone further than the search. The best match is taken, and noted
            self.log.setdefault("checks", []).append({"ghost": ghost, "pattern": [(round(a, 1), round(b, 2)) for a, b in alts], "found": False})
            if ghost:
                return {"ghost": ghost, "centre": np.array(c, float), "clear_nm": 1e3, "slope_deg": 0.0, "tip": {}, "frame": r["frame"],
                        "t": float(f["t_start_s"]) + 0.5 * float(r["duration_s"]), "x_e": 0.0, "n_good": 0}
            off, corr = alts[0]
            ed = crop_edges(zb, npx, SITE_NM / 2.0 + want + off, EDGE_WINDOW_NM)
            found = (0, off, corr, ed, ed["offset_nm"] + ed["slope_nm_per_row"] * zb.shape[0] / 2.0)
        alt, off, corr, ed, pos_mid = found
        x_e = pos_mid - SITE_NM / 2.0                       # the edge along the fast axis, from the centre
        r0 = n_rows // 2 - half
        ed_frame = dict(ed, offset_nm=float(ed["offset_nm"]) - float(ed["slope_nm_per_row"]) * r0)     # the line in the whole frame's rows
        atoms = list(locate(f)) + frame_adatoms(f, ed_frame)
        shifted = c + (x_e - want) * plan["e_u"]
        self.edge_track.append((float(f["t_start_s"]) + 0.5 * float(r["duration_s"]), float(np.array(c, float) @ plan["e_u"]) + x_e))
        clear = min([math.hypot(a[0] - shifted[0], a[1] - shifted[1]) for a in atoms] + [1e3])
        tip = diagnose(f)
        slope_deg = math.degrees(math.atan(ed["slope_nm_per_row"] / float(f["nm_per_px"])))
        # the edge's slope through a frame is mostly the sample's drift along the normal (the place
        # was chosen where the edge runs along the rows); a map's rows take 2 * line time each
        self.slope_prior = float(ed["slope_nm_per_row"]) * MAP_LINE_S / SITE_LINE_S
        info = {"ghost": ghost, "pattern_shift_nm": round(off, 2), "pattern_r": round(corr, 2), "alternative": alt, "edge_shift_nm": round(float(x_e - want), 2), "slope_deg": round(slope_deg, 2), "n_good": ed["n_good"],
                "atoms": len(atoms), "clear_nm": round(float(clear), 1), "tip": tip["verdict"], "jumps": tip["jumps"]}
        self.log.setdefault("checks", []).append(info)
        return {"ghost": ghost, "centre": shifted, "clear_nm": float(clear), "slope_deg": slope_deg, "tip": tip, "frame": r["frame"],
                "t": float(f["t_start_s"]) + 0.5 * float(r["duration_s"]), "x_e": float(x_e), "n_good": ed["n_good"]}

    def edge_velocity(self) -> float | None:
        """How fast the edge moves along the fast axis (nm/s) in the scan frame, from the frames
        since the compensation last changed; None when there are not two of them a minute apart."""
        tr = self.edge_track[self.track_from:][-3:]
        if len(tr) < 2 or tr[-1][0] - tr[0][0] < 60.0:
            return None
        return float(np.polyfit([a[0] for a in tr], [a[1] for a in tr], 1)[0])

    def set_comp(self, plan: dict) -> None:
        """Programme the drift compensation along the edge's normal, the only component the edge
        shows: features move through the scan frame at the velocity the edge was seen to
        (what a two-frame drift measurement reports). Called again, it adds what is left."""
        v = self.edge_velocity()
        if v is None:
            return
        self.comp_nm_s += v
        vec = self.comp_nm_s * 60.0 * plan["e_u"]
        self.v.act("drift_comp", on=True, vx_nm_per_min=float(vec[0]), vy_nm_per_min=float(vec[1]))
        self.track_from = len(self.edge_track)
        self.log.setdefault("drift_comp_nm_per_min", []).append(round(self.comp_nm_s * 60.0, 2))

    def predict_centre(self, plan: dict, duration_s: float) -> np.ndarray:
        """The centre for a map that starts now: the edge's position along the fast axis at the
        frame's start (the rule measures the distance of the centre to the edge there), extrapolated
        from the last frames, put the planned distance from the centre."""
        want = -plan["side_col"] * EDGE_FROM_CENTRE_NM
        e_u = plan["e_u"]
        c = np.array(plan["centre"], float)
        if not self.edge_track:
            return c
        t0, p0 = self.edge_track[-1]
        v = self.edge_velocity()
        pred = p0 + (v if v is not None else 0.0) * (self.ops.now() - t0)
        return c + ((pred - want) - float(c @ e_u)) * e_u

    def take_map(self, plan: dict, bias_v: float) -> dict:
        """One map at the place: scanned with the edge expected where the last frames say it will
        be, fitted, and the edge's position noted for the next one."""
        want = -plan["side_col"] * EDGE_FROM_CENTRE_NM
        centre = self.predict_centre(plan, MAP_PX * 2 * MAP_LINE_S)
        m = self.scan_map(bias_v, centre, plan["angle_deg"], want)
        # a dI/dV map that is mostly NaN is taken again at the same place. The cause it was written for, a
        # pixel on scipy 1.18.1's struve(0, z) hole in the step term, is gone: the simulator steps over the
        # hole since 2026-11-27 (surface_state._struve0). It stays, capped, for a map blank for another reason
        for _ in range(2):
            if float(np.mean(~np.isfinite(m["df"]))) <= 0.2:
                break
            self.log["nan_retakes"] = self.log.get("nan_retakes", 0) + 1
            m = self.scan_map(bias_v, centre, plan["angle_deg"], want)
        fit = self.fit_map(m, plan["side_col"], self.slope_prior)
        m["fit"] = fit
        plan["centre"] = centre
        ed = fit.get("edge")
        if ed is not None:
            self.slope_prior = float(ed["slope_nm_per_row"])
            x_e = ed["offset_nm"] + ed["slope_nm_per_row"] * m["zf"].shape[0] / 2.0 - MAP_NM / 2.0
            self.edge_track.append((m["t_mid"], float(centre @ plan["e_u"]) + x_e))
            m["x_e"] = float(x_e)
            m["slope_deg"] = math.degrees(math.atan(ed["slope_nm_per_row"] / m["npx"]))
            # the rule measures the centre's distance to the edge when the frame starts: the line fitted
            # to the rows gives where the edge stood in the first one
            m["start_nm"] = abs(float(ed["offset_nm"]) - MAP_NM / 2.0)
        return m

    @staticmethod
    def sound(m: dict) -> bool:
        """A map to keep: the wave fitted, to a chi-square a sound tip gives, within a percent in
        k, and the edge no further from the frame's centre than the rule allows when it began."""
        f = m["fit"]
        return bool(f.get("ok") and f["chi2_red"] <= CHI2_MAX and f["k_err"] <= 0.015 * f["k"]
                    and m.get("start_nm", 99.0) <= START_MAX_NM)

    def trial_map(self, plan: dict) -> dict:
        """One map in the middle of the energy range at the place, to judge the tip and the place
        by how well the standing wave of one straight edge fits it (a second apex shows as a
        reduced chi-square of tens, a sound tip as 1-3)."""
        self.v.act("lockin", on=True, amplitude_mv=10.0)
        fit = self.take_map(plan, TRIAL_V)["fit"]
        info = {"chi2_red": round(_nan(fit.get("chi2_red")), 2), "k": round(_nan(fit.get("k")), 4), "ok": bool(fit.get("ok"))}
        self.log.setdefault("trials", []).append(info)
        return fit

    def repair_tip(self, plan: dict, k: int) -> None:
        """Reshape the apex away from the place: a 4 V, 50 ms pulse on fresh ground 150 nm along the
        steps (a pulse sprays clusters ~40 nm around), two shallow pokes beside it (a pulse leaves the
        apex metastable, a shallow contact settles it), back, and the creep of the two long moves
        waited out."""
        c = np.array(plan["centre"], float)
        sgn = 1.0 if k % 2 else -1.0
        x0 = c + plan["along"] * sgn * (REPAIR_AWAY_NM + 10.0 * (k // 2)) + plan["e_u"] * 8.0 * k
        self.ops.go(float(x0[0]), float(x0[1]))
        self.v.act("pulse", bias_v=4.0, width_ms=50.0)
        for j in (1, 2):
            y = x0 + plan["e_u"] * 2.0 * j
            self.ops.go(float(y[0]), float(y[1]))
            self.v.act("poke", depth_nm=0.4, dwell_s=0.5)
        self.ops.go(float(c[0]), float(c[1]))
        self.v.act("wait", seconds=REPAIR_WAIT_S)
        # the compensation stays on through the repair (switched off, the sample would drift away
        # from the place by a nanometre a minute for the ten it takes); the frames before it say
        # nothing of the edge's velocity after it
        self.track_from = len(self.edge_track)
        self.log["repairs"] = self.log.get("repairs", 0) + 1

    @staticmethod
    def quality(maps: list[dict]) -> dict:
        """How well the maps agree with one tip and one straight edge, as a set. Three things stay the
        same across energies when they do, and a second apex breaks them (its wave is the first one
        displaced by a nanometre or two, so the sum is another standing wave, of a phase and an
        amplitude that turn with k and of a k that is a percent or two off):
        ``resid_rms``, the scatter of k about the dispersion's parabola (relative; a sound set is
        within 0.3 %); ``amp_cv``, the spread of the waves' amplitudes (a sound set is within 6 %:
        the step's reflection does not change much over 0.4 eV); ``phase_rms`` the scatter of their
        phases about a straight line in k (a sound set within 0.05 rad). The reflection model is
        one reflection amplitude and phase for all energies, which the P2 step edge has; a surface
        whose edge reflected otherwise would have to be fitted otherwise. ``chi2_max`` is the
        worst reduced chi-square of the profile fits (a sound tip 1-3)."""
        good = [m for m in maps if m["fit"].get("ok")]
        out = {"maps_ok": len(good), "maps": len(maps), "resid_rms": float("inf"), "amp_cv": float("inf"),
               "phase_rms": float("inf"), "chi2_max": float("inf"), "suspect": True}
        if len(good) < 5:
            return out
        e = np.array([m["bias_v"] for m in good])
        k = np.array([m["fit"]["k"] for m in good])
        c = np.linalg.lstsq(np.vstack([np.ones(len(k)), k ** 2]).T, e, rcond=None)[0]
        rel = (k - np.sqrt(np.clip((e - c[0]) / c[1], 1e-9, None))) / k
        out["resid_rms"] = float(np.sqrt(np.mean(rel ** 2)))
        amp = np.array([m["fit"]["amp"] for m in good], float)
        out["amp_cv"] = float(np.std(np.abs(amp)) / max(float(np.mean(np.abs(amp))), 1e-9))
        ph = np.array([m["fit"]["phase"] for m in good], float) + np.where(amp < 0, math.pi, 0.0)
        mean = math.atan2(float(np.mean(np.sin(ph))), float(np.mean(np.cos(ph))))
        dev = (ph - mean + math.pi) % (2 * math.pi) - math.pi
        out["phase_rms"] = float(np.std(dev - np.polyval(np.polyfit(k, dev, 1), k)))
        out["chi2_max"] = float(max(m["fit"]["chi2_red"] for m in good))
        out["suspect"] = bool(out["resid_rms"] > RESID_MAX or out["chi2_max"] > CHI2_MAX or out["amp_cv"] > AMP_CV_MAX
                              or out["phase_rms"] > PHASE_RMS_MAX or len(good) < len(maps) - 1)
        return out

    def run_maps(self, plan: dict, energies=ENERGIES_V) -> list[dict]:
        """A map at each energy. One whose edge began further from the centre than the rule allows
        (the sample moved faster than it was told) is taken again at once, with the edge's
        position now known."""
        self.v.act("lockin", on=True, amplitude_mv=10.0)
        for e in energies:
            m = self.take_map(plan, e)
            if m.get("start_nm", 0.0) > START_MAX_NM:
                self.maps.append(m)
                self.log["retaken"] = self.log.get("retaken", 0) + 1
                m = self.take_map(plan, e)
            self.maps.append(m)
            v = self.edge_velocity()
            if v is not None and abs(v) > COMP_UPDATE_NM_S:
                self.set_comp(plan)
        return self.maps

    def prepare(self, plan: dict, n_rep: int, chk: dict | None = None) -> int:
        """Find the edge again, judge the tip (a frame whose steps are drawn twice, or a trial map
        that does not fit one straight edge's standing wave, says a second apex) and repair it
        while it is not sound, ``MAX_REPAIRS`` times at most over the session; then programme the
        drift compensation. Returns the repairs made so far."""
        chk = chk or self.check_place(plan)
        while True:
            plan["centre"] = chk["centre"]
            bad = bool(chk.get("ghost"))
            if not bad:
                fit = self.trial_map(plan)
                bad = not (fit.get("ok") and fit["chi2_red"] <= CHI2_TRIAL)
            if not bad or n_rep >= MAX_REPAIRS or self.ops.now() > BUDGET_STOP_S:
                break
            n_rep += 1
            self.repair_tip(plan, n_rep)
            chk = self.check_place(plan)
        self.set_comp(plan)
        return n_rep

    def dispersion(self) -> dict:
        good = [m for m in self.maps if m["fit"].get("ok")]
        if len(good) < 5:
            raise RuntimeError(f"only {len(good)} of {len(self.maps)} maps gave a wavevector")
        e = np.array([m["bias_v"] for m in good])
        k = np.array([m["fit"]["k"] for m in good])
        ke = np.array([m["fit"]["k_err"] for m in good])
        ke = np.where(np.isfinite(ke) & (ke > 0), ke, 0.02 * k)
        d = fit_dispersion(e, k, ke)
        d["maps"] = [m["frame"] for m in good]
        return d


def claim_ids(text: str) -> list[str]:
    import re
    return re.findall(r"^\d+\. (\w+)[:：]", text, flags=re.M)


def _nan(x) -> float:
    return float(x) if x is not None else float("nan")


def _map_log(m: dict) -> dict:
    f = m["fit"]
    return {"bias_v": m["bias_v"], "k": round(_nan(f.get("k")), 4), "k_err": round(_nan(f.get("k_err")), 4), "n_rows": f.get("n_rows"),
            "chi2_red": round(_nan(f.get("chi2_red")), 2), "slope_deg": round(m.get("slope_deg", float("nan")), 2), "tilt_deg": round(_nan(f.get("tilt_deg")), 2),
            "x_e": round(m.get("x_e", float("nan")), 2)}


def play(v: Blind, log: dict) -> dict:
    """One P2 session through ``v``: what to report and the evidence for it."""
    s = Standing(v, log)
    log["stage"] = "survey"
    cand, st = s.find_place()
    log["stage"] = "place"
    refused: list[dict] = []
    for _ in range(MAX_PLACES):
        plan = s.plan(cand, st)
        try:
            chk = s.check_place(plan)
        except NoSite:                               # the staircase of the survey is not where the check looked
            chk = None
        if chk is not None and chk["clear_nm"] >= ADATOM_CLEAR_NM - 3.0:
            break
        # an adatom the survey did not show: the place is given up for the next one that is not near it
        refused.append(cand)
        later = [c for c in s.cands if all(math.hypot(c["centre"][0] - r["centre"][0], c["centre"][1] - r["centre"][1]) > 40.0 for r in refused)]
        if not later:
            break
        cand = later[0]
        s.edge_track, s.track_from = [], 0
    log["place"] = {"width_nm": round(cand["width_nm"], 1), "slope": round(cand["slope"], 4), "clear_nm": round(cand["clear_nm"], 1),
                    "axis_deg": round(st["theta_deg"], 2), "refused": len(refused)}
    kept: dict[float, dict] = {}                   # the sound map at each energy
    spare: dict[float, dict] = {}                  # the best unsound one, if no other
    passes: list[tuple[dict, list[dict]]] = []     # the set of maps after each pass, and how good it is
    todo = list(ENERGIES_V)
    n_rep = 0
    need_repair = True
    for k in range(MAX_ATTEMPTS):
        log["stage"] = "tip"
        if k and need_repair:
            n_rep += 1
            s.repair_tip(plan, n_rep)
        n_rep = s.prepare(plan, n_rep, chk if k == 0 else None)
        log["stage"] = "maps"
        s.maps = []
        s.run_maps(plan, todo)
        for m in s.maps:
            e = m["bias_v"]
            if s.sound(m):
                kept[e] = m
            elif e not in spare or m["fit"].get("chi2_red", 1e9) < spare[e]["fit"].get("chi2_red", 1e9):
                spare[e] = m
        log.setdefault("attempts", []).append({"energies": list(todo), "maps": [_map_log(m) for m in s.maps]})
        todo = [e for e in ENERGIES_V if e not in kept]
        have = [kept.get(e) or spare[e] for e in ENERGIES_V if e in kept or e in spare]
        q = s.quality(have)
        passes.append((q, have))
        log.setdefault("qualities", []).append({a: (round(b, 4) if isinstance(b, float) else b) for a, b in q.items()})
        # a map unsound for a reason but where the edge stood is the tip's (or the place's); the sample's drift is not
        need_repair = bool(todo and any(m["fit"].get("ok") and (m["fit"]["chi2_red"] > CHI2_MAX or m["fit"]["k_err"] > 0.015 * m["fit"]["k"])
                                        for m in s.maps if m["bias_v"] in todo)) or (not todo and q["suspect"])
        if not todo and q["suspect"]:                # sound maps that do not agree with each other: take them all again
            kept, spare, todo = {}, {}, list(ENERGIES_V)
        if (not todo and not q["suspect"]) or v.status()["instrument_time_s"] > BUDGET_STOP_S:
            break
    best_q, best = min(passes, key=lambda a: (a[0]["suspect"], a[0]["resid_rms"] + a[0]["amp_cv"] + a[0]["phase_rms"] + 0.001 * min(a[0]["chi2_max"], 100.0)))
    s.maps = best
    first = (log.get("trials") or [{}])[0]
    if (log.get("checks") or [{}])[0].get("ghost"):
        log["tip_judged"] = "second apex (the steps of the check frame drawn twice)"
    else:
        log["tip_judged"] = ("sound" if first.get("ok") and first["chi2_red"] <= CHI2_TRIAL else "not sound (a second apex, or the place)") if first else "not judged"
    log["stage"] = "dispersion"
    disp = s.dispersion()
    log["dispersion"] = {"e0_mev": round(disp["e0_ev"] * 1e3, 1), "m_star": round(disp["m_star"], 4), "e0_err_mev": round(disp["e0_err"] * 1e3, 2),
                         "m_err": round(disp["m_err"], 4), "dropped": disp["dropped"]}
    log["stage"] = "reported"
    ids = claim_ids(v.task())
    ev = [m["frame"] for m in s.maps]
    out = {}
    for cid in ids:
        if cid == "e0_mev":
            out[cid] = (disp["e0_ev"] * 1e3, "meV", ev)
        elif cid == "m_eff":
            out[cid] = (disp["m_star"], "m_e", ev)
    return out


def run(seed: int, *, out_dir: str | Path | None = None, scenario: str = SCENARIO) -> dict:
    """One P2 episode by the baseline; what it reported and how it scored, and (from the truth,
    after the episode only) how far off it was."""
    inst = Instrument(scenario, seed, out_dir)
    log: dict = {}
    got, error = {}, None
    try:
        got = play(Blind(inst), log)
    except Exception as exc:  # noqa: BLE001 — a failed session is a result
        error = f"{type(exc).__name__}: {exc}"
    for cid, (val, unit, ev) in got.items():
        inst.report(cid, val, unit=unit, evidence=ev)
    out = inst.finish()
    det = (inst.verdict.details or {}).get("claims", {})
    values = {cid: float(val) for cid, (val, _, _) in got.items()}
    ss = (inst.truth_after or {}).get("surface_state", {})
    want = {"e0_mev": float(ss["e0_mev"]), "m_eff": float(ss["m_star"])} if ss else {}
    err = {k: values[k] - want[k] for k in want if k in values}
    rel = {k: err[k] / want[k] for k in err if k == "m_eff"}
    return {"paper": "P2", "seed": int(seed), "success": bool(out["success"]), "partial": float(out.get("partial", 0.0)),
            "error": error, "values": values, "truth": want, "abs_err": err, "rel_err": rel,
            "claims_ok": {k: bool(c.get("value_ok")) and bool(c.get("evidence_ok", True)) for k, c in det.items()},
            "instrument_s": float(inst.used_s), "actions": int(out.get("actions", 0)),
            "tip_condition": inst.scenario.draw_hidden(seed).get("tip", {}).get("condition"), **log}
