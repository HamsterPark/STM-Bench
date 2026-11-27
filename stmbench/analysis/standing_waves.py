"""Standing waves at a step edge: the surface state's dispersion from dI/dV maps.

A step edge reflects the Shockley surface state, and a dI/dV map at bias E shows the
interference as ripples of period pi/k(E) running away from the edge. With the fast scan
direction across the edge every row is one profile; the profiles of many rows, each aligned
at its own edge, average the noise down. Fitting k at several energies gives the parabola
E = E0 + hbar^2 k^2 / (2 m* m_e), whose intercept is the band bottom and whose slope is the
effective mass.

Design choices, each from a failure it avoids:

* The edge is located per row and the rows are aligned at their own edge. A step is never
  quite straight and the sample drifts slowly during a frame; a common edge position would
  smear the ripples at the far end of the profile, where the wave carries the most k leverage.
* The fit carries a free phase, ``cos(phi) J0(2kd) + sin(phi) H0(2kd)``. The edge position is
  only known to a fraction of a pixel and the reflection has a phase of its own; an error
  delta in the edge position is, far from the edge, a phase 2 k delta. A J0-only model turns
  that into a wrong k (a 0.15 nm error is a few percent in m*); a free phase absorbs it.
* A decay length is fitted as well: the wave is damped over the phase-coherence length, which
  depends on energy and is not known to the measurer.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, special

HBAR2_OVER_2M_EV_NM2 = 0.0381   # ħ²/2mₑ, eV·nm² (a physical constant; do not import it from stmsim)


def _smooth_rows(z: np.ndarray, px: int) -> np.ndarray:
    z = np.asarray(z, float)
    if px <= 1:
        return z
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(z, size=int(px), axis=1, mode="nearest")


def _theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope and offset (y = offset + slope * x) as the median of pairwise slopes."""
    n = x.size
    if n < 2:
        return 0.0, float(np.median(y)) if n else 0.0
    i, j = np.triu_indices(n, 1)
    dx = x[j] - x[i]
    ok = dx != 0
    slope = float(np.median((y[j] - y[i])[ok] / dx[ok])) if ok.any() else 0.0
    return slope, float(np.median(y - slope * x))


def edge_positions(z: np.ndarray, nm_per_px: float, *, smooth_px: int = 3) -> dict:
    """Each row is one scan line crossing one step edge: the sub-pixel position (nm from the
    row's start) of its steepest z change, the sign of the step (+1 up, -1 down, the majority),
    a robust straight-line fit of position against row (offset, slope_nm_per_row) and a mask of
    the rows that agree with it (row jumps, a doubled edge, an adsorbate on the edge fall out).
    Keys: pos_nm (n_rows,), sign, slope_nm_per_row, offset_nm, good (bool mask), n_good.

    The steepest change is searched near the column where the whole frame changes most, so that
    a bump elsewhere in a noisy row cannot win; the row mean of z is removed first, which makes
    a slow tilt along the row harmless."""
    z = np.asarray(z, float)
    n_rows, n_cols = z.shape
    zs = _smooth_rows(z, smooth_px)
    g = np.gradient(zs, axis=1)
    gm = np.nanmean(g, axis=0)
    c0 = int(np.nanargmax(np.abs(gm)))
    half = max(4, int(round(3.0 / nm_per_px)), n_cols // 5)
    lo, hi = max(1, c0 - half), min(n_cols - 1, c0 + half + 1)
    pos = np.full(n_rows, np.nan)
    sgn = np.zeros(n_rows)
    for r in range(n_rows):
        seg = g[r, lo:hi]
        if not np.isfinite(seg).any():
            continue
        i = int(np.nanargmax(np.abs(seg)))
        c = lo + i
        off = 0.0
        if 0 < c < n_cols - 1:
            a, b, d = np.abs(g[r, c - 1]), np.abs(g[r, c]), np.abs(g[r, c + 1])
            den = a - 2 * b + d
            if den != 0:
                off = float(np.clip(0.5 * (a - d) / den, -1.0, 1.0))
        pos[r] = (c + off + 0.5) * nm_per_px
        sgn[r] = np.sign(g[r, c])
    sign = 1 if np.nansum(sgn) >= 0 else -1
    ok = np.isfinite(pos) & (sgn == sign)
    rows = np.arange(n_rows, dtype=float)
    good = ok.copy()
    slope = offset = 0.0
    for _ in range(3):
        if good.sum() < 3:
            break
        slope, offset = _theil_sen(rows[good], pos[good])
        res = pos - (offset + slope * rows)
        mad = 1.4826 * np.median(np.abs(res[good] - np.median(res[good])))
        tol = max(3.0 * mad, 1.5 * nm_per_px)
        new = ok & (np.abs(res) <= tol)
        if new.sum() < 3 or np.array_equal(new, good):
            good = new if new.sum() >= 3 else good
            break
        good = new
    return {"pos_nm": pos, "sign": int(sign), "slope_nm_per_row": float(slope),
            "offset_nm": float(offset), "good": good, "n_good": int(good.sum())}


def step_profile(didv: np.ndarray, edges: dict, nm_per_px: float, *, side: int,
                 d_min_nm: float = 0.6, d_max_nm: float = 18.0, step_nm: float | None = None) -> dict:
    """The good rows aligned at their own edge, on one side (+1: towards higher column index),
    each normalised by its own mean, resampled to one distance grid and averaged. Keys: d_nm,
    y (mean), sem, n_rows. ``step_nm`` defaults to the pixel size.

    The row mean is taken over the stretch that is used (``d_min_nm`` to ``d_max_nm``): a
    dI/dV row is a product of the bias-dependent lock-in gain and the local density of states,
    and only the shape carries the wave. The distance grid is cut where fewer than half the
    rows reach (the frame ends, or the edge sits near the far end)."""
    a = np.asarray(didv, float)
    n_rows, n_cols = a.shape
    step = float(step_nm) if step_nm else float(nm_per_px)
    d = np.arange(d_min_nm, d_max_nm + 0.5 * step, step)
    x = (np.arange(n_cols) + 0.5) * nm_per_px
    pos = np.asarray(edges["pos_nm"], float)
    good = np.asarray(edges["good"], bool)
    rows = []
    for r in np.flatnonzero(good):
        p = pos[r]
        xs = p + side * d
        inside = (xs >= x[0]) & (xs <= x[-1])
        if inside.sum() < 5:
            continue
        v = np.full(d.size, np.nan)
        v[inside] = np.interp(xs[inside], x, a[r])
        m = np.nanmean(v)
        if not np.isfinite(m) or m == 0:
            continue
        rows.append(v / m)
    if not rows:
        return {"d_nm": d[:0], "y": d[:0], "sem": d[:0], "n_rows": 0}
    arr = np.vstack(rows)
    cnt = np.isfinite(arr).sum(axis=0)
    keep = cnt >= max(2, 0.5 * len(rows))
    arr, d, cnt = arr[:, keep], d[keep], cnt[keep]
    mean = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0, ddof=1) if len(rows) > 1 else np.full(d.size, np.nan)
    return {"d_nm": d, "y": mean, "sem": sd / np.sqrt(np.maximum(cnt, 1)), "n_rows": len(rows)}


def _basis(d: np.ndarray, k: float, L: float) -> np.ndarray:
    z = 2.0 * k * d
    damp = np.exp(-2.0 * d / L)
    return np.column_stack([np.ones_like(d), special.j0(z) * damp, special.struve(0, z) * damp])


def _linear_part(d, y, w, k, L):
    B = _basis(d, k, L)
    coef, *_ = np.linalg.lstsq(B * w[:, None], y * w, rcond=None)
    return coef, float(np.sum((w * (B @ coef - y)) ** 2))


def fit_step_wave(d_nm: np.ndarray, y: np.ndarray, *, k_guess: float, sem: np.ndarray | None = None) -> dict:
    """c + A·[cos φ·J0(2k d) + sin φ·H0(2k d)]·exp(-2 d / L): the standing wave of one straight
    edge with a complex reflection, multistart in k (±10 %) and φ. Keys: k, k_err, phase, amp,
    decay_nm, offset, chi2_red, ok.

    Given k and L the model is linear in (c, A cos φ, A sin φ), so φ needs no starting values:
    it is solved exactly at every trial (variable projection). k is scanned over ±10 % of the
    guess and L over a log grid, the best cell is refined with all parameters free and k_err
    comes from the covariance, scaled by the reduced chi² when that exceeds 1. ``ok`` is False
    for a fit that ran into its bounds or lacks data."""
    d = np.asarray(d_nm, float)
    y = np.asarray(y, float)
    fin = np.isfinite(d) & np.isfinite(y)
    if sem is not None:
        s = np.asarray(sem, float)
        fin &= np.isfinite(s) & (s > 0)
    d, y = d[fin], y[fin]
    bad = {"k": float("nan"), "k_err": float("nan"), "phase": float("nan"), "amp": float("nan"),
           "decay_nm": float("nan"), "offset": float("nan"), "chi2_red": float("nan"), "ok": False}
    if d.size < 12:
        return bad
    if sem is not None:
        sig = np.asarray(sem, float)[fin]
        # an error bar below a tenth of the median is a numerical accident, not information
        sig = np.maximum(sig, 0.1 * np.median(sig))
    else:
        sig = np.full(d.size, 1.0)
    w = 1.0 / sig
    L_lo, L_hi = 8.0, 400.0
    ks = k_guess * np.linspace(0.9, 1.1, 81)
    Ls = np.geomspace(L_lo, L_hi, 9)
    best = (np.inf, None, None)
    for k in ks:
        for L in Ls:
            _, chi = _linear_part(d, y, w, k, L)
            if chi < best[0]:
                best = (chi, k, L)
    _, k0, L0 = best

    def resid(p):
        c, a, b, k, logL = p
        B = _basis(d, k, float(np.exp(logL)))
        return w * (B @ np.array([c, a, b]) - y)

    coef0, _ = _linear_part(d, y, w, k0, L0)
    p0 = np.array([*coef0, k0, np.log(L0)])
    lb = [-np.inf, -np.inf, -np.inf, 0.8 * k_guess, np.log(L_lo)]
    ub = [np.inf, np.inf, np.inf, 1.2 * k_guess, np.log(L_hi)]
    try:
        sol = optimize.least_squares(resid, p0, bounds=(lb, ub), x_scale=[1, 1, 1, 0.05, 0.5])
    except (ValueError, RuntimeError):
        return bad
    c, a, b, k, logL = sol.x
    dof = max(d.size - 5, 1)
    chi2_red = float(np.sum(sol.fun ** 2) / dof)
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * max(chi2_red, 1.0)
        k_err = float(np.sqrt(max(cov[3, 3], 0.0)))
    except np.linalg.LinAlgError:
        k_err = float("nan")
    at_bound = (abs(k - lb[3]) < 1e-6 * k_guess) or (abs(k - ub[3]) < 1e-6 * k_guess)
    amp = float(np.hypot(a, b))
    return {"k": float(k), "k_err": k_err, "phase": float(np.arctan2(b, a)), "amp": amp,
            "decay_nm": float(np.exp(logL)), "offset": float(c), "chi2_red": chi2_red,
            "ok": bool(np.isfinite(k_err) and not at_bound and amp > 0)}


def k_guess(e_ev: float, *, e0_ev: float = -0.44, m_star: float = 0.38) -> float:
    """A starting k from the literature dispersion of Cu(111) (knowledge the task allows)."""
    return float(np.sqrt(max(m_star * (e_ev - e0_ev), 1e-6) / HBAR2_OVER_2M_EV_NM2))


def fit_dispersion(e_ev, k, k_err=None, *, drop_outliers: bool = True) -> dict:
    """E = E0 + (ħ²/2m*mₑ)·k²: a weighted straight line in k². Drops at most one point beyond
    3 sigma and says which. Keys: e0_ev, m_star, e0_err, m_err, used (mask), dropped, resid_mev.

    The line is k² = a·E + b with errors on k² (σ = 2 k σ_k; the energy is set, not measured),
    so m* = ħ²/2mₑ · a and E0 = -b/a. An error bar below 0.5 % of k is raised to that: the
    fit's own error bars are the least reliable thing about it, and one point should not be able
    to pin the line by being claimed to be exact. ``dropped`` is the index of the removed point
    or None."""
    e = np.asarray(e_ev, float)
    k = np.asarray(k, float)
    ke = np.zeros_like(k) if k_err is None else np.asarray(k_err, float)
    sk = np.maximum(np.where(np.isfinite(ke), ke, 0.0), 0.005 * k)
    sig = 2.0 * k * sk
    used = np.isfinite(e) & np.isfinite(k) & (sig > 0)

    def line(mask):
        A = np.column_stack([e[mask], np.ones(mask.sum())])
        wv = 1.0 / sig[mask]
        coef, *_ = np.linalg.lstsq(A * wv[:, None], k[mask] ** 2 * wv, rcond=None)
        res = (A @ coef - k[mask] ** 2) * wv
        dof = max(mask.sum() - 2, 1)
        cov = np.linalg.inv((A * wv[:, None]).T @ (A * wv[:, None])) * max(float(np.sum(res ** 2)) / dof, 1.0)
        return coef, cov

    dropped = None
    coef, cov = line(used)
    if drop_outliers and used.sum() >= 5:
        z = np.abs((coef[0] * e + coef[1] - k ** 2) / sig)
        z[~used] = 0.0
        i = int(np.argmax(z))
        if z[i] > 3.0:
            used[i] = False
            dropped = i
            coef, cov = line(used)
    a, b = float(coef[0]), float(coef[1])
    m_star = HBAR2_OVER_2M_EV_NM2 * a
    e0 = -b / a
    m_err = HBAR2_OVER_2M_EV_NM2 * float(np.sqrt(cov[0, 0]))
    # e0 = -b/a: first-order propagation
    g = np.array([b / a ** 2, -1.0 / a])
    e0_err = float(np.sqrt(max(g @ cov[[0, 1]][:, [0, 1]] @ g, 0.0)))
    model_e = e0 + HBAR2_OVER_2M_EV_NM2 * k ** 2 / m_star
    return {"e0_ev": float(e0), "m_star": float(m_star), "e0_err": e0_err, "m_err": m_err,
            "used": used, "dropped": dropped, "resid_mev": (e - model_e) * 1e3}
