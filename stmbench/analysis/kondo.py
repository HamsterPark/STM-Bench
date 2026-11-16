"""Fitting a Kondo resonance from measured bias sweeps, the way an experimenter does.

The line shape over a Kondo atom is the Fano form (Madhavan et al., Science 280, 567 (1998);
Knorr et al., PRL 88, 096804 (2002))::

    dI/dV  ~  b(V) * (1 + A * [ (q + e)^2 / (1 + e^2) - 1 ]),     e = (eV - eps_K) / Gamma

with a smooth background ``b(V)`` (here linear: ``c0 + c1 V``). The sweep sees it through two
broadenings: the thermal one (derivative of the Fermi function, FWHM ~ 3.53 kT) and the lock-in
one (a sinusoidal modulation of zero-to-peak amplitude ``V_p`` read at the first harmonic has the semicircle instrument
function, FWHM sqrt(3) V_p). Because ``(q+e)^2/(1+e^2) - 1 = (q^2-1) L + 2 q D`` with
``L = 1/(1+e^2)`` and ``D = e/(1+e^2)``, the broadening is applied to ``L`` and ``D``, the two
pieces that are not linear in the bias.

This module deliberately shares no code with the simulator (it imports nothing from ``stmsim``):
an independent analysis is what checks the simulator's physics.

Units: biases, energies and ``gamma_v``/``e_k_v`` are in volts (eV per e). The P6 claims are in
meV, so multiply by 1e3 when comparing.

The width Gamma is the half width at half maximum of the Fano feature as used in the papers
above; the fitted values are the *intrinsic* ones, with the instrument's broadening removed by
the model. If the true modulation at the junction is smaller than the one the operator set, a
fit that assumes the set value over-corrects and reads Gamma too narrow; keep the modulation
small compared with Gamma (see the tests that document this).
"""
from __future__ import annotations

from dataclasses import dataclass

import warnings

import numpy as np
from scipy.optimize import least_squares

KB_EV_PER_K = 8.617333262e-5  # Boltzmann constant in eV/K (CODATA 2018)

# Bounds the brief asks for: Gamma 0.3-40 meV, |q| <= 3, amplitude 0-2.
GAMMA_BOUNDS_V = (0.3e-3, 40e-3)
Q_BOUND = 3.0
AMP_BOUNDS = (0.0, 2.0)

# Grid of the internal fine bias axis (V). A thermal FWHM of ~1.3 mV at 4.36 K and the 0.3 mV
# minimum Gamma are both resolved at 0.1 mV; the kernel is integrated exactly per cell.
_FINE_DE_V = 1e-4


def _thermal_cdf(e: np.ndarray, kt: float) -> np.ndarray:
    # The thermal kernel -f'(E) has the Fermi function itself as its cumulative distribution.
    return 0.5 * (1.0 + np.tanh(e / (2.0 * kt)))


def _lockin_cdf(v: np.ndarray, vp: float) -> np.ndarray:
    # Cumulative distribution of the semicircle instrument function 2/(pi Vp^2) sqrt(Vp^2 - v^2).
    # Integrating it over each cell (rather than sampling the density) tames the edge singularity
    # of the derivative and keeps unit area exactly.
    x = np.clip(v / vp, -1.0, 1.0)
    return 0.5 + (x * np.sqrt(1.0 - x * x) + np.arcsin(x)) / np.pi


def _cell_weights(cdf, half_cells: int, de: float) -> np.ndarray:
    edges = (np.arange(-half_cells, half_cells + 2) - 0.5) * de
    c = cdf(edges)
    return np.diff(c)


def instrument_kernel(de_v: float, temp_k: float, vmod_peak_v: float) -> np.ndarray:
    """Thermal (x) lock-in kernel on a grid of spacing ``de_v`` (V), unit area.

    The array has odd length and its centre element sits at zero energy. ``vmod_peak_v = 0``
    gives the thermal kernel alone, ``temp_k = 0`` the lock-in alone (both zero: a delta).
    """
    de = float(de_v)
    kt = KB_EV_PER_K * float(temp_k)
    parts = []
    if kt > 0:
        # cosh^-2 tails fall as exp(-E/kT); 20 kT leaves < 1e-8 of the area outside.
        n = int(np.ceil(20.0 * kt / de))
        parts.append(_cell_weights(lambda e: _thermal_cdf(e, kt), n, de))
    if vmod_peak_v > 0:
        n = int(np.ceil(vmod_peak_v / de)) + 1
        parts.append(_cell_weights(lambda e: _lockin_cdf(e, float(vmod_peak_v)), n, de))
    if not parts:
        return np.ones(1)
    k = parts[0]
    for p in parts[1:]:
        k = np.convolve(k, p)
    return k / k.sum()


def _fine_axis(v_lo: float, v_hi: float, half_kernel: int, de: float) -> np.ndarray:
    i0 = int(np.floor(v_lo / de)) - half_kernel - 1
    i1 = int(np.ceil(v_hi / de)) + half_kernel + 1
    return np.arange(i0, i1 + 1) * de


def _broadened_ld(gamma, e_k, v_fine, kernel):
    e = (v_fine - e_k) / gamma
    d = 1.0 / (1.0 + e * e)
    ll = np.convolve(d, kernel, mode="same")
    dd = np.convolve(e * d, kernel, mode="same")
    return ll, dd


def fano_didv(v, gamma_v, q, e_k_v, amp, *, temp_k, vmod_peak_v, c0=1.0, c1=0.0,
              kernel=None, de_v=_FINE_DE_V):
    """Model ``(c0 + c1 v)(1 + amp [(q^2-1) L~ + 2 q D~])`` at biases ``v`` (V).

    ``L~``, ``D~`` are the Lorentzian and dispersive parts broadened by the thermal x lock-in
    kernel (computed on an internal 0.1 mV grid padded by the kernel's half width, then
    interpolated to ``v``). ``kernel`` may be passed to skip rebuilding it.
    """
    v = np.asarray(v, float)
    if kernel is None:
        kernel = instrument_kernel(de_v, temp_k, vmod_peak_v)
    half = len(kernel) // 2
    vf = _fine_axis(float(np.nanmin(v)), float(np.nanmax(v)), half, de_v)
    ll, dd = _broadened_ld(gamma_v, e_k_v, vf, kernel)
    shape = 1.0 + amp * ((q * q - 1.0) * ll + 2.0 * q * dd)
    return (c0 + c1 * v) * np.interp(v, vf, shape)


# ---- sweeps ------------------------------------------------------------------------------------

def on_grid(v, y, v_new):
    """Resample ``y(v)`` onto ``v_new`` (linear), ignoring non-finite points of the source;
    points of ``v_new`` outside the source range are NaN."""
    v = np.asarray(v, float)
    y = np.asarray(y, float)
    ok = np.isfinite(v) & np.isfinite(y)
    o = np.argsort(v[ok])
    return np.interp(np.asarray(v_new, float), v[ok][o], y[ok][o], left=np.nan, right=np.nan)


def mean_sweep(sweeps):
    """Average repeated sweeps of one grid: ``(mean, standard error of the mean)`` per point.

    NaNs are ignored point by point. With a single sweep the error is NaN (unknown, not zero).
    """
    a = np.asarray(sweeps, float)
    if a.ndim != 2:
        raise ValueError("sweeps must be a 2-D array: one row per sweep")
    n = np.sum(np.isfinite(a), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(a, axis=0)
        sd = np.nanstd(a, axis=0, ddof=1)
        sem = np.where(n > 1, sd / np.sqrt(n), np.nan)
    m = np.where(n > 0, m, np.nan)
    return m, sem


# ---- the fit -----------------------------------------------------------------------------------

@dataclass
class FanoFit:
    """Result of :func:`fit_fano`. Energies in volts; ``*_err`` are 1-sigma from the Jacobian
    (NaN when unavailable). ``c0``/``c1`` are in the units of the data (of the ratio, when a
    reference was used)."""
    gamma_v: float = float("nan")
    q: float = float("nan")
    e_k_v: float = float("nan")
    amp: float = float("nan")
    c0: float = float("nan")
    c1: float = float("nan")
    gamma_err: float = float("nan")
    q_err: float = float("nan")
    e_k_err: float = float("nan")
    amp_err: float = float("nan")
    r2: float = float("nan")
    rms: float = float("nan")
    n: int = 0
    success: bool = False
    note: str = ""

    @property
    def gamma_mev(self) -> float:
        return self.gamma_v * 1e3

    @property
    def e_k_mev(self) -> float:
        return self.e_k_v * 1e3


def _prepare(v, didv, reference, sigma, window):
    v = np.asarray(v, float)
    y = np.asarray(didv, float)
    s = None if sigma is None else np.broadcast_to(np.asarray(sigma, float), y.shape).copy()
    if reference is not None:
        if isinstance(reference, tuple):
            rv, ry = reference
            ref = on_grid(rv, ry, v)
        else:
            ref = np.asarray(reference, float)
            if ref.shape != y.shape:
                raise ValueError("a bare reference array must share the data's grid; pass (v, y)")
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bad = ~(ref > 0)
            y = np.where(bad, np.nan, y / ref)
            if s is not None:
                s = s / ref
    ok = np.isfinite(v) & np.isfinite(y)
    if s is not None:
        ok &= np.isfinite(s) & (s > 0)
    if window is not None:
        ok &= (v >= window[0]) & (v <= window[1])
    return v[ok], y[ok], (None if s is None else s[ok])


def fit_fano(v, didv, *, temp_k=4.36, vmod_peak_v=0.0, reference=None, sigma=None, window=None):
    """Fit a Fano resonance to a bias sweep.

    ``v`` in V, ``didv`` in any units. ``vmod_peak_v`` is the zero-to-peak lock-in amplitude the
    model assumes at the junction. ``reference`` is a bare-surface sweep, either an array on the
    same grid or a ``(v_ref, didv_ref)`` pair; the data are divided by it first (removing the
    tip's and substrate's smooth structure) and a linear background is fitted regardless.
    ``sigma`` (scalar or per point) weights the residuals; ``window=(v_lo, v_hi)`` restricts the
    points used. Non-finite points are dropped.

    Gamma, eps_K are returned in volts (the P6 claims are in meV). Starts cover several widths,
    signs of q and positions of the resonance; the best few are refined with bounded least
    squares (Gamma 0.3-40 meV, |q| <= 3, eps_K inside the fitted bias range, amp 0-2).
    """
    vv, yy, ss = _prepare(v, didv, reference, sigma, window)
    n = len(vv)
    res = FanoFit(n=n)
    if n < 12:
        res.note = "too few finite points"
        return res
    scale = float(np.median(np.abs(yy)))
    if not np.isfinite(scale) or scale <= 0:
        res.note = "data have no positive scale"
        return res
    y = yy / scale
    w = np.ones(n) if ss is None else scale / ss  # weights on the normalised residuals
    vmin, vmax = float(vv.min()), float(vv.max())

    kernel = instrument_kernel(_FINE_DE_V, temp_k, vmod_peak_v)
    half = len(kernel) // 2
    vf = _fine_axis(vmin, vmax, half, _FINE_DE_V)

    def model(p):
        g, q, ek, a, c0, c1 = p
        ll, dd = _broadened_ld(g, ek, vf, kernel)
        shape = np.interp(vv, vf, 1.0 + a * ((q * q - 1.0) * ll + 2.0 * q * dd))
        return (c0 + c1 * vv) * shape

    def resid(p):
        return (model(p) - y) * w

    lo = np.array([GAMMA_BOUNDS_V[0], -Q_BOUND, vmin, AMP_BOUNDS[0], -np.inf, -np.inf])
    hi = np.array([GAMMA_BOUNDS_V[1], Q_BOUND, vmax, AMP_BOUNDS[1], np.inf, np.inf])

    # Cheap screening of a start grid, then refinement of the best few.
    starts = []
    span = vmax - vmin
    for g in (1.5e-3, 4e-3, 10e-3, 25e-3):
        for q in (-1.5, -0.3, 0.5, 1.5):
            for ek in (-4e-3, 0.0, 4e-3):
                if vmin < ek < vmax:
                    starts.append([g, q, ek, 0.2, 1.0, 0.0])
    if not starts:
        starts.append([min(max(span / 8, GAMMA_BOUNDS_V[0] * 2), 0.02), 0.5, 0.5 * (vmin + vmax),
                       0.2, 1.0, 0.0])
    # The linear background is solved for each start so that screening compares line shapes.
    scored = []
    for p in starts:
        shape = model([p[0], p[1], p[2], p[3], 1.0, 0.0])
        design = np.vstack([shape, shape * vv]).T
        try:
            c, *_ = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)
        except np.linalg.LinAlgError:
            continue
        p2 = list(p)
        p2[4], p2[5] = float(c[0]), float(c[1])
        scored.append((float(np.sum(resid(p2) ** 2)), p2))
    scored.sort(key=lambda t: t[0])

    best = None
    for _, p0 in scored[:8]:
        p0 = np.clip(p0, lo + 1e-12, hi - 1e-12)
        try:
            sol = least_squares(resid, p0, bounds=(lo, hi), x_scale=[5e-3, 0.5, 2e-3, 0.1, 1.0, 5.0],
                                method="trf", max_nfev=200)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        res.note = "no start converged"
        return res

    p = best.x
    g, q, ek, a, c0, c1 = p
    fit = model(p)
    ss_res = float(np.sum((y - fit) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    res.gamma_v, res.q, res.e_k_v, res.amp = float(g), float(q), float(ek), float(a)
    res.c0, res.c1 = float(c0) * scale, float(c1) * scale
    res.rms = float(np.sqrt(ss_res / n)) * scale
    res.r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    dof = max(n - len(p), 1)
    try:
        jac = best.jac
        cov = np.linalg.inv(jac.T @ jac) * (2.0 * best.cost / dof)
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
        res.gamma_err, res.q_err, res.e_k_err, res.amp_err = (float(x) for x in err[:4])
    except np.linalg.LinAlgError:
        res.note = "covariance singular"
    at_bound = []
    if g <= lo[0] * 1.001 or g >= hi[0] * 0.999:
        at_bound.append("gamma")
    if abs(q) >= Q_BOUND * 0.999:
        at_bound.append("q")
    if a <= 1e-3 or a >= AMP_BOUNDS[1] * 0.999:
        at_bound.append("amp")
    if ek <= vmin + 1e-6 or ek >= vmax - 1e-6:
        at_bound.append("eps_K")
    res.success = bool(best.success) and not at_bound
    if at_bound:
        res.note = ("parameter at its bound: " + ", ".join(at_bound) + ("; " + res.note if res.note else ""))
    return res
