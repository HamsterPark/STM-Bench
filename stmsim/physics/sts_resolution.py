"""What a bias sweep resolves: the instrument's energy kernel, the Kondo shapes seen through it,
and the lateral average over the tip's apexes (the ``sts_resolution`` physics switch).

Pure numerics, no world state: :meth:`stmsim.physics.world.World._sts_curve_resolved` feeds it.

**Energy kernel.** A dI/dV reading is the sample's conductance convolved with

* the thermal kernel ``-df/dE = 1 / (4 k_B T cosh²(E / 2 k_B T))`` (FWHM ≈ 3.53 k_B T): the tip's
  Fermi edge, a textbook relation. The resonance's own temperature dependence (a real Kondo line
  broadens with T beyond the tip's edge) is NOT modelled: Γ is the line's width at the rig's T;
* with the lock-in on, the first-harmonic instrument function of a sinusoidal modulation of
  zero-to-peak amplitude ``V_p``, ``χ(v) = 2 sqrt(V_p² - v²) / (π V_p²)`` for ``|v| < V_p`` (a
  semicircle: unit area, FWHM ``sqrt(3) V_p``). It follows from integrating by parts the first
  Fourier coefficient of ``I(V_0 + V_p sin ωt)``: the reading is ``∫ G(V_0 + v) χ(v) dv``. The
  convention (zero to peak, not peak to peak) is the one the instrument's manual states.

Both are built as exact bin weights (differences of their cumulative distributions), so the
discrete kernel has unit area at any grid spacing.

**Kondo shapes.** ``L = 1/(1 + ε²)`` and ``D = ε/(1 + ε²)`` with ``ε = (E - ε_K)/Γ`` (the two
parts of the Fano line, :mod:`stmsim.physics.kondo`), convolved with the kernel on a uniform
energy grid by FFT. The grid covers the sweep's window (never more than ±:data:`W_CAP_V` of it; a
point beyond sees the unbroadened shape, whose broadening is a second-order correction there)
plus the kernel's width, so that the convolution has no edge effect inside the window.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from .kondo import KB_EV_PER_K

#: the thermal kernel is cut at this many k_B T on each side (logistic tail 6e-6) and renormalised
THERMAL_CUT_KT = 12.0
#: smallest energy step of the grid (eV): below the lock-in / thermal / Kondo resolutions of any rig
H_MIN_V = 2e-6
#: the grid never has more than about 2**18 points: a wide window gets a coarser step instead
MAX_GRID = 2 ** 18
#: half width of the window the convolved shapes are built for (V); beyond it, the bare shapes
W_CAP_V = 0.4
#: a magnetic atom farther than this from every apex position is not summed (m): a(r) < 1e-5 there
ATOM_REACH_M = 2.5e-9


# ───────────────────────────── kernels ─────────────────────────────

def thermal_kernel(e: np.ndarray, kt: float) -> np.ndarray:
    """``-df/dE`` at energy ``e`` (same unit as ``kt``)."""
    x = np.abs(np.asarray(e, float)) / (2.0 * kt)
    # 1/(4 kT cosh²(x)) written with exp(-2x) so that it cannot overflow
    return np.exp(-2.0 * x) / (kt * (1.0 + np.exp(-2.0 * x)) ** 2)


def lockin_kernel(v: np.ndarray, vp: float) -> np.ndarray:
    """First-harmonic instrument function of a modulation of zero-to-peak amplitude ``vp``."""
    v = np.asarray(v, float)
    inside = np.abs(v) < vp
    out = np.zeros_like(v)
    out[inside] = 2.0 * np.sqrt(vp * vp - v[inside] ** 2) / (math.pi * vp * vp)
    return out


def _thermal_cdf(x: np.ndarray, kt: float) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(np.asarray(x, float) / (2.0 * kt)))


def _lockin_cdf(x: np.ndarray, vp: float) -> np.ndarray:
    u = np.clip(np.asarray(x, float) / vp, -1.0, 1.0)
    return 0.5 + (u * np.sqrt(1.0 - u * u) + np.arcsin(u)) / math.pi


def thermal_weights(h: float, kt: float) -> np.ndarray:
    """Weights of the thermal kernel on bins of width ``h`` (odd length, centred, sum 1)."""
    m = int(math.ceil(THERMAL_CUT_KT * kt / h))
    x = np.arange(-m, m + 1) * h
    w = _thermal_cdf(x + 0.5 * h, kt) - _thermal_cdf(x - 0.5 * h, kt)
    return w / w.sum()


def lockin_weights(h: float, vp: float) -> np.ndarray:
    """Weights of the lock-in kernel on bins of width ``h`` (odd length, centred, sum 1)."""
    m = int(math.ceil(vp / h))
    x = np.arange(-m, m + 1) * h
    w = _lockin_cdf(x + 0.5 * h, vp) - _lockin_cdf(x - 0.5 * h, vp)
    return w / w.sum()


def instrument_weights(h: float, kt: float, vp: float | None) -> np.ndarray:
    """The kernel a reading is convolved with: thermal alone (``vp`` None, lock-in off), or the
    thermal kernel convolved with the lock-in's."""
    w = thermal_weights(h, kt)
    if vp is not None and vp > 0.0:
        w = np.convolve(w, lockin_weights(h, vp))
        w /= w.sum()
    return w


def kernel_pad(kt: float, vp: float | None) -> float:
    """Half width (V) beyond which the kernel carries no weight."""
    return THERMAL_CUT_KT * kt + (vp or 0.0)


# ───────────────────────────── Fano parts ─────────────────────────────

def fano_parts(e: np.ndarray, e_k: float, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """``L = 1/(1 + ε²)`` and ``D = ε/(1 + ε²)``, ``ε = (e - e_k)/gamma``."""
    eps = (np.asarray(e, float) - e_k) / gamma
    d = 1.0 + eps * eps
    return 1.0 / d, eps / d


def _fano_primitives(e: np.ndarray, e_k: float, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Antiderivatives of L and D: ``Γ atan ε`` and ``(Γ/2) ln(1 + ε²)``."""
    eps = (np.asarray(e, float) - e_k) / gamma
    return gamma * np.arctan(eps), 0.5 * gamma * np.log1p(eps * eps)


def grid_step(kt: float, vp: float | None, gamma: float | None, span: float) -> float:
    """Energy step: a tenth of the narrowest of the kernel widths and Γ, not below
    :data:`H_MIN_V` and not so small that the window needs more than :data:`MAX_GRID` points."""
    narrow = min(kt, vp if vp else kt, gamma if gamma else kt)
    return max(narrow / 10.0, H_MIN_V, span / MAX_GRID)


@dataclass
class Window:
    """The energy window of one sweep: the bias range the convolved shapes are valid for, and the
    uniform grid (valid range plus the kernel's width on each side)."""
    lo: float                  # valid range (V): the sweep's range with 0 included, clipped to ±W_CAP_V
    hi: float
    h: float
    grid: np.ndarray           # uniform, lo - pad ... hi + pad
    weights: np.ndarray        # the kernel on that step

    @classmethod
    def build(cls, v_min: float, v_max: float, kt: float, vp: float | None,
              gamma: float | None) -> "Window":
        lo = max(min(v_min, 0.0), -W_CAP_V)
        hi = min(max(v_max, 0.0), W_CAP_V)
        pad = kernel_pad(kt, vp)
        h = grid_step(kt, vp, gamma, hi - lo + 2.0 * pad)
        n = int(math.ceil((hi - lo + 2.0 * pad) / h)) + 1
        grid = lo - pad + h * np.arange(n)
        return cls(lo, hi, h, grid, instrument_weights(h, kt, vp))

    def smooth(self, y: np.ndarray) -> np.ndarray:
        """``y`` (sampled on :attr:`grid`) convolved with the kernel."""
        return fftconvolve(y, self.weights, mode="same")

    def inside(self, v: np.ndarray) -> np.ndarray:
        return (v >= self.lo) & (v <= self.hi)


def kondo_shapes(win: Window, v: np.ndarray, e_k: float, gamma: float, kt: float,
                 vp: float | None) -> dict[str, np.ndarray]:
    """The Fano parts at the sweep biases ``v``, as the instrument sees them.

    ``L_k``, ``D_k``: broadened by the thermal kernel and, with the lock-in on (``vp``), the
    lock-in's. ``lam_L``, ``lam_D``: ``∫₀^V`` of the *thermally* broadened shapes (a DC current
    carries no modulation). Beyond the window the shapes are the bare Fano parts and the integrals
    continue with their analytic antiderivatives from the window's edge."""
    v = np.asarray(v, float)
    l_bare, d_bare = fano_parts(win.grid, e_k, gamma)
    th = Window(win.lo, win.hi, win.h, win.grid, thermal_weights(win.h, kt))
    l_th, d_th = th.smooth(l_bare), th.smooth(d_bare)
    if vp:
        l_k, d_k = win.smooth(l_bare), win.smooth(d_bare)
    else:
        l_k, d_k = l_th, d_th
    inside = win.inside(v)
    out: dict[str, np.ndarray] = {}
    l_bv, d_bv = fano_parts(v, e_k, gamma)
    out["L_k"] = np.where(inside, np.interp(v, win.grid, l_k), l_bv)
    out["D_k"] = np.where(inside, np.interp(v, win.grid, d_k), d_bv)
    out["L_th"] = np.where(inside, np.interp(v, win.grid, l_th), l_bv)      # thermal only
    out["D_th"] = np.where(inside, np.interp(v, win.grid, d_th), d_bv)
    # cumulative integrals of the thermal shapes from the grid's start, then re-zeroed at E = 0
    for name, y, prim in (("lam_L", l_th, 0), ("lam_D", d_th, 1)):
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1])) * win.h])
        at = lambda e: np.interp(e, win.grid, cum)                      # noqa: E731
        val = at(np.clip(v, win.lo, win.hi)) - at(0.0)
        ana = _fano_primitives
        edge_hi = ana(np.array([win.hi]), e_k, gamma)[prim][0]
        edge_lo = ana(np.array([win.lo]), e_k, gamma)[prim][0]
        vv = ana(v, e_k, gamma)[prim]
        val = np.where(v > win.hi, val + vv - edge_hi, val)
        val = np.where(v < win.lo, val + vv - edge_lo, val)
        out[name] = val
    return out


def smoothed_background(win: Window, v: np.ndarray, v_nodes: np.ndarray, g_nodes: np.ndarray) -> np.ndarray:
    """``G_bg`` convolved with the kernel, at the sweep biases ``v``; beyond the window the
    background itself (its curvature scale there is far above the kernel's width)."""
    v = np.asarray(v, float)
    g_grid = np.interp(win.grid, v_nodes, g_nodes)
    g_s = win.smooth(g_grid)
    return np.where(win.inside(v), np.interp(v, win.grid, g_s), np.interp(v, v_nodes, g_nodes))


# ───────────────────────────── lateral average ─────────────────────────────

def _blur(r2: np.ndarray, s: float, sigma_t: float) -> np.ndarray:
    """Gaussian ``exp(-r²/2s²)`` averaged over a Gaussian of width ``sigma_t``:
    ``s²/(s² + σ_t²) · exp(-r²/(2 (s² + σ_t²)))`` (two Gaussians' overlap, closed form)."""
    s2 = s * s + sigma_t * sigma_t
    return (s * s / s2) * np.exp(-r2 / (2.0 * s2))


def kondo_coefficients(points: np.ndarray, apex_xy: np.ndarray, apex_w: np.ndarray, atoms: np.ndarray,
                       *, a0: float, q0: float, sigma_a: float, sigma_q: float,
                       sigma_t: float, chunk: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``C0, C1, C2`` at every sweep point: sums over apexes (weighted) and magnetic atoms of the
    blurred ``a``, ``a q`` and ``a q²``, so the conductance factor is
    ``1 + (C2 - C0) L + 2 C1 D``.

    ``points`` (n, 2) sample-frame positions, ``apex_xy`` (A, 2) the apexes' offsets, ``apex_w``
    (A, n) their normalised weights at each point, ``atoms`` (M, 2). With ``a(r) = a0 exp(-r²/2σ_a²)``
    and ``q(r) = q0 exp(-r²/2σ_q²)``: ``a`` has width σ_a, ``a q`` has ``s1`` with
    ``1/s1² = 1/σ_a² + 1/σ_q²`` and ``a q²`` has ``s2`` with ``1/s2² = 1/σ_a² + 2/σ_q²``."""
    n = points.shape[0]
    c0, c1, c2 = np.zeros(n), np.zeros(n), np.zeros(n)
    if atoms.size == 0:
        return c0, c1, c2
    s1 = (1.0 / sigma_a ** 2 + 1.0 / sigma_q ** 2) ** -0.5
    s2 = (1.0 / sigma_a ** 2 + 2.0 / sigma_q ** 2) ** -0.5
    for a in range(apex_xy.shape[0]):
        pos = points + apex_xy[a]
        for k0 in range(0, n, chunk):
            sl = slice(k0, min(n, k0 + chunk))
            d = pos[sl, None, :] - atoms[None, :, :]
            r2 = d[..., 0] ** 2 + d[..., 1] ** 2                      # (chunk, M)
            w = apex_w[a, sl]
            c0[sl] += w * a0 * _blur(r2, sigma_a, sigma_t).sum(axis=1)
            c1[sl] += w * a0 * q0 * _blur(r2, s1, sigma_t).sum(axis=1)
            c2[sl] += w * a0 * q0 * q0 * _blur(r2, s2, sigma_t).sum(axis=1)
    return c0, c1, c2
