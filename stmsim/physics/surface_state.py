"""Shockley surface state: dispersion, standing waves and quantum corrals.

A (111) noble-metal surface carries a two-dimensional free-electron gas that starts at
``E0`` below the Fermi level. Electrons scatter off step edges and adsorbates, and the
interference is what Crommie/Lutz/Eigler (Nature 363, 524) and Hasegawa/Avouris (PRL 71,
1071) imaged in 1993 — ripples of period ``π/k(E)`` around every scatterer, from which the
band bottom ``E0`` and the effective mass ``m*`` follow.

Conventions (SI unless a name says otherwise; energies in eV relative to E_F):

* ``k[nm⁻¹] = 5.123·sqrt(m*·(E − E0)[eV])`` — the same ``1/sqrt(ħ²/2mₑ)`` constant the
  tunnelling barrier uses; below the band bottom ``k = 0`` and there is no standing wave.
* lifetime ``Γ(E) = Γ_F + β(E − E_F)²`` with ``β = (Γ_0 − Γ_F)/E0²`` (widest at the band
  bottom, narrowest at E_F), broadened by temperature to ``Γ_eff = hypot(Γ, 3.5 k_B T)``;
  the phase-coherence length is ``L_φ = 0.0762·k/(m*·Γ)`` nm (Cu at E_F: ≈66 nm, Bürgi).
* the LDOS is ``ρ(E) = 1 + h·L(E)·(1 + S(x, y, E))`` with a Lorentzian-broadened 2D band
  edge ``L(E) = ½ + (1/π)·atan((E − E0)/(Γ_eff/2))`` and the interference term ``S``.

``S`` has three forms, all damped by ``exp(−2d/L_φ)`` (the wave travels out and back):

* **straight step**, reflection ``r = |r|e^{iφ}``, averaged over incidence angle:
  ``S = |r|·[cos φ·J₀(2kd) + sin φ·H₀(2kd)]`` — the hard wall ``φ = π`` gives ``1 − J₀``;
* **isolated point scatterer**, s-wave with phase shift δ and absorption α ∈ [0, 1]
  (``c = α·e^{2iδ} − 1``): ``S = c_r J₀² − c_i J₀Y₀ + (|c|²/4)(J₀² + Y₀²)`` at ``kr``;
* **a corral** (`multiple=True`): the T-matrix ``M = I − (C/2)∘H`` with
  ``H_ij = H₀⁽¹⁾(k r_ij)·e^{−r_ij/L_φ}``, giving the angle-averaged ``⟨|ψ|²⟩ − 1`` that a
  dI/dV map measures (Heller et al., Nature 369, 464 — absorbing scatterers included). With
  one scatterer it reduces term-by-term to the single-scattering form above.

For the **topograph** the interference has to be integrated over the bias window. Because a
2D band has a constant density of states, ``dE ∝ k dk`` and every single-scattering integral
is closed form (Lommel / Struve recurrences), so a constant-current image costs no energy
grid at all. Multiple scattering has no such primitive: corrals are integrated numerically on
a coarse grid once per frame (:meth:`SurfaceState.corral_map`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy import special

from .junction import LDOSTemplate

HBAR2_OVER_2M_EV_NM2 = 0.0381        # ħ²/2mₑ in eV·nm²
K_CONST = 1.0 / math.sqrt(HBAR2_OVER_2M_EV_NM2)      # 5.123 nm⁻¹ per sqrt(eV)
KB_EV_PER_K = 8.617333e-5
THERMAL_FWHM = 3.5                   # the usual 3.5 k_B T instrumental width

# literature defaults; a scenario's ``hidden`` block draws around them per seed
SURFACE_STATE_DEFAULTS: dict[str, dict[str, float]] = {
    "Cu(111)": {"e0_ev": -0.44, "m_star": 0.38, "gamma_f_ev": 0.006, "gamma_0_ev": 0.018},
    "Au(111)": {"e0_ev": -0.49, "m_star": 0.26, "gamma_f_ev": 0.005, "gamma_0_ev": 0.016},
    "Ag(111)": {"e0_ev": -0.065, "m_star": 0.40, "gamma_f_ev": 0.004, "gamma_0_ev": 0.012},
}

# (phase shift δ, absorption α) by the surface feature that is scattering
SCATTERER_DEFAULTS: dict[str, tuple[float, float]] = {
    "adsorbate": (1.2, 0.45),
    "adatom": (1.2, 0.45),
    "cluster": (math.pi / 2, 0.30),
    "spatter": (math.pi / 2, 0.30),
    "pit": (math.pi / 2, 0.30),
    "crater": (math.pi / 2, 0.30),
}


@dataclass(frozen=True)
class SurfaceStateParams:
    e0_ev: float = -0.44
    m_star: float = 0.38
    gamma_f_ev: float = 0.006
    gamma_0_ev: float = 0.018
    step_height: float = 0.6            # relative LDOS step at the band edge
    step_r_up: float = 0.50             # |r| of the ascending step edge
    step_r_down: float = 0.40
    step_phi_up_rad: float = math.pi
    step_phi_down_rad: float = math.pi
    point_delta_rad: float = 1.2
    point_absorption: float = 0.45
    cutoff_m: float = 60e-9             # ignore scatterers further away than this
    r_min_m: float = 0.3e-9             # regularises the Y₀ log divergence at the scatterer
    temperature_k: float = 4.36

    @classmethod
    def for_material(cls, material: str, **overrides) -> "SurfaceStateParams":
        base = SURFACE_STATE_DEFAULTS.get(str(material), {})
        return cls(**{**base, **{k: v for k, v in overrides.items() if v is not None}})

    @classmethod
    def from_yaml(cls, cfg: dict[str, Any]) -> "SurfaceStateParams":
        known = {f for f in cls.__dataclass_fields__}
        material = cfg.get("material")
        base = cls.for_material(material) if material else cls()
        return replace(base, **{k: float(v) for k, v in cfg.items() if k in known})


@dataclass(frozen=True)
class PointScatterer:
    x: float
    y: float
    delta: float = 1.2
    alpha: float = 0.45
    kind: str = "adsorbate"
    ident: int | None = None


@dataclass(frozen=True)
class LineScatterer:
    """A straight step edge: distance is measured along the staircase coordinate."""
    distance_m: np.ndarray              # signed distance of each evaluated point to the edge
    r: float
    phi: float


class SurfaceState:
    """The electronic model. Pure physics: it never touches a Site or a Feature."""

    def __init__(self, params: SurfaceStateParams):
        self.p = params
        self._ss_fraction_cache: dict[int, float] = {}

    # ── dispersion & lifetime ──
    def k_of_e(self, e_ev):
        e = np.asarray(e_ev, float)
        return K_CONST * np.sqrt(np.clip(self.p.m_star * (e - self.p.e0_ev), 0.0, None))  # nm⁻¹

    def gamma(self, e_ev):
        p = self.p
        e = np.asarray(e_ev, float)
        beta = (p.gamma_0_ev - p.gamma_f_ev) / max(p.e0_ev ** 2, 1e-12)
        return np.maximum(p.gamma_f_ev + beta * e ** 2, 1e-6)

    def gamma_eff(self, e_ev):
        thermal = THERMAL_FWHM * KB_EV_PER_K * self.p.temperature_k
        return np.hypot(self.gamma(e_ev), thermal)

    def l_phi_nm(self, e_ev):
        k = self.k_of_e(e_ev)
        g = self.gamma(e_ev)
        return np.where(k > 0, 2 * HBAR2_OVER_2M_EV_NM2 * k / (self.p.m_star * g), 1e-9)

    def band_edge(self, e_ev):
        """L(E): the Lorentzian-broadened 2D step at the band bottom, 0…1."""
        e = np.asarray(e_ev, float)
        half = self.gamma_eff(e) / 2.0
        return 0.5 + np.arctan((e - self.p.e0_ev) / half) / np.pi

    # ── energy-resolved interference ──
    def line_modulation(self, d_m, e_ev, r: float, phi: float):
        """Standing wave from one straight edge at distance ``d`` (metres)."""
        d_nm = np.abs(np.asarray(d_m, float)) * 1e9
        k = self.k_of_e(e_ev)
        z = 2 * k * d_nm
        damp = np.exp(-2 * d_nm / np.maximum(self.l_phi_nm(e_ev), 1e-6))
        return r * (math.cos(phi) * special.j0(z) + math.sin(phi) * special.struve(0, z)) * damp

    @staticmethod
    def _c_of(delta: float, alpha: float) -> complex:
        return complex(alpha * math.cos(2 * delta) - 1.0, alpha * math.sin(2 * delta))

    def point_modulation(self, x, y, e_ev, points, *, multiple: bool = False):
        """Interference from point scatterers at sample-frame (x, y).

        ``multiple=False`` sums the isolated-scatterer form (right when the scatterers are far
        apart); ``multiple=True`` solves the T-matrix, which is what a corral needs."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if not points:
            return np.zeros(np.broadcast(x, np.asarray(e_ev, float)).shape)
        if multiple and len(points) > 1:
            return self._multiple_scattering(x, y, e_ev, points)
        out = None
        for s in points:
            r_nm = np.hypot(x - s.x, y - s.y) * 1e9
            r_nm = np.maximum(r_nm, self.p.r_min_m * 1e9)
            k = self.k_of_e(e_ev)
            z = np.asarray(k * r_nm, float)
            # Below the band bottom there is no propagating state and no interference: the
            # term is zero there. Evaluating Y0 at z = 0 instead gave −inf and a NaN that ran
            # through every spectrum below E0. A NaN edge at E0 would also expose hidden
            # reference values through downstream diagnostics.
            live = z > 0
            zs = np.where(live, z, 1.0)
            j0, y0 = special.j0(zs), special.y0(zs)
            c = self._c_of(s.delta, s.alpha)
            damp = np.exp(-2 * r_nm / np.maximum(self.l_phi_nm(e_ev), 1e-6))
            term = (c.real * j0 ** 2 - c.imag * j0 * y0 + (abs(c) ** 2 / 4.0) * (j0 ** 2 + y0 ** 2)) * damp
            term = np.where(live, term, 0.0)
            out = term if out is None else out + term
        return out

    def _multiple_scattering(self, x, y, e_ev, points):
        """Angle-averaged ⟨|ψ|²⟩ − 1 for N scatterers, one N×N solve per energy."""
        pos = np.array([[s.x, s.y] for s in points], float) * 1e9        # nm
        cs = np.array([self._c_of(s.delta, s.alpha) for s in points], complex)
        energies = np.atleast_1d(np.asarray(e_ev, float))
        pts = np.column_stack([np.asarray(x, float).ravel() * 1e9,
                               np.asarray(y, float).ravel() * 1e9])
        rij = np.hypot(pos[:, None, 0] - pos[None, :, 0], pos[:, None, 1] - pos[None, :, 1])
        np.fill_diagonal(rij, 0.0)
        rho = np.hypot(pts[:, None, 0] - pos[None, :, 0], pts[:, None, 1] - pos[None, :, 1])
        rho = np.maximum(rho, self.p.r_min_m * 1e9)
        out = np.zeros((energies.size, pts.shape[0]))
        eye = np.eye(len(points))
        off = ~eye.astype(bool)
        big = pts.shape[0] >= self.TABLE_THRESHOLD
        for n, e in enumerate(energies):
            k = float(self.k_of_e(e))
            if k <= 0:
                continue
            lphi = float(np.maximum(self.l_phi_nm(e), 1e-6))
            H = np.zeros_like(rij, dtype=complex)
            zz = k * rij[off]
            H[off] = special.hankel1(0, zz) * np.exp(-rij[off] / lphi)
            J = np.eye(len(points), dtype=complex)
            J[off] = special.j0(k * rij[off]) * np.exp(-rij[off] / lphi)
            M = np.eye(len(points), dtype=complex) - (cs[:, None] / 2.0) * H
            if big:
                # a frame is millions of (point, scatterer) pairs but each term depends only on
                # the distance, so one table per scatterer replaces the Hankel calls entirely
                h = np.empty(rho.shape, dtype=complex)
                j = np.empty(rho.shape)
                for c in range(rho.shape[1]):
                    col = rho[:, c]
                    grid = np.linspace(float(col.min()), float(col.max()), self.TABLE_POINTS)
                    damp_g = np.exp(-grid / lphi)
                    hg = special.hankel1(0, k * grid) * damp_g
                    h[:, c] = np.interp(col, grid, hg.real) + 1j * np.interp(col, grid, hg.imag)
                    j[:, c] = np.interp(col, grid, special.j0(k * grid) * damp_g)
            else:
                damp = np.exp(-rho * (1.0 / lphi))
                h = special.hankel1(0, k * rho) * damp          # (n_pts, N)
                j = special.j0(k * rho) * damp
            v = (cs[None, :] / 2.0) * np.linalg.solve(M.T, h.T).T
            out[n] = 2.0 * np.real(np.einsum("pi,pi->p", v, j)) \
                + np.real(np.einsum("pi,ij,pj->p", np.conj(v), J, v))
        shape = np.shape(x)
        if energies.size == 1:
            return out[0].reshape(shape)
        return out.reshape((energies.size,) + tuple(shape))

    # ── bias-integrated interference (closed form, single scattering) ──
    def _k_window(self, bias_v: float) -> tuple[float, float]:
        lo, hi = (min(0.0, bias_v), max(0.0, bias_v))
        return float(self.k_of_e(lo)), float(self.k_of_e(hi))

    #: above this many points a radial function is tabulated once and interpolated: the
    #: Struve/Bessel calls are the cost of a frame, and the value depends only on distance
    TABLE_THRESHOLD = 4000
    TABLE_POINTS = 4096

    def _radial(self, dist, fn):
        """``fn`` evaluated at every distance — through a 1-D table when there are many."""
        dist = np.asarray(dist, float)
        if dist.size < self.TABLE_THRESHOLD:
            return fn(dist)
        lo, hi = float(np.min(dist)), float(np.max(dist))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
            return fn(dist)
        grid = np.linspace(lo, hi, self.TABLE_POINTS)
        return np.interp(dist, grid, fn(grid))

    def line_modulation_integrated(self, d_m, bias_v: float, r: float, phi: float):
        k1, k2 = self._k_window(bias_v)
        if k2 <= k1:
            return np.zeros_like(np.asarray(d_m, float))
        d_nm = np.maximum(np.abs(np.asarray(d_m, float)) * 1e9, 1e-3)
        e_mid = bias_v / 2.0
        lphi = max(float(self.l_phi_nm(e_mid)), 1e-6)
        norm = (k2 ** 2 - k1 ** 2) / 2.0

        def whole(dd):
            def F(k):
                z = 2 * k * dd
                return k * (math.cos(phi) * special.j1(z)
                            + math.sin(phi) * special.struve(1, z)) / (2 * dd)
            return r * (F(k2) - F(k1)) / norm * np.exp(-2 * dd / lphi)

        return self._radial(d_nm, whole)

    def point_modulation_integrated(self, x, y, bias_v: float, points):
        k1, k2 = self._k_window(bias_v)
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if k2 <= k1 or not points:
            return np.zeros_like(x)
        e_mid = bias_v / 2.0
        lphi = max(float(self.l_phi_nm(e_mid)), 1e-6)
        out = np.zeros_like(x)
        norm = (k2 ** 2 - k1 ** 2) / 2.0
        for s in points:
            r_nm = np.maximum(np.hypot(x - s.x, y - s.y) * 1e9, self.p.r_min_m * 1e9)
            c = self._c_of(s.delta, s.alpha)

            def whole(rr, c=c):
                def F(k):
                    if k <= 0:
                        # the window starts below the band bottom: nothing propagates there
                        # (and 0 · Y0(0) is NaN, not 0)
                        return np.zeros_like(np.asarray(rr, float))
                    z = k * rr
                    j0, j1 = special.j0(z), special.j1(z)
                    y0, y1 = special.y0(z), special.y1(z)
                    k2h = k ** 2 / 2.0
                    return (c.real * k2h * (j0 ** 2 + j1 ** 2)
                            - c.imag * k2h * (j0 * y0 + j1 * y1)
                            + (abs(c) ** 2 / 4.0) * k2h * (j0 ** 2 + j1 ** 2 + y0 ** 2 + y1 ** 2))
                return (F(k2) - F(k1)) / norm * np.exp(-2 * rr / lphi)

            out = out + self._radial(r_nm, whole)
        return out

    def ss_fraction(self, bias_v: float) -> float:
        """How much of the tunnel current inside the window comes from the surface state."""
        key = int(round(bias_v * 1e6))
        hit = self._ss_fraction_cache.get(key)
        if hit is not None:
            return hit
        lo, hi = (min(0.0, bias_v), max(0.0, bias_v))
        if hi - lo < 1e-9:
            self._ss_fraction_cache[key] = 0.0
            return 0.0
        grid = np.linspace(lo, hi, 129)
        integral = float(np.trapezoid(self.band_edge(grid), grid)) * self.p.step_height
        frac = abs(integral) / (abs(hi - lo) + abs(integral))
        self._ss_fraction_cache[key] = frac
        return frac

    def apparent_height(self, s_bar, bias_v: float, kappa_m: float):
        """Constant-current topograph offset from an interference contrast ``s_bar``."""
        f = self.ss_fraction(bias_v)
        if f <= 0.0:
            return np.zeros_like(np.asarray(s_bar, float))
        arg = np.maximum(1.0 + f * np.asarray(s_bar, float), 0.05)
        return np.log(arg) / (2.0 * max(kappa_m, 1.0))

    def corral_map(self, xs: np.ndarray, ys: np.ndarray, bias_v: float, points,
                   *, n_energy: int = 0):
        """Bias-integrated multiple-scattering contrast on a grid (no closed form exists)."""
        lo, hi = (min(0.0, bias_v), max(0.0, bias_v))
        if hi - lo < 1e-9 or not points:
            return np.zeros(np.shape(xs))
        n = n_energy or int(np.clip(math.ceil(abs(bias_v) / 0.010), 1, 8))
        nodes, weights = np.polynomial.legendre.leggauss(int(n))
        e_nodes = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
        # dE ∝ k dk in 2D, so weight each energy by k(E)
        kk = self.k_of_e(e_nodes)
        if float(np.sum(kk * weights)) <= 0:
            return np.zeros(np.shape(xs))
        s = self._multiple_scattering(xs, ys, e_nodes, points)
        s = np.atleast_2d(s.reshape(len(e_nodes), -1))
        acc = np.tensordot(weights * kk, s, axes=(0, 0)) / float(np.sum(weights * kk))
        return acc.reshape(np.shape(xs))


def local_ldos(state: SurfaceState, x: float, y: float, *, lines=(), points=(),
               material: str = "", multiple: bool = False, tip_like=None) -> "LocalLDOS":
    return LocalLDOS(state, x, y, lines=tuple(lines), points=tuple(points),
                     material=material, multiple=multiple)


class LocalLDOS(LDOSTemplate):
    """The sample LDOS **at one position**: the material's band edge plus interference.

    ``rho(e)`` keeps :class:`LDOSTemplate`'s contract, so ``iv_curve`` / ``didv_curve`` are
    unchanged; ``S(E)`` is evaluated once on a fixed grid and interpolated afterwards, which
    is what keeps a corral spectrum (an N×N solve per energy) inside one second."""

    #: fine 1 meV steps across the band edge and the useful window, 10 meV elsewhere
    FINE_STEP_EV = 0.001
    COARSE_STEP_EV = 0.010
    FINE_HI_EV = 0.6
    RANGE_EV = 1.5

    def __init__(self, state: SurfaceState, x: float, y: float, *, lines=(), points=(),
                 material: str = "", multiple: bool = False):
        super().__init__(name=material or "surface_state", onset_ev=state.p.e0_ev,
                         step_height=state.p.step_height,
                         broadening_ev=float(state.gamma_eff(state.p.e0_ev)) / 2.0,
                         source=f"surface_state:e0={state.p.e0_ev:.3f},m*={state.p.m_star:.3f}")
        self.ss = state
        self.x = float(x)
        self.y = float(y)
        self.lines = tuple(lines)
        self.points = tuple(points)
        self.multiple = bool(multiple)
        self._grid: np.ndarray | None = None
        self._s: np.ndarray | None = None

    def _energy_grid(self) -> np.ndarray:
        p = self.ss.p
        fine = np.arange(p.e0_ev - 0.05, self.FINE_HI_EV, self.FINE_STEP_EV)
        lo = np.arange(-self.RANGE_EV, p.e0_ev - 0.05, self.COARSE_STEP_EV)
        hi = np.arange(self.FINE_HI_EV, self.RANGE_EV, self.COARSE_STEP_EV)
        return np.unique(np.concatenate([lo, fine, hi]))

    def _ensure(self) -> None:
        if self._s is not None:
            return
        grid = self._energy_grid()
        s = np.zeros_like(grid)
        for ln in self.lines:
            s = s + self.ss.line_modulation(ln.distance_m, grid, ln.r, ln.phi)
        if self.points:
            pm = self.ss.point_modulation(np.array(self.x), np.array(self.y), grid,
                                          self.points, multiple=self.multiple)
            s = s + np.asarray(pm, float).reshape(grid.shape)
        self._grid, self._s = grid, s

    def modulation(self, e_ev):
        self._ensure()
        return np.interp(np.asarray(e_ev, float), self._grid, self._s)

    def rho(self, e: np.ndarray) -> np.ndarray:
        e = np.asarray(e, float)
        out = 1.0 + self.step_height * self.ss.band_edge(e) * (1.0 + self.modulation(e))
        for e0, amp, w in self.extra_peaks:
            out = out + amp * np.exp(-0.5 * ((e - e0) / w) ** 2)
        return np.maximum(out, 0.02)
