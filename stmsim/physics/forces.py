"""Tip–sample force and the frequency shift a qPlus sensor reports.

A qPlus sensor is a stiff quartz tuning fork oscillating at ``f0`` with amplitude ``A``; the
force gradient it feels shifts its resonance. What the instrument records is **Δf, not force**
— recovering ``F(z)`` from ``Δf(z)`` is the inversion Sader and Jarvis published in 2004, and
the thing Huber et al. (Science 366, 235, 2019) used to watch a bond form between a CO-tipped
probe and an iron adatom.

Model:

* long-range van der Waals, sphere over a plane: ``F = −A_H·R / (6(d + z0)²)``;
* short-range chemical bond, Morse: ``U = D_e[(1 − e^{−a(z − z_e)})² − 1]``, so
  ``F = −2aD_e·u(1 − u)`` with ``u = e^{−a(z − z_e)}``. The minimum sits at
  ``z_e + ln2/a`` and is worth ``−a·D_e/2``;
* laterally the bond fades as a Gaussian of width ``sigma_lat`` — 0.15 nm, representing the
  lateral range of a single bond.

**z is the closest approach**, i.e. the lower turning point of the oscillation. The simulator's
``gap`` is the mean position, so ``z_close = gap − A``. Δf follows from the exact large-amplitude
integral (Giessibl 2001), evaluated with Gauss–Chebyshev quadrature because the weight
``1/√(1−u²)`` is exactly the one that rule integrates:

    Δf(z) = −(f0 / (π k A)) ∫₋₁¹ F(z + A(1 + u)) · u / √(1 − u²) du

Small amplitudes recover ``Δf ≈ −(f0/2k)·F′``; the code switches to that branch below 0.1 pm
rather than dividing by a vanishing A.

The **binding energy** the benchmark asks for is defined the way an experimenter defines it:
the atom curve minus the clean-surface curve at the same tip height. The van der Waals term is
identical on both and cancels exactly, so with no short-range background ``E_b = D_e`` and
``F_min = −a·D_e/2`` in closed form — which is what the test pins.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

SMALL_AMPLITUDE_M = 1e-13           # below this Δf is the force-gradient limit
GAUSS_CHEBYSHEV_N = 64


@dataclass(frozen=True)
class ForceParams:
    D_e_j: float = 1.6022e-20           # 100 meV
    a_per_m: float = 1.2e10             # 12 / nm
    z_e_m: float = 0.30e-9
    hamaker_j: float = 2.0e-19          # 200 zJ, a metal–metal Hamaker constant
    z0_m: float = 0.3e-9
    bg_frac: float = 0.0                # short-range strength on the clean surface
    sigma_lat_m: float = 0.15e-9
    sigma_df_hz: float = 0.2
    bump_m: float = 70e-12              # apparent height of the adatom under the tip

    @classmethod
    def from_yaml(cls, cfg: dict[str, Any]) -> "ForceParams":
        d: dict[str, float] = {}
        if "D_e_mev" in cfg:
            d["D_e_j"] = float(cfg["D_e_mev"]) * 1e-3 * 1.602176634e-19
        if "a_per_nm" in cfg:
            d["a_per_m"] = float(cfg["a_per_nm"]) * 1e9
        if "z_e_nm" in cfg:
            d["z_e_m"] = float(cfg["z_e_nm"]) * 1e-9
        if "hamaker_zj" in cfg:
            d["hamaker_j"] = float(cfg["hamaker_zj"]) * 1e-21
        if "z0_nm" in cfg:
            d["z0_m"] = float(cfg["z0_nm"]) * 1e-9
        for k in ("bg_frac", "sigma_df_hz"):
            if k in cfg:
                d[k] = float(cfg[k])
        if "sigma_lat_nm" in cfg:
            d["sigma_lat_m"] = float(cfg["sigma_lat_nm"]) * 1e-9
        if "bump_pm" in cfg:
            d["bump_m"] = float(cfg["bump_pm"]) * 1e-12
        return cls(**d)

    @property
    def D_e_mev(self) -> float:
        return self.D_e_j / 1.602176634e-19 * 1e3


def morse_force(z, D_j: float, a_per_m: float, z_e_m: float):
    """F = −dU/dz of a Morse well (negative = attractive)."""
    z = np.asarray(z, float)
    u = np.exp(-a_per_m * (z - z_e_m))
    return -2.0 * a_per_m * D_j * u * (1.0 - u)


def morse_energy(z, D_j: float, a_per_m: float, z_e_m: float):
    z = np.asarray(z, float)
    u = np.exp(-a_per_m * (z - z_e_m))
    return D_j * ((1.0 - u) ** 2 - 1.0)


def vdw_force(d, hamaker_j: float, radius_m: float, z0_m: float):
    d = np.asarray(d, float)
    return -hamaker_j * radius_m / (6.0 * np.maximum(d + z0_m, 1e-12) ** 2)


class ForceLaw:
    """The total force over one lateral position: vdW plus the locally weighted bond."""

    def __init__(self, params: ForceParams, radius_m: float, w_lat: float = 1.0,
                 bump_m: float | None = None):
        self.p = params
        self.radius_m = float(radius_m)
        self.w_lat = float(np.clip(w_lat, 0.0, 1.0))
        self.bump_m = float(params.bump_m if bump_m is None else bump_m) * self.w_lat

    def F(self, z):
        p = self.p
        z = np.asarray(z, float)
        f = vdw_force(z + self.bump_m, p.hamaker_j, self.radius_m, p.z0_m)
        if self.w_lat > 0:
            f = f + self.w_lat * morse_force(z, p.D_e_j, p.a_per_m, p.z_e_m)
        if p.bg_frac > 0 and self.w_lat < 1.0:
            f = f + (1.0 - self.w_lat) * morse_force(z + self.bump_m, p.bg_frac * p.D_e_j,
                                                     p.a_per_m, p.z_e_m)
        return f

    def U(self, z):
        p = self.p
        z = np.asarray(z, float)
        u = -p.hamaker_j * self.radius_m / (6.0 * np.maximum(z + self.bump_m + p.z0_m, 1e-12))
        if self.w_lat > 0:
            u = u + self.w_lat * morse_energy(z, p.D_e_j, p.a_per_m, p.z_e_m)
        return u


def delta_f(force: Callable[[np.ndarray], np.ndarray], z, amplitude_m: float, f0_hz: float,
            k_n_per_m: float, n: int = GAUSS_CHEBYSHEV_N):
    """Frequency shift (Hz) at closest-approach distance ``z`` for a given force law."""
    z = np.atleast_1d(np.asarray(z, float))
    if amplitude_m < SMALL_AMPLITUDE_M:
        h = 0.5e-12
        grad = (force(z + amplitude_m + h) - force(z + amplitude_m - h)) / (2 * h)
        out = -(f0_hz / (2.0 * k_n_per_m)) * grad
        return out
    j = np.arange(1, n + 1)
    u = np.cos((2 * j - 1) * np.pi / (2 * n))
    zz = z[:, None] + amplitude_m * (1.0 + u)[None, :]
    fm = force(zz)
    return -(f0_hz / (k_n_per_m * amplitude_m * n)) * (fm @ u)


def force_truth(params: ForceParams, *, radius_m: float, amplitude_m: float, f0_hz: float,
                k_n_per_m: float, bump_m: float | None = None) -> dict:
    """Reference force quantities: F_min, its position relative to the Δf minimum, and E_b.

    The background-subtracted (short-range) curve is the atom's Morse minus the clean
    surface's, at the same tip height; the van der Waals term is common and cancels."""
    p = params
    bump = float(p.bump_m if bump_m is None else bump_m)
    z = np.arange(max(p.z_e_m - 0.15e-9, 0.02e-9), p.z_e_m + 3e-9, 0.5e-12)
    on_atom = ForceLaw(p, radius_m, w_lat=1.0, bump_m=bump)
    df = delta_f(on_atom.F, z, amplitude_m, f0_hz, k_n_per_m)
    i_df = int(np.argmin(df))
    f_short = morse_force(z, p.D_e_j, p.a_per_m, p.z_e_m)
    u_short = morse_energy(z, p.D_e_j, p.a_per_m, p.z_e_m)
    if p.bg_frac > 0:
        f_short = f_short - morse_force(z + bump, p.bg_frac * p.D_e_j, p.a_per_m, p.z_e_m)
        u_short = u_short - morse_energy(z + bump, p.bg_frac * p.D_e_j, p.a_per_m, p.z_e_m)
    i_f = int(np.argmin(f_short))
    f_vdw_05 = float(vdw_force(0.5e-9, p.hamaker_j, radius_m, p.z0_m))
    # the Morse tail is a single exponential, so 1/a is the decay length an experimenter reads
    # from the inverted short-range force. It spans 50–125 pm over the drawn range, while
    # z(F_min) − z(Δf_min) does not.
    return {
        "D_e_mev": p.D_e_mev, "a_per_nm": p.a_per_m * 1e-9, "z_e_nm": p.z_e_m * 1e9,
        "hamaker_zj": p.hamaker_j * 1e21, "z0_nm": p.z0_m * 1e9, "bg_frac": p.bg_frac,
        "sigma_df_hz": p.sigma_df_hz,
        "f_min_pn": float(f_short[i_f]) * 1e12,
        "z_fmin_nm": float(z[i_f]) * 1e9,
        "z_dfmin_nm": float(z[i_df]) * 1e9,
        "z_fmin_offset_pm": float(z[i_f] - z[i_df]) * 1e12,
        "f_decay_pm": 1e12 / p.a_per_m,
        "e_bind_mev": float(-np.min(u_short)) / 1.602176634e-19 * 1e3,
        "df_min_hz": float(df[i_df]),
        "f_vdw_at_0p5nm_pn": f_vdw_05 * 1e12,
    }


def averaged_current_factor(kappa_m: float, amplitude_m: float) -> float:
    """⟨I⟩ / I(z_mean) for a tip oscillating with amplitude A: the modified Bessel I₀(2κA)."""
    if amplitude_m <= 0:
        return 1.0
    from scipy.special import i0
    return float(i0(2.0 * kappa_m * amplitude_m))
