"""The Kondo resonance of a magnetic adatom, as bias spectroscopy sees it (P6).

A single magnetic atom (Co) on a noble-metal surface is screened by the substrate's conduction
electrons below its Kondo temperature, and tunnelling into it shows a narrow resonance at the
Fermi level. Because the tip can tunnel into the screened d level directly or through the
substrate's conduction electrons, the two paths interfere and the resonance appears as a Fano
line shape (Madhavan et al., Science 280, 567 (1998); Li et al., PRL 80, 2893 (1998); for
Co/Cu(111) and Co/Cu(100), Knorr et al., PRL 88, 096804 (2002))::

    dI/dV ∝ (q + ε)² / (1 + ε²),        ε = (eV − ε_K) / Γ

with Γ the half width (Γ = k_B·T_K in the convention used here), q the asymmetry and ε_K the
resonance position.

**Local model.** At a lateral offset r from a magnetic atom the differential conductance is the
background's times ``1 + a(r)·[(q(r) + ε)²/(1 + ε²) − 1]``, which, with ``L = 1/(1 + ε²)`` and
``D = ε/(1 + ε²)``, is ``1 + a(q² − 1)·L + 2aq·D``. Modelling assumptions (not fits):

* ``a(r) = a0·exp(−r²/2σ_a²)`` — the fraction of the conductance through the resonance falls off
  within about a nanometre (Madhavan 1998: gone by ~1 nm);
* ``q(r) = q0·exp(−r²/2σ_q²)`` — the direct path into the d level is more localized than the
  path through the substrate, so the line turns into a dip off the atom (Plihal & Gadzuk,
  PRB 63, 085404 (2001); Újsághy et al., PRL 85, 2557 (2000)); a smooth Gaussian stands in for
  the Friedel-like oscillation those theories give farther out.

``Γ``, ``q0``, ``ε_K``, ``a0``, ``σ_a`` and ``σ_q`` are drawn per seed (``hidden.kondo``): the
published Co fits on noble metals span T_K ≈ 50–100 K (Γ ≈ 4–9 meV), q ≈ 0.2–1.1 and ε_K within
a few meV of E_F, and they depend on the substrate and the adsorption site, so a fixed answer
from the literature cannot pass.

Only the parameters and their truth live here so far; how a sweep sees them (thermal and
lock-in broadening, the apex blur, the drift along the sweep) comes with the ``sts_resolution``
physics switch (:mod:`stmsim.physics.switches`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Boltzmann constant (eV/K)
KB_EV_PER_K = 8.617333e-5


@dataclass(frozen=True)
class KondoParams:
    """One sample's Kondo resonance: every adatom of ``species`` carries it."""
    # defaults near Co/Cu(111) (Knorr 2002: T_K ≈ 54 K, q ≈ 0.18); a scenario draws its own
    gamma_ev: float = 0.0047          # Γ, the Fano half width (eV); 4.7 meV = k_B · 54 K
    q: float = 0.18                   # q0, the asymmetry on top of the atom
    e_k_ev: float = 0.002             # ε_K, the resonance position (eV from E_F)
    amplitude: float = 0.2            # a0, the fraction of the conductance through the resonance on top
    sigma_m: float = 0.35e-9          # σ_a, lateral extent of a(r)
    q_sigma_m: float = 0.55e-9        # σ_q, lateral extent of q(r)
    species: str = "Co"

    @classmethod
    def from_yaml(cls, cfg: dict[str, Any]) -> "KondoParams":
        """From a scenario's ``initial.kondo`` merged with its ``hidden.kondo`` draws (meV, nm)."""
        d: dict[str, Any] = {}
        if "species" in cfg:
            d["species"] = str(cfg["species"])
        for src, dst, scale in (("gamma_mev", "gamma_ev", 1e-3), ("q", "q", 1.0),
                                ("e_k_mev", "e_k_ev", 1e-3), ("amplitude", "amplitude", 1.0),
                                ("sigma_nm", "sigma_m", 1e-9), ("q_sigma_nm", "q_sigma_m", 1e-9)):
            if src in cfg and cfg[src] is not None:
                d[dst] = float(cfg[src]) * scale
        p = cls(**d)
        if p.gamma_ev <= 0 or p.sigma_m <= 0 or p.q_sigma_m <= 0 or not 0.0 <= p.amplitude < 1.0:
            raise ValueError(f"kondo: Γ, σ_a and σ_q must be positive and 0 ≤ a0 < 1, got {p}")
        return p

    def truth(self) -> dict[str, Any]:
        """What the judge compares with (meV, nm, K)."""
        return {"species": self.species, "gamma_mev": self.gamma_ev * 1e3, "q": self.q,
                "e_k_mev": self.e_k_ev * 1e3, "amplitude": self.amplitude,
                "sigma_nm": self.sigma_m * 1e9, "q_sigma_nm": self.q_sigma_m * 1e9,
                "t_k_k": self.gamma_ev / KB_EV_PER_K}
