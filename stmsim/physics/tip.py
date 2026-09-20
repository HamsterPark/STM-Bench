"""Tip state and transitions used by the simulator.

State (docs/DESIGN.md §4.2):

* ``apexes`` — list of ``(dx, dy, dz, weight)``; one entry is a single tip, two or more of
  comparable weight within ~6 nm is a multi-tip. ``dz`` is quantised to multiples of the
  material step so a double tip on a stepped surface produces integer-step ghost terraces
  (the stepped-surface ghost terraces used by the double-tip analysis);
* ``radius_m`` — bluntness; images are blurred with ``sigma_eff = sqrt(R / 2κ)``;
* ``axis_ratio``, ``angle`` — anisotropy;
* ``phi_ev`` — the tip's contribution to the apparent barrier (contaminated tips lower it);
* ``lambda_per_s`` — spontaneous change hazard; each change perturbs the apexes (a row
  DC jump ≥ tens of pm, what ``tip_change`` v2 looks for) — metastable after a pulse
  until a shallow poke stabilises it;
* ``flicker_dz_m`` / ``flicker_rate_hz`` — a *crashed* tip's loosely bound apex cluster
  hops between configurations faster than a scan line: the secondary apexes' ``dz`` is
  re-drawn for every pass (:meth:`pass_apexes`) and the apex length hops mid-line at
  ``flicker_rate_hz`` (:meth:`flicker_series`), forward and backward separately, so trace
  and retrace image different tips and the trace/retrace correlation falls to the
  low values expected for an unstable verification frame (0.43 in the 2026-08-28
  verification reference), instead of the high correlation of a static blunt multi-apex tip
  (0.962 median in the same calibration). Cleared by an operation that reforms the apex (a
  shallow poke, a reshaping pulse);
* ``ldos`` — tip DOS (flat = spectroscopically clean);
* ``material``/``form`` — W / PtIr / qPlus-W; qPlus rings up when poked at high bias.

Transitions ``pulse`` / ``poke`` / ``crash`` draw from the calibrated distributions described
in ``docs/DESIGN.md`` §4.2 and the calibration table §4.5.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .junction import FLAT, LDOSTemplate, kappa_per_m


@dataclass
class Apex:
    dx: float
    dy: float
    dz: float
    w: float

    def as_tuple(self):
        return (self.dx, self.dy, self.dz, self.w)


@dataclass
class TipEvent:
    sim_s: float
    kind: str
    detail: dict = field(default_factory=dict)
    state: dict | None = None          # the tip right after the event (:meth:`Tip.snapshot`)


@dataclass
class Tip:
    material: str = "W"
    form: str = "qplus"               # etched | cut | qplus
    apexes: list[Apex] = field(default_factory=lambda: [Apex(0.0, 0.0, 0.0, 1.0)])
    radius_m: float = 2e-9
    apex_sigma_m: float = 0.08e-9     # lateral smearing of the terminating apex atom (atomic contrast)
    apex_radius_ref_m: float = -1.0   # radius at which apex_sigma_m was last (re)sampled
    axis_ratio: float = 1.0
    angle: float = 0.0
    phi_ev: float = 4.2               # tip-side barrier; combined with the surface via mean
    lambda_per_s: float = 1e-4        # spontaneous change hazard (1/s)
    ldos: LDOSTemplate = field(default_factory=lambda: LDOSTemplate("flat"))
    qplus_q: float = 25000.0
    qplus_f0_hz: float = 25296.2      # calibrated sensor resonance; the PLL starts centred here
    qplus_k_n_per_m: float = 1800.0   # nominal qPlus stiffness (not in any controller header)
    carried: str | None = None        # species of an atom picked up off the surface
    _carry_backup: dict | None = field(default=None, repr=False)
    dead: bool = False
    metastable: bool = False          # after a pulse, until a shallow poke
    flicker_dz_m: float = 0.0         # rms of the crashed cluster's configuration hops (m); 0 = static apex
    flicker_rate_hz: float = 0.0      # hop rate within a scan line (Poisson); 0 = one configuration per pass
    contact_depth_m: float = 0.25e-9  # poke depth at which the apex touches (critical depth)
    length_m: float = 0.0             # cumulative apex length change (the persistent part of Z jumps)
    last_persistent_dz_m: float = 0.0
    persist_frac: float = 0.12        # fraction of a pulse's Z jump that stays (rest relaxes in ~1 s)
    events: list[TipEvent] = field(default_factory=list)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    # ── derived ──
    @property
    def is_qplus(self) -> bool:
        return self.form == "qplus"

    @property
    def n_apex(self) -> int:
        return len(self.apexes)

    def is_multi(self, step_m: float = 0.2354e-9) -> bool:
        """≥2 apexes of comparable weight (≥25 % of the strongest) within 6 nm."""
        if len(self.apexes) < 2:
            return False
        wmax = max(a.w for a in self.apexes)
        strong = [a for a in self.apexes if a.w >= 0.25 * wmax]
        if len(strong) < 2:
            return False
        a0 = max(strong, key=lambda a: a.w)
        return any(math.hypot(a.dx - a0.dx, a.dy - a0.dy) <= 6e-9 for a in strong if a is not a0)

    def sigma_eff_m(self, phi_ev_junction: float) -> float:
        k = kappa_per_m(phi_ev_junction)
        return math.sqrt(max(self.radius_m, 0.2e-9) / (2.0 * k))

    # ── two sharpness scales ──────────────────────────────────────────────────
    # ``radius_m`` is the mesoscopic radius: it broadens steps and features through
    # ``sigma_eff_m`` and drives MAST's trace/retrace similarity. ``apex_sigma_m`` is the
    # lateral smearing of the terminating apex atom's orbital: it alone decides whether
    # the atomic lattice is transferred into the image. They are correlated (a blunt
    # tip rarely ends in one clean atom) but not identical — the correlation is the
    # ASSUMED map below, and ``derive_thresholds`` reports the sharpness threshold in
    # apex_sigma, not in radius.
    APEX_SIGMA_FLOOR_M = 0.05e-9
    APEX_SIGMA_CEIL_M = 0.60e-9

    def apex_sigma_from_radius(self, rng) -> float:
        """ASSUMED map σ_a = (0.06 + 0.04·R[nm]) nm × lognormal(0, 0.25), clipped."""
        r_nm = self.radius_m * 1e9
        mean = (0.06 + 0.04 * r_nm) * 1e-9
        val = mean * math.exp(rng.normal(0.0, 0.25))
        return float(min(max(val, self.APEX_SIGMA_FLOOR_M), self.APEX_SIGMA_CEIL_M))

    def _refresh_apex(self, rng) -> None:
        """Resample the apex smearing after an operation that changed the radius."""
        if self.radius_m != self.apex_radius_ref_m:
            self.apex_sigma_m = self.apex_sigma_from_radius(rng)
            self.apex_radius_ref_m = self.radius_m

    def atomic_transfer(self, period_m: float) -> float:
        """Fraction of the bare lattice corrugation that reaches the image:
        a Gaussian apex of width σ_a convolved with a lattice of first-order period d."""
        if period_m <= 0:
            return 0.0
        return float(math.exp(-2.0 * math.pi ** 2 * (self.apex_sigma_m / period_m) ** 2))

    def pass_apexes(self, rng=None) -> list[Apex]:
        """The apex configuration one scan pass sees.

        A static tip returns ``self.apexes`` itself. A flickering tip (``flicker_dz_m > 0``,
        set by :meth:`crash`) returns a copy in which every *secondary* apex's ``dz`` is
        re-drawn around its base value — the cluster on a crashed apex hops between
        configurations faster than a line takes — so the forward and backward passes of one
        row (each calls this once) image different apex arrangements. The base ``apexes``
        are never mutated here; spontaneous changes (:meth:`spontaneous_change`) are the
        persistent part."""
        if self.flicker_dz_m <= 0 or len(self.apexes) < 2:
            return self.apexes
        r = rng if rng is not None else self.rng
        out = [Apex(self.apexes[0].dx, self.apexes[0].dy, self.apexes[0].dz, self.apexes[0].w)]
        for a in self.apexes[1:]:
            out.append(Apex(a.dx, a.dy, a.dz + float(r.normal(0.0, self.flicker_dz_m)), a.w))
        return out

    def flicker_series(self, n: int, dt_px_s: float, rng=None) -> np.ndarray | None:
        """Tip-length offset (m) at each pixel of one scan pass, or ``None`` for a static tip.

        The crashed cluster hops between configurations *within* a line at
        ``flicker_rate_hz`` (Poisson); every hop is a new apex length drawn from
        N(0, ``flicker_dz_m``) — the mid-line Z segments (telegraph steps of tens to hundreds
        of pm) a real unstable tip writes into both passes independently. With
        ``flicker_rate_hz == 0`` the pass gets one constant offset (the row-DC part, which
        MAST's row-median detrend removes; the per-pass apex re-draw in :meth:`pass_apexes`
        then carries the disagreement)."""
        if self.flicker_dz_m <= 0 or n <= 0:
            return None
        r = rng if rng is not None else self.rng
        out = np.empty(n)
        level = float(r.normal(0.0, self.flicker_dz_m))
        if self.flicker_rate_hz <= 0:
            out.fill(level)
            return out
        p_hop = 1.0 - math.exp(-self.flicker_rate_hz * dt_px_s)
        hops = np.flatnonzero(r.random(n) < p_hop)
        start = 0
        for h in hops:
            out[start:h] = level
            level = float(r.normal(0.0, self.flicker_dz_m))
            start = int(h)
        out[start:] = level
        return out

    def _electronic(self, surface, x: np.ndarray, y: np.ndarray, kappa_m: float,
                    bias_v: float) -> np.ndarray | None:
        """Standing-wave apparent height, when the surface models a surface state.

        It rides in the *atomic* channel: it is an LDOS modulation right under the apex atom,
        so the apex smearing filters it, while the mesoscopic radius kernel (which broadens
        steps) must not. ``getattr`` because tests drive ``height_parts`` with duck-typed
        surfaces that only implement the two height layers."""
        fn = getattr(surface, "electronic_height", None)
        if fn is None or not bias_v:
            return None
        out = fn(x, y, bias_v, kappa_m, self.apex_sigma_m)
        return out if out is not None and np.any(out) else None

    def height_parts(self, surface, x: np.ndarray, y: np.ndarray, kappa_m: float,
                     apexes: list[Apex] | None = None,
                     bias_v: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """(smooth, atomic): the soft-max over apexes of the smooth surface, and the
        apex-transferred atomic lattice (weighted over apexes, so a double tip images a
        doubled lattice of lower contrast). ``apexes`` overrides the tip's own list (the
        renderer passes :meth:`pass_apexes` so a flickering tip differs pass to pass)."""
        apexes = self.apexes if apexes is None else apexes
        transfer = self.atomic_transfer(surface.material.first_order_period_m)
        if len(apexes) == 1 and apexes[0].dx == 0 and apexes[0].dy == 0:
            smooth = surface.height_smooth(x, y) + apexes[0].dz + self.length_m
            atomic = surface.atomic_height(x, y) * transfer if transfer > 1e-4 else np.zeros_like(smooth)
            elec = self._electronic(surface, x, y, kappa_m, bias_v)
            if elec is not None:
                atomic = atomic + elec
            return smooth, atomic
        hs = []
        for a in apexes:
            hs.append(surface.height_smooth(x + a.dx, y + a.dy) + a.dz)
        hs = np.stack(hs)
        hmax = hs.max(axis=0)
        wsum = sum(a.w for a in apexes) or 1.0
        ws = np.array([a.w for a in apexes])[:, None, None] if hs.ndim == 3 else np.array([a.w for a in apexes])[:, None]
        acc = np.log(np.sum(ws * np.exp(2 * kappa_m * (hs - hmax)), axis=0)) / (2 * kappa_m)
        smooth = hmax + acc + self.length_m
        atomic = np.zeros_like(smooth)
        if transfer > 1e-4:
            for a in apexes:
                atomic += (a.w / wsum) * surface.atomic_height(x + a.dx, y + a.dy)
            atomic *= transfer
        if bias_v:
            for a in apexes:
                elec = self._electronic(surface, x + a.dx, y + a.dy, kappa_m, bias_v)
                if elec is not None:
                    atomic = atomic + (a.w / wsum) * elec
        return smooth, atomic

    def effective_height(self, surface, x: np.ndarray, y: np.ndarray, kappa_m: float,
                         bias_v: float = 0.0) -> np.ndarray:
        """Soft-max over apexes of the smooth surface + apex-transferred lattice."""
        smooth, atomic = self.height_parts(surface, x, y, kappa_m, bias_v=bias_v)
        return smooth + atomic

    # ── how the tip arrived ──
    #: possible tip states at the start of a session; the initial state is not assumed
    #: to be known in advance and is characterized during operation
    CONDITIONS = ("good", "blunt", "double", "unstable", "dirty")

    def apply_condition(self, kind: str, scale: float = 0.5, angle_deg: float = 0.0) -> None:
        """Put the tip in the state it arrived in. ``scale`` (0…1) is how bad within the kind;
        ``angle_deg`` orients a second apex. Deterministic (no RNG draw), so the rest of the
        tip's history is the one the seed gives.

        * ``blunt`` — a 3–8 nm radius, and the apex smearing that goes with it: blurred
          images, no atomic lattice, a weak grip on adatoms;
        * ``double`` — a second apex 0.8–4 nm off, 20–80 pm shorter: every feature doubled,
          and a second place that can grab an adatom;
        * ``unstable`` — metastable with a 3e-3–3e-2 /s change hazard: row jumps, spectra
          that jump, a tip that changes under a manipulation current (a shallow poke
          stabilises it, a pulse alone does not);
        * ``dirty`` — something adsorbed on the apex: a peak in the tip's density of states
          inside the usual sweep window and a lowered barrier, so every spectrum carries a
          feature that is not the sample's."""
        if kind not in self.CONDITIONS:
            raise ValueError(f"unknown tip condition {kind!r} (known: {self.CONDITIONS})")
        s = min(max(float(scale), 0.0), 1.0)
        if kind == "blunt":
            self.radius_m = (3.0 + 5.0 * s) * 1e-9
            mean = (0.06 + 0.04 * self.radius_m * 1e9) * 1e-9          # the radius map's centre
            self.apex_sigma_m = float(min(max(mean, self.APEX_SIGMA_FLOOR_M), self.APEX_SIGMA_CEIL_M))
            self.apex_radius_ref_m = self.radius_m
        elif kind == "double":
            d = (0.8 + 3.2 * s) * 1e-9
            th = math.radians(float(angle_deg))
            self.apexes = [Apex(0.0, 0.0, 0.0, 1.0),
                           Apex(d * math.cos(th), d * math.sin(th), -(0.02 + 0.06 * s) * 1e-9, 0.9 - 0.3 * s)]
        elif kind == "unstable":
            self.metastable = True
            self.lambda_per_s = max(self.lambda_per_s, 3e-3 * 10.0 ** s)
        elif kind == "dirty":
            self.ldos.extra_peaks.append((-0.45 + 0.7 * s, 0.5 + 0.3 * s, 0.05))
            self.ldos.name = "featured"
            self.phi_ev = 3.2 + 0.6 * (1.0 - s)

    # ── vertical manipulation ──
    def pick_up(self, sim_s: float, species: str) -> None:
        """An atom transfers to the apex: sharper, lower barrier, and a tip resonance that
        will show up in every spectrum taken afterwards."""
        if self.carried is not None:
            return
        self._carry_backup = {"apex_sigma_m": self.apex_sigma_m, "phi_ev": self.phi_ev,
                              "lambda_per_s": self.lambda_per_s,
                              "ldos_peaks": list(self.ldos.extra_peaks), "ldos_name": self.ldos.name}
        self.carried = str(species)
        self.apex_sigma_m = max(0.03e-9, 0.7 * self.apex_sigma_m)
        self.phi_ev = max(1.0, self.phi_ev - 0.4)
        self.lambda_per_s = min(0.05, self.lambda_per_s * 2.0)
        self.ldos = LDOSTemplate(name="featured", onset_ev=self.ldos.onset_ev,
                                 step_height=self.ldos.step_height,
                                 broadening_ev=self.ldos.broadening_ev,
                                 gap_ev=self.ldos.gap_ev,
                                 extra_peaks=list(self.ldos.extra_peaks) + [(-0.15, 0.4, 0.06)],
                                 table_e_ev=list(self.ldos.table_e_ev),
                                 table_rho=list(self.ldos.table_rho), source=self.ldos.source)
        self._log(sim_s, "pick_up", species=species)

    def drop(self, sim_s: float) -> None:
        b = self._carry_backup
        if b is None:
            self.carried = None
            return
        self.apex_sigma_m = b["apex_sigma_m"]
        self.phi_ev = b["phi_ev"]
        self.lambda_per_s = b["lambda_per_s"]
        self.ldos = LDOSTemplate(name=b["ldos_name"], onset_ev=self.ldos.onset_ev,
                                 step_height=self.ldos.step_height,
                                 broadening_ev=self.ldos.broadening_ev,
                                 gap_ev=self.ldos.gap_ev, extra_peaks=list(b["ldos_peaks"]),
                                 table_e_ev=list(self.ldos.table_e_ev),
                                 table_rho=list(self.ldos.table_rho), source=self.ldos.source)
        self._carry_backup = None
        species, self.carried = self.carried, None
        self._log(sim_s, "drop", species=species)

    # ── bookkeeping ──
    def _log(self, sim_s: float, kind: str, **detail) -> None:
        # every caller logs after the change, so the snapshot is the tip the event left
        # behind — what a replay shows; the detail alone does not say it (a spontaneous
        # change records only its Z jump)
        self.events.append(TipEvent(sim_s, kind, detail, self.snapshot()))

    def snapshot(self) -> dict:
        return {
            "material": self.material, "form": self.form,
            "apexes": [a.as_tuple() for a in self.apexes], "n_apex": self.n_apex,
            "multi": self.is_multi(), "radius_nm": self.radius_m * 1e9,
            "apex_sigma_nm": self.apex_sigma_m * 1e9,
            "axis_ratio": self.axis_ratio, "phi_ev": self.phi_ev,
            "lambda_per_s": self.lambda_per_s, "metastable": self.metastable,
            "flicker_dz_pm": self.flicker_dz_m * 1e12, "flicker_rate_hz": self.flicker_rate_hz,
            "ldos": self.ldos.name, "dead": self.dead,
            "contact_depth_pm": self.contact_depth_m * 1e12, "length_nm": self.length_m * 1e9,
            "carried": self.carried,
            "qplus_f0_hz": self.qplus_f0_hz, "qplus_k_n_per_m": self.qplus_k_n_per_m,
            "qplus_q": self.qplus_q,
        }

    # ── spontaneous evolution ──
    def maybe_change(self, dt_sim: float, sim_s: float, *, current_a: float = 0.0,
                     bias_v: float = 0.0) -> bool:
        """Poisson hazard over ``dt_sim``; boosted by current dose and high bias."""
        boost = 1.0 + abs(current_a) / 1e-9 + (abs(bias_v) / 2.0) ** 2
        p = 1.0 - math.exp(-self.lambda_per_s * boost * dt_sim)
        if self.rng.random() < p:
            self.spontaneous_change(sim_s)
            return True
        return False

    def spontaneous_change(self, sim_s: float) -> None:
        r = self.rng
        a = self.apexes[0]
        dz = float(r.choice([-1, 1]) * r.uniform(20e-12, 150e-12))
        a.dz += dz
        a.dx += float(r.normal(0, 0.05e-9))
        a.dy += float(r.normal(0, 0.05e-9))
        if r.random() < 0.15 and self.n_apex < 4:
            self.apexes.append(Apex(float(r.normal(0, 1.5e-9)), float(r.normal(0, 1.5e-9)),
                                    -abs(dz), float(r.uniform(0.3, 0.8))))
        self._log(sim_s, "spontaneous_change", dz_pm=dz * 1e12, n_apex=self.n_apex)

    # ── transitions ──
    def pulse(self, sim_s: float, v: float, width_s: float, envelope_v: float = 10.0) -> dict:
        """Bias pulse. Returns {'dz_m': Z jump, 'outcome': ...}."""
        r = self.rng
        av = abs(v)
        thresh = 2.5 if self.material == "W" else 3.5
        if av < thresh:
            out = {"outcome": "no_effect", "dz_m": float(r.normal(0, 0.3e-9))}
        elif av <= envelope_v * 1.02:      # float32 round-trip of "10 V" must not read as over-envelope
            # reshape: apexes resampled, mostly sharper, sometimes worse
            p_good = 0.55 if not self.is_qplus else 0.5
            u = r.random()
            if u < p_good:
                self.apexes = [Apex(0.0, 0.0, 0.0, 1.0)]
                self.radius_m = float(max(0.3e-9, self.radius_m * r.uniform(0.3, 0.8)))
                self.axis_ratio = float(min(1.0, self.axis_ratio + r.uniform(0.1, 0.5)))
                self.phi_ev = float(max(self.phi_ev, r.uniform(3.6, 4.4)))
                self.ldos = LDOSTemplate("flat")
                out = {"outcome": "reshaped_better"}
            elif u < 0.85:
                n = int(r.integers(2, 4))
                self.apexes = [Apex(0.0, 0.0, 0.0, 1.0)] + [
                    Apex(float(r.normal(0, 2e-9)), float(r.normal(0, 2e-9)),
                         float(-r.integers(0, 2)) * 0.2354e-9, float(r.uniform(0.3, 0.9)))
                    for _ in range(n - 1)]
                self.radius_m = float(self.radius_m * r.uniform(0.8, 1.5))
                out = {"outcome": "reshaped_multi"}
            else:
                self.apexes = [Apex(0.0, 0.0, 0.0, 1.0)]
                self.radius_m = float(self.radius_m * r.uniform(1.5, 3.0))
                out = {"outcome": "reshaped_blunt"}
            self.metastable = True
            self.lambda_per_s = max(self.lambda_per_s, 3e-3)
            self.flicker_dz_m = 0.0            # the apex was re-formed: whatever flickered is gone
            self.flicker_rate_hz = 0.0
            out["dz_m"] = float(r.choice([-1, 1], p=[0.3, 0.7]) * r.uniform(20e-9, 50e-9))
        else:
            self.radius_m = float(self.radius_m * r.uniform(2.0, 5.0))
            self.apexes = [Apex(0.0, 0.0, 0.0, 1.0), Apex(float(r.normal(0, 3e-9)), float(r.normal(0, 3e-9)), -0.2354e-9, 0.7)]
            self.flicker_dz_m = float(r.uniform(*self.CRASH_FLICKER_DZ_M))   # over-envelope: a crash-like apex
            self.flicker_rate_hz = float(r.uniform(*self.CRASH_FLICKER_RATE_HZ))
            self.metastable = True
            out = {"outcome": "destroyed", "dz_m": float(r.uniform(30e-9, 80e-9))}
            if av > 1.6 * envelope_v:
                self.dead = True
        self.last_persistent_dz_m = self.persist_frac * float(out.get("dz_m", 0.0))
        self.length_m += self.last_persistent_dz_m
        self._refresh_apex(r)
        self._log(sim_s, "pulse", v=v, width_s=width_s, **out)
        return out

    def poke(self, sim_s: float, depth_m: float, bias_v: float, step_m: float = 0.2354e-9) -> dict:
        """Poke (tip shaper) of ``depth`` (positive = into the surface).

        Outcome kinds are ``no_change`` / ``cluster`` / ``pit``.
        A qPlus poked at |bias| > ~50 mV rings up: Z bounces, clusters spray, tip damaged.
        """
        r = self.rng
        d = abs(depth_m)
        out: dict = {"depth_m": d}
        if self.is_qplus and abs(bias_v) > 0.05 and r.random() < 0.8:
            self.radius_m = float(self.radius_m * r.uniform(1.5, 3.0))
            self.apexes = [Apex(0.0, 0.0, 0.0, 1.0)] + [
                Apex(float(r.normal(0, 2.5e-9)), float(r.normal(0, 2.5e-9)), -step_m, float(r.uniform(0.4, 0.9)))
                for _ in range(int(r.integers(1, 3)))]
            out.update(outcome="ring_up", dz_m=float(r.uniform(5e-9, 30e-9)), n_clusters=int(r.integers(6, 15)))
            self._refresh_apex(r)
            self._log(sim_s, "poke", bias_v=bias_v, **out)
            return out
        if d < self.contact_depth_m * r.uniform(0.7, 1.0):
            out.update(outcome="no_change", dz_m=float(r.normal(0, 5e-12)))
        else:
            # material transfer: a cluster is left; deeper pokes → bigger, less round
            depth_nm = d * 1e9
            axis_ratio_median = float(np.interp(depth_nm, [0.2, 0.5, 1.0, 2.0], [0.74, 0.64, 0.44, 0.35]))
            ratio = float(np.clip(r.normal(axis_ratio_median, 0.12), 0.15, 1.0))
            dz = float(r.uniform(0.3e-9, 2.5e-9) * min(depth_nm / 0.5, 3.0))
            pit = r.random() < 0.15 * min(depth_nm, 2.0)
            out.update(outcome="pit" if pit else "cluster", dz_m=(-dz if pit else dz),
                       cluster_axis_ratio=ratio, cluster_height_m=float(r.uniform(0.2e-9, 0.8e-9) * min(depth_nm / 0.5, 2.5)),
                       cluster_sigma_m=float(r.uniform(0.6e-9, 1.5e-9) * (1 + depth_nm)))
            # tip side
            if depth_nm <= 0.7:
                # shallow: stabilises, often sharpens — and pins a flickering cluster
                self.metastable = False
                self.flicker_dz_m = 0.0
                self.flicker_rate_hz = 0.0
                if self.carried is None and self.ldos.name == "featured":
                    # the apex comes back coated with fresh substrate metal: whatever was
                    # adsorbed on it is buried (no draw, so the tip's random history is kept)
                    self.ldos = LDOSTemplate("flat")
                    self.phi_ev = max(self.phi_ev, 4.0)
                self.lambda_per_s = float(max(1e-5, self.lambda_per_s * r.uniform(0.2, 0.6)))
                if r.random() < 0.45:
                    self.radius_m = float(max(0.3e-9, self.radius_m * r.uniform(0.6, 0.95)))
                if r.random() < 0.2 and self.n_apex > 1:
                    self.apexes = [self.apexes[0]]
                if r.random() < 0.25:
                    self.axis_ratio = float(min(1.0, self.axis_ratio + 0.2))
            else:
                if r.random() < 0.35 * min(depth_nm / 2.0, 1.5):
                    self.radius_m = float(self.radius_m * r.uniform(1.3, 2.5))
                if r.random() < 0.25 * min(depth_nm / 2.0, 1.5) and self.n_apex < 4:
                    self.apexes.append(Apex(float(r.normal(0, 2e-9)), float(r.normal(0, 2e-9)), -step_m, float(r.uniform(0.3, 0.8))))
                self.metastable = False
            self.contact_depth_m = float(max(0.1e-9, self.contact_depth_m * r.uniform(0.9, 1.1)))
        self.length_m += 0.3 * float(out.get("dz_m", 0.0))
        self._log(sim_s, "poke", bias_v=bias_v, **out)
        return out

    # Calibrated 2026-08-28 against the trace/retrace correlation on a verification frame
    # (100 nm / 256 px / 0.586 s). Flicker rms 0.15–0.30 nm gave a median of 0.38,
    # 0.06–0.10 nm 0.77, and 0.10–0.18 nm 0.58; the range below targets a median near
    # 0.5 and keeps the verification gate below 0.80 (tests/test_fidelity.py pins it).
    CRASH_LAMBDA_PER_S = 0.05          # a crashed apex keeps changing: ~15 % of verify rows see a change
    CRASH_FLICKER_DZ_M = (0.12e-9, 0.20e-9)   # rms of the cluster's configuration hops
    CRASH_FLICKER_RATE_HZ = (1.0, 4.0)        # hops per second within a line (0.6–2.3 per 0.586 s line)

    def crash(self, sim_s: float, severity: float = 1.0, step_m: float = 0.2354e-9) -> dict:
        """Tip into the sample: blunt radius, 1–3 extra apexes 2–6 nm off (the picked-up
        cluster), a lowered barrier — and an *unstable* apex: metastable, λ ≥ 0.05/s and
        a per-pass dz flicker of the cluster apexes (:meth:`pass_apexes`), which is what
        separates a crashed tip's verify frame from a merely blunt one under MAST's
        trace/retrace detector. The extra apexes' dz is quantised to whole steps."""
        r = self.rng
        self.radius_m = float(self.radius_m * r.uniform(2.0, 6.0) * severity)
        self._refresh_apex(r)
        k = int(r.integers(1, 4))
        extra = []
        for _ in range(k):
            rho = float(r.uniform(2e-9, 6e-9))
            ang = float(r.uniform(0.0, 2.0 * math.pi))
            extra.append(Apex(rho * math.cos(ang), rho * math.sin(ang),
                              -step_m * float(r.integers(0, 3)), float(r.uniform(0.5, 1.0))))
        self.apexes = [Apex(0.0, 0.0, 0.0, 1.0)] + extra
        self.metastable = True
        self.lambda_per_s = max(self.lambda_per_s, self.CRASH_LAMBDA_PER_S)
        self.flicker_dz_m = float(r.uniform(*self.CRASH_FLICKER_DZ_M) * min(max(severity, 1.0), 2.0))
        self.flicker_rate_hz = float(r.uniform(*self.CRASH_FLICKER_RATE_HZ))
        self.phi_ev = float(min(self.phi_ev, r.uniform(1.0, 3.0)))
        out = {"outcome": "crashed", "severity": severity, "n_apex": self.n_apex,
               "flicker_dz_pm": self.flicker_dz_m * 1e12, "flicker_rate_hz": self.flicker_rate_hz}
        self._log(sim_s, "crash", **out)
        return out

    def ghost_contrast(self, corrugation_m: float, kappa_m: float) -> float:
        """Visibility of the strongest secondary apex relative to the primary (0..1)."""
        if len(self.apexes) < 2:
            return 0.0
        a0 = max(self.apexes, key=lambda a: a.w)
        best = 0.0
        for a in self.apexes:
            if a is a0:
                continue
            best = max(best, (a.w / a0.w) * math.exp(2 * kappa_m * min(a.dz - a0.dz, 0.0)))
        return best
