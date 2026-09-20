"""Tunnelling junction: current as a function of gap, bias, barrier and densities of states.

Conventions (SI unless noted):

* ``kappa`` — inverse decay length, 1/m; ``kappa = 5.123 * sqrt(phi_eV)`` per nm
  (``mast.vision.spectroscopy.assess_iz`` uses the same constant);
* ``gap`` — tip-apex to surface distance, m;
* metallic junction: ``I = G_contact * V_eff * exp(-2 kappa gap)``, with ``G_contact``
  chosen so that 100 pA at 1 V sits near a 0.57 nm gap for phi = 4 eV;
* I(V) with structure: ``I(V) = G ∫₀^V rho_s(E) rho_t(E-V) T(E, V, gap) dE`` on a grid;
  ``rho_s`` is a material template (Shockley surface-state step for the noble (111)s),
  ``rho_t`` flat for a spectroscopically clean tip.
* preamp: gain range → full scale; readings clamp at ``full_scale * 1.0004`` (10.004 nA
  for the calibrated 10 nA range); de-saturation is exponential with ``desat_tau``.
"""
from __future__ import annotations

import json
import math
import re
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

KAPPA_PER_NM_PER_SQRT_EV = 5.123
G_CONTACT_A_PER_V = 10e-6   # ≈ 0.13 G0; sets the absolute gap scale


def kappa_per_m(phi_ev: float) -> float:
    return KAPPA_PER_NM_PER_SQRT_EV * math.sqrt(max(phi_ev, 1e-3)) * 1e9


def phi_from_kappa_per_m(kappa: float) -> float:
    return (kappa / 1e9 / KAPPA_PER_NM_PER_SQRT_EV) ** 2


@dataclass
class LDOSTemplate:
    """Sample density of states vs energy (eV, relative to E_F), normalised ~1.

    Two forms: the analytic one (flat + Shockley step at ``onset_ev``) and a tabulated one
    (``table_e_ev`` / ``table_rho`` from a calibrated dI/dV template — see
    :func:`load_ldos_template`). When the table is present it replaces the analytic shape
    (edge values are held outside its range); ``gap_ev`` and ``extra_peaks`` apply to both.
    """
    name: str = "flat"
    onset_ev: float | None = None      # Shockley surface-state onset (Au −0.49, Ag −0.065, Cu −0.44)
    step_height: float = 0.6           # relative DOS increase above onset
    broadening_ev: float = 0.02
    gap_ev: float = 0.0                # semiconducting gap half-width (0 = metal)
    extra_peaks: list[tuple[float, float, float]] = field(default_factory=list)  # (E, amp, width)
    table_e_ev: list[float] = field(default_factory=list)   # tabulated rho (calibrated template)
    table_rho: list[float] = field(default_factory=list)
    source: str = "analytic"           # "analytic" | "calibrated:<file>" | "analytic:fallback(<why>)"

    def rho(self, e: np.ndarray) -> np.ndarray:
        e = np.asarray(e, float)
        if self.table_e_ev:
            out = np.interp(e, self.table_e_ev, self.table_rho)
        else:
            out = np.ones_like(e)
            if self.onset_ev is not None:
                out += self.step_height / (1.0 + np.exp(-(e - self.onset_ev) / self.broadening_ev))
        if self.gap_ev > 0:
            out *= (np.abs(e) > self.gap_ev).astype(float) * 1.0 + 0.02
        for e0, amp, w in self.extra_peaks:
            out += amp * np.exp(-0.5 * ((e - e0) / w) ** 2)
        return out


FLAT = LDOSTemplate("flat")

# literature Shockley onsets (V) — the analytic fallback and the check the calibrated file ran
MATERIAL_ONSET_EV = {"Au": -0.49, "Ag": -0.065, "Cu": -0.44}
STS_TEMPLATES_FILE = "sts_templates.json"
_STS_CACHE: dict[tuple[str, float], dict] = {}


def _material_key(material: str) -> str | None:
    m = re.match(r"\s*(Au|Ag|Cu)\b", str(material or ""), re.IGNORECASE)
    return m.group(1).capitalize() if m else None


def _read_sts_file(path) -> dict | None:
    """Parsed ``sts_templates.json`` (cached on mtime); None if absent or unreadable."""
    try:
        p = Path(path)
        mtime = p.stat().st_mtime
    except OSError:
        return None
    key = (str(p), mtime)
    if key not in _STS_CACHE:
        try:
            _STS_CACHE[key] = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return _STS_CACHE[key]


def load_ldos_template(material: str, path=None, min_points: int = 20,
                       rho_floor: float = 0.05) -> LDOSTemplate:
    """LDOS template for ``material`` ("Au(111)", "Ag", …): calibrated table when available.

    Preference order, recorded in ``template.source`` so a scenario ledger can say which one
    ran:

    1. ``sts_templates.json`` (``path`` or ``stmsim.paths.calib_dir()``) has the material
       with ≥ ``min_points`` finite grid points **and** the table covers the literature
       onset ±0.1 V (a ±200 mV template cannot stand in for Au's −0.49 V step) →
       ``source="calibrated:<file>"`` (suffix ``;onset_ok=<bool>`` copies the file's own check);
    2. otherwise today's analytic step at :data:`MATERIAL_ONSET_EV` →
       ``source="analytic"`` (unknown material: flat) or ``"analytic:fallback(<why>)"``.

    Table values are clipped at ``rho_floor`` (a negative or zero DOS is a lock-in artefact).
    """
    key = _material_key(material)
    onset = MATERIAL_ONSET_EV.get(key) if key else None
    analytic = LDOSTemplate(str(material), onset_ev=onset)
    if key is None:
        return analytic
    if path is None:
        from ..paths import calib_dir
        path = calib_dir() / STS_TEMPLATES_FILE
    data = _read_sts_file(path)
    if data is None:
        analytic.source = "analytic:fallback(no_file)"
        return analytic
    entry = (data.get("materials") or {}).get(key)
    if not entry:
        analytic.source = f"analytic:fallback(no_{key})"
        return analytic
    grid = np.asarray(data.get("v_grid") or entry.get("v_grid") or [], float)
    vals = np.asarray([np.nan if v is None else v for v in entry.get("didv") or []], float)
    ok = np.isfinite(vals) & np.isfinite(grid) if grid.size == vals.size else np.zeros(0, bool)
    if ok.sum() < min_points:
        analytic.source = f"analytic:fallback(too_few_points:{int(ok.sum())})"
        return analytic
    e, r = grid[ok], np.clip(vals[ok], rho_floor, None)
    if onset is not None and not (e.min() <= onset - 0.1 and e.max() >= onset + 0.1):
        analytic.source = f"analytic:fallback(onset_not_covered:{e.min():+.2f}..{e.max():+.2f})"
        return analytic
    onset_ok = (entry.get("onset") or {}).get("ok")
    return LDOSTemplate(str(material), onset_ev=onset, table_e_ev=[float(x) for x in e],
                        table_rho=[float(x) for x in r],
                        source=f"calibrated:{Path(path).name};onset_ok={onset_ok}")


@dataclass
class Preamp:
    full_scale_a: float = 10e-9
    clamp_factor: float = 1.0004
    desat_tau_s: float = 0.45
    #: the panel index Current.GainsGet reports. The gain names are transimpedances
    #: 1E6…1E11 V/A and the DAC swings ±10 V, so index i means a full scale of
    #: 10 V / 10^(6+i) — index 3 is the 10 nA range. It has to agree with ``full_scale_a``
    #: or ``Current.GainSet`` moves the reported range without moving the physical range;
    #: a wider range must therefore be selected before a manipulation.
    gain_index: int = 3

    @staticmethod
    def index_for(full_scale_a: float) -> int:
        return max(0, min(5, int(round(math.log10(10.0 / float(full_scale_a)) - 6))))

    @staticmethod
    def full_scale_for(index: int) -> float:
        """The range of gain index ``index`` (0 = 1E6 V/A = 10 µA … 5 = 1E11 V/A = 100 pA)."""
        return 10.0 / 10 ** (6 + max(0, min(5, int(index))))

    @property
    def max_full_scale_a(self) -> float:
        """The widest switchable preamp range (index 0)."""
        return self.full_scale_for(0)

    @property
    def clamp_a(self) -> float:
        return self.full_scale_a * self.clamp_factor

    def clamp(self, i: float | np.ndarray):
        return np.clip(i, -self.clamp_a, self.clamp_a)


def current_metal(gap_m: float | np.ndarray, bias_v: float, phi_ev: float,
                  g_contact: float = G_CONTACT_A_PER_V) -> float | np.ndarray:
    k = kappa_per_m(phi_ev)
    return g_contact * bias_v * np.exp(-2.0 * k * np.asarray(gap_m, float))


def gap_for_current(i_set_a: float, bias_v: float, phi_ev: float,
                    g_contact: float = G_CONTACT_A_PER_V) -> float:
    """Gap at which the metallic junction carries ``i_set`` at ``bias``.

    Bias of exactly 0 V has no tunnelling solution; treat |V| < 1 mV as 1 mV so the loop
    still finds *some* gap (controller behaves similarly: at 0 V the tip crashes into the
    surface, which the caller models as a crash hazard, not here).
    """
    v = max(abs(bias_v), 1e-3)
    k = kappa_per_m(phi_ev)
    ratio = max(abs(i_set_a), 1e-15) / (g_contact * v)
    return max(-math.log(ratio) / (2.0 * k), 0.05e-9)


def iv_curve(vs: np.ndarray, gap_m: float, phi_ev: float, rho_s: LDOSTemplate = FLAT,
             rho_t: LDOSTemplate = FLAT, g_contact: float = G_CONTACT_A_PER_V,
             n_grid: int = 200) -> np.ndarray:
    """I(V) from a simplified DOS-product model inspired by Tersoff–Hamann."""
    vs = np.asarray(vs, float)
    k = kappa_per_m(phi_ev)
    out = np.empty_like(vs)
    t0 = math.exp(-2.0 * k * gap_m)
    for i, v in enumerate(vs):
        if abs(v) < 1e-6:
            out[i] = 0.0
            continue
        e = np.linspace(0.0, v, n_grid)
        # mild bias dependence of the barrier: T ∝ exp(-2κ gap sqrt(1 - (E - V/2)/phi))
        corr = np.sqrt(np.clip(1.0 - (e - v / 2.0) / max(phi_ev, 0.1) * 0.5, 0.2, 2.0))
        integrand = rho_s.rho(e) * rho_t.rho(e - v) * t0 ** (corr - 1.0)
        out[i] = g_contact * t0 * np.trapezoid(integrand, e)
    return out


def didv_curve(vs: np.ndarray, gap_m: float, phi_ev: float, rho_s: LDOSTemplate = FLAT,
               rho_t: LDOSTemplate = FLAT, mod_amp_v: float = 0.01) -> np.ndarray:
    """Lock-in style dI/dV: finite difference over ±mod_amp."""
    ip = iv_curve(np.asarray(vs) + mod_amp_v, gap_m, phi_ev, rho_s, rho_t)
    im = iv_curve(np.asarray(vs) - mod_amp_v, gap_m, phi_ev, rho_s, rho_t)
    return (ip - im) / (2 * mod_amp_v)


def apparent_barrier_from_iz(z_m: np.ndarray, i_a: np.ndarray, floor_a: float) -> tuple[float, float, int]:
    """(phi_eV, r², n_ok) — the fit MAST's ``assess_iz`` / ``MeasureBarrierHeight`` do."""
    z = np.asarray(z_m, float) * 1e9
    i = np.abs(np.asarray(i_a, float))
    ok = np.isfinite(i) & (i > floor_a)
    if ok.sum() < 3:
        return float("nan"), float("nan"), int(ok.sum())
    y = np.log(i[ok])
    A = np.vstack([z[ok], np.ones(ok.sum())]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1e-30
    kappa_nm = abs(coef[0]) / 2.0
    return (kappa_nm / KAPPA_PER_NM_PER_SQRT_EV) ** 2, 1 - ss_res / ss_tot, int(ok.sum())
