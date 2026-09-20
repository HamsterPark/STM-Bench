"""Scenario = initial world + faults + task text + budget + success criterion (YAML).

Example (``stmbench/trackB/scenarios/B5_repair_blunt.yaml``)::

    id: B5_repair_blunt
    family: B5
    variant: blunt
    rig: reference-stm
    material: Au(111)
    time_scale: 20
    initial:
      approached: true
      tip: {radius_nm: 8.0, lambda_per_s: 2.0e-3}
      surface: {contamination: 0.0, adsorbate_density_per_um2: 3.0}
    faults: []
    task: 针尖状态未知……
    budget: {sim_hours: 2.0, wire_cmds: 300000}
    success: {kind: tip_repaired}

``seed`` is supplied at run time (paired design across models), not in the file.

**Paper scenarios (v2).** A scenario that reproduces a published measurement carries three
more blocks::

    paper: {id: P1, title: "Barth et al. 1990 …", ref: "PRB 42, 9307"}
    hidden:
      herringbone: {period_nm: [6.0, 6.6], chevron_period_nm: [22, 34]}
    claims:
      - {id: stripe_period_nm, kind: scalar, unit: nm, tol: {abs: 0.4},
         truth: herringbone.period_nm, evidence: {kind: frame, min_fov_nm: 40}}
    success: {kind: claims_verified}

``hidden`` draws one value per seed from a **physically sensible** range, so a model that
recites the literature value instead of measuring gets no credit. Every key must appear in
:data:`HIDDEN_SCHEMA` (a typo would otherwise silently draw nothing). The draws use their own
RNG stream — ``default_rng([seed, HIDDEN_STREAM, crc32(scenario.id)])`` — so adding a hidden
block never moves the world / tip / noise / site streams, and a scenario with no ``hidden``
block builds a bit-identical world to the one it built before v2.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .faults import Fault, FaultScheduler
from .physics.rig import RigProfile
from .physics.tip import Apex, Tip
from .physics.world import World

# ── hidden parameters ───────────────────────────────────────────────────────
# The whitelist of what a scenario may randomise per seed. Nested dicts mirror the YAML;
# a leaf is the set of legal keys for that block. An unknown key raises rather than being
# silently ignored, because a silent miss looks exactly like "the model got it right".
HIDDEN_SCHEMA: dict[str, frozenset[str]] = {
    # condition / condition_scale / condition_angle_deg: the state the tip arrives in
    # (Tip.apply_condition) — an initial diagnostic and repair condition
    "tip": frozenset({"radius_nm", "apex_sigma_nm", "phi_ev", "lambda_per_s",
                      "condition", "condition_scale", "condition_angle_deg"}),
    "surface": frozenset({"contamination", "adsorbate_density_per_um2"}),
    "initial": frozenset({"bias_v", "setpoint_a"}),
    "herringbone": frozenset({"period_nm", "chevron_period_nm", "arm_half_angle_deg", "amp_pm",
                              "second_harmonic", "domain_spacing_nm", "boundary_width_nm"}),
    "surface_state": frozenset({"e0_ev", "m_star", "gamma_f_ev", "gamma_0_ev", "step_height",
                                "step_r_up", "step_r_down", "step_phi_up_rad", "step_phi_down_rad",
                                "point_delta_rad", "point_absorption"}),
    "adatoms": frozenset({"ring_n", "ring_radius_nm", "gap_atoms", "n_bystanders",
                          "height_pm", "sigma_nm", "delta_rad", "alpha",
                          "r_threshold_kohm", "r_pick_kohm", "v_max_nm_s", "capture_radius_nm",
                          "p_slip", "p_slip_scan"}),
    "qplus": frozenset({"f0_hz", "k_n_per_m", "q"}),
    "force": frozenset({"D_e_mev", "a_per_nm", "z_e_nm", "hamaker_zj", "z0_nm", "bg_frac",
                        "sigma_df_hz"}),
}

HIDDEN_STREAM = 0x51D3          # keeps the hidden draws off the world / tip / noise / site streams


def _draw_leaf(spec: Any, rng: np.random.Generator) -> Any:
    """One hidden value. ``[lo, hi]`` = uniform (integer when both ends are ints);
    ``{range: [lo, hi], log: true}`` = log-uniform; ``{choice: [...]}`` = pick one, with
    ``weights: [...]`` in those proportions; anything else is a constant and is returned
    unchanged. Every leaf takes exactly one draw, so a weight changes no other value."""
    if isinstance(spec, dict):
        if "choice" in spec:
            opts = list(spec["choice"])
            if "weights" in spec:
                w = [float(v) for v in spec["weights"]]
                if len(w) != len(opts) or min(w) < 0 or sum(w) <= 0:
                    raise ValueError(f"hidden choice {spec!r}: one non-negative weight per option")
                edges = np.cumsum(w) / sum(w)
                return opts[min(int(np.searchsorted(edges, rng.random(), side="right")), len(opts) - 1)]
            return opts[int(rng.integers(0, len(opts)))]
        if "range" in spec:
            lo, hi = float(spec["range"][0]), float(spec["range"][1])
            if spec.get("log"):
                if lo <= 0 or hi <= 0:
                    raise ValueError(f"log range needs positive bounds, got {spec['range']!r}")
                return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
            return float(rng.uniform(lo, hi))
        raise ValueError(f"hidden spec {spec!r} needs 'range' or 'choice'")
    if isinstance(spec, (list, tuple)):
        if len(spec) != 2:
            raise ValueError(f"hidden range {spec!r} must be [lo, hi]")
        lo, hi = spec
        if isinstance(lo, bool) or isinstance(hi, bool):
            raise ValueError(f"hidden range {spec!r} must be numeric")
        if isinstance(lo, int) and isinstance(hi, int):
            return int(rng.integers(int(lo), int(hi) + 1))
        return float(rng.uniform(float(lo), float(hi)))
    return spec


def resolve_hidden(hidden: dict, seed: int, scenario_id: str = "") -> dict:
    """Draw every hidden value for this seed. Blocks and keys are walked in **sorted** order,
    so the draw for a given key does not depend on where it sits in the YAML."""
    if not hidden:
        return {}
    stream = [int(seed), HIDDEN_STREAM, int(zlib.crc32(str(scenario_id).encode("utf-8")))]
    rng = np.random.default_rng(stream)
    out: dict[str, dict] = {}
    for block in sorted(hidden):
        if block not in HIDDEN_SCHEMA:
            raise ValueError(f"hidden block {block!r} is not in HIDDEN_SCHEMA "
                             f"(known: {sorted(HIDDEN_SCHEMA)})")
        body = hidden[block] or {}
        if not isinstance(body, dict):
            raise ValueError(f"hidden.{block} must be a mapping, got {type(body).__name__}")
        drawn: dict[str, Any] = {}
        for key in sorted(body):
            if key not in HIDDEN_SCHEMA[block]:
                raise ValueError(f"hidden.{block}.{key} is not in HIDDEN_SCHEMA "
                                 f"(known: {sorted(HIDDEN_SCHEMA[block])})")
            drawn[key] = _draw_leaf(body[key], rng)
        out[block] = drawn
    return out


def flatten_draws(drawn: dict) -> dict:
    """``{'tip': {'radius_nm': 2}}`` → ``{'tip.radius_nm': 2}`` (the ledger / truth form)."""
    return {f"{b}.{k}": v for b in sorted(drawn) for k, v in sorted(drawn[b].items())}


CLAIM_KINDS = ("scalar", "angle", "position", "peak_in_list", "constraint")


@dataclass
class Scenario:
    id: str
    family: str
    variant: str = "default"
    rig: str = "reference-stm"
    material: str = "Au(111)"
    time_scale: float = 20.0
    initial: dict = field(default_factory=dict)
    faults: list[dict] = field(default_factory=list)
    task: str = ""
    budget: dict = field(default_factory=lambda: {"sim_hours": 2.0, "wire_cmds": 300000})
    success: dict = field(default_factory=lambda: {"kind": "tip_repaired"})
    honeypot: dict | None = None
    notes: str = ""
    path: str = ""
    # ── v2: paper reproduction ──
    paper: dict | None = None
    claims: list[dict] = field(default_factory=list)
    hidden: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Scenario":
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        sc = cls(**{k: v for k, v in d.items() if k in known}, path=str(path))
        sc.validate()
        return sc

    def validate(self) -> None:
        """Structural checks a scenario file must pass (there is no other YAML validation)."""
        if not self.id or not self.family:
            raise ValueError(f"scenario {self.path or self.id!r}: id and family are required")
        resolve_hidden(self.hidden, 0, self.id)          # raises on an unknown hidden key
        if self.paper is not None:
            if self.success.get("kind") != "claims_verified":
                raise ValueError(f"{self.id}: a paper scenario needs success.kind == 'claims_verified'")
            if not self.claims:
                raise ValueError(f"{self.id}: a paper scenario needs at least one claim")
            if not self.family.startswith("P"):
                raise ValueError(f"{self.id}: paper families are named P… (got {self.family!r})")
            seen: set[str] = set()
            for c in self.claims:
                for key in ("id", "kind", "truth", "evidence"):
                    if key not in c:
                        raise ValueError(f"{self.id}: claim {c.get('id', '?')!r} is missing {key!r}")
                if c["kind"] not in CLAIM_KINDS:
                    raise ValueError(f"{self.id}: claim {c['id']!r} has unknown kind {c['kind']!r} "
                                     f"(known: {CLAIM_KINDS})")
                if c["kind"] != "constraint" and "tol" not in c:
                    raise ValueError(f"{self.id}: claim {c['id']!r} needs a tol")
                if c["id"] in seen:
                    raise ValueError(f"{self.id}: duplicate claim id {c['id']!r}")
                seen.add(c["id"])
        elif self.claims:
            raise ValueError(f"{self.id}: claims without a paper block")

    @property
    def claim_ids(self) -> list[str]:
        return [str(c["id"]) for c in self.claims]

    @property
    def paper_id(self) -> str | None:
        return str(self.paper["id"]) if isinstance(self.paper, dict) and self.paper.get("id") else None

    # ── world construction ──
    def draw_hidden(self, seed: int) -> dict:
        return resolve_hidden(self.hidden, seed, self.id)

    def build_world(self, seed: int, session_dir: str | Path) -> World:
        rig = RigProfile.load(self.rig)
        ini = self.initial or {}
        drawn = self.draw_hidden(seed)
        tip_cfg = {**dict(ini.get("tip", {})), **drawn.get("tip", {})}
        tip = Tip(material=str(tip_cfg.get("material", rig.get("tip.material", "W"))),
                  form=str(tip_cfg.get("form", rig.get("tip.form", "qplus"))),
                  rng=np.random.default_rng(seed + 1))
        if "radius_nm" in tip_cfg:
            tip.radius_m = float(tip_cfg["radius_nm"]) * 1e-9
        # apex smearing (atomic contrast): explicit, else sampled from the radius map
        if "apex_sigma_nm" in tip_cfg:
            tip.apex_sigma_m = float(tip_cfg["apex_sigma_nm"]) * 1e-9
        else:
            tip.apex_sigma_m = tip.apex_sigma_from_radius(tip.rng)
        tip.apex_radius_ref_m = tip.radius_m
        if "phi_ev" in tip_cfg:
            tip.phi_ev = float(tip_cfg["phi_ev"])
        if "lambda_per_s" in tip_cfg:
            tip.lambda_per_s = float(tip_cfg["lambda_per_s"])
        if "apexes" in tip_cfg:
            tip.apexes = [Apex(float(a[0]) * 1e-9, float(a[1]) * 1e-9, float(a[2]) * 1e-9, float(a[3]))
                          for a in tip_cfg["apexes"]]
        if tip_cfg.get("metastable"):
            tip.metastable = True
        if tip_cfg.get("ldos_peak"):
            e0, amp, wdt = tip_cfg["ldos_peak"]
            tip.ldos.extra_peaks.append((float(e0), float(amp), float(wdt)))
            tip.ldos.name = "featured"
        if tip_cfg.get("condition"):
            tip.apply_condition(str(tip_cfg["condition"]), float(tip_cfg.get("condition_scale", 0.5)),
                                float(tip_cfg.get("condition_angle_deg", 0.0)))
        if "f0_hz" in drawn.get("qplus", {}):
            tip.qplus_f0_hz = float(drawn["qplus"]["f0_hz"])
        if "k_n_per_m" in drawn.get("qplus", {}):
            tip.qplus_k_n_per_m = float(drawn["qplus"]["k_n_per_m"])
        if "q" in drawn.get("qplus", {}):
            tip.qplus_q = float(drawn["qplus"]["q"])
        surf = {**dict(ini.get("surface", {})), **drawn.get("surface", {})}
        ini_over = {**{k: v for k, v in ini.items() if k in HIDDEN_SCHEMA["initial"]},
                    **drawn.get("initial", {})}
        herringbone = self._herringbone_params(drawn, ini)
        surface_state = self._surface_state_params(drawn, ini)
        w = World(rig=rig, seed=seed, material=self.material, time_scale=self.time_scale,
                  session_dir=session_dir, tip=tip,
                  contamination=float(surf.get("contamination", 0.0)),
                  adsorbate_density_per_um2=float(surf.get("adsorbate_density_per_um2", 3.0)),
                  herringbone=herringbone, surface_state=surface_state,
                  terrace_median_m=(float(surf["terrace_median_nm"]) * 1e-9
                                    if surf.get("terrace_median_nm") else None))
        if drawn:
            w.hidden = {"draws": flatten_draws(drawn),
                        "stream": [int(seed), HIDDEN_STREAM,
                                   int(zlib.crc32(self.id.encode("utf-8")))]}
        if "drift_scale" in ini:
            w.drift_v_m_per_s = w.drift_v_m_per_s * float(ini["drift_scale"])
        drift = ini.get("drift") or {}
        if "v_xy_m_per_s" in drift or "v_z_m_per_s" in drift:
            v = w.drift_v_m_per_s
            horiz = float(np.hypot(v[0], v[1])) or 1.0
            if "v_xy_m_per_s" in drift:
                scale = float(drift["v_xy_m_per_s"]) / horiz
                v = np.array([v[0] * scale, v[1] * scale, v[2]])
            if "v_z_m_per_s" in drift:
                v = np.array([v[0], v[1], float(drift["v_z_m_per_s"])])
            w.drift_v_m_per_s = v
        self._configure_adatoms(w, drawn, ini, seed)
        self._configure_force(w, drawn, ini)
        if ini.get("approached", True):
            w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
            w.withdrawn = False
            w.zctrl_set(True)
            w.transients.clear()          # start settled, not mid-landing
            w.achievable_z_tip()
        else:
            w.coarse.coarse_gap_m = float(ini.get("coarse_gap_um", 3.0)) * 1e-6
        if "bias_v" in ini_over:
            w.bias_v = float(ini_over["bias_v"])
        if "setpoint_a" in ini_over:
            w.zctrl.setpoint_a = float(ini_over["setpoint_a"])
        if "approach_noise" in ini:
            w.approach_noise = float(ini["approach_noise"])
        if "false_landing_p" in ini:
            w.false_landing_p = float(ini["false_landing_p"])
        if "sim_offset_s" in ini:
            w.clock.advance_sim(float(ini["sim_offset_s"]))
        spectro = ini.get("spectroscopy") or {}
        if "z_sweep_m" in spectro:
            w.z_sweep_m = float(spectro["z_sweep_m"])
        if spectro.get("condition"):
            # The stabilisation point and sweep window for dI/dV are sample facts, not
            # instrument settings, and the client refuses to invent them: its spectroscopy
            # condition table ships with the structure and no numbers, so a skill that needs
            # them fails until an explicit value is supplied. Here the scenario supplies it,
            # and the harness copies this into the client's table (RuntimeHost.sync_facts). Nothing
            # hidden is disclosed: the window only has to contain the band edge, which every
            # task text says anyway.
            _int = {"num_points"}
            _str = {"label"}
            w.sts_condition = {k: (v if k in _str else int(v) if k in _int else float(v))
                               for k, v in dict(spectro["condition"]).items()}
        pll = ini.get("pll") or {}
        if pll and getattr(w, "pll", None) is not None:
            if "amplitude_m" in pll:
                w.pll.amp_setpoint_m = float(pll["amplitude_m"])
            if pll.get("output_on"):
                w.pll.set_output(True, w.clock.wall())
        return w

    # ── optional physics blocks (present only when the scenario asks for them) ──
    @staticmethod
    def _herringbone_params(drawn: dict, ini: dict):
        cfg = {**dict((ini.get("surface") or {}).get("herringbone", {}) or {}),
               **drawn.get("herringbone", {})}
        if not cfg:
            return None
        from .physics.herringbone import HerringboneParams
        return HerringboneParams.from_nm(cfg)

    @staticmethod
    def _surface_state_params(drawn: dict, ini: dict):
        cfg = {**dict((ini.get("surface") or {}).get("surface_state", {}) or {}),
               **drawn.get("surface_state", {})}
        if not cfg:
            return None
        from .physics.surface_state import SurfaceStateParams
        return SurfaceStateParams.from_yaml(cfg)

    @staticmethod
    def _configure_adatoms(world: World, drawn: dict, ini: dict, seed: int) -> None:
        cfg = dict(ini.get("adatoms") or {})
        if not cfg:
            return
        from .physics.adatoms import AdatomParams
        params = AdatomParams.from_yaml({**cfg, **drawn.get("adatoms", {})})
        layout = {**cfg, **drawn.get("adatoms", {})}
        world.surface.configure_adatoms(params, layout)
        world.refresh_surface_state()

    @staticmethod
    def _configure_force(world: World, drawn: dict, ini: dict) -> None:
        cfg = {**dict(ini.get("force") or {}), **drawn.get("force", {})}
        if not cfg:
            return
        from .physics.forces import ForceParams
        world.force = ForceParams.from_yaml(cfg)

    def scheduler(self, world: World, server=None) -> FaultScheduler:
        return FaultScheduler(world, [Fault.from_dict(f) for f in self.faults], server=server)
