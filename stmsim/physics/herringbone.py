"""Au(111) 22×√3 herringbone: stripes, chevron zigzag and **rotational domains**.

The reconstruction compresses the top layer along one of the three ⟨1̄10⟩ directions, so a
real Au(111) surface is tiled with domains whose stripe direction differs by 60° (the three
are equivalent modulo 180°). Within a domain the stripe pattern zigzags: the compression
direction alternates between two arms ±α about the mean, with the elbows repeating every
``chevron_period``. Both are what Barth et al. (PRB 42, 9307, 1990) measured.

Model (all vectorised; one orientation per evaluated point):

    label, d_edge = layout.query(x, y)            # Voronoi cell + distance to its border
    th   = lattice_angle + π/6 + label·π/3        # the domain's compression direction
    u, v = rotate((x, y), th)                     # u ∥ stripe wavevector, v ∥ stripe lines
    zig  = sign(sin(2π(v/L + phase_v)))           # which arm we are on
    ph   = 2π((u + shear·zig·v)/T + phase_u)
    env  = 1 − 0.5·exp(−d_edge²/2w²)              # amplitude dips at a domain wall
    h    = A·env·(cos ph + h2·cos 2ph)

With ``domain_spacing_m=None`` (the default for every pre-existing scenario), ``env ≡ 1``,
``phase_u = phase_v = 0`` and ``second_harmonic = 0``, the expression is **bit-identical** to
the single-orientation cosine the simulator used before domains existed — the degenerate case
of one function, not a second code path (``tests/test_herringbone.py`` pins that to 1e-15).

The domain layout is drawn from ``default_rng([site.seed, LAYOUT_STREAM])`` and never touches
``Site.rng``: ``Site.__post_init__`` and ``Surface._populate`` share that generator, so a draw
from it would move every existing scenario's adsorbates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

LAYOUT_STREAM = 0xD0
LEGACY_SHEAR = 0.15                 # the pre-domain formula's arm shear (α ≈ 8.5°)
BOUNDARY_DIP = 0.5                  # amplitude fraction lost at the centre of a domain wall
K_CONVENTION = "k_deg = stripe_deg - 90 (mod 180); stage frame, CCW from +x"


@dataclass(frozen=True)
class HerringboneParams:
    """SI. ``domain_spacing_m=None`` ⇒ a single domain (the legacy field)."""
    period_m: float = 6.3e-9
    chevron_period_m: float = 30e-9
    shear: float = LEGACY_SHEAR
    amp_m: float = 0.010e-9
    second_harmonic: float = 0.0
    domain_spacing_m: float | None = None
    boundary_width_m: float = 3e-9

    @property
    def arm_half_angle_deg(self) -> float:
        return math.degrees(math.atan(self.shear))

    @classmethod
    def from_material(cls, material) -> "HerringboneParams | None":
        if not getattr(material, "herringbone_period_m", None):
            return None
        return cls(period_m=float(material.herringbone_period_m),
                   chevron_period_m=float(material.chevron_period_m or 30e-9),
                   amp_m=float(material.herringbone_amp_m))

    @classmethod
    def from_nm(cls, cfg: dict[str, Any]) -> "HerringboneParams":
        """YAML block in lab units (nm / pm / deg) → SI."""
        d: dict[str, Any] = {}
        if "period_nm" in cfg:
            d["period_m"] = float(cfg["period_nm"]) * 1e-9
        if "chevron_period_nm" in cfg:
            d["chevron_period_m"] = float(cfg["chevron_period_nm"]) * 1e-9
        if "arm_half_angle_deg" in cfg:
            d["shear"] = math.tan(math.radians(float(cfg["arm_half_angle_deg"])))
        if "amp_pm" in cfg:
            d["amp_m"] = float(cfg["amp_pm"]) * 1e-12
        if "second_harmonic" in cfg:
            d["second_harmonic"] = float(cfg["second_harmonic"])
        if cfg.get("domain_spacing_nm") is not None:
            d["domain_spacing_m"] = float(cfg["domain_spacing_nm"]) * 1e-9
        if "boundary_width_nm" in cfg:
            d["boundary_width_m"] = float(cfg["boundary_width_nm"]) * 1e-9
        return cls(**d)

    def with_defaults_from(self, material) -> "HerringboneParams":
        """Fill period / chevron / amplitude the YAML did not name from the material."""
        base = self.from_material(material)
        if base is None:
            return self
        changes: dict[str, Any] = {}
        if self.period_m == HerringboneParams.period_m and base.period_m:
            changes["period_m"] = base.period_m
        if self.chevron_period_m == HerringboneParams.chevron_period_m and base.chevron_period_m:
            changes["chevron_period_m"] = base.chevron_period_m
        if self.amp_m == HerringboneParams.amp_m and base.amp_m:
            changes["amp_m"] = base.amp_m
        return replace(self, **changes) if changes else self


@dataclass
class DomainLayout:
    """Voronoi tiling of one site into rotational domains."""
    seeds: np.ndarray                      # (N, 2) sample-frame metres
    labels: np.ndarray                     # (N,) ∈ {0, 1, 2}
    phase_u: np.ndarray                    # (N,) stripe phase, cycles
    phase_v: np.ndarray                    # (N,) chevron phase, cycles
    spacing_m: float
    _tree: Any = field(default=None, repr=False)

    def __post_init__(self):
        from scipy.spatial import cKDTree
        self._tree = cKDTree(self.seeds)

    def query(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(label, distance to the nearest Voronoi border) for each point."""
        pts = np.column_stack([np.asarray(x, float).ravel(), np.asarray(y, float).ravel()])
        k = 2 if len(self.seeds) > 1 else 1
        d, idx = self._tree.query(pts, k=k)
        if k == 1:
            lab = self.labels[idx]
            edge = np.full(pts.shape[0], np.inf)
        else:
            lab = self.labels[idx[:, 0]]
            edge = (d[:, 1] - d[:, 0]) / 2.0
        shape = np.shape(x)
        return lab.reshape(shape), edge.reshape(shape)

    def nearest_index(self, x: float, y: float) -> int:
        return int(self._tree.query(np.array([[x, y]]))[1][0])


def layout_for_site(site, params: HerringboneParams) -> DomainLayout | None:
    """Build (and cache on the site) the domain tiling. ``None`` for a single-domain field."""
    if params.domain_spacing_m is None:
        return None
    cached = getattr(site, "hb_layout", None)
    if cached is not None and abs(cached.spacing_m - params.domain_spacing_m) < 1e-15:
        return cached
    rng = np.random.default_rng([int(site.seed), LAYOUT_STREAM])
    spacing = float(params.domain_spacing_m)
    half = float(getattr(site, "span_m", 12e-6)) / 2.0
    half = min(half, 3e-6) + 2 * spacing          # the site's terrain extent plus a margin
    area = (2 * half) ** 2
    n = max(4, int(rng.poisson(area / spacing ** 2)))
    n = min(n, 20000)
    seeds = rng.uniform(-half, half, size=(n, 2))
    labels = rng.integers(0, 3, size=n)
    phase_u = rng.uniform(0.0, 1.0, size=n)
    phase_v = rng.uniform(0.0, 1.0, size=n)
    # Ensure that a domain-enabled scenario has at least two orientations near the start
    # position; if the local seeds are uniform, re-label the second-nearest seed.
    reach = 600e-9
    near = np.flatnonzero((np.abs(seeds[:, 0]) < reach) & (np.abs(seeds[:, 1]) < reach))
    if near.size and len(set(labels[near].tolist())) == 1:
        order = np.argsort(seeds[:, 0] ** 2 + seeds[:, 1] ** 2)
        pick = int(order[1]) if order.size > 1 else int(order[0])
        labels[pick] = (int(labels[pick]) + 1) % 3
    layout = DomainLayout(seeds=seeds, labels=labels, phase_u=phase_u, phase_v=phase_v,
                          spacing_m=spacing)
    try:
        site.hb_layout = layout
    except Exception:  # noqa: BLE001 — duck-typed surfaces in tests
        pass
    return layout


def _angles(site, params: HerringboneParams, label) -> np.ndarray:
    return site.lattice_angle + math.pi / 6 + np.asarray(label, float) * (math.pi / 3)


def height(x: np.ndarray, y: np.ndarray, site, params: HerringboneParams) -> np.ndarray:
    """The reconstruction's height field (metres) at sample-frame (x, y)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if not params.period_m or params.amp_m == 0.0:
        return np.zeros_like(x)
    layout = layout_for_site(site, params)
    if layout is None:
        th = site.lattice_angle + math.pi / 6
        ph_u = 0.0
        ph_v = 0.0
        env = 1.0
    else:
        pts = np.column_stack([x.ravel(), y.ravel()])
        d, idx = layout._tree.query(pts, k=2 if len(layout.seeds) > 1 else 1)
        if idx.ndim == 1:                       # a single seed: no border anywhere
            near = idx
            d_edge = np.full(pts.shape[0], np.inf)
        else:
            near = idx[:, 0]
            d_edge = (d[:, 1] - d[:, 0]) / 2.0
        th = _angles(site, params, layout.labels[near]).reshape(x.shape)
        ph_u = layout.phase_u[near].reshape(x.shape)
        ph_v = layout.phase_v[near].reshape(x.shape)
        w = max(params.boundary_width_m, 1e-12)
        env = 1.0 - BOUNDARY_DIP * np.exp(-0.5 * (d_edge.reshape(x.shape) / w) ** 2)
    c, s = np.cos(th), np.sin(th)
    u = x * c + y * s
    v = -x * s + y * c
    # the grouping matters: with zero phases this has to reduce *exactly* to the pre-domain
    # expression, so the phase is added as a separate term rather than inside the division
    zig = np.sign(np.sin(2 * np.pi * v / params.chevron_period_m + 2 * np.pi * ph_v))
    ph = 2 * np.pi * (u + params.shear * zig * v) / params.period_m + 2 * np.pi * ph_u
    out = np.cos(ph)
    if params.second_harmonic:
        out = out + params.second_harmonic * np.cos(2 * ph)
    return params.amp_m * env * out


def orientation_at(x: float, y: float, site, params: HerringboneParams) -> dict:
    """Which domain sits under (x, y) and which way its stripes run (stage frame, mod 180°)."""
    layout = layout_for_site(site, params)
    if layout is None:
        label, d_edge, seed_i = 0, float("inf"), -1
    else:
        lab, edge = layout.query(np.array([x]), np.array([y]))
        label, d_edge = int(np.ravel(lab)[0]), float(np.ravel(edge)[0])
        seed_i = layout.nearest_index(x, y)
    th = float(site.lattice_angle + math.pi / 6 + label * math.pi / 3)
    stripe = math.degrees(th + math.pi / 2) % 180.0
    ph_v = float(layout.phase_v[seed_i]) if layout is not None and seed_i >= 0 else 0.0
    c, s = math.cos(th), math.sin(th)
    v = -x * s + y * c
    arm = 1.0 if math.sin(2 * math.pi * (v / params.chevron_period_m + ph_v)) >= 0 else -1.0
    return {"domain": label, "seed_index": seed_i,
            "stripe_deg": stripe,
            "stripe_local_deg": (stripe + arm * params.arm_half_angle_deg) % 180.0,
            "arm_sign": arm,
            # the two arms of the zigzag, which is what an FFT of a frame shows: two satellite
            # peaks at ±alpha about the mean direction, never a peak on the mean itself
            "arm_half_angle_deg": params.arm_half_angle_deg,
            "arms_deg": [(stripe + params.arm_half_angle_deg) % 180.0,
                         (stripe - params.arm_half_angle_deg) % 180.0],
            "boundary_distance_nm": d_edge * 1e9 if math.isfinite(d_edge) else None}


#: how many domain seeds a snapshot carries — enough to judge any position an agent can reach
#: in an episode, few enough that the ledger's truth block stays readable
SNAPSHOT_MAX_DOMAINS = 64


def snapshot(site, params: HerringboneParams, *, cx: float = 0.0, cy: float = 0.0,
             half_m: float = 600e-9) -> dict:
    """Reference reconstruction around (cx, cy), retained for judging and replay."""
    layout = layout_for_site(site, params)
    base = math.degrees(site.lattice_angle + math.pi / 6 + math.pi / 2)
    orientations = [round((base + k * 60.0) % 180.0, 6) for k in range(3)]
    out = {
        "enabled": True,
        "period_nm": params.period_m * 1e9,
        # what an FFT of the frame actually measures: the stripes run at ±α to the mean
        # direction, so their wavevector is longer by 1/cos α and the period shorter by cos α.
        # That is the number Barth et al. report — the spacing between soliton walls measured
        # perpendicular to them — and it is the one a claim can be checked against.
        "period_along_arm_nm": params.period_m * 1e9 * math.cos(math.atan(params.shear)),
        "chevron_period_nm": params.chevron_period_m * 1e9,
        "arm_half_angle_deg": params.arm_half_angle_deg,
        "amp_pm": params.amp_m * 1e12,
        "second_harmonic": params.second_harmonic,
        "orientations_deg": orientations,
        "k_convention": K_CONVENTION,
        "domain_spacing_nm": params.domain_spacing_m * 1e9 if params.domain_spacing_m else None,
        "boundary_width_nm": params.boundary_width_m * 1e9,
        "single_domain": layout is None,
        "domains": [],
        "n_orientations_in_range": 1,
    }
    if layout is None:
        out["domains"] = [{"id": 0, "k": 0, "stripe_deg": orientations[0],
                           "seed_xy_nm": [0.0, 0.0], "phase_u": 0.0, "phase_v": 0.0}]
        return out
    m = ((np.abs(layout.seeds[:, 0] - cx) <= half_m + 2 * layout.spacing_m)
         & (np.abs(layout.seeds[:, 1] - cy) <= half_m + 2 * layout.spacing_m))
    idx = np.flatnonzero(m)
    if idx.size > SNAPSHOT_MAX_DOMAINS:                 # keep the ones nearest the centre
        d = (layout.seeds[idx, 0] - cx) ** 2 + (layout.seeds[idx, 1] - cy) ** 2
        idx = idx[np.argsort(d)[:SNAPSHOT_MAX_DOMAINS]]
    out["domains"] = [{"id": int(i), "k": int(layout.labels[i]),
                       "stripe_deg": orientations[int(layout.labels[i])],
                       "seed_xy_nm": [float(layout.seeds[i, 0] * 1e9), float(layout.seeds[i, 1] * 1e9)],
                       "phase_u": float(layout.phase_u[i]), "phase_v": float(layout.phase_v[i])}
                      for i in idx]
    out["n_orientations_in_range"] = len({d["k"] for d in out["domains"]}) or 1
    return out


def orientation_from_snapshot(snap: dict | None, x_nm: float, y_nm: float) -> float | None:
    """Stripe direction (deg, mod 180) at a sample-frame point, from the truth snapshot alone.

    ``stmbench`` judges a reported orientation with this: it may import ``stmsim``, and the
    snapshot travels in the ledger, so a judge never needs the live world."""
    if not snap or not snap.get("enabled"):
        return None
    domains = snap.get("domains") or []
    if not domains:
        return None
    if snap.get("single_domain") or len(domains) == 1:
        return float(domains[0]["stripe_deg"])
    best, best_d = None, float("inf")
    for d in domains:
        sx, sy = d["seed_xy_nm"]
        dist = (sx - x_nm) ** 2 + (sy - y_nm) ** 2
        if dist < best_d:
            best, best_d = d, dist
    return float(best["stripe_deg"]) if best else None
