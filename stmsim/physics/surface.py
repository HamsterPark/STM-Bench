"""Sample surface: a persistent height field per coarse-motion site.

``height(x, y)`` (metres in, metres out, vectorised) is the sum of three layers:

1. **terraces** — monoatomic steps of the material's step height along wandering step
   lines (terrace width drawn from the material's distribution; Au(111) median 17 nm in the
   calibrated sample);
2. **atomic layer** — hexagonal lattice corrugation (nearest-neighbour distance is the
   source of truth, row spacing ``a·√3/2`` derived), Au(111) herringbone (6.3 nm) with a
   30 nm chevron modulation, HOPG's triangular lattice;
3. **dynamic layer** — adsorbates, contamination patches, and everything the operator
   leaves behind: poke clusters (one per apex), pulse craters + spatter, crash pits,
   scratches. These persist for the life of the site.

A separate ``phi_factor(x, y)`` field (0.2–1.0) scales the local apparent barrier so a
contaminated area gives the low-φ junction observed during extended calibration.

**Tilt.** Each site carries a physical sample tilt (``Site.tilt``, a slope per axis,
rad ≈ m/m) that ``terrace_height`` adds as a plane. The *instrument* compensates it with
the piezo tilt correction (controller ``Piezo.TiltSet``): ``Surface.tilt_comp`` is the
compensating slope that :meth:`World.set_piezo_tilt` pushes down here, and the height
every consumer sees — the tip, the renderer, the Z the controller reports — is the
**residual** ``(tilt − tilt_comp)`` plane. ``Site.tilt`` itself never changes: the
sample is what it is; levelling is a property of the scanner.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Material:
    name: str
    nn_m: float                 # nearest-neighbour distance
    step_m: float               # single-atom step height
    corrugation_m: float        # lattice corrugation amplitude (pm-scale) at low bias
    phi_ev: float               # clean-surface apparent barrier
    terrace_median_m: float
    ldos_onset_ev: float | None = None
    herringbone_period_m: float | None = None
    chevron_period_m: float | None = None
    herringbone_amp_m: float = 0.0
    lattice: str = "hex"        # hex | triangular (HOPG) | none

    @property
    def first_order_period_m(self) -> float:
        """Row spacing of the close-packed lattice (a·√3/2): the first-order FFT period."""
        if self.lattice == "none":
            return 0.0
        return self.nn_m * math.sqrt(3) / 2.0


MATERIALS: dict[str, Material] = {
    "Au(111)": Material("Au(111)", 0.28837e-9, 0.2354e-9, 0.030e-9, 4.0, 17e-9,
                        ldos_onset_ev=-0.49, herringbone_period_m=6.3e-9,
                        chevron_period_m=30e-9, herringbone_amp_m=0.010e-9),
    "Cu(111)": Material("Cu(111)", 0.255e-9, 0.208e-9, 0.025e-9, 4.3, 30e-9, ldos_onset_ev=-0.44),
    "Ag(111)": Material("Ag(111)", 0.289e-9, 0.236e-9, 0.025e-9, 4.1, 25e-9, ldos_onset_ev=-0.065),
    "HOPG": Material("HOPG", 0.246e-9, 0.335e-9, 0.060e-9, 4.5, 200e-9, lattice="triangular"),
}


@dataclass
class Feature:
    """A localised height modification (Gaussian / elliptical Gaussian)."""
    x: float
    y: float
    height: float               # + bump, − pit (metres)
    sigma_x: float
    sigma_y: float | None = None
    angle: float = 0.0
    kind: str = "adsorbate"     # adsorbate | cluster | pit | crater | spatter | scratch
    born_sim_s: float = 0.0

    def eval(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        sy = self.sigma_y if self.sigma_y is not None else self.sigma_x
        dx, dy = x - self.x, y - self.y
        if self.angle:
            c, s = math.cos(self.angle), math.sin(self.angle)
            dx, dy = c * dx + s * dy, -s * dx + c * dy
        return self.height * np.exp(-0.5 * ((dx / self.sigma_x) ** 2 + (dy / sy) ** 2))


@dataclass
class Site:
    """One coarse-motion site: its own random terrain and its own memory of damage."""
    seed: int
    material: Material
    rng: np.random.Generator = field(init=False)
    step_angle: float = 0.0
    terrace_w: float = 17e-9
    wander_amp: float = 5e-9
    wander_period: float = 80e-9
    lattice_angle: float = 0.0
    base_phase: float = 0.0
    features: list[Feature] = field(default_factory=list)
    phi_blobs: list[tuple[float, float, float, float]] = field(default_factory=list)  # x, y, sigma, factor
    tilt: tuple[float, float] = (0.0, 0.0)   # residual sample tilt (rad) — AutoTilt target
    boundaries: np.ndarray = field(init=False)   # step positions along the staircase axis (m)
    span_m: float = 12e-6
    # lazily built, each from its own RNG stream so Site.rng stays untouched
    hb_layout: object | None = field(default=None, repr=False)      # herringbone domains
    adatoms: object | None = field(default=None, repr=False)        # AdatomRegistry

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        r = self.rng
        self.step_angle = float(r.uniform(0, math.pi))
        # terrace widths vary along the staircase: log-normal around the material median with a
        # heavy tail (calibrated Au(111): median 17 nm, with 35 nm flat windows in 5/6 frames)
        self.terrace_w = float(self.material.terrace_median_m)
        widths = self.material.terrace_median_m * np.exp(r.normal(0.0, 0.7, 4000))
        b = np.cumsum(widths)
        b = b - b[len(b) // 2] + r.uniform(0, self.terrace_w)
        self.boundaries = b[(b > -self.span_m) & (b < self.span_m)]
        self.wander_amp = float(self.terrace_w * r.uniform(0.05, 0.2))
        self.wander_period = float(r.uniform(6, 12) * self.terrace_w)
        self.lattice_angle = float(r.uniform(0, math.pi / 3))
        self.base_phase = float(r.uniform(0, 1))
        self.tilt = (float(r.normal(0, 2e-3)), float(r.normal(0, 2e-3)))


class Surface:
    def __init__(self, material: str | Material = "Au(111)", *, seed: int = 0,
                 adsorbate_density_per_um2: float = 3.0, contamination: float = 0.0,
                 herringbone=None, surface_state=None, terrace_median_m: float | None = None):
        self.material = MATERIALS[material] if isinstance(material, str) else material
        if terrace_median_m:
            # how well the sample was annealed. The calibrated Au(111) reference has 17 nm terraces, and
            # a 235 pm step every 17 nm buries a 10 pm reconstruction; the samples the 1990
            # herringbone work used were annealed to terraces of many tens of nm.
            from dataclasses import replace as _replace

            self.material = _replace(self.material, terrace_median_m=float(terrace_median_m))
        self.seed = seed
        self.adsorbate_density = adsorbate_density_per_um2
        self.contamination = contamination      # 0 clean … 1 heavily contaminated
        self._sites: dict[tuple[int, int], Site] = {}
        self.site_index: tuple[int, int] = (0, 0)
        self.site_extent_m = 3e-6           # square area each site's terrain covers
        # ── optional physics (None ⇒ exactly the field the simulator had before) ──
        from .herringbone import HerringboneParams
        if herringbone is None:
            self.herringbone = HerringboneParams.from_material(self.material)
        else:
            self.herringbone = herringbone.with_defaults_from(self.material)
        self.surface_state = None
        if surface_state is not None:
            from .surface_state import SurfaceState
            self.surface_state = SurfaceState(surface_state)
        self._ldos_cache: dict[tuple, object] = {}
        self._corral_map: dict | None = None
        self._frame_map: dict | None = None
        # instrument-side compensating plane (slope per axis, same units as Site.tilt):
        # the scanner's tilt correction, owned by World.set_piezo_tilt. Subtracted from the
        # terrace plane so everything downstream sees the residual tilt only.
        self.tilt_comp: tuple[float, float] = (0.0, 0.0)

    # ── sites ──
    @property
    def site(self) -> Site:
        return self.site_at(self.site_index)

    def site_at(self, idx: tuple[int, int]) -> Site:
        if idx not in self._sites:
            s = Site(seed=self.seed * 100003 + idx[0] * 1009 + idx[1] * 7 + 12345,
                     material=self.material)
            self._populate(s)
            self._sites[idx] = s
        return self._sites[idx]

    def move_site(self, di: int, dj: int) -> None:
        self.site_index = (self.site_index[0] + di, self.site_index[1] + dj)

    def _populate(self, s: Site) -> None:
        r = s.rng
        area_um2 = (self.site_extent_m * 1e6) ** 2
        n = int(r.poisson(self.adsorbate_density * area_um2))
        half = self.site_extent_m / 2
        for _ in range(min(n, 5000)):
            s.features.append(Feature(
                x=float(r.uniform(-half, half)), y=float(r.uniform(-half, half)),
                height=float(r.uniform(0.05e-9, 0.3e-9)), sigma_x=float(r.uniform(0.8e-9, 2.0e-9)),
                kind="adsorbate"))
        n_blob = int(round(self.contamination * 12))
        for _ in range(n_blob):
            s.phi_blobs.append((float(r.uniform(-half, half)), float(r.uniform(-half, half)),
                                float(r.uniform(50e-9, 400e-9)), float(r.uniform(0.15, 0.5))))

    # ── height field ──
    def _staircase_u(self, x: np.ndarray, y: np.ndarray, site: Site | None = None) -> np.ndarray:
        """Coordinate along the staircase axis, with the step lines' wander folded in, so that
        ``searchsorted(boundaries, u)`` is the terrace index and ``u − boundary`` is the
        distance to a step edge (what the surface state scatters off)."""
        s = site or self.site
        c, sn = math.cos(s.step_angle), math.sin(s.step_angle)
        u = x * c + y * sn
        v = -x * sn + y * c
        wander = s.wander_amp * np.sin(2 * np.pi * v / s.wander_period) \
            + 0.4 * s.wander_amp * np.sin(2 * np.pi * v / (0.37 * s.wander_period) + 1.3)
        return u + wander

    def terrace_level(self, x: np.ndarray, y: np.ndarray, site: Site | None = None) -> np.ndarray:
        """Integer terrace index under each (x, y) — the staircase without step height or tilt."""
        s = site or self.site
        return np.searchsorted(s.boundaries, self._staircase_u(x, y, s))

    def terrace_height(self, x: np.ndarray, y: np.ndarray, site: Site | None = None) -> np.ndarray:
        """Staircase + the **residual** tilt plane (sample tilt minus the scanner's compensation)."""
        s = site or self.site
        level = self.terrace_level(x, y, s).astype(float)
        rx, ry = self.residual_tilt(s)
        tilt = rx * x + ry * y
        return level * self.material.step_m + tilt

    def residual_tilt(self, site: Site | None = None) -> tuple[float, float]:
        """Slope per axis (rad ≈ m/m) left after the piezo tilt correction — the plane a
        frame shows and what a levelling routine is trying to drive to zero."""
        s = site or self.site
        return (s.tilt[0] - self.tilt_comp[0], s.tilt[1] - self.tilt_comp[1])

    # ── truth queries (benchmark criteria, not rendering) ──
    def step_free_window(self, x: float, y: float, w: float, n: int = 49) -> bool:
        """True when the w×w square centred on sample-frame (x, y) crosses no step edge."""
        half = w / 2
        g = np.linspace(-half, half, n)
        gx, gy = np.meshgrid(x + g, y + g)
        lv = self.terrace_level(gx, gy)
        return bool(lv.min() == lv.max())

    def damage_in_window(self, x: float, y: float, w: float) -> int:
        """Operator-made features (anything but native adsorbates) overlapping the window."""
        half = w / 2
        n = 0
        for f in self.site.features:
            if f.kind == "adsorbate":
                continue
            r = 2.0 * max(f.sigma_x, f.sigma_y or f.sigma_x)
            if abs(f.x - x) <= half + r and abs(f.y - y) <= half + r:
                n += 1
        return n

    def atomic_height(self, x: np.ndarray, y: np.ndarray, site: Site | None = None) -> np.ndarray:
        """The atomic lattice alone — the *bare* corrugation, before the apex transfer.

        Rendered separately from everything else because atomic contrast is carried by
        the terminating apex atom (``Tip.atomic_transfer``), not by the mesoscopic tip
        radius that broadens steps: a 1 nm-radius tip blurs a 0.25 nm lattice to nothing
        under the geometric ``sqrt(R/2κ)`` kernel, while nm-radius tips can resolve Au(111)
        atoms. Two scales, two paths; this separation is required for atomic-phase diagnostics."""
        m = self.material
        if m.lattice == "none" or m.corrugation_m <= 0:
            return np.zeros_like(x, dtype=float)
        s = site or self.site
        a = m.nn_m
        # three reciprocal vectors of a hexagonal lattice, row spacing a*sqrt(3)/2
        k = 4 * np.pi / (math.sqrt(3) * a)
        out = np.zeros_like(x, dtype=float)
        for j in range(3):
            th = s.lattice_angle + j * np.pi / 3
            out += np.cos(k * (x * math.cos(th) + y * math.sin(th)))
        if m.lattice == "triangular":
            out = out - 0.5 * np.abs(out)   # accentuate one sublattice (HOPG-like)
        out *= m.corrugation_m / 3.0
        return out

    def herringbone_height(self, x: np.ndarray, y: np.ndarray, site: Site | None = None) -> np.ndarray:
        """The 22×√3 reconstruction (see :mod:`stmsim.physics.herringbone`).

        With the material's own parameters and no domain spacing this is bit-identical to the
        single-orientation cosine the simulator used before rotational domains existed."""
        if self.herringbone is None:
            return np.zeros_like(x, dtype=float)
        from . import herringbone as hb
        return hb.height(x, y, site or self.site, self.herringbone)

    def orientation_at(self, x: float, y: float, site: Site | None = None) -> dict | None:
        """Which rotational domain sits at (x, y) and how its stripes run — the P1 truth."""
        if self.herringbone is None:
            return None
        from . import herringbone as hb
        return hb.orientation_at(float(x), float(y), site or self.site, self.herringbone)

    def herringbone_snapshot(self, cx: float = 0.0, cy: float = 0.0,
                             half_m: float = 1.5e-6) -> dict | None:
        if self.herringbone is None:
            return None
        from . import herringbone as hb
        return hb.snapshot(self.site, self.herringbone, cx=cx, cy=cy, half_m=half_m)

    def lattice_height(self, x: np.ndarray, y: np.ndarray, site: Site | None = None) -> np.ndarray:
        """Atomic lattice + herringbone (kept for callers that want both)."""
        return self.atomic_height(x, y, site) + self.herringbone_height(x, y, site)

    def height_smooth(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Everything except the atomic lattice: terraces, herringbone, features, adatoms."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        return (self.terrace_height(x, y) + self.herringbone_height(x, y)
                + self.feature_height(x, y) + self.adatom_height(x, y))

    def adatom_height(self, x: np.ndarray, y: np.ndarray, site: Site | None = None) -> np.ndarray:
        """Registered adatoms as apparent-height bumps (zero cost when there are none)."""
        s = site or self.site
        reg = s.adatoms
        if reg is None:
            return np.zeros_like(x, dtype=float)
        return reg.height(np.asarray(x, float), np.asarray(y, float))

    def feature_height(self, x: np.ndarray, y: np.ndarray, site: Site | None = None) -> np.ndarray:
        s = site or self.site
        out = np.zeros_like(x, dtype=float)
        if not s.features:
            return out
        x0, x1 = float(np.min(x)), float(np.max(x))
        y0, y1 = float(np.min(y)), float(np.max(y))
        for f in s.features:
            sy = f.sigma_y if f.sigma_y is not None else f.sigma_x
            r = 3.5 * max(f.sigma_x, sy)
            if f.x + r < x0 or f.x - r > x1 or f.y + r < y0 or f.y - r > y1:
                continue
            out += f.eval(x, y)
        return out

    def height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        return self.terrace_height(x, y) + self.lattice_height(x, y) + self.feature_height(x, y)

    def phi_factor(self, x: float, y: float) -> float:
        s = self.site
        f = 1.0
        for bx, by, sig, fac in s.phi_blobs:
            w = math.exp(-0.5 * ((x - bx) ** 2 + (y - by) ** 2) / sig ** 2)
            f = min(f, 1.0 - (1.0 - fac) * w)
        return f

    # ── damage ──
    def add_feature(self, f: Feature) -> Feature:
        self.site.features.append(f)
        return f

    def features_near(self, x: float, y: float, r: float) -> list[Feature]:
        return [f for f in self.site.features if (f.x - x) ** 2 + (f.y - y) ** 2 <= r * r]

    def damage_area_nm2(self) -> float:
        return sum(2 * math.pi * f.sigma_x * (f.sigma_y or f.sigma_x) * 1e18
                   for f in self.site.features if f.kind != "adsorbate")

    # ── surface state (standing waves, corrals) ──────────────────────────────
    def _step_distances(self, x: float, y: float, site: Site | None = None):
        """Signed distance to the two step edges bounding the terrace under (x, y)."""
        s = site or self.site
        u = float(self._staircase_u(np.array([x]), np.array([y]), s)[0])
        b = s.boundaries
        if b.size == 0:
            return None, None
        idx = int(np.searchsorted(b, u))
        right = float(b[idx] - u) if idx < b.size else None          # ascending edge
        left = float(u - b[idx - 1]) if idx > 0 else None            # descending edge
        return left, right

    def _line_terms(self, x: float, y: float, site: Site | None = None):
        from .surface_state import LineScatterer
        if self.surface_state is None:
            return ()
        p = self.surface_state.p
        left, right = self._step_distances(x, y, site)
        out = []
        if right is not None and abs(right) < p.cutoff_m:
            out.append(LineScatterer(np.array(abs(right)), p.step_r_up, p.step_phi_up_rad))
        if left is not None and abs(left) < p.cutoff_m:
            out.append(LineScatterer(np.array(abs(left)), p.step_r_down, p.step_phi_down_rad))
        return tuple(out)

    def scatterers_near(self, x: float, y: float, radius: float | None = None):
        """Point scatterers within reach of (x, y): Poisson adsorbates plus registered adatoms."""
        from .surface_state import SCATTERER_DEFAULTS, PointScatterer
        if self.surface_state is None:
            return ()
        p = self.surface_state.p
        r = radius if radius is not None else p.cutoff_m
        s = self.site
        out = []
        for f in s.features:
            if abs(f.x - x) > r or abs(f.y - y) > r:
                continue
            delta, alpha = SCATTERER_DEFAULTS.get(f.kind, (p.point_delta_rad, p.point_absorption))
            if f.kind == "adsorbate":
                delta, alpha = p.point_delta_rad, p.point_absorption
            out.append(PointScatterer(f.x, f.y, delta, alpha, kind=f.kind))
        if s.adatoms is not None:
            for sc in s.adatoms.scatterers():
                if abs(sc.x - x) <= r and abs(sc.y - y) <= r:
                    out.append(sc)
        return tuple(out)

    @property
    def scatterer_epoch(self) -> int:
        reg = self.site.adatoms
        return int(getattr(reg, "epoch", 0)) if reg is not None else 0

    def invalidate_electronic_cache(self) -> None:
        self._ldos_cache.clear()
        self._corral_map = None
        self._frame_map = None

    def has_corral(self) -> bool:
        reg = self.site.adatoms
        return reg is not None and getattr(reg, "ring", None) is not None

    def ldos_at(self, x: float, y: float):
        """The sample LDOS **at this position** — a plain material template when the scenario
        never switched the surface state on, so B6 and every earlier scenario are unchanged."""
        from .junction import LDOSTemplate
        if self.surface_state is None:
            return LDOSTemplate(self.material.name, onset_ev=self.material.ldos_onset_ev)
        key = (round(float(x), 11), round(float(y), 11), self.scatterer_epoch)
        hit = self._ldos_cache.get(key)
        if hit is not None:
            return hit
        from .surface_state import LocalLDOS
        pts = self.scatterers_near(x, y)
        if self.has_corral():
            # multiple scattering: the adsorbates near the point and the corral's own atoms —
            # not the whole deposit (same order as before, so a corral without a field is unchanged)
            r = self.surface_state.p.cutoff_m
            pts = tuple(p for p in pts if getattr(p, "kind", "") != "adatom") + tuple(
                p for p in self.corral_scatterers() if abs(p.x - x) <= r and abs(p.y - y) <= r)
        out = LocalLDOS(self.surface_state, x, y, lines=self._line_terms(x, y), points=pts,
                        material=self.material.name, multiple=self.has_corral())
        if len(self._ldos_cache) > 64:
            self._ldos_cache.clear()
        self._ldos_cache[key] = out
        return out

    def ldos_map(self, x: np.ndarray, y: np.ndarray, e_ev: float) -> np.ndarray:
        """ρ(E) over a grid at one energy (dI/dV map channel and the physics tests)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if self.surface_state is None:
            return np.ones_like(x)
        ss = self.surface_state
        cx, cy = float(np.mean(x)), float(np.mean(y))
        s = np.zeros_like(x)
        for ln in self._line_terms(cx, cy):
            pass
        # step distance is position-dependent across a map, so recompute it per point
        u = self._staircase_u(x, y)
        b = self.site.boundaries
        if b.size:
            idx = np.searchsorted(b, u)
            right = np.where(idx < b.size, b[np.clip(idx, 0, b.size - 1)] - u, np.inf)
            left = np.where(idx > 0, u - b[np.clip(idx - 1, 0, b.size - 1)], np.inf)
            p = ss.p
            s = s + ss.line_modulation(np.where(np.isfinite(right), right, 1.0), e_ev,
                                       p.step_r_up, p.step_phi_up_rad) * np.isfinite(right)
            s = s + ss.line_modulation(np.where(np.isfinite(left), left, 1.0), e_ev,
                                       p.step_r_down, p.step_phi_down_rad) * np.isfinite(left)
        pts = self.scatterers_near(cx, cy, radius=ss.p.cutoff_m + float(np.ptp(x) + np.ptp(y)))
        if pts:
            s = s + np.asarray(ss.point_modulation(x, y, e_ev, pts, multiple=self.has_corral()),
                               float).reshape(x.shape)
        return 1.0 + ss.p.step_height * np.asarray(ss.band_edge(e_ev), float) * (1.0 + s)

    #: standing-wave map resolution: λ/2 is ~1.5 nm on Cu(111), so this is ~15 points a period
    ELECTRONIC_GRID_M = 0.1e-9
    ELECTRONIC_GRID_MAX = 400          # per axis, so a wide frame coarsens instead of exploding

    def prepare_frame(self, x0: float, x1: float, y0: float, y1: float, bias_v: float) -> None:
        """Build the frame's standing-wave map once, instead of once per scan line.

        A row of a 256 px frame is 1024 oversampled points and every one of them costs a
        handful of Bessel/Struve evaluations; done per row, a frame takes longer than the 5 s
        the controller client waits for ``Scan.Action`` to answer. One coarse map interpolated
        afterwards is the same physics at a fraction of the cost."""
        if self.surface_state is None or abs(bias_v) < 1e-3:
            self._frame_map = None
            return
        from scipy.interpolate import RegularGridInterpolator
        pad = 2e-9
        key = (round(x0, 12), round(x1, 12), round(y0, 12), round(y1, 12),
               round(bias_v, 6), self.scatterer_epoch, self.site_index)
        if (self._frame_map or {}).get("key") == key:
            return
        step = self.ELECTRONIC_GRID_M
        nx = int(np.clip((x1 - x0 + 2 * pad) / step, 8, self.ELECTRONIC_GRID_MAX)) + 1
        ny = int(np.clip((y1 - y0 + 2 * pad) / step, 8, self.ELECTRONIC_GRID_MAX)) + 1
        gx = np.linspace(x0 - pad, x1 + pad, nx)
        gy = np.linspace(y0 - pad, y1 + pad, ny)
        gxx, gyy = np.meshgrid(gx, gy, indexing="ij")
        vals = self._electronic_raw(gxx, gyy, bias_v)
        self._frame_map = {"key": key,
                           "interp": RegularGridInterpolator((gx, gy), vals, bounds_error=False,
                                                             fill_value=None)}

    def electronic_height(self, x: np.ndarray, y: np.ndarray, bias_v: float, kappa_m: float,
                          apex_sigma_m: float = 0.0, site: Site | None = None) -> np.ndarray:
        """Apparent-height contribution of the standing waves at the imaging bias.

        Rides in the tip's *atomic* channel (filtered by the apex smearing, not by the
        mesoscopic radius kernel), because it is an LDOS modulation under the apex atom."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        ss = self.surface_state
        if ss is None or abs(bias_v) < 1e-3:
            return np.zeros_like(x)
        fm = self._frame_map
        if fm is not None and abs(fm["key"][4] - round(bias_v, 6)) < 1e-9 \
                and fm["key"][5] == self.scatterer_epoch:
            x0, x1, y0, y1 = fm["key"][:4]
            if (x.size and float(np.min(x)) >= x0 - 3e-9 and float(np.max(x)) <= x1 + 3e-9
                    and float(np.min(y)) >= y0 - 3e-9 and float(np.max(y)) <= y1 + 3e-9):
                acc = fm["interp"](np.column_stack([x.ravel(), y.ravel()])).reshape(x.shape)
                return self._apex_filter(ss.apparent_height(acc, bias_v, kappa_m),
                                         bias_v, apex_sigma_m)
        acc = self._electronic_raw(x, y, bias_v, site)
        return self._apex_filter(ss.apparent_height(acc, bias_v, kappa_m), bias_v, apex_sigma_m)

    def _apex_filter(self, dz: np.ndarray, bias_v: float, apex_sigma_m: float) -> np.ndarray:
        if apex_sigma_m <= 0:
            return dz
        ss = self.surface_state
        k = float(np.max(ss.k_of_e(max(abs(bias_v) / 2.0, 1e-6))))
        if k <= 0:
            return dz
        d_sw = math.pi / k * 1e-9
        return dz * math.exp(-2 * math.pi ** 2 * (apex_sigma_m / d_sw) ** 2)

    def _electronic_raw(self, x: np.ndarray, y: np.ndarray, bias_v: float,
                        site: Site | None = None) -> np.ndarray:
        """The dimensionless interference contrast (before it becomes an apparent height)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        ss = self.surface_state
        if ss is None or abs(bias_v) < 1e-3:
            return np.zeros_like(x)
        s = self.site if site is None else site
        u = self._staircase_u(x, y, s)
        b = s.boundaries
        acc = np.zeros_like(x)
        if b.size:
            idx = np.searchsorted(b, u)
            right = np.where(idx < b.size, b[np.clip(idx, 0, b.size - 1)] - u, np.inf)
            left = np.where(idx > 0, u - b[np.clip(idx - 1, 0, b.size - 1)], np.inf)
            p = ss.p
            fin_r = np.isfinite(right) & (right < p.cutoff_m)
            fin_l = np.isfinite(left) & (left < p.cutoff_m)
            if fin_r.any():
                acc = acc + np.where(fin_r, ss.line_modulation_integrated(
                    np.where(fin_r, right, 1.0), bias_v, p.step_r_up, p.step_phi_up_rad), 0.0)
            if fin_l.any():
                acc = acc + np.where(fin_l, ss.line_modulation_integrated(
                    np.where(fin_l, left, 1.0), bias_v, p.step_r_down, p.step_phi_down_rad), 0.0)
        cx, cy = float(np.mean(x)), float(np.mean(y))
        span = float(np.ptp(x)) + float(np.ptp(y))
        if self.has_corral():
            acc = acc + self._corral_contrast(x, y, bias_v)
        else:
            pts = self.scatterers_near(cx, cy, radius=ss.p.cutoff_m + span)
            if pts:
                acc = acc + ss.point_modulation_integrated(x, y, bias_v, pts)
        return acc

    CORRAL_GRID_M = 0.2e-9

    #: atoms further than this outside the ring take no part in the corral's multiple
    #: scattering: its levels are the ring's, and the rest of the deposit (a field over
    #: hundreds of nm) would only blow up the grid and the scattering matrix
    CORRAL_REACH_M = 10e-9

    def corral_scatterers(self) -> list:
        """The registered adatoms that make up the corral: the ring, the spares by its gap and
        anything else within :data:`CORRAL_REACH_M` of the ring."""
        reg = self.site.adatoms
        if reg is None:
            return []
        pts = reg.scatterers()
        ring = getattr(reg, "ring", None)
        if not ring:
            return pts
        cx, cy = (float(v) * 1e-9 for v in ring["centre_nm"])
        reach = float(ring["radius_nm"]) * 1e-9 + self.CORRAL_REACH_M
        return [p for p in pts if (p.x - cx) ** 2 + (p.y - cy) ** 2 <= reach * reach]

    def _corral_contrast(self, x: np.ndarray, y: np.ndarray, bias_v: float) -> np.ndarray:
        """Bias-integrated multiple scattering, interpolated off a coarse per-frame grid."""
        from scipy.interpolate import RegularGridInterpolator
        ss = self.surface_state
        pts = self.corral_scatterers()
        if not pts or ss is None:
            return np.zeros_like(x)
        if x.size < 256:
            # a point reading (Current.Get, the equilibrium gap) must stay O(1): building the
            # frame map here would put four seconds inside a one-command reply
            return np.asarray(ss.corral_map(x, y, bias_v, pts), float).reshape(x.shape)
        cxs = [p.x for p in pts]
        cys = [p.y for p in pts]
        pad = 5e-9
        box = (min(cxs) - pad, max(cxs) + pad, min(cys) - pad, max(cys) + pad)
        key = (round(bias_v, 6), self.scatterer_epoch, tuple(round(v, 12) for v in box))
        cache = self._corral_map
        if cache is None or cache["key"] != key:
            nx = int(np.clip((box[1] - box[0]) / self.CORRAL_GRID_M, 8, 200)) + 1
            ny = int(np.clip((box[3] - box[2]) / self.CORRAL_GRID_M, 8, 200)) + 1
            gx = np.linspace(box[0], box[1], nx)
            gy = np.linspace(box[2], box[3], ny)
            gxx, gyy = np.meshgrid(gx, gy, indexing="ij")
            s = ss.corral_map(gxx, gyy, bias_v, pts)
            cache = {"key": key, "interp": RegularGridInterpolator((gx, gy), s, bounds_error=False,
                                                                   fill_value=0.0)}
            self._corral_map = cache
        return cache["interp"](np.column_stack([x.ravel(), y.ravel()])).reshape(x.shape)

    def nearest_scatterer(self, x: float, y: float) -> tuple[str, float]:
        """(kind, distance) of the closest thing an electron can scatter off — the ``sts`` /
        ``zspec`` event field a claims judge uses to know where a spectrum was taken."""
        best_kind, best_d = "none", float("inf")
        reg = self.site.adatoms
        if reg is not None:
            atom, d = reg.nearest(x, y)
            if atom is not None and d < best_d:
                best_kind, best_d = "adatom", d
        for f in self.site.features:
            d = math.hypot(f.x - x, f.y - y)
            if d < best_d:
                best_kind, best_d = f.kind, d
        left, right = self._step_distances(x, y)
        for d in (left, right):
            if d is not None and abs(d) < best_d:
                best_kind, best_d = "step", abs(d)
        return best_kind, best_d

    def surface_state_snapshot(self) -> dict | None:
        ss = self.surface_state
        if ss is None:
            return None
        p = ss.p
        k_f = float(ss.k_of_e(0.0))
        pts = self.scatterers_near(0.0, 0.0, radius=200e-9)
        return {"e0_ev": p.e0_ev, "m_star": p.m_star, "gamma_f_ev": p.gamma_f_ev,
                "gamma_0_ev": p.gamma_0_ev, "step_height": p.step_height,
                "step_r_up": p.step_r_up, "step_r_down": p.step_r_down,
                "step_phi_up_rad": p.step_phi_up_rad, "step_phi_down_rad": p.step_phi_down_rad,
                "point_delta_rad": p.point_delta_rad, "point_absorption": p.point_absorption,
                "k_f_per_nm": k_f,
                "half_wavelength_f_nm": (math.pi / k_f) if k_f > 0 else None,
                "l_phi_f_nm": float(ss.l_phi_nm(0.0)),
                "e0_mev": p.e0_ev * 1e3,
                "points": [{"x_nm": s.x * 1e9, "y_nm": s.y * 1e9, "kind": s.kind} for s in pts[:64]]}

    # ── adatoms ──
    def configure_adatoms(self, params, layout: dict) -> None:
        """Build this site's adatom registry and place the scenario's layout."""
        from . import adatoms as ad
        reg = ad.AdatomRegistry(self.site, params)
        self.site.adatoms = reg
        kind = str(layout.get("layout", "single"))
        centre = (0.0, 0.0)
        clear: list[tuple[float, float, float]] = []
        if kind == "corral":
            radius = float(layout.get("ring_radius_nm", 7.13)) * 1e-9
            centre = self._clear_area(radius * 2 + 10e-9)
            ad.build_corral(reg, centre=centre, radius_m=radius,
                            n_atoms=int(layout.get("ring_n", 48)),
                            gap_atoms=int(layout.get("gap_atoms", 0) or 0))
            clear.append((centre[0], centre[1], radius + 8e-9))        # the corral's own room
        else:
            centre = self._clear_area(25e-9)
            d = (float(layout.get("target_dx_nm", 4.0)) * 1e-9, float(layout.get("target_dy_nm", 0.0)) * 1e-9)
            ad.build_single(reg, centre=centre, target_d=d,
                            n_bystanders=int(layout.get("n_bystanders", 3)))
            if reg.target is not None:
                # the placed atom keeps an isolated start and a clear corridor to its goal
                iso = float(layout.get("isolation_nm", 3.0)) * 1e-9 + 0.5e-9
                sx, sy = (v * 1e-9 for v in reg.target["start_nm"])
                clear += [(sx + f * d[0], sy + f * d[1], iso) for f in (0.0, 0.5, 1.0)]
                if str(layout.get("target", "")) == "any":
                    # the task names no atom: any isolated one moved by d onto its lattice site
                    # counts; the placed one is only an example that is guaranteed to exist
                    reg.target = {"rule": "any", "atom_id": reg.target["atom_id"],
                                  "d_nm": [d[0] * 1e9, d[1] * 1e9],
                                  "isolation_nm": float(layout.get("isolation_nm", 3.0)),
                                  "bystander_nm": float(layout.get("bystander_nm", 10.0))}
        per100 = float(layout.get("field_per_100nm2", 0.0) or 0.0)
        if per100 > 0:
            ad.scatter_field(reg, density_per_nm2=per100 / 100.0,
                             half_m=float(layout.get("field_half_nm", 150.0)) * 1e-9, clear=tuple(clear))
        self.invalidate_electronic_cache()

    #: how far from the origin a layout may be put. The task text promises the agent where to
    #: look, and a spot outside this would break that promise — an agent that surveyed exactly
    #: where it was told would find nothing. Scenarios that need a clear area this close raise
    #: ``terrace_median_nm``, which is what annealing a crystal does.
    CLEAR_AREA_REACH_M = 30e-9

    def _clear_area(self, size_m: float) -> tuple[float, float]:
        """A step-free spot near the origin, with the Poisson adsorbates inside it removed."""
        cx, cy = 0.0, 0.0
        n = int((self.CLEAR_AREA_REACH_M / 5e-9) ** 2)
        for k in range(n):
            ang = 2.399963 * k                       # golden-angle spiral
            r = 5e-9 * math.sqrt(k)
            x, y = r * math.cos(ang), r * math.sin(ang)
            if self.step_free_window(x, y, size_m, n=21):
                cx, cy = x, y
                break
        half = size_m / 2
        self.site.features = [f for f in self.site.features
                              if f.kind != "adsorbate"
                              or abs(f.x - cx) > half or abs(f.y - cy) > half]
        return cx, cy

    def adatoms_snapshot(self) -> dict | None:
        reg = self.site.adatoms
        return reg.snapshot() if reg is not None else None

    def corral_snapshot(self) -> dict | None:
        """Ring geometry, occupancy and the resonance energies at its centre (P3 truth)."""
        reg = self.site.adatoms
        if reg is None or reg.ring is None or self.surface_state is None:
            return None
        cache_key = (self.scatterer_epoch, "corral")
        cached = self._ldos_cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        centre = (reg.ring["centre_nm"][0] * 1e-9, reg.ring["centre_nm"][1] * 1e-9)
        ss = self.surface_state
        lo = ss.p.e0_ev + 0.02
        grid = np.arange(lo, 0.6, 0.002)
        rho = self.ldos_at(*centre)
        y = np.asarray(rho.rho(grid), float)
        peaks = _find_peaks(grid, y, prominence_frac=0.08)
        out = {"centre_nm": reg.ring["centre_nm"], "radius_nm": reg.ring["radius_nm"],
               "n_sites": reg.ring["n_sites"], "ring_sites": reg.ring["sites"],
               "ring_occupancy": reg.ring_occupancy(),
               "gap_sites": reg.ring.get("gap_sites", []),
               "spares_remaining": sum(1 for a in reg.atoms.values()
                                       if a.role == "spare" and a.status == "on_surface"),
               "peaks_ev": [float(p[0]) for p in peaks],
               "peaks_mev": [float(p[0]) * 1e3 for p in peaks],
               "peak_prominence": [float(p[1]) for p in peaks],
               "hard_wall_ev": _hard_wall_levels(ss.p.e0_ev, ss.p.m_star,
                                                 reg.ring["radius_nm"] * 1e-9),
               "alpha": ss.p.point_absorption, "delta_rad": ss.p.point_delta_rad}
        self._ldos_cache[cache_key] = out
        return dict(out)

    def corral_centre_distance_nm(self, x: float, y: float) -> float | None:
        reg = self.site.adatoms
        if reg is None or reg.ring is None:
            return None
        cx, cy = reg.ring["centre_nm"]
        return math.hypot(x * 1e9 - cx, y * 1e9 - cy)


def _find_peaks(x: np.ndarray, y: np.ndarray, *, prominence_frac: float = 0.08):
    from scipy.signal import find_peaks
    span = float(np.ptp(y))
    if span <= 0:
        return []
    idx, props = find_peaks(y, prominence=prominence_frac * span)
    return [(float(x[i]), float(p) / span) for i, p in zip(idx, props["prominences"])]


def _hard_wall_levels(e0_ev: float, m_star: float, radius_m: float, n: int = 6) -> list[float]:
    """E_n = E0 + (ħ²/2m*)(j_{0,n}/R)² — the textbook circular box, for comparison only."""
    from scipy.special import jn_zeros
    from .surface_state import HBAR2_OVER_2M_EV_NM2
    r_nm = radius_m * 1e9
    if r_nm <= 0:
        return []
    return [float(e0_ev + (HBAR2_OVER_2M_EV_NM2 / m_star) * (z / r_nm) ** 2)
            for z in jn_zeros(0, n)]
