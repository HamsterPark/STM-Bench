"""Adsorbed atoms on lattice sites, and moving them with the tip.

The model represents the imaging/manipulation distinction described by Eigler and Schweizer
(1990): lowering
the junction resistance can make an atom follow the tip, while a higher resistance supports
imaging without intentional displacement. The operating regime is set by ``R = |V| / I``.

* Each atom occupies an fcc hollow site of the substrate lattice — a registry entry, not a
  height bump that happens to be somewhere. Moving one changes its ``(i, j)``; the image, the
  surface-state scattering and the truth all read the same registry.
* A lateral move (``FolMe`` in controller terms) is integrated along the path: while the tip is
  within ``capture_radius`` of an atom, the atom hops to the neighbouring site nearest the
  tip's next position with probability ``p_follow(R, v)``, which falls off as ``(R/R_th)⁸``
  and as ``(v/v_max)²``. Below ``R_pick`` the atom transfers to the tip instead — and a tip
  carrying an atom images and performs spectroscopy with the corresponding altered tip state.
* Imaging is the same physics with a fast tip: at 171 nm/s the follow probability is
  negligible, so a corral survives being scanned at 10 mV / 1 nA, and gets destroyed by
  scanning at 10 mV / 50 nA. That asymmetry is the point of the task.
* **The tip decides how well it grabs** (:func:`grip_of`). A blunt apex needs a lower
  resistance, grabs from further off and now and then pushes the atom onto the wrong
  neighbour; every apex of a multiple tip grabs on its own, so a double tip can drag an atom
  that is not under the selected apex; a metastable or flickering apex loses its
  grip at random; an atom on the apex changes the threshold.
* **No two atoms are alike.** Each carries its own threshold and grab-radius factor, and an
  atom next to a step edge or crowded by another atom holds on harder.
* **A field**: the rest of the deposit — a sparse random coverage on lattice sites over a
  few hundred nanometres (:func:`scatter_field`), so measurements can continue outside a
  locally disturbed area.

RNG discipline: layout, dynamics, per-atom traits and the field each have their own stream,
all derived from the site seed, so nothing here moves the adsorbate positions or the tip
trajectory of an existing scenario, and adding the field moves none of the placed atoms.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

LAYOUT_STREAM = 0xADA0
DYNAMICS_STREAM = 0xADA1
TRAITS_STREAM = 0xADA2
FIELD_STREAM = 0xADA3
PATH_STEP_M = 0.05e-9           # path sampling step for the drag integration
HOP_EXPONENT = 8.0
SPEED_EXPONENT = 2.0
#: atom-to-atom spread (lognormal σ) of the pull threshold and of the grab radius
TRAIT_SIGMA_R = 0.18
TRAIT_SIGMA_CAP = 0.10
#: an atom within this of a step edge, or of another atom, is held harder: its threshold ×
STEP_NEAR_M, STEP_FACTOR = 1.0e-9, 0.6
CROWD_NEAR_M, CROWD_FACTOR = 0.6e-9, 0.8
#: a moved atom that is lost or on the tip counts as having moved this far (bystander check)
GONE_SHIFT_NM = 99.0


@dataclass(frozen=True)
class GripPoint:
    """One apex that can grab: its lateral offset from the tip position, and how much higher
    the junction resistance looks from it (a shorter or lighter apex sees a larger gap)."""
    dx: float = 0.0
    dy: float = 0.0
    r_factor: float = 1.0


@dataclass(frozen=True)
class Grip:
    """How a tip grabs adatoms. ``Grip()`` is the ideal single, sharp apex."""
    points: tuple[GripPoint, ...] = (GripPoint(),)
    threshold_scale: float = 1.0     # × every atom's pull and pick-up thresholds
    capture_scale: float = 1.0       # × every atom's grab radius
    fumble: float = 0.0              # chance a hop attempt slips out of an unsteady apex
    jitter: float = 0.0              # chance a hop lands on the second-best neighbour


#: apexes seeing more than this many times the junction resistance cannot manipulate
GRIP_R_FACTOR_MAX = 8.0


def grip_of(tip, kappa_m: float) -> Grip:
    """The grip of a :class:`~stmsim.physics.tip.Tip` (duck-typed: apexes, radius_m,
    flicker_dz_m, metastable, carried, dead).

    * apexes: the one nearest the surface leads; another reaches the atoms with the junction
      resistance multiplied by ``exp(2κ·Δz)`` for being Δz further away, and by the weight ratio;
    * a radius over 2.5 nm lowers the thresholds (``(2.5/R)^0.8``), widens the grab (up to ×1.6)
      and makes hops wander (``jitter`` up to 0.35);
    * a flickering apex fumbles 35 % of hop attempts, a metastable one 12 %;
    * an atom on the apex: thresholds × 0.7; a dead tip hardly grabs at all."""
    apexes = list(getattr(tip, "apexes", None) or [])
    if not apexes:
        points = (GripPoint(),)
    else:
        def reach(a):
            return a.dz + math.log(max(a.w, 1e-9)) / (2.0 * kappa_m)
        lead = max(apexes, key=reach)
        pts = []
        for a in apexes:
            rf = math.exp(2.0 * kappa_m * (reach(lead) - reach(a)))
            if rf <= GRIP_R_FACTOR_MAX:
                pts.append(GripPoint(float(a.dx), float(a.dy), float(rf)))
        points = tuple(pts) or (GripPoint(float(lead.dx), float(lead.dy), 1.0),)
    r_nm = float(getattr(tip, "radius_m", 1e-9)) * 1e9
    over = max(0.0, r_nm - 2.5)
    thr = min(1.0, (2.5 / max(r_nm, 2.5)) ** 0.8)
    if getattr(tip, "carried", None):
        thr *= 0.7
    if getattr(tip, "dead", False):
        thr *= 0.05
    fumble = 0.35 if float(getattr(tip, "flicker_dz_m", 0.0) or 0.0) > 0 else (0.12 if getattr(tip, "metastable", False) else 0.0)
    return Grip(points=points, threshold_scale=thr, capture_scale=min(1.6, 1.0 + 0.08 * over),
                fumble=fumble, jitter=min(0.35, 0.06 * over))


@dataclass(frozen=True)
class AdatomParams:
    species: str = "Fe"
    height_m: float = 70e-12
    sigma_m: float = 0.30e-9
    delta_rad: float = 1.2
    alpha: float = 0.45
    r_threshold_ohm: float = 200e3
    r_pick_ohm: float = 30e3
    v_max_m_s: float = 1.0e-9
    capture_radius_m: float = 0.45e-9
    p_slip: float = 0.005
    p_slip_scan: float = 0.05

    @classmethod
    def from_yaml(cls, cfg: dict[str, Any]) -> "AdatomParams":
        d: dict[str, Any] = {}
        if "species" in cfg:
            d["species"] = str(cfg["species"])
        for src, dst, scale in (("height_pm", "height_m", 1e-12), ("sigma_nm", "sigma_m", 1e-9),
                                ("delta_rad", "delta_rad", 1.0), ("alpha", "alpha", 1.0),
                                ("r_threshold_kohm", "r_threshold_ohm", 1e3),
                                ("r_pick_kohm", "r_pick_ohm", 1e3),
                                ("v_max_nm_s", "v_max_m_s", 1e-9),
                                ("capture_radius_nm", "capture_radius_m", 1e-9),
                                ("p_slip", "p_slip", 1.0), ("p_slip_scan", "p_slip_scan", 1.0)):
            if src in cfg and cfg[src] is not None:
                d[dst] = float(cfg[src]) * scale
        return cls(**d)


@dataclass
class Adatom:
    id: int
    species: str
    i: int
    j: int
    sub: int = 0                       # 0 = fcc (the only one used); 1 = hcp, reserved
    born_sim_s: float = 0.0
    role: str = ""                     # ring | spare | target | bystander | field | ""
    status: str = "on_surface"         # on_surface | on_tip | lost
    x0: float = 0.0                    # birth position (sample frame) — bystander shift truth
    y0: float = 0.0
    history: list[tuple] = field(default_factory=list)   # (sim_s, i, j, cause)
    r_scale: float = 1.0               # this atom's pull / pick-up threshold factor
    cap_scale: float = 1.0             # this atom's grab-radius factor

    @property
    def n_hops(self) -> int:
        return len(self.history)


class AdatomRegistry:
    """Which lattice sites are occupied on one coarse-motion site, and what moves them."""

    def __init__(self, site, params: AdatomParams):
        self.site = site
        self.params = params
        a = float(site.material.nn_m)
        th = float(site.lattice_angle) + math.pi / 6      # nearest-neighbour direction
        self.a1 = np.array([a * math.cos(th), a * math.sin(th)])
        self.a2 = np.array([a * math.cos(th + math.pi / 3), a * math.sin(th + math.pi / 3)])
        self.offset = (self.a1 + self.a2) / 3.0           # atop → fcc hollow
        self._B = np.column_stack([self.a1, self.a2])
        self._Binv = np.linalg.inv(self._B)
        self.rng_layout = np.random.default_rng([int(site.seed), LAYOUT_STREAM])
        self.rng_dyn = np.random.default_rng([int(site.seed), DYNAMICS_STREAM])
        self.rng_traits = np.random.default_rng([int(site.seed), TRAITS_STREAM])
        self.atoms: dict[int, Adatom] = {}
        self.occ: dict[tuple[int, int, int], int] = {}
        self.epoch = 0
        self.carried: Adatom | None = None
        self.ring: dict | None = None
        self.target: dict | None = None
        self._next_id = 1
        self._cache: tuple[int, list, np.ndarray] | None = None   # (epoch, atoms on surface, xy)

    # ── lattice geometry ──
    def site_xy(self, i: int, j: int, sub: int = 0) -> tuple[float, float]:
        p = i * self.a1 + j * self.a2 + self.offset * (1 if sub == 0 else 2)
        return float(p[0]), float(p[1])

    def site_of(self, x: float, y: float, sub: int = 0) -> tuple[int, int]:
        """Nearest lattice site. The oblique basis makes rounding the wrong answer, so the
        four surrounding cells are compared in real space."""
        rel = np.array([x, y]) - self.offset * (1 if sub == 0 else 2)
        uv = self._Binv @ rel
        best, best_d = (0, 0), float("inf")
        for di in (0, 1):
            for dj in (0, 1):
                i, j = int(math.floor(uv[0])) + di, int(math.floor(uv[1])) + dj
                px, py = self.site_xy(i, j, sub)
                d = (px - x) ** 2 + (py - y) ** 2
                if d < best_d:
                    best, best_d = (i, j), d
        return best

    def neighbours(self, i: int, j: int) -> list[tuple[int, int]]:
        return [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1), (i + 1, j - 1), (i - 1, j + 1)]

    # ── population ──
    @property
    def n_on_surface(self) -> int:
        return len(self._arrays()[0])

    def add(self, i: int, j: int, *, role: str = "", sub: int = 0, sim_s: float = 0.0) -> Adatom | None:
        if (i, j, sub) in self.occ:
            return None
        x, y = self.site_xy(i, j, sub)
        # traits come from their own stream, in creation order: which atom is which never
        # depends on how many draws the layout made
        r_scale = float(math.exp(self.rng_traits.normal(0.0, TRAIT_SIGMA_R)))
        cap_scale = float(math.exp(self.rng_traits.normal(0.0, TRAIT_SIGMA_CAP)))
        atom = Adatom(id=self._next_id, species=self.params.species, i=i, j=j, sub=sub,
                      born_sim_s=sim_s, role=role, x0=x, y0=y, r_scale=r_scale, cap_scale=cap_scale)
        self._next_id += 1
        self.atoms[atom.id] = atom
        self.occ[(i, j, sub)] = atom.id
        self.epoch += 1
        return atom

    def _arrays(self) -> tuple[list, np.ndarray]:
        """The atoms on the surface and their positions (N×2, m), rebuilt only when the
        registry changed — a field of a few hundred atoms is read on every scan line."""
        if self._cache is None or self._cache[0] != self.epoch:
            on = [a for a in self.atoms.values() if a.status == "on_surface"]
            xy = np.array([self.site_xy(a.i, a.j, a.sub) for a in on], float).reshape(-1, 2)
            self._cache = (self.epoch, on, xy)
        return self._cache[1], self._cache[2]

    def remove_within(self, x: float, y: float, radius: float) -> list[int]:
        on, xy = self._arrays()
        if not on:
            return []
        hit = np.flatnonzero((xy[:, 0] - x) ** 2 + (xy[:, 1] - y) ** 2 <= radius * radius)
        gone = []
        for k in hit:
            atom = on[int(k)]
            self.occ.pop((atom.i, atom.j, atom.sub), None)
            atom.status = "lost"
            gone.append(atom.id)
        if gone:
            self.epoch += 1
        return gone

    def nearest(self, x: float, y: float, radius: float | None = None) -> tuple[Adatom | None, float]:
        on, xy = self._arrays()
        if not on:
            return None, float("inf")
        d = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
        k = int(np.argmin(d))
        best_d = float(d[k])
        if radius is not None and best_d > radius:
            return None, best_d
        return on[k], best_d

    def positions(self) -> np.ndarray:
        return self._arrays()[1]

    def grab(self, x: float, y: float, grip: Grip | None = None) -> tuple[Adatom | None, GripPoint | None]:
        """The atom one of the tip's apexes has hold of with the tip at (x, y): the one
        deepest inside its own grab radius (``capture_radius`` × the atom's and the tip's
        factors), or ``(None, None)``."""
        grip = grip or Grip()
        on, xy = self._arrays()
        if not on:
            return None, None
        cap = self.params.capture_radius_m * grip.capture_scale * np.array([a.cap_scale for a in on])
        best, best_q, best_p = None, 1.0, None
        for gp in grip.points:
            q = np.hypot(xy[:, 0] - (x + gp.dx), xy[:, 1] - (y + gp.dy)) / cap
            k = int(np.argmin(q))
            if float(q[k]) <= best_q:
                best, best_q, best_p = on[k], float(q[k]), gp
        return best, best_p

    # ── rendering ──
    def height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=float)
        pts = self.positions()
        if not pts.size:
            return out
        s = self.params.sigma_m
        r = 4.0 * s
        x0, x1 = float(np.min(x)) - r, float(np.max(x)) + r
        y0, y1 = float(np.min(y)) - r, float(np.max(y)) + r
        m = ((pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1))
        for px, py in pts[m]:
            out += self.params.height_m * np.exp(-0.5 * (((x - px) ** 2 + (y - py) ** 2) / s ** 2))
        return out

    def scatterers(self):
        from .surface_state import PointScatterer
        out = []
        for atom in self.atoms.values():
            if atom.status != "on_surface":
                continue
            ax, ay = self.site_xy(atom.i, atom.j, atom.sub)
            out.append(PointScatterer(ax, ay, self.params.delta_rad, self.params.alpha,
                                      kind="adatom", ident=atom.id))
        return out

    # ── manipulation ──
    def hold_factor(self, atom: Adatom | None, terrace=None) -> float:
        """How much lower than the nominal the resistance has to be to move this atom: its
        own trait, and harder next to a step edge or another atom."""
        if atom is None:
            return 1.0
        f = atom.r_scale
        x, y = self.site_xy(atom.i, atom.j, atom.sub)
        if terrace is not None:
            d = STEP_NEAR_M
            lv = terrace(np.array([x, x + d, x - d, x, x]), np.array([y, y, y, y + d, y - d]))
            if len({int(v) for v in lv}) > 1:
                f *= STEP_FACTOR
        on, xy = self._arrays()
        if len(on) > 1:
            dd = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
            if int(np.count_nonzero(dd <= CROWD_NEAR_M)) > 1:       # itself plus a neighbour
                f *= CROWD_FACTOR
        return float(f)

    def p_follow(self, r_ohm: float, v_m_s: float, atom: Adatom | None = None, grip: Grip | None = None,
                 terrace=None) -> float:
        """Chance one hop attempt carries the atom along: the resistance against this atom's
        threshold (with the tip's scale), the speed against ``v_max``, and the tip's fumbles."""
        p = self.params
        grip = grip or Grip()
        r_th = p.r_threshold_ohm * grip.threshold_scale * self.hold_factor(atom, terrace)
        a = 1.0 / (1.0 + (max(r_ohm, 1.0) / max(r_th, 1.0)) ** HOP_EXPONENT)
        b = 1.0 / (1.0 + (max(v_m_s, 0.0) / p.v_max_m_s) ** SPEED_EXPONENT)
        return float(a * b * (1.0 - grip.fumble))

    def p_pick(self, r_ohm: float, atom: Adatom | None = None, grip: Grip | None = None) -> float:
        r_pick = self.params.r_pick_ohm * (grip or Grip()).threshold_scale * (atom.r_scale if atom else 1.0)
        return float(1.0 / (1.0 + (max(r_ohm, 1.0) / max(r_pick, 1.0)) ** HOP_EXPONENT))

    def _hop(self, atom: Adatom, toward: tuple[float, float], *, sim_s: float, cause: str,
             terrace=None, blocked: Callable[[float, float], bool] | None = None, jitter: float = 0.0) -> bool:
        """One site hop toward a point. False when no free neighbour is closer to it (an
        occupied site, debris on the site, a step edge). With ``jitter`` the atom now and then
        lands on the second-best neighbour — a blunt apex pushes as much as it pulls."""
        cx, cy = self.site_xy(atom.i, atom.j, atom.sub)
        here_d = (cx - toward[0]) ** 2 + (cy - toward[1]) ** 2
        options = []
        for i, j in self.neighbours(atom.i, atom.j):
            if (i, j, atom.sub) in self.occ:
                continue
            px, py = self.site_xy(i, j, atom.sub)
            if blocked is not None and blocked(px, py):
                continue
            options.append(((px - toward[0]) ** 2 + (py - toward[1]) ** 2, (i, j), (px, py)))
        options.sort()
        options = [o for o in options if o[0] < here_d]                # only hops that follow
        if not options:
            return False
        pick = options[0]
        if jitter > 0 and len(options) > 1 and self.rng_dyn.random() < jitter:
            pick = options[1]
        _, best, (nx, ny) = pick
        if terrace is not None:
            a = terrace(np.array([cx]), np.array([cy]))
            b = terrace(np.array([nx]), np.array([ny]))
            if int(a[0]) != int(b[0]):
                return False                                 # atoms do not climb steps
        self.occ.pop((atom.i, atom.j, atom.sub), None)
        atom.i, atom.j = best
        self.occ[(atom.i, atom.j, atom.sub)] = atom.id
        atom.history.append((float(sim_s), atom.i, atom.j, cause))
        self.epoch += 1
        return True

    def pick_up(self, atom: Adatom, *, sim_s: float) -> None:
        self.occ.pop((atom.i, atom.j, atom.sub), None)
        atom.status = "on_tip"
        atom.history.append((float(sim_s), atom.i, atom.j, "pickup"))
        self.carried = atom
        self.epoch += 1

    def drop(self, x: float, y: float, *, sim_s: float, cause: str = "approach") -> Adatom | None:
        atom = self.carried
        if atom is None:
            return None
        i, j = self.site_of(x, y, atom.sub)
        if (i, j, atom.sub) in self.occ:
            for ni, nj in self.neighbours(i, j):
                if (ni, nj, atom.sub) not in self.occ:
                    i, j = ni, nj
                    break
            else:
                return None
        atom.i, atom.j = i, j
        atom.status = "on_surface"
        self.occ[(i, j, atom.sub)] = atom.id
        atom.history.append((float(sim_s), i, j, cause))
        self.carried = None
        self.epoch += 1
        return atom

    def drag(self, start: tuple[float, float], end: tuple[float, float], *, r_ohm: float,
             v_m_s: float, sim_s: float, terrace=None, cause: str = "folme", grip: Grip | None = None,
             blocked: Callable[[float, float], bool] | None = None) -> list[dict]:
        """Integrate one tip path; returns aggregated events (one per atom touched).

        Whichever apex of ``grip`` gets an atom inside its grab radius first holds it; the atom
        then follows that apex (offset and all) hop by hop, each hop with this atom's own
        ``p_follow`` at the resistance that apex sees."""
        p = self.params
        grip = grip or Grip()
        x0, y0 = start
        x1, y1 = end
        length = math.hypot(x1 - x0, y1 - y0)
        if length <= 0:
            return []
        n = max(2, int(length / PATH_STEP_M) + 1)
        ts = np.linspace(0.0, 1.0, n)
        xs = x0 + (x1 - x0) * ts
        ys = y0 + (y1 - y0) * ts
        events: list[dict] = []
        attached: Adatom | None = None
        held_by: GripPoint = grip.points[0]
        moved: dict[int, dict] = {}
        nn = float(self.site.material.nn_m)
        for idx in range(n):
            tx, ty = float(xs[idx]), float(ys[idx])
            if self.carried is not None:
                continue
            if attached is None:
                cand, gp = self.grab(tx, ty, grip)
                if cand is None:
                    continue
                r_eff = r_ohm * gp.r_factor
                p_p = self.p_pick(r_eff, cand, grip)
                if p_p > 1e-6 and self.rng_dyn.random() < p_p:
                    ax, ay = self.site_xy(cand.i, cand.j, cand.sub)
                    self.pick_up(cand, sim_s=sim_s)
                    events.append({"kind": "adatom_picked", "atom_id": cand.id,
                                   "r_kohm": r_eff / 1e3, "xy_nm": [ax * 1e9, ay * 1e9]})
                    return events
                attached, held_by = cand, gp
                if cand.id not in moved:
                    ax, ay = self.site_xy(cand.i, cand.j, cand.sub)
                    moved[cand.id] = {"kind": "adatom_hop", "atom_id": cand.id, "cause": cause,
                                      "n_hops": 0, "from_site": [cand.i, cand.j],
                                      "from_xy_nm": [ax * 1e9, ay * 1e9],
                                      "r_kohm": r_ohm / 1e3, "v_nm_s": v_m_s * 1e9}
                continue
            ax, ay = self.site_xy(attached.i, attached.j, attached.sub)
            hx, hy = tx + held_by.dx, ty + held_by.dy                  # where the holding apex is
            if math.hypot(hx - ax, hy - ay) <= 0.6 * nn:
                continue
            nxt = (float(xs[min(idx + 1, n - 1)]) + held_by.dx, float(ys[min(idx + 1, n - 1)]) + held_by.dy)
            p_f = self.p_follow(r_ohm * held_by.r_factor, v_m_s, attached, grip, terrace)
            if self.rng_dyn.random() >= p_f or not self._hop(attached, nxt, sim_s=sim_s, cause=cause,
                                                             terrace=terrace, blocked=blocked,
                                                             jitter=grip.jitter):
                attached = None                              # dropped behind
                continue
            moved[attached.id]["n_hops"] += 1
            if self.rng_dyn.random() < p.p_slip:
                attached = None
        for atom_id, rec in moved.items():
            atom = self.atoms[atom_id]
            ax, ay = self.site_xy(atom.i, atom.j, atom.sub)
            rec["to_site"] = [atom.i, atom.j]
            rec["to_xy_nm"] = [ax * 1e9, ay * 1e9]
            if rec["n_hops"]:
                events.append(rec)
        return events

    def scan_row(self, p0: tuple[float, float], p1: tuple[float, float], *, r_ohm: float,
                 v_m_s: float, sim_s: float, grip: Grip | None = None,
                 blocked: Callable[[float, float], bool] | None = None, terrace=None) -> list[dict]:
        """Imaging pass over one line: mostly nothing, unless R is a manipulation resistance.
        Every apex of ``grip`` sweeps its own copy of the line."""
        p = self.params
        grip = grip or Grip()
        worst = max(a.r_scale for a in self.atoms.values()) if self.atoms else 1.0
        ceiling = 2.0 * max(p.r_threshold_ohm, p.r_pick_ohm) * grip.threshold_scale * worst
        if r_ohm * min(gp.r_factor for gp in grip.points) >= ceiling:
            return []
        events: list[dict] = []
        ux, uy = p1[0] - p0[0], p1[1] - p0[1]
        length = math.hypot(ux, uy)
        if length <= 0:
            return []
        ux, uy = ux / length, uy / length
        on, xy = self._arrays()
        reach = p.capture_radius_m * grip.capture_scale * 1.5
        swept: list[tuple[Adatom, GripPoint]] = []
        seen: set[int] = set()
        for gp in grip.points:
            q0 = (p0[0] + gp.dx, p0[1] + gp.dy)
            t = (xy[:, 0] - q0[0]) * ux + (xy[:, 1] - q0[1]) * uy if len(on) else np.zeros(0)
            perp = np.abs(-(xy[:, 0] - q0[0]) * uy + (xy[:, 1] - q0[1]) * ux) if len(on) else np.zeros(0)
            for k in np.flatnonzero((t >= 0) & (t <= length) & (perp <= reach)):
                atom = on[int(k)]
                cap = p.capture_radius_m * grip.capture_scale * atom.cap_scale
                if float(perp[k]) <= cap and atom.id not in seen:
                    seen.add(atom.id)
                    swept.append((atom, gp))
        for atom, gp in swept:
            if atom.status != "on_surface":
                continue
            ax, ay = self.site_xy(atom.i, atom.j, atom.sub)
            r_eff = r_ohm * gp.r_factor
            p_p = self.p_pick(r_eff, atom, grip)
            p_f = self.p_follow(r_eff, v_m_s, atom, grip, terrace) * (1.0 - p.p_slip_scan)
            if p_p > 1e-6 and self.carried is None and self.rng_dyn.random() < p_p:
                self.pick_up(atom, sim_s=sim_s)
                events.append({"kind": "adatom_picked", "atom_id": atom.id, "cause": "scan",
                               "r_kohm": r_eff / 1e3, "xy_nm": [ax * 1e9, ay * 1e9]})
                continue
            if p_f <= 1e-9:
                continue
            hops = 0
            while self.rng_dyn.random() < p_f and hops < 12:
                toward = (ax + ux * (hops + 1) * float(self.site.material.nn_m),
                          ay + uy * (hops + 1) * float(self.site.material.nn_m))
                if not self._hop(atom, toward, sim_s=sim_s, cause="scan", terrace=terrace,
                                 blocked=blocked, jitter=grip.jitter):
                    break
                hops += 1
            if hops:
                nx, ny = self.site_xy(atom.i, atom.j, atom.sub)
                events.append({"kind": "adatom_hop", "atom_id": atom.id, "cause": "scan",
                               "n_hops": hops, "from_xy_nm": [ax * 1e9, ay * 1e9],
                               "to_xy_nm": [nx * 1e9, ny * 1e9], "r_kohm": r_ohm / 1e3,
                               "v_nm_s": v_m_s * 1e9})
        return events

    # ── truth ──
    def any_target(self) -> dict:
        """Truth for "move any isolated atom by d to a lattice site": the rule, plus every atom
        that moved at all with the site d from where it was born (its goal), whether it sits
        there now, whether it was isolated when born, and how far the atoms born within
        ``bystander_nm`` of its start or goal have moved (a lost or picked-up one counts as
        :data:`GONE_SHIFT_NM`). The judge picks the candidate the report points at."""
        t = self.target or {}
        dx, dy = (float(v) * 1e-9 for v in t.get("d_nm", (0.0, 0.0)))
        iso = float(t.get("isolation_nm", 3.0)) * 1e-9
        by_r = float(t.get("bystander_nm", 10.0)) * 1e-9
        atoms = list(self.atoms.values())
        births = np.array([(a.x0, a.y0) for a in atoms], float).reshape(-1, 2)

        def shift_nm(a: Adatom) -> float:
            if a.status != "on_surface":
                return GONE_SHIFT_NM
            x, y = self.site_xy(a.i, a.j, a.sub)
            return math.hypot(x - a.x0, y - a.y0) * 1e9

        cands = []
        for a in atoms:
            if not a.history:
                continue
            gi, gj = self.site_of(a.x0 + dx, a.y0 + dy, a.sub)
            gx, gy = self.site_xy(gi, gj, a.sub)
            d_birth = np.hypot(births[:, 0] - a.x0, births[:, 1] - a.y0)
            d_goal = np.hypot(births[:, 0] - gx, births[:, 1] - gy)
            others = [atoms[k] for k in np.flatnonzero((d_birth <= by_r) | (d_goal <= by_r)) if atoms[k] is not a]
            now = self.site_xy(a.i, a.j, a.sub) if a.status == "on_surface" else None
            cands.append({"id": a.id, "role": a.role, "status": a.status, "n_hops": a.n_hops,
                          "start_nm": [a.x0 * 1e9, a.y0 * 1e9],
                          "now_nm": [now[0] * 1e9, now[1] * 1e9] if now else None,
                          "goal_site": [gi, gj], "goal_nm": [gx * 1e9, gy * 1e9],
                          "on_goal": bool(now is not None and (a.i, a.j) == (gi, gj)),
                          "isolated": bool(np.count_nonzero(d_birth <= iso) <= 1),
                          "n_bystanders": len(others),
                          "bystander_max_shift_nm": max((shift_nm(o) for o in others), default=0.0)})
        ex = self.atoms.get(t.get("atom_id", -1))
        example = None
        if ex is not None:
            gi, gj = self.site_of(ex.x0 + dx, ex.y0 + dy, ex.sub)
            gx, gy = self.site_xy(gi, gj, ex.sub)
            example = {"atom_id": ex.id, "start_nm": [ex.x0 * 1e9, ex.y0 * 1e9], "goal_nm": [gx * 1e9, gy * 1e9]}
        return {"rule": "any", "d_nm": [dx * 1e9, dy * 1e9], "isolation_nm": iso * 1e9,
                "bystander_nm": by_r * 1e9, "example": example, "candidates": cands}

    def snapshot(self) -> dict:
        sites = []
        for atom in self.atoms.values():
            ax, ay = self.site_xy(atom.i, atom.j, atom.sub)
            sites.append({"id": atom.id, "species": atom.species, "role": atom.role,
                          "status": atom.status, "x_nm": ax * 1e9, "y_nm": ay * 1e9,
                          "i": atom.i, "j": atom.j, "sublattice": atom.sub,
                          "n_hops": atom.n_hops,
                          "shift_nm": math.hypot(ax - atom.x0, ay - atom.y0) * 1e9,
                          "x0_nm": atom.x0 * 1e9, "y0_nm": atom.y0 * 1e9,
                          "r_scale": round(atom.r_scale, 4), "cap_scale": round(atom.cap_scale, 4)})
        by = [a for a in self.atoms.values() if a.role == "bystander"]
        shift = 0.0
        for atom in by:
            ax, ay = self.site_xy(atom.i, atom.j, atom.sub)
            shift = max(shift, math.hypot(ax - atom.x0, ay - atom.y0) * 1e9)
        out = {"n": len(self.atoms), "n_on_surface": self.n_on_surface, "epoch": self.epoch,
               "lattice": {"a_nm": float(self.site.material.nn_m) * 1e9,
                           "angle_deg": math.degrees(float(self.site.lattice_angle)),
                           "origin_nm": [float(self.offset[0]) * 1e9, float(self.offset[1]) * 1e9]},
               "params": {"height_pm": self.params.height_m * 1e12,
                          "sigma_nm": self.params.sigma_m * 1e9,
                          "delta_rad": self.params.delta_rad, "alpha": self.params.alpha,
                          "r_threshold_kohm": self.params.r_threshold_ohm / 1e3,
                          "r_pick_kohm": self.params.r_pick_ohm / 1e3,
                          "v_max_nm_s": self.params.v_max_m_s * 1e9,
                          "capture_radius_nm": self.params.capture_radius_m * 1e9},
               "sites": sites,
               "carried": self.carried.id if self.carried is not None else None,
               "bystander_ids": [a.id for a in by],
               "bystander_max_shift_nm": shift,
               "n_hops_total": sum(a.n_hops for a in self.atoms.values()),
               "n_picked": sum(1 for a in self.atoms.values()
                               if any(h[3] == "pickup" for h in a.history)),
               "n_lost": sum(1 for a in self.atoms.values() if a.status == "lost")}
        if self.target is not None and self.target.get("rule") == "any":
            out["target"] = self.any_target()
        elif self.target is not None:
            t = dict(self.target)
            atom = self.atoms.get(t.get("atom_id", -1))
            if atom is not None:
                ax, ay = self.site_xy(atom.i, atom.j, atom.sub)
                t["atom_now_nm"] = [ax * 1e9, ay * 1e9]
                t["on_target"] = bool((atom.i, atom.j) == tuple(t["site"]) and atom.status == "on_surface")
            out["target"] = t
        if self.ring is not None:
            out["ring"] = dict(self.ring)
        return out

    #: a ring site counts as held when an atom sits within this fraction of the spacing
    #: between neighbouring ring sites — i.e. closer to it than to the next site
    RING_SITE_TOL = 0.5

    def ring_occupancy(self) -> float | None:
        """Fraction of the designed ring sites that hold an atom.

        "Hold" is geometric, not by lattice index: an atom within half a site spacing of
        the designed position counts. The confinement that sets the corral's levels comes
        from atoms sitting on the ring. Closing a gap on the hollow next to the designed one
        therefore closes the gap geometrically, and levels are recomputed from the actual
        atom positions. In the 2026-09-11 P3 repair trial, six atoms were restored on the
        circle; counting exact indices would have rejected this geometrically valid repair."""
        if not self.ring:
            return None
        sites = [tuple(s) for s in self.ring["sites"]]
        if not sites:
            return None
        pts = self.positions()
        if pts.shape[0] == 0:
            return 0.0
        n = len(sites)
        spacing = 2 * math.pi * float(self.ring["radius_nm"]) * 1e-9 / n
        tol = self.RING_SITE_TOL * spacing
        held = 0
        for (i, j) in sites:
            sx, sy = self.site_xy(i, j)
            d = np.hypot(pts[:, 0] - sx, pts[:, 1] - sy)
            if float(d.min()) <= tol:
                held += 1
        return held / n


# ── layout generators ────────────────────────────────────────────────────────
def build_corral(reg: AdatomRegistry, *, centre: tuple[float, float], radius_m: float,
                 n_atoms: int, gap_atoms: int = 0, spares: int | None = None) -> None:
    r = reg.rng_layout
    phi0 = float(r.uniform(0, 2 * math.pi))
    gap_start = int(r.integers(0, max(n_atoms, 1))) if gap_atoms else 0
    missing = {(gap_start + k) % n_atoms for k in range(gap_atoms)}
    sites: list[tuple[int, int]] = []
    for idx in range(n_atoms):
        ang = phi0 + 2 * math.pi * idx / n_atoms
        i, j = reg.site_of(centre[0] + radius_m * math.cos(ang),
                           centre[1] + radius_m * math.sin(ang))
        if (i, j) in sites:
            continue
        sites.append((i, j))
        if idx not in missing:
            reg.add(i, j, role="ring")
    reg.ring = {"centre_nm": [centre[0] * 1e9, centre[1] * 1e9], "radius_nm": radius_m * 1e9,
                "n_sites": len(sites), "sites": [list(s) for s in sites],
                "gap_sites": [list(sites[k]) for k in sorted(missing) if k < len(sites)]}
    n_spare = spares if spares is not None else gap_atoms
    placed = 0
    guard = 0
    while placed < n_spare and guard < 400:
        guard += 1
        ang = phi0 + 2 * math.pi * (gap_start + gap_atoms / 2.0) / max(n_atoms, 1) \
            + float(r.uniform(-0.7, 0.7))
        rad = radius_m + float(r.uniform(3e-9, 6e-9))
        i, j = reg.site_of(centre[0] + rad * math.cos(ang), centre[1] + rad * math.sin(ang))
        x, y = reg.site_xy(i, j)
        if any(math.hypot(x - px, y - py) < 1.5e-9 for px, py in reg.positions()):
            continue
        if reg.add(i, j, role="spare") is not None:
            placed += 1


def build_single(reg: AdatomRegistry, *, centre: tuple[float, float], target_d: tuple[float, float],
                 n_bystanders: int = 3) -> None:
    r = reg.rng_layout
    ang = float(r.uniform(0, 2 * math.pi))
    rad = float(r.uniform(0, 3e-9))
    i, j = reg.site_of(centre[0] + rad * math.cos(ang), centre[1] + rad * math.sin(ang))
    atom = reg.add(i, j, role="target")
    x0, y0 = reg.site_xy(i, j)
    ti, tj = reg.site_of(x0 + target_d[0], y0 + target_d[1])
    tx, ty = reg.site_xy(ti, tj)
    reg.target = {"atom_id": atom.id if atom else None, "site": [ti, tj],
                  "site_nm": [tx * 1e9, ty * 1e9],
                  "start_nm": [x0 * 1e9, y0 * 1e9],
                  "path_dx_nm": target_d[0] * 1e9, "path_dy_nm": target_d[1] * 1e9}
    ux, uy = target_d[0], target_d[1]
    plen = math.hypot(ux, uy) or 1.0
    ux, uy = ux / plen, uy / plen
    placed, guard = 0, 0
    while placed < n_bystanders and guard < 400:
        guard += 1
        a = float(r.uniform(0, 2 * math.pi))
        d = float(r.uniform(5e-9, 10e-9))
        px, py = x0 + d * math.cos(a), y0 + d * math.sin(a)
        t = (px - x0) * ux + (py - y0) * uy
        perp = abs(-(px - x0) * uy + (py - y0) * ux)
        if -1e-9 <= t <= plen + 1e-9 and perp < 2e-9:
            continue                                     # keep the corridor clear
        bi, bj = reg.site_of(px, py)
        bx, by = reg.site_xy(bi, bj)
        if any(math.hypot(bx - qx, by - qy) < 2e-9 for qx, qy in reg.positions()):
            continue
        if reg.add(bi, bj, role="bystander") is not None:
            placed += 1


#: no two field atoms closer than this (a pair on neighbouring hollows is a dimer, not two atoms)
FIELD_MIN_SEP_M = 0.6e-9


def scatter_field(reg: AdatomRegistry, *, density_per_nm2: float, half_m: float,
                  clear: tuple[tuple[float, float, float], ...] = ()) -> int:
    """The rest of the deposit: a Poisson number of atoms, uniform over the square
    ``±half_m`` around the origin, each on its nearest lattice site, none inside a ``clear``
    circle ``(x, y, r)`` (the placed layout keeps its working room) and none within
    :data:`FIELD_MIN_SEP_M` of another atom. Its own stream, drawn after the layout, so the
    placed atoms and their traits are the ones the scenario had without a field.
    Returns how many were placed."""
    if density_per_nm2 <= 0 or half_m <= 0:
        return 0
    r = np.random.default_rng([int(reg.site.seed), FIELD_STREAM])
    n = int(r.poisson(density_per_nm2 * (2.0 * half_m * 1e9) ** 2))
    xy = r.uniform(-half_m, half_m, size=(n, 2))
    taken = [tuple(p) for p in reg.positions()]
    placed = 0
    for x, y in xy:
        if any((x - cx) ** 2 + (y - cy) ** 2 < cr * cr for cx, cy, cr in clear):
            continue
        i, j = reg.site_of(float(x), float(y))
        sx, sy = reg.site_xy(i, j)
        if taken and min((sx - px) ** 2 + (sy - py) ** 2 for px, py in taken) < FIELD_MIN_SEP_M ** 2:
            continue
        if reg.add(i, j, role="field") is not None:
            taken.append((sx, sy))
            placed += 1
    return placed
