"""The pull threshold of an adatom: the junction resistance at which a standard drag carries it.

This is the quantity of P7, the lateral-manipulation threshold of Bartels, Meyer & Rieder
(PRL 79, 697 (1997)) and of the thresholds tabulated by Hla (JVST B 23, 1351 (2005)). A drag
(:meth:`~stmsim.physics.adatoms.AdatomRegistry.drag`) is a chain of chances: while the gripping
apex has hold of the atom and is more than 0.6 of a site spacing ahead of it, each path sample
is one hop attempt, which succeeds with ``p_follow`` (the resistance against the atom's threshold
with the tip's scale, the speed against ``v_max``, the tip's fumbles); a failed attempt lets go,
and the apex takes hold again at the next sample while the atom is still inside its grab radius;
every grab may lift the atom onto the tip instead (``p_pick``); a hop may land on the
second-best neighbour (``jitter``) and may slip out of the grip (``p_slip``).

No measurement sees a single hop attempt, so the law's own ``R_th`` (where one attempt succeeds
half the time) is not what an operator can measure. What an operator measures is a property of a
whole drag: **the panel resistance at which a drag of a set length and speed, started with the
gripping apex over the atom, carries the atom the whole way in half of the attempts** - "the whole
way" meaning that the atom ends within 0.6 of a site spacing of the apex's end point, the distance
inside which the drag counts an atom as being with the tip. That resistance depends on the atom's
own factors (threshold and grab radius), how it is held where it sits (a step edge, a neighbour),
the tip's grip, the speed against ``v_max``, and the chances of slipping and of being picked up.

:func:`drag_outcome` computes the outcome of one drag exactly for an isolated atom on a flat
terrace: a forward pass over the drag's path samples of the probability of every state the atom
can be in (its site; held or not; on the tip), with the geometry and the chances of
:meth:`AdatomRegistry.drag` taken from the same registry methods. It is the model's own answer,
not an approximation of it (``tests/test_drag_threshold.py`` checks it against ``drag()`` by Monte
Carlo), and it draws nothing. :func:`pull_threshold` finds the resistance at which the
full-carry probability is one half. Left out: other atoms (none is within reach of an isolated
one), steps and debris (a flat terrace), and every apex but the one that holds the atom.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import numpy as np

from .adatoms import HOP_EXPONENT, SPEED_EXPONENT, AdatomParams, AdatomRegistry, path_points

#: the drag counts an atom whose site is within this fraction of the site spacing of the apex as
#: being with the tip (``AdatomRegistry.drag``); the same distance decides "carried the whole way"
WITH_TIP_FRACTION = 0.6
#: P7's standard drag: 1.0 nm at 0.1 nm/s - about four sites along a close-packed row, slow enough
#: that the speed factor is at least 0.96 over v_max = 0.5-2 nm/s
STANDARD_LENGTH_M = 1.0e-9
STANDARD_SPEED_M_S = 0.1e-9


@dataclass(frozen=True)
class DragConditions:
    """Everything :meth:`AdatomRegistry.drag` reads to decide one atom's fate in one drag, apart
    from the geometry: the scenario's parameters, the atom's traits and hold, the tip's grip, and
    the resistance factor of the apex that holds the atom."""
    r_threshold_ohm: float
    r_pick_ohm: float
    v_max_m_s: float
    capture_radius_m: float
    p_slip: float
    r_scale: float                 # the atom's own factor on both thresholds
    cap_scale: float               # the atom's own factor on the grab radius
    hold: float                    # AdatomRegistry.hold_factor where it sits (r_scale × step × crowd)
    threshold_scale: float = 1.0   # Grip.threshold_scale
    capture_scale: float = 1.0     # Grip.capture_scale
    fumble: float = 0.0            # Grip.fumble
    jitter: float = 0.0            # Grip.jitter
    r_factor: float = 1.0          # GripPoint.r_factor of the holding apex

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DragConditions":
        return cls(**{k: float(d[k]) for k in cls.__dataclass_fields__ if k in d})


def lattice(nn_m: float, lattice_angle: float) -> AdatomRegistry:
    """An empty registry on the hollow-site lattice of a surface with nearest-neighbour spacing
    ``nn_m`` and lattice angle ``lattice_angle`` (rad): the geometry :func:`drag_outcome` reads.
    Its random streams are never used."""
    site = SimpleNamespace(seed=0, lattice_angle=float(lattice_angle),
                           material=SimpleNamespace(nn_m=float(nn_m)))
    return AdatomRegistry(site, AdatomParams())


def _p_follow(r_ohm: float, v_m_s: float, c: DragConditions) -> float:
    """``AdatomRegistry.p_follow`` for the holding apex (its ``r_factor`` already in ``r_ohm``)."""
    r_th = c.r_threshold_ohm * c.threshold_scale * c.hold
    a = 1.0 / (1.0 + (max(r_ohm, 1.0) / max(r_th, 1.0)) ** HOP_EXPONENT)
    b = 1.0 / (1.0 + (max(v_m_s, 0.0) / c.v_max_m_s) ** SPEED_EXPONENT)
    return float(a * b * (1.0 - c.fumble))


def _p_pick(r_ohm: float, c: DragConditions) -> float:
    """``AdatomRegistry.p_pick``; below 1e-6 the drag does not even draw for it."""
    r_pick = c.r_pick_ohm * c.threshold_scale * c.r_scale
    p = float(1.0 / (1.0 + (max(r_ohm, 1.0) / max(r_pick, 1.0)) ** HOP_EXPONENT))
    return p if p > 1e-6 else 0.0


def _hop_options(geom: AdatomRegistry, i: int, j: int, toward: tuple[float, float]) -> list:
    """The free neighbours a hop from (i, j) toward ``toward`` may land on, best first: the
    candidates and the order of ``AdatomRegistry._hop`` for an atom alone on a flat terrace."""
    cx, cy = geom.site_xy(i, j)
    here_d = (cx - toward[0]) ** 2 + (cy - toward[1]) ** 2
    options = []
    for ni, nj in geom.neighbours(i, j):
        px, py = geom.site_xy(ni, nj)
        options.append(((px - toward[0]) ** 2 + (py - toward[1]) ** 2, (ni, nj), (px, py)))
    options.sort()
    return [o for o in options if o[0] < here_d]


def drag_outcome(geom: AdatomRegistry, site: tuple[int, int], start_xy: tuple[float, float],
                 end_xy: tuple[float, float], r_ohm: float, v_m_s: float, c: DragConditions) -> dict:
    """The exact outcome distribution of one drag of the holding apex from ``start_xy`` to
    ``end_xy`` (sample frame, m) over an isolated atom born at lattice ``site``, at panel resistance
    ``r_ohm`` and speed ``v_m_s``.

    Returns ``{"full": P(the atom ends within 0.6 site spacings of the end point, on the surface),
    "picked": P(it went onto the tip), "sites": {(i, j): P(it ends there, on the surface)}}``."""
    nn = float(geom.site.material.nn_m)
    with_tip = WITH_TIP_FRACTION * nn
    cap = c.capture_radius_m * c.capture_scale * c.cap_scale
    r_eff = float(r_ohm) * c.r_factor
    p_f = _p_follow(r_eff, v_m_s, c)
    p_p = _p_pick(r_eff, c)
    x0, y0 = start_xy
    x1, y1 = end_xy
    length = math.hypot(x1 - x0, y1 - y0)
    n = path_points(length)
    ts = np.linspace(0.0, 1.0, n)
    xs = x0 + (x1 - x0) * ts
    ys = y0 + (y1 - y0) * ts
    states: dict[tuple[int, int, bool], float] = {(int(site[0]), int(site[1]), False): 1.0}
    picked = 0.0
    options_cache: dict = {}
    for idx in range(n):
        tx, ty = float(xs[idx]), float(ys[idx])
        k = min(idx + 1, n - 1)
        nxt = (float(xs[k]), float(ys[k]))
        new: dict[tuple[int, int, bool], float] = defaultdict(float)
        for (i, j, held), pr in states.items():
            if pr <= 0.0:
                continue
            ax, ay = geom.site_xy(i, j)
            if not held:
                # grab(): the atom inside the apex's grab radius is taken hold of (q <= 1), and
                # may be lifted instead
                if math.hypot(ax - tx, ay - ty) / cap <= 1.0:
                    picked += pr * p_p
                    new[(i, j, True)] += pr * (1.0 - p_p)
                else:
                    new[(i, j, False)] += pr
                continue
            if math.hypot(tx - ax, ty - ay) <= with_tip:
                new[(i, j, True)] += pr
                continue
            # one hop attempt: it fails (and lets go) with 1 - p_follow, or when no free neighbour
            # is closer to where the apex goes next
            new[(i, j, False)] += pr * (1.0 - p_f)
            key = (i, j, idx)
            if key not in options_cache:
                options_cache[key] = _hop_options(geom, i, j, nxt)
            opts = options_cache[key]
            if not opts:
                new[(i, j, False)] += pr * p_f
                continue
            branches = [(opts[0][1], 1.0)]
            if c.jitter > 0.0 and len(opts) > 1:
                branches = [(opts[0][1], 1.0 - c.jitter), (opts[1][1], c.jitter)]
            for (ni, nj), w in branches:
                new[(ni, nj, True)] += pr * p_f * w * (1.0 - c.p_slip)
                new[(ni, nj, False)] += pr * p_f * w * c.p_slip
        states = new
    sites: dict[tuple[int, int], float] = defaultdict(float)
    full = 0.0
    for (i, j, _held), pr in states.items():
        sites[(i, j)] += pr
        sx, sy = geom.site_xy(i, j)
        if math.hypot(sx - x1, sy - y1) <= with_tip:
            full += pr
    return {"full": float(full), "picked": float(picked), "sites": dict(sites)}


def full_carry_probability(geom: AdatomRegistry, site: tuple[int, int], direction_rad: float,
                           r_ohm: float, c: DragConditions, *, length_m: float = STANDARD_LENGTH_M,
                           speed_m_s: float = STANDARD_SPEED_M_S) -> float:
    """P(a drag of ``length_m`` at ``speed_m_s`` along ``direction_rad`` (sample frame), started
    with the holding apex over the atom, carries it the whole way)."""
    x0, y0 = geom.site_xy(*site)
    end = (x0 + length_m * math.cos(direction_rad), y0 + length_m * math.sin(direction_rad))
    return drag_outcome(geom, site, (x0, y0), end, r_ohm, speed_m_s, c)["full"]


def pull_threshold(geom: AdatomRegistry, site: tuple[int, int], direction_rad: float,
                   c: DragConditions, *, length_m: float = STANDARD_LENGTH_M,
                   speed_m_s: float = STANDARD_SPEED_M_S, target: float = 0.5,
                   rel_tol: float = 1e-5) -> float | None:
    """The panel resistance (ohm) at which the standard drag carries the atom the whole way with
    probability ``target``, on the high-resistance side of the manipulation window (lower still,
    the atom may be lifted onto the tip instead, which also stops it being carried). ``None``
    when no resistance carries it that often (a tip that fumbles almost every hop, an atom that is
    picked up before it follows)."""
    def p(r: float) -> float:
        return full_carry_probability(geom, site, direction_rad, r, c, length_m=length_m,
                                      speed_m_s=speed_m_s)

    # the law's own threshold for the holding apex, where the window's upper edge lies
    r_law = c.r_threshold_ohm * c.threshold_scale * c.hold / max(c.r_factor, 1e-12)
    grid = r_law * np.exp(np.linspace(math.log(1 / 30), math.log(4.0), 81))
    vals = [p(float(r)) for r in grid]
    k = int(np.argmax(vals))
    if vals[k] < target:
        return None
    m = next((q for q in range(k + 1, len(grid)) if vals[q] < target), None)
    if m is None:                   # cannot happen: at 4 R_th a hop attempt succeeds 1.5e-5 of the time
        return None
    lo, hi = float(grid[m - 1]), float(grid[m])
    while hi / lo - 1.0 > rel_tol:
        mid = math.sqrt(lo * hi)
        if p(mid) >= target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


_LATTICES: dict[tuple[float, float], AdatomRegistry] = {}
_RECORD_THRESHOLDS: dict[tuple, float | None] = {}


def record_threshold(lattice_info: dict, record: dict, *, length_m: float = STANDARD_LENGTH_M,
                     speed_m_s: float = STANDARD_SPEED_M_S) -> float | None:
    """R½ (ohm) of the atom a recorded drag held, under that drag's conditions and along its
    direction: the truth one cited drag stands for (P7's judge). ``lattice_info`` and
    ``record`` are an entry of ``truth()["drags"]`` (:meth:`World.drag_truth`). ``None`` for a
    drag that held no atom, or when no resistance carries the atom half the time. Cached: a
    series of drags under one condition is computed once per direction."""
    cond = record.get("conditions")
    if not cond or record.get("atom_id") is None:
        return None
    key_l = (float(lattice_info["nn_m"]), float(lattice_info["angle_rad"]))
    geom = _LATTICES.get(key_l)
    if geom is None:
        geom = _LATTICES[key_l] = lattice(*key_l)
    c = DragConditions.from_dict(cond)
    # the lattice has sixfold symmetry about a hollow, so a drag and its return are the same
    direction = float(record.get("direction_rad", 0.0))
    key = (key_l, tuple(sorted(c.as_dict().items())), round(direction % math.pi, 9),
           float(length_m), float(speed_m_s))
    if key not in _RECORD_THRESHOLDS:
        _RECORD_THRESHOLDS[key] = pull_threshold(geom, (0, 0), direction, c, length_m=length_m,
                                                 speed_m_s=speed_m_s)
    return _RECORD_THRESHOLDS[key]


def example_thresholds(world, n: int = 2, *, length_m: float = STANDARD_LENGTH_M,
                       speed_m_s: float = STANDARD_SPEED_M_S, isolation_m: float = 3e-9,
                       direction_rad: float = 0.0) -> list[float | None]:
    """R½ (ohm) of the ``n`` isolated atoms (no other within ``isolation_m``) nearest the sample
    origin, as the world is built, under an ideal grip (one sharp, steady apex) and along
    ``direction_rad``: what a fixed answer sheet is up against, seed by seed (the chance audit).
    The ideal grip gives the narrowest spread a seed's truth can have, so the audit's figure is
    an upper bound for agents whose tips differ."""
    from .adatoms import Grip
    reg = world.surface.site.adatoms
    if reg is None:
        return [None] * n
    on, xy = reg._arrays()
    if not on:
        return [None] * n
    d0 = np.hypot(xy[:, 0], xy[:, 1])
    picked: list = []
    for k in np.argsort(d0):
        others = np.hypot(xy[:, 0] - xy[k, 0], xy[:, 1] - xy[k, 1])
        if int(np.count_nonzero(others <= isolation_m)) == 1:
            picked.append(on[int(k)])
        if len(picked) == n:
            break
    grip = Grip()
    geom = lattice(float(world.surface.site.material.nn_m), float(world.surface.site.lattice_angle))
    out: list[float | None] = []
    for atom in picked:
        c = conditions_of(reg, atom, grip, grip.points[0], world.surface.terrace_level)
        out.append(pull_threshold(geom, (0, 0), direction_rad, c, length_m=length_m, speed_m_s=speed_m_s))
    return out + [None] * (n - len(out))


def conditions_of(reg: AdatomRegistry, atom, grip, point, terrace=None) -> DragConditions:
    """The conditions of a drag of ``atom`` held by grip ``point`` of ``grip`` on registry
    ``reg``, as the registry is now (its hold factor where the atom sits)."""
    p = reg.params
    return DragConditions(r_threshold_ohm=p.r_threshold_ohm, r_pick_ohm=p.r_pick_ohm,
                          v_max_m_s=p.v_max_m_s, capture_radius_m=p.capture_radius_m,
                          p_slip=p.p_slip, r_scale=float(atom.r_scale), cap_scale=float(atom.cap_scale),
                          hold=float(reg.hold_factor(atom, terrace)),
                          threshold_scale=float(grip.threshold_scale), capture_scale=float(grip.capture_scale),
                          fumble=float(grip.fumble), jitter=float(grip.jitter),
                          r_factor=float(point.r_factor))
