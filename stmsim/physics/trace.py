"""The trace of a lateral move: what the controller records while the tip crosses its path.

An operator dragging an atom is not told that it hopped. The controller records the tip height
and the current while the tip moves, and the hops are read from that record: in constant-current
mode an atom pulled along by the tip gives a sawtooth in Z with the period of the lattice
(Bartels, Meyer & Rieder, PRL 79, 697 (1997)); with no atom under the tip the trace shows the
substrate and noise. :func:`begin` and :func:`finish` bracket :meth:`World.move_xy_path`, which
runs the drag itself as before, and draw that record from the instrument's own state.

**Where the height comes from.** The path is sampled on the drag integration's grid
(:func:`~stmsim.physics.adatoms.path_points`, one sample per 0.05 nm), and at every sample the height
under the tip is what :meth:`Tip.height_parts` gives for the tip as it is (every apex, so a
double tip shows two signatures) over the surface with every atom *where it was at that sample*:
the drag records the path index of each hop and pick-up (``AdatomRegistry.drag(log=...)``).
Atoms are drawn as a frame draws them once the tip's size has blurred them: a Gaussian bump of the
atom's height and width, widened by the tip's blur and lowered with it (covariance ``sigma² I +
Sigma_blur``, height × ``sigma² / sqrt(det(sigma² I + Sigma_blur))``; for the round tip,
``Sigma_blur = s² I`` with ``s`` from :meth:`Tip.sigma_eff_m`, that is the width
``sqrt(sigma² + s²)`` and the factor ``sigma² / (sigma² + s²)``), which is what filtering an
isolated bump with the renderer's Gaussian gives, so an atom has the same apparent height in a
trace and in a frame. With ``tip_anisotropy`` on (:mod:`stmsim.physics.switches`) the tip's blur
is the renderer's ellipse: long axis ``s`` along ``Tip.angle``, short axis ``axis_ratio`` of it.
The substrate (terraces and steps, herringbone, deposits) is blurred along the path by the
width the blur has in the path's direction; the atomic lattice is not blurred (it is already the
apex's transfer, as in a frame).

**What the controller does with it.** Feedback on: the Z loop follows the height towards the gap
:meth:`World.equilibrium_gap` gives, starting at its first target as a scan does (the loop of a
scan line, discretised exactly: see :func:`_track`); the current is what the junction carries
across the gap that results, through the preamplifier clamp. Feedback off: Z is held and the current
follows the height. Withdrawn: Z sits at the retract limit and nothing tunnels. While the loop
presses the tip into the sample (:meth:`World.over_range`) the gap :meth:`World.equilibrium_gap`
gives is contact and the current is what the world reads then (:meth:`World._dc_current`): the
preamplifier's clamp. Z and the current carry the instrument's noise, drawn from the trace's own
stream, and so does the flicker of a crashed tip.

**What is simplified.** The path is the drag's, in sample coordinates fixed by the drift at the
move's start (drift moves the sample under the tip during the move only along Z); a picked-up atom
leaves the surface at its sample and the tip is not lengthened by it (``Tip.pick_up`` does not do
so in a frame either), so the trace steps down by the atom's apparent height there; the substrate
is blurred along the path rather than over the plane; the junction (barrier, the gap the loop
holds) is the one the move began with, whatever happens to the tip during it; an open-loop move
cannot crash into a step. A change of the tip during the move is a step in the trace: at the time
the tip's own log gives it when the move tunnelled through an interval (the tip's hazards and
recoveries outside scans, :meth:`World.tunnel_interval`) and at the start for the crash of a
tip pressed into the sample; for the change of a drag (:meth:`Tip.maybe_change`, as before, which
stamps it at the move's start) at a time drawn from the trace's stream. Every such step is the tip
as the event left it (``TipEvent.state``); the drag itself is unchanged by any of it.

**Streams.** Nothing here draws from ``world.rng``, ``world.noise``, ``tip.rng`` or the adatom
streams: the noise, the time of a tip change and the flicker come from one generator,
``[seed, MOVE_TRACE_STREAM]``, so a trace changes none of the draws the rest of an episode gets.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, signal

from .adatoms import path_points
from .feedback import Loop
from .junction import current_metal
from .noise import NoiseModel
from .tip import Apex

#: the move trace's own random stream, ``np.random.default_rng([seed, MOVE_TRACE_STREAM])``
MOVE_TRACE_STREAM = 0x7ACE
#: a trace has at most this many samples; a longer path is sampled coarser (each sample is the
#: mean over its stretch of the drag grid) so that a file and a reply stay small
MAX_POINTS = 1000
#: the tunnelling gap never closes below this when a current is worked out (as in a frame)
MIN_GAP_M = 0.05e-9
#: at most this many changes of the tip are drawn into one trace (a tip that changed more often
#: during one move keeps its first ones and the state the move left it in)
MAX_TIP_STEPS = 16
#: a change of the tip logged within this long (s) of the move's start is stamped at it, as a
#: drag's is; later ones lie inside the interval the move tunnelled through, with times of their own.
#: (On the virtual clock the two are exactly apart; the host clock moves a little between calls.)
AT_START_S = 1e-3


def new_stream(seed: int) -> np.random.Generator:
    """The generator a world draws its move traces' noise from."""
    return np.random.default_rng([int(seed), MOVE_TRACE_STREAM])


# ── the tip, as it was ──────────────────────────────────────────────────────
def tip_view(tip):
    """A copy of ``tip`` to evaluate heights with later, whatever happens to the tip itself.

    Its random generators (the tip's own and those a scenario's mechanisms give it) and its event
    log are shared, not copied: they are never used on the copy, and copying them would cost a
    lot and mean nothing."""
    shared = {id(tip.events): tip.events}
    shared.update({id(v): v for v in vars(tip).values() if isinstance(v, np.random.Generator)})
    return copy.deepcopy(tip, shared)


def _round(v) -> float:
    """``v`` to ten significant digits: a number that went through a snapshot (metres to
    nanometres and back) is the same number for what a trace shows."""
    return float(f"{float(v):.10g}")


def _height_signature(tip) -> tuple:
    """Everything of a tip that a trace's height depends on; two tips with the same signature
    show the same heights."""
    return (tuple(tuple(_round(c) for c in a.as_tuple()) for a in tip.apexes), _round(tip.length_m),
            _round(tip.apex_sigma_m), _round(tip.radius_m), _round(tip.axis_ratio), _round(tip.angle),
            _round(tip.flicker_dz_m), _round(tip.flicker_rate_hz))


def _tip_at(base, state: dict):
    """``base`` (a view of a tip) with what a tip event's snapshot (:meth:`Tip.snapshot`) says of
    the apexes, the size and shape of the blur, the apex's smearing, its length and its flicker:
    the tip as that event left it. Whatever a trace's height does not depend on stays the base's."""
    v = copy.copy(base)
    v.apexes = [Apex(*a) for a in state["apexes"]]
    v.radius_m = float(state["radius_nm"]) * 1e-9
    v.apex_sigma_m = float(state["apex_sigma_nm"]) * 1e-9
    v.length_m = float(state["length_nm"]) * 1e-9
    v.axis_ratio = float(state["axis_ratio"])
    if "angle_deg" in state:                       # recorded only while anisotropy is on
        v.angle = math.radians(float(state["angle_deg"]))
    v.flicker_dz_m = float(state["flicker_dz_pm"]) * 1e-12
    v.flicker_rate_hz = float(state["flicker_rate_hz"])
    v.lambda_per_s = float(state["lambda_per_s"])
    return v


def blur_covariance(tip, phi_ev: float, anisotropic: bool) -> np.ndarray:
    """The covariance (m², x then y of the sample frame) of the Gaussian with which ``tip`` blurs
    the smooth surface: round, ``s² I`` with ``s = Tip.sigma_eff_m`` - or, with anisotropy on and a
    tip that is not round, the renderer's ellipse (``Renderer._anisotropic_blur_cov``): long axis
    ``s`` along ``tip.angle``, short axis ``axis_ratio`` of it."""
    s = tip.sigma_eff_m(phi_ev)
    if anisotropic and tip.axis_ratio < 1.0 - 1e-9:
        major, minor = s * s, (s * max(float(tip.axis_ratio), 0.0)) ** 2
        c, sn = math.cos(float(tip.angle)), math.sin(float(tip.angle))
        return np.array([[major * c * c + minor * sn * sn, (major - minor) * sn * c],
                         [(major - minor) * sn * c, major * sn * sn + minor * c * c]])
    return np.eye(2) * s * s


# ── atoms along the path ────────────────────────────────────────────────────
class _Atoms:
    """The adatoms' bumps at every sample of a path.

    ``log`` is what :meth:`AdatomRegistry.drag` recorded; an event decided at path index ``idx``
    holds from sample ``idx`` on. Atoms the drag did not touch stand where they are now. The
    bump is the renderer's: a Gaussian blurred by the tip (see the module's docstring).

    ``p0``, ``p1``: the path's ends, ``n``: its samples, ``reach``: how far from the path an
    atom still shows in some apex's height (a few widths and the apexes' offset)."""

    def __init__(self, reg, log: list[dict], p0: tuple[float, float], p1: tuple[float, float], n: int,
                 reach: float):
        self.n = n
        self.height = float(reg.params.height_m) if reg is not None else 0.0
        self.sigma = float(reg.params.sigma_m) if reg is not None else 1.0
        self.static: list[tuple[float, float, int, int]] = []        # x, y, and the samples it shows in
        self.moving: list[tuple[np.ndarray, np.ndarray]] = []        # n x 2 positions, n gone-flags
        self.events: list[dict] = []                                 # {"sample", "kind", "atom_id"}
        if reg is None:
            return
        d = np.array([p1[0] - p0[0], p1[1] - p0[1]])
        length2 = float(d @ d)
        step = math.sqrt(length2) / (n - 1)
        touched: dict[int, list[dict]] = {}
        for e in log:
            touched.setdefault(int(e["atom_id"]), []).append(e)
        on, xy = reg._arrays()
        if len(on):
            rel = xy - np.array(p0)
            u = np.clip((rel @ d) / length2, 0.0, 1.0) if length2 > 0 else np.zeros(len(on))
            near = np.hypot(rel[:, 0] - u * d[0], rel[:, 1] - u * d[1]) <= reach
            half = reach / step if step > 0 else float(n)
            for k in np.flatnonzero(near):
                if on[int(k)].id in touched:
                    continue
                c = float((rel[k] @ d) / length2) * (n - 1) if length2 > 0 else 0.0
                self.static.append((float(xy[k, 0]), float(xy[k, 1]), max(0, int(math.floor(c - half))),
                                    min(n, int(math.ceil(c + half)) + 1)))
        for atom_id, evs in touched.items():
            sub = reg.atoms[atom_id].sub
            pos = np.zeros((n, 2))
            gone = np.zeros(n, bool)
            site, start = evs[0]["from"], 0
            for e in evs:
                idx = int(e["idx"])
                pos[start:idx] = reg.site_xy(site[0], site[1], sub)
                if e["kind"] == "pickup":
                    gone[idx:] = True
                    start = n
                    self.events.append({"sample": idx, "kind": "pickup", "atom_id": atom_id, "site": list(site)})
                    break
                site, start = e["to"], idx
                self.events.append({"sample": idx, "kind": "hop", "atom_id": atom_id, "site": list(site)})
            if start < n:
                pos[start:] = reg.site_xy(site[0], site[1], sub)
            self.moving.append((pos, gone))
        self.events.sort(key=lambda e: e["sample"])

    def blurred(self, cov_blur: np.ndarray) -> tuple[float, np.ndarray]:
        """(apparent height, inverse covariance) of one atom after the tip's blur ``cov_blur``
        (2 x 2, m²): a Gaussian of covariance ``sigma² I`` filtered with one of ``cov_blur`` is
        a Gaussian of the sum, lowered by ``sigma² / sqrt(det(sum))``."""
        cov = np.eye(2) * self.sigma ** 2 + np.asarray(cov_blur, float)
        return self.height * self.sigma ** 2 / math.sqrt(float(np.linalg.det(cov))), np.linalg.inv(cov)

    def bumps(self, x: np.ndarray, y: np.ndarray, cov_blur: np.ndarray) -> np.ndarray:
        """Summed apparent height of all atoms at the points ``(x[k], y[k])``, atoms as they are
        at sample ``k``."""
        out = np.zeros_like(x, dtype=float)
        if not (self.static or self.moving):
            return out
        amp, inv = self.blurred(cov_blur)

        def gauss(dx, dy):
            return np.exp(-0.5 * (inv[0, 0] * dx * dx + 2.0 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy))

        for px, py, lo, hi in self.static:
            out[lo:hi] += gauss(x[lo:hi] - px, y[lo:hi] - py)
        for pos, gone in self.moving:
            out += np.where(gone, 0.0, gauss(x - pos[:, 0], y - pos[:, 1]))
        return amp * out


class _PathSurface:
    """The surface along one path, in the form :meth:`Tip.height_parts` takes: arrays of the
    path's samples in order. Atoms are the path's own (:class:`_Atoms`), so it is the real
    surface less its registry, blurred along the path, plus the atoms as they were at each sample."""

    def __init__(self, surface, atoms: _Atoms, step_m: float, cov_blur: np.ndarray, direction: np.ndarray):
        self._surface = surface
        self._atoms = atoms
        self._cov = np.asarray(cov_blur, float)
        # a straight edge or a smooth feature crossed along ``direction`` is blurred by the width
        # the Gaussian has along that direction
        width = math.sqrt(max(float(direction @ self._cov @ direction), 0.0))
        self._sigma_samples = width / step_m if step_m > 0 else 0.0
        self.material = surface.material

    def height_smooth(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        base = self._surface.height_smooth(x, y) - self._surface.adatom_height(x, y)
        if self._sigma_samples > 0.3 and base.size > 1:
            base = ndimage.gaussian_filter1d(base, self._sigma_samples, mode="nearest")
        return base + self._atoms.bumps(x, y, self._cov)

    def atomic_height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._surface.atomic_height(x, y)

    def __getattr__(self, name):
        return getattr(self._surface, name)


# ── the move ────────────────────────────────────────────────────────────────
@dataclass
class MoveStart:
    """The instrument as a move found it, taken before the move changes anything: the junction
    (bias, barrier, the gap the loop holds, the qPlus oscillation's current factor), the Z that
    would be held, the drift and the tip."""
    x0: float
    y0: float
    x1: float
    y1: float
    speed_m_s: float
    wall_s: float
    ox: float
    oy: float
    oz: float
    oz_end: float
    feedback: bool
    withdrawn: bool
    z_held_tip: float
    bias_v: float
    kappa_m: float
    phi_ev: float
    gap_m: float
    osc: float
    over_range: bool
    anisotropic: bool
    loop: object
    tip: object
    n_tip_events: int
    sim_s: float
    drag_log: list[dict] = field(default_factory=list)
    #: what the tip went through during the move (set by :func:`changed`): one ``(sim_s or None,
    #: kind, view)`` per change that shows in a trace, in the order they happened; the last view is
    #: the tip as the move left it
    tip_steps: list = field(default_factory=list)

    @property
    def length_m(self) -> float:
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)

    @property
    def duration_s(self) -> float:
        return self.length_m / max(self.speed_m_s, 1e-12)


def begin(world, x: float, y: float, speed_m_s: float) -> MoveStart:
    """Take what a trace of the move to ``(x, y)`` needs from the instrument as it is before the
    move changes anything."""
    sim0 = world.clock.sim()
    ox, oy, oz = world.drift_offset(sim0)
    length = math.hypot(x - world.tip_x, y - world.tip_y)
    oz_end = world.drift_offset(sim0 + length / max(speed_m_s, 1e-12))[2]
    return MoveStart(x0=world.tip_x, y0=world.tip_y, x1=float(x), y1=float(y), speed_m_s=float(speed_m_s),
                     wall_s=world.clock.wall(), ox=ox, oy=oy, oz=oz, oz_end=oz_end,
                     feedback=bool(world.zctrl.on and not world.withdrawn), withdrawn=bool(world.withdrawn),
                     z_held_tip=float(world.tip_height_m()), bias_v=float(world.bias_v),
                     kappa_m=world.kappa_m(), phi_ev=world.phi_junction(), gap_m=world.equilibrium_gap(),
                     osc=world.osc_factor(), over_range=bool(world.over_range()),
                     anisotropic=bool(world.physics.tip_anisotropy), loop=world.loop_params(),
                     tip=tip_view(world.tip), n_tip_events=len(world.tip.events), sim_s=sim0)


def changed(start: MoveStart, world) -> None:
    """Note what the move did to the tip, before anything a pick-up does to it: every change
    its log recorded since the move began, with the tip as that event left it (its snapshot),
    keeping those that show in a trace's height; the last one is the tip as it is now. A tip
    that differs from the one the move began with without having logged why gets one step."""
    tip = world.tip
    new = tip.events[start.n_tip_events:]
    if not new and _height_signature(tip) == _height_signature(start.tip):
        return
    steps, prev = [], _height_signature(start.tip)
    for ev in new:
        view = _tip_at(start.tip, ev.state)
        sig = _height_signature(view)
        if sig != prev:
            steps.append([ev.sim_s, ev.kind, view])
            prev = sig
    final = tip_view(tip)
    if _height_signature(final) != prev:
        steps.append([None, "unlogged", final])
    elif steps:
        steps[-1][2] = final                       # exact where the snapshot is not
    if len(steps) > MAX_TIP_STEPS:
        steps = steps[:MAX_TIP_STEPS - 1] + steps[-1:]
    start.tip_steps = steps


def _track(params, z_eq: np.ndarray, dt: float, z0: float) -> np.ndarray:
    """The Z loop's output at the end of each sample interval ``dt``, for a target that holds
    still within one: :meth:`Loop.track` with the loop discretised exactly (zero-order hold).

    A scan line steps the loop at a fraction of its time constant (``Loop.track`` does it by
    the bilinear transform at a pixel's eighth). A trace of a slow drag has 0.2 s between
    samples, thousands of time constants, and the bilinear transform then maps the loop's fast
    poles next to z = -1: the output would ring from sample to sample. The exact discretisation
    settles within the sample, as the real loop does."""
    if not params.stable:
        return Loop(params).track(z_eq, dt, z0=z0)
    num, den = params.tf()
    b, a, _ = signal.cont2discrete((num, den), dt, method="zoh")
    b, a = np.asarray(b).ravel(), np.asarray(a).ravel()
    zi = signal.lfiltic(b, a, [z0] * (len(a) - 1), [z0] * (len(b) - 1))
    # the exact discretisation of a strictly proper loop has no direct term: the output at a
    # step's end reads the target held during it, one entry of the filter's output further on
    y, _ = signal.lfilter(b, a, np.append(z_eq, z_eq[-1]), zi=zi)
    return y[1:]


def _reduce(h: np.ndarray, n_out: int) -> np.ndarray:
    """``h`` (the drag grid) on ``n_out`` samples spread over the same path, each the mean over
    its stretch of the grid."""
    n = h.size
    if n_out >= n:
        return h
    width = max(1, int(round(n / n_out)))
    if width > 1:
        h = ndimage.uniform_filter1d(h, size=width, mode="nearest")
    return np.interp(np.linspace(0.0, n - 1.0, n_out), np.arange(n), h)


def finish(world, start: MoveStart, rng: np.random.Generator) -> dict:
    """The record of the move that ``start`` began: arrays ``t`` (s since the move's start),
    ``dist`` (m along the path), ``z_abs`` (m, the controller's Z output) and ``i`` (A, as
    measured), the move's settings, and ``events`` - where hops and pick-ups fell, for tests and
    validation, which no observation of the instrument carries.

    ``rng`` is the trace's generator (:func:`new_stream`)."""
    length = start.length_m
    n_fine = path_points(length)
    n = min(n_fine, MAX_POINTS)
    step = length / (n_fine - 1)
    ts = np.linspace(0.0, 1.0, n_fine)
    p0 = (start.x0 + start.ox, start.y0 + start.oy)
    p1 = (start.x1 + start.ox, start.y1 + start.oy)
    xs = p0[0] + (p1[0] - p0[0]) * ts
    ys = p0[1] + (p1[1] - p0[1]) * ts
    duration = max(start.duration_s, 1e-6)
    dt = duration / (n - 1)
    dist = np.linspace(0.0, length, n)
    t = np.linspace(0.0, duration, n)

    # the height under the tip along the drag grid. The tip is the one the move began with until
    # a change the move went through, then the tip as that change left it: where the tip's own
    # log puts the change in the interval it tunnelled through, at the start for a crash, and
    # at a time of the trace stream's choosing for the change a drag stamps at its start.
    # (Times are drawn first, in the order of the changes, so that the stream's use is the same
    # whatever else is computed.)
    reg = world.surface.site.adatoms
    views, ks = [start.tip], [0]
    for sim_s, kind, view in start.tip_steps:
        if kind == "crash":                                    # the only crash a move makes is the start's plough
            k = 0
        elif sim_s is not None and sim_s - start.sim_s > AT_START_S:
            k = int(math.ceil((sim_s - start.sim_s) / duration * (n_fine - 1)))
        else:                                                  # stamped at the start: a drag's change
            k = 1 + int(rng.random() * (n_fine - 1))
        views.append(view)
        ks.append(min(max(k, 0), n_fine - 1))
    order = sorted(range(1, len(views)), key=lambda j: ks[j])                    # by time, ties as logged
    views, ks = [views[0]] + [views[j] for j in order], [0] + [ks[j] for j in order]
    covs = [blur_covariance(v, start.phi_ev, start.anisotropic) for v in views]
    reach = max((abs(c) for v in views for a in v.apexes for c in (a.dx, a.dy)), default=0.0)
    if reg is not None:
        reach += 4.0 * math.hypot(reg.params.sigma_m, max(math.sqrt(float(np.linalg.eigvalsh(c)[-1])) for c in covs))
    atoms = _Atoms(reg, start.drag_log, p0, p1, n_fine, reach)
    direction = np.array([p1[0] - p0[0], p1[1] - p0[1]]) / max(length, 1e-30)

    def heights(tp, cov) -> np.ndarray:
        surf = _PathSurface(world.surface, atoms, step, cov, direction)
        apexes = tp.pass_apexes(rng) if tp.flicker_dz_m > 0 else None
        smooth, atomic = tp.height_parts(surf, xs, ys, start.kappa_m, apexes=apexes, bias_v=start.bias_v)
        return smooth + atomic

    h = heights(views[0], covs[0])
    fine = np.arange(n_fine)
    for v, c, k in zip(views[1:], covs[1:], ks[1:]):
        h = np.where(fine >= k, heights(v, c), h)
    h = _reduce(h, n)
    # on the trace's samples: which tip is at each, its flicker (a crashed apex's cluster hops
    # between configurations) and how noisy a tip that keeps changing makes Z
    first = [_sample_of(k, n_fine, n) for k in ks]
    which = np.searchsorted(np.array(first), np.arange(n), side="right") - 1
    lam = np.array([v.lambda_per_s for v in views])[which]
    for j, v in enumerate(views):
        if v.flicker_dz_m > 0:
            h = h + np.where(which == j, v.flicker_series(n, dt, rng), 0.0)
    h = h + np.interp(t, [0.0, duration], [start.oz, start.oz_end])          # Z drift, linear in time

    # what the controller makes of it
    noise = NoiseModel(rng, floor_a=world.noise.floor_a, bandwidth_hz=world.noise.bandwidth_hz,
                       mains_hz=world.noise.mains_hz, mains_amp_a=world.noise.mains_amp_a,
                       pink_amp_a=world.noise.pink_amp_a, spectral_lines=world.noise.spectral_lines,
                       z_floor_m=world.noise.z_floor_m)
    if start.withdrawn:
        z_ctrl = np.full(n, float(world.zctrl.z_n))
        i_dc = np.zeros(n)
    else:
        if start.feedback:
            # the loop is at its target where the move starts, as at the start of a scan: whatever
            # the world's equilibrium Z says of the tip standing over an atom (the unblurred
            # bump), the trace is the blurred landscape's, and a transient between the two
            # would be an artefact at the start of every fast move that leaves an atom
            z_tip = _track(start.loop, h + start.gap_m, dt, float(h[0]) + start.gap_m)
        else:
            z_tip = np.full(n, start.z_held_tip)
        z_ctrl = np.asarray(world.z_ctrl_from_tip(z_tip), float)
        if start.over_range:
            # the loop presses the tip into the sample (its target is contact) and the reading is
            # pinned at the preamplifier's clamp, as World._dc_current reads it
            i_dc = np.full(n, world.preamp.clamp_a * math.copysign(1.0, start.bias_v or 1.0))
        else:
            # an oscillating qPlus tip carries the current its cycle averages to, which is what the
            # loop holds at the setpoint (World.osc_factor; 1 without oscillation)
            i_dc = current_metal(np.clip(z_tip - h, MIN_GAP_M, None), start.bias_v, start.phi_ev) * start.osc
    if not (start.withdrawn or float(z_ctrl[0]) in (world.zctrl.limit_high_m, world.zctrl.limit_low_m)):
        # a reading at the rail is exactly the limit; elsewhere it carries the Z noise, more of it
        # on a tip that keeps changing (as in a frame)
        z_ctrl = z_ctrl + noise.z_series(n) * (1.0 + 0.3 * lam * 1e2)
    i = np.asarray(world.preamp.clamp(noise.current_series(i_dc, start.wall_s + t)), float)

    world.move_counter += 1
    return {"index": world.move_counter, "t": t, "dist": dist, "z_abs": z_ctrl, "i": i,
            "from_nm": [start.x0 * 1e9, start.y0 * 1e9], "to_nm": [start.x1 * 1e9, start.y1 * 1e9],
            "speed_nm_s": start.speed_m_s * 1e9, "bias_v": start.bias_v,
            "setpoint_pa": float(world.zctrl.setpoint_a) * 1e12, "feedback": start.feedback,
            "withdrawn": start.withdrawn, "over_range": start.over_range,
            "events": [{**e, "sample": _sample_of(e["sample"], n_fine, n)} for e in atoms.events],
            "tip_change_samples": first[1:],
            "tip_change_sample": first[1] if len(first) > 1 else None}


def _sample_of(fine_idx: int, n_fine: int, n: int) -> int:
    """The trace sample a drag-grid index shows at (the first one at or after it)."""
    if n >= n_fine:
        return int(fine_idx)
    return int(min(n - 1, math.ceil(fine_idx * (n - 1) / (n_fine - 1))))
