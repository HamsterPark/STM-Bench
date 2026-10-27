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
atom's height and width, widened to ``sqrt(sigma² + s²)`` and lowered by ``sigma² / (sigma² +
s²)`` for the tip's blur ``s`` (:meth:`Tip.sigma_eff_m`), which is what filtering an isolated
bump with the renderer's Gaussian gives, so an atom has the same apparent height in a trace and
in a frame. The substrate (terraces and steps, herringbone, deposits) is blurred by the same
``s`` along the path; the atomic lattice is not (it is already the apex's transfer, as in a
frame).

**What the controller does with it.** Feedback on: the Z loop follows the height towards the gap
:meth:`World.equilibrium_gap` gives, starting at its first target as a scan does (the loop of a
scan line, discretised exactly: see :func:`_track`); the current is what the junction carries
across the gap that results, through the preamplifier clamp. Feedback off: Z is held and the current
follows the height. Withdrawn: Z sits at the retract limit and nothing tunnels. Z and the
current carry the instrument's noise, drawn from the trace's own stream, and so does the flicker
of a crashed tip.

**What is simplified.** The path is the drag's, in sample coordinates fixed by the drift at the
move's start (drift moves the sample under the tip during the move only along Z); a picked-up atom
leaves the surface at its sample and the tip is not lengthened by it (``Tip.pick_up`` does not do
so in a frame either), so the trace steps down by the atom's apparent height there; the substrate
is blurred along the path rather than over the plane; the tip's anisotropy, if the renderer has
any, is not applied; an open-loop move cannot crash into a step. A tip change during the move
(:meth:`Tip.maybe_change`, as before) is a step in the trace at a time drawn from the trace's
stream; the drag itself is unchanged by it.

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

#: the move trace's own random stream, ``np.random.default_rng([seed, MOVE_TRACE_STREAM])``
MOVE_TRACE_STREAM = 0x7ACE
#: a trace has at most this many samples; a longer path is sampled coarser (each sample is the
#: mean over its stretch of the drag grid) so that a file and a reply stay small
MAX_POINTS = 1000
#: the tunnelling gap never closes below this when a current is worked out (as in a frame)
MIN_GAP_M = 0.05e-9


def new_stream(seed: int) -> np.random.Generator:
    """The generator a world draws its move traces' noise from."""
    return np.random.default_rng([int(seed), MOVE_TRACE_STREAM])


# ── the tip, as it was ──────────────────────────────────────────────────────
def tip_view(tip):
    """A copy of ``tip`` to evaluate heights with later, whatever happens to the tip itself.

    Its random generator and its event log are shared, not copied (they are never used on the
    copy, and copying them would cost a lot and mean nothing)."""
    return copy.deepcopy(tip, {id(tip.rng): tip.rng, id(tip.events): tip.events})


def _height_signature(tip) -> tuple:
    """Everything of a tip that a trace's height depends on; two tips with the same signature
    show the same heights."""
    return (tuple(a.as_tuple() for a in tip.apexes), float(tip.length_m), float(tip.apex_sigma_m),
            float(tip.radius_m), float(tip.flicker_dz_m), float(tip.flicker_rate_hz))


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

    def blurred(self, blur_m: float) -> tuple[float, float]:
        """(apparent height, width) of one atom after the tip's blur ``blur_m``."""
        s2, b2 = self.sigma ** 2, blur_m ** 2
        return self.height * s2 / (s2 + b2), math.sqrt(s2 + b2)

    def bumps(self, x: np.ndarray, y: np.ndarray, blur_m: float) -> np.ndarray:
        """Summed apparent height of all atoms at the points ``(x[k], y[k])``, atoms as they are
        at sample ``k``."""
        out = np.zeros_like(x, dtype=float)
        if not (self.static or self.moving):
            return out
        amp, width = self.blurred(blur_m)
        w2 = 2.0 * width ** 2
        for px, py, lo, hi in self.static:
            out[lo:hi] += np.exp(-((x[lo:hi] - px) ** 2 + (y[lo:hi] - py) ** 2) / w2)
        for pos, gone in self.moving:
            g = np.exp(-((x - pos[:, 0]) ** 2 + (y - pos[:, 1]) ** 2) / w2)
            out += np.where(gone, 0.0, g)
        return amp * out


class _PathSurface:
    """The surface along one path, in the form :meth:`Tip.height_parts` takes: arrays of the
    path's samples in order. Atoms are the path's own (:class:`_Atoms`), so it is the real
    surface less its registry, blurred along the path, plus the atoms as they were at each sample."""

    def __init__(self, surface, atoms: _Atoms, step_m: float, blur_m: float):
        self._surface = surface
        self._atoms = atoms
        self._blur = float(blur_m)
        self._sigma_samples = float(blur_m) / step_m if step_m > 0 else 0.0
        self.material = surface.material

    def height_smooth(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        base = self._surface.height_smooth(x, y) - self._surface.adatom_height(x, y)
        if self._sigma_samples > 0.3 and base.size > 1:
            base = ndimage.gaussian_filter1d(base, self._sigma_samples, mode="nearest")
        return base + self._atoms.bumps(x, y, self._blur)

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
    loop: object
    tip: object
    drag_log: list[dict] = field(default_factory=list)
    tip_after: object = None

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
                     osc=world.osc_factor(), loop=world.loop_params(), tip=tip_view(world.tip))


def changed(start: MoveStart, world) -> None:
    """Note the tip as the move leaves it, before anything a pick-up does to it: if it differs
    from the tip the move began with (a change during the move), the trace shows a step."""
    if _height_signature(world.tip) != _height_signature(start.tip):
        start.tip_after = tip_view(world.tip)


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

    # the height under the tip along the drag grid: with the tip the move began with, and from
    # a time of the trace stream's choosing with the tip it ended with when that differs
    before, after = start.tip, start.tip_after
    change_at = None if after is None else 1 + int(rng.random() * (n_fine - 1))      # sample 1 .. n_fine - 1
    reg = world.surface.site.adatoms
    tips = [before] + ([after] if after is not None else [])
    reach = max((abs(c) for tp in tips for a in tp.apexes for c in (a.dx, a.dy)), default=0.0)
    if reg is not None:
        reach += 4.0 * math.hypot(reg.params.sigma_m, max(tp.sigma_eff_m(start.phi_ev) for tp in tips))
    atoms = _Atoms(reg, start.drag_log, p0, p1, n_fine, reach)

    def heights(tp) -> np.ndarray:
        surf = _PathSurface(world.surface, atoms, step, tp.sigma_eff_m(start.phi_ev))
        apexes = tp.pass_apexes(rng) if tp.flicker_dz_m > 0 else None
        smooth, atomic = tp.height_parts(surf, xs, ys, start.kappa_m, apexes=apexes, bias_v=start.bias_v)
        return smooth + atomic

    h = heights(before)
    if change_at is not None:
        h = np.where(np.arange(n_fine) >= change_at, heights(after), h)
    h = _reduce(h, n)
    if before.flicker_dz_m > 0:
        h = h + before.flicker_series(n, dt, rng)
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
        # an oscillating qPlus tip carries the current its cycle averages to, which is what the
        # loop holds at the setpoint (World.osc_factor; 1 without oscillation)
        i_dc = current_metal(np.clip(z_tip - h, MIN_GAP_M, None), start.bias_v, start.phi_ev) * start.osc
    if not (start.withdrawn or float(z_ctrl[0]) in (world.zctrl.limit_high_m, world.zctrl.limit_low_m)):
        # a reading at the rail is exactly the limit; elsewhere it carries the Z noise, more of it
        # on a tip that keeps changing (as in a frame)
        z_ctrl = z_ctrl + noise.z_series(n) * (1.0 + 0.3 * before.lambda_per_s * 1e2)
    i = np.asarray(world.preamp.clamp(noise.current_series(i_dc, start.wall_s + t)), float)

    world.move_counter += 1
    return {"index": world.move_counter, "t": t, "dist": dist, "z_abs": z_ctrl, "i": i,
            "from_nm": [start.x0 * 1e9, start.y0 * 1e9], "to_nm": [start.x1 * 1e9, start.y1 * 1e9],
            "speed_nm_s": start.speed_m_s * 1e9, "bias_v": start.bias_v,
            "setpoint_pa": float(world.zctrl.setpoint_a) * 1e12, "feedback": start.feedback,
            "withdrawn": start.withdrawn,
            "events": [{**e, "sample": _sample_of(e["sample"], n_fine, n)} for e in atoms.events],
            "tip_change_sample": None if change_at is None else _sample_of(change_at, n_fine, n)}


def _sample_of(fine_idx: int, n_fine: int, n: int) -> int:
    """The trace sample a drag-grid index shows at (the first one at or after it)."""
    if n >= n_fine:
        return int(fine_idx)
    return int(min(n - 1, math.ceil(fine_idx * (n - 1) / (n_fine - 1))))
