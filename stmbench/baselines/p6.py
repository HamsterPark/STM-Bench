"""P6 scripted baseline: the Kondo resonance of one Co atom, measured with nothing but what the
instrument shows — no truth.

What the privileged reference (``stmbench.reference.p6``) reads from the truth, this reads from
frames and sweeps:

* **the tip**: images judge it as in P4 (:func:`stmbench.analysis.tip.diagnose`: blunt, double),
  three shallow pokes go in whatever it shows (an unstable or dirty tip does not show in one
  frame; a shallow contact stabilises an apex and buries what sits on it), and blunt or double
  tips are pulsed away from home and judged again. Spectroscopy then judges what images cannot:
  a sweep over bare Cu(111) shows the surface-state onset and nothing else, and an extra peak
  (:func:`spectral_flaw`) means something sits on the apex — poked away on fresh ground, then
  the sweep is taken again;
* **which atom**: one of the survey's own atoms with no other bump within 3 nm, no step within
  3 nm (a step leaves the background uneven by far more than a terrace does), and a patch of
  bare surface 3.5-5.5 nm away on the same terrace (the reference spectrum goes there);
* **the drift**: two frames 30 s apart for the velocity, the compensation, a third pair that
  checks what is left and takes it up;
* **the centre**: a slow 5 nm frame immediately before every sweep on the atom, whose centroid
  (:func:`stmbench.analysis.adatoms.locate`) the tip is moved to — the resonance falls off over
  ~0.4 nm, and the judge asks for the sweep to start within 0.15 nm of the atom;
* **the spectra**: lock-in on at a small amplitude (a large one widens the line, and the
  fraction of it that reaches the junction is not known), the stabilisation point of imaging
  (+0.1 V, outside the resonance), ±50 mV, several sweeps on the atom and several on the bare
  patch with the same settings; a sweep whose current steps (an unstable tip) is left out;
* **the fit**: the atom's mean divided by the bare patch's mean, fitted with the Fano line shape
  through the instrument's thermal and lock-in broadening
  (:func:`stmbench.analysis.kondo.fit_fano`). The temperature is the one ``info`` reports; the
  amplitude is the one set. Γ, q and ε_K are reported citing the sweeps that went into them.

What the baseline reaches is what a script reaches from the observations alone
(``stmbench.cli baseline --paper P6``).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..analysis.adatoms import locate
from ..analysis.tip import diagnose
from ..instrument import Instrument
# pure measurement helpers and settings of the reference; none of them reads the world
from ..reference._manip import SLOW_LINE_S, SLOW_NM, SLOW_PX, nearest
from ._common import JUDGE_NM, JUDGE_PX, Blind, Ops
from .p4 import SEARCH, step_free

SCENARIO = "P6_kondo_co_cu111"
LINE_S = 0.1
#: no other bump within this of the atom (nm), no step within STEP_NM, and the atom and its
#: bare patch at least EDGE_NM inside the survey
ISOLATION_NM, STEP_NM, EDGE_NM = 3.0, 3.0, 5.0
#: where the bare patch is looked for: distances from the atom (nm), the first that is clear of
#: bumps (BARE_CLEAR_NM) and steps wins. The judge asks for ≥ 2.0 nm from any atom; the margin
#: is for the drift and for a second bump the survey did not resolve.
BARE_RADII_NM, BARE_CLEAR_NM = (3.5, 4.5, 5.5), 3.0
#: an atom lost to the tip or to a poke: the next one starts this far from it (nm)
AVOID_NM = 4.0

#: the sweeps over the resonance. ±50 mV holds a line of up to ~20 meV half width with a
#: baseline either side; 101 points make 1 mV steps (thermal FWHM at 4.4 K is 1.3 meV).
#: The modulation amplitude and the integration time are the trade the task is about: the
#: line is broadened by √3·V_p (FWHM), the noise grows as 1/V_p and 1/√t_int.
WINDOW_V, POINTS, SETTLE_MS = 0.05, 201, 5.0
AMP_MV, INTEG_MS = 3.0, 100.0
N_ATOM, N_BARE = 6, 4
#: the tip hangs on a qPlus fork: a poke at more than ~50 mV rings it up (a blunt, multiple tip
#: and a spray of clusters that takes every atom within 15 nm), so pokes are made at 20 mV and
#: 20 pA (the operator's rule of thumb for this sensor), imaging stays at the task's 0.1 V.
POKE_V, POKE_PA = 0.02, 20.0
#: the stabilisation point of the sweeps (V): their window ends at +50 mV, outside the resonance
SWEEP_BIAS_V = 0.05
#: what is kept back when sweeps stop (s): the fit, the report
RESERVE_S = 120.0

#: the check of the tip over bare Cu(111): −0.6…+0.3 V, the onset at −0.44 V in the middle
CHECK_V = (-0.6, 0.3)
CHECK_POINTS, CHECK_INTEG_MS, CHECK_AMP_MV = 150, 20.0, 10.0
#: a bump in the check's residual above this fraction of the spectrum's level is not the
#: sample's (a tip's peak is 0.5-0.8 of the density of states; the noise at 20 ms is ~1.5 %)
FLAW_FRACTION = 0.10
#: the residual within this of the fitted edge (V) is not read
EDGE_MASK_V = 0.07
TIP_ROUNDS = 3


class Lost(Exception):
    """The atom cannot be found where it was: taken by the tip, drifted away, or gone."""


# ── judging the tip from a spectrum ──

def spectral_flaw(v, didv, *, edge_v: float = -0.44) -> dict:
    """A bare-Cu(111) sweep read for what is not the sample's: the surface-state step
    (an error function at ``edge_v``, free to move ±60 mV and to be 2-50 mV wide) on a
    quadratic background is fitted, and the largest smoothed residual, as a fraction of the
    fitted level, is the flaw. Returns ``{"flaw", "clean", "v_peak"}``; a sweep that cannot be
    fitted is not clean."""
    from scipy.optimize import curve_fit
    from scipy.special import erf

    from ..analysis.spectra import smooth

    v = np.asarray(v, float)
    y = np.asarray(didv, float)
    ok = np.isfinite(v) & np.isfinite(y)
    v, y = v[ok], y[ok]
    if v.size < 30 or float(np.ptp(v)) < 0.5:
        return {"flaw": float("nan"), "clean": False, "v_peak": float("nan")}
    def model(vv, e0, w, a, b, c, d):
        return a + b * 0.5 * (1.0 + erf((vv - e0) / w)) + c * (vv + 0.1) + d * (vv + 0.1) ** 2

    lo = float(np.median(y[: max(3, y.size // 10)]))
    hi = float(np.median(y[-max(3, y.size // 10):]))
    best = None
    for e0 in (edge_v, edge_v - 0.03, edge_v + 0.03):
        try:
            p, _ = curve_fit(model, v, y, p0=(e0, 0.01, lo, max(hi - lo, 1e-15), 0.0, 0.0),
                             bounds=([edge_v - 0.06, 0.002, -np.inf, -np.inf, -np.inf, -np.inf],
                                     [edge_v + 0.06, 0.05, np.inf, np.inf, np.inf, np.inf]), maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        res = float(np.sum((model(v, *p) - y) ** 2))
        if best is None or res < best[0]:
            best = (res, p)
    if best is None:
        return {"flaw": float("nan"), "clean": False, "v_peak": float("nan")}
    fit = model(v, *best[1])
    r = smooth(y - fit, 9)
    # the step's own shape is not an error function (the dev2 probe, seeds 0-7: every tip, clean
    # or not, left 0.09-0.15 of residual within 40 mV of the edge), so only the residual away
    # from the edge is read; a peak right on the edge is not seen
    r = np.where(np.abs(v - float(best[1][0])) > EDGE_MASK_V, r, 0.0)
    k = int(np.argmax(np.abs(r)))
    flaw = float(abs(r[k]) / max(float(np.max(np.abs(fit))), 1e-30))
    return {"flaw": flaw, "clean": bool(flaw < FLAW_FRACTION), "v_peak": float(v[k])}


def outlier_sweeps(currents, v, *, step: float = 0.06, min_v: float = 0.005) -> list[int]:
    """Which of several sweeps of one place had its current step (the tip changed inside it):
    each sweep's current over the median sweep's, a step in that ratio above ``step`` (compared
    over runs of four points either side, where |V| > ``min_v``). Needs three sweeps."""
    c = np.asarray(currents, float)
    if c.shape[0] < 3:
        return []
    v = np.asarray(v, float)
    med = np.median(c, axis=0)
    use = np.abs(v) > min_v
    bad = []
    for s in range(c.shape[0]):
        r = (c[s] / np.where(med == 0.0, np.nan, med))[use]
        r = r[np.isfinite(r)]
        if r.size < 20:
            continue
        sc = max(abs(float(np.median(r[k:k + 4])) - float(np.median(r[k - 4:k]))) for k in range(4, r.size - 3))
        if sc > step:
            bad.append(s)
    return bad


# ── choosing the atom and its bare patch ──

def _bare_patches(frame: dict, atoms, obstacles, around, radii, *, band_nm: float = 1.5):
    """Points on circles of the given radii about ``around`` that are clear of every bump
    (``BARE_CLEAR_NM``), inside the survey by ``EDGE_NM``, and on the same terrace: no step
    within ``band_nm`` of the point or of the line to ``around``. Nearest radius first."""
    cx, cy = (float(t) for t in frame["center_nm"])
    half = float(frame["size_nm"]) / 2
    obs = np.array([a[:2] for a in obstacles], float).reshape(-1, 2)
    ax, ay = around
    for r in radii:
        for deg in range(0, 360, 20):
            px, py = ax + r * math.cos(math.radians(deg)), ay + r * math.sin(math.radians(deg))
            if max(abs(px - cx), abs(py - cy)) > half - EDGE_NM:
                continue
            if len(obs) and float(np.min(np.hypot(obs[:, 0] - px, obs[:, 1] - py))) < BARE_CLEAR_NM:
                continue
            if step_free(frame, (ax, ay), (px, py), atoms, band_nm=band_nm) and \
                    step_free(frame, (px, py), (px + 1e-3, py), atoms, band_nm=band_nm):
                yield (float(px), float(py))


def candidates(frame: dict, atoms, obstacles=None, *, avoid=(), why: dict | None = None):
    """Atoms of a survey that can carry the experiment, nearest the frame's centre first, each
    with its bare patch: ``[((x, y, h), (bx, by))]``. No other bump (compact or not,
    ``obstacles``) within ``ISOLATION_NM``, no step within ``STEP_NM`` (the background of the
    square around the atom stays within 70 pm of a plane, atoms left out), not near an atom
    lost, a height like the others', inside the survey by ``EDGE_NM``, and a bare patch
    (:func:`_bare_patches`). ``why`` counts the reasons the others were passed over."""
    why = {} if why is None else why

    def no(reason: str) -> None:
        why[reason] = why.get(reason, 0) + 1

    pts = np.array([a[:2] for a in atoms], float).reshape(-1, 2)
    if not len(pts):
        return []
    obs = pts if obstacles is None else np.array([a[:2] for a in obstacles], float).reshape(-1, 2)
    typical = float(np.median([a[2] for a in atoms]))
    cx, cy = (float(t) for t in frame["center_nm"])
    half = float(frame["size_nm"]) / 2
    out = []
    for x, y, h in atoms:
        if len(atoms) >= 3 and not 0.6 * typical <= h <= 1.5 * typical:
            no("height")
            continue
        if max(abs(x - cx), abs(y - cy)) > half - EDGE_NM:
            no("edge")
            continue
        others = obs[np.hypot(obs[:, 0] - x, obs[:, 1] - y) > 0.3] if len(obs) else obs
        if len(others) and float(np.min(np.hypot(others[:, 0] - x, others[:, 1] - y))) <= ISOLATION_NM:
            no("neighbour")
            continue
        if any(math.hypot(px - x, py - y) <= AVOID_NM for px, py in avoid):
            no("avoid")
            continue
        if not step_free(frame, (x, y), (x + 1e-3, y), atoms, band_nm=STEP_NM):
            no("step")
            continue
        bare = next(iter(_bare_patches(frame, atoms, obs, (x, y), BARE_RADII_NM)), None)
        if bare is None:
            no("no bare patch")
            continue
        out.append((math.hypot(x - cx, y - cy), (float(x), float(y), float(h)), bare))
    out.sort(key=lambda c: c[0])
    return [(c[1], c[2]) for c in out]


def bare_spot(frame: dict, atoms, obstacles, near) -> tuple[float, float] | None:
    """Bare terrace of a survey nearest ``near`` for the check of the tip: clear of every bump
    and step. ``None`` when the survey has none."""
    grid = [(near[0] + dx, near[1] + dy) for dx in range(-16, 17, 4) for dy in range(-16, 17, 4)]
    grid.sort(key=lambda p: math.hypot(p[0] - near[0], p[1] - near[1]))
    cx, cy = (float(t) for t in frame["center_nm"])
    half = float(frame["size_nm"]) / 2
    obs = np.array([a[:2] for a in obstacles], float).reshape(-1, 2)
    for px, py in grid:
        if max(abs(px - cx), abs(py - cy)) > half - 4.0:
            continue
        if len(obs) and float(np.min(np.hypot(obs[:, 0] - px, obs[:, 1] - py))) < 4.0:
            continue
        if step_free(frame, (px, py), (px + 1e-3, py), atoms, band_nm=2.0):
            return (float(px), float(py))
    return None


# ── operating ──

class Bench:
    """The session's instrument settings, the sweeps taken so far, and the places to sweep."""

    def __init__(self, ops: Ops, log: dict, temperature_k: float):
        self.ops, self.v, self.log, self.temp_k = ops, ops.v, log, float(temperature_k)
        self.lockin_mv: float | None = None

    def lockin(self, amp_mv: float) -> None:
        if self.lockin_mv != amp_mv:
            self.v.act("lockin", on=True, amplitude_mv=amp_mv)
            self.lockin_mv = amp_mv

    def sweep(self, at, v0: float, v1: float, points: int, integ_ms: float) -> dict:
        """The tip to ``at``, then one sweep; the spectrum as the instrument reports it."""
        self.ops.go(*at)
        r = self.v.act("sts", v_start=v0, v_end=v1, points=points, settle_ms=SETTLE_MS,
                       integration_ms=integ_ms, backward=False)
        s = self.v.spectrum(r["spectrum"])
        self.log["n_sweeps"] = self.log.get("n_sweeps", 0) + 1
        return s

    def low_bias(self, on: bool) -> None:
        """20 mV / 20 pA for conditioning (see ``POKE_V``), or back to the imaging point."""
        v, pa = (POKE_V, POKE_PA) if on else (self.ops.imaging_v, self.ops.imaging_pa)
        if on:
            self.v.act("set_setpoint", current_pa=pa)
            self.v.act("set_bias", bias_v=v)
        else:
            self.v.act("set_bias", bias_v=v)
            self.v.act("set_setpoint", current_pa=pa)

    def condition(self):
        """:meth:`Ops.condition` (pulses and shallow pokes) with the pokes at the low bias."""
        self.low_bias(True)
        try:
            return self.ops.condition()
        finally:
            self.low_bias(False)

    def poke_ground(self, n: int, depth_nm: float) -> None:
        """Pokes on fresh ground 60 nm below home, as :meth:`Ops.poke_near` but of a chosen depth
        (a poke that does not reach contact changes nothing on the tip)."""
        ops = self.ops
        hx, hy = ops.home
        for _ in range(n):
            k = ops.n_pokes
            ops.n_pokes += 1
            ops.go(hx + 3.0 * (k % 10) - 13.5, hy - 60.0 - 3.0 * (k // 10))
            self.v.act("poke", depth_nm=depth_nm, dwell_s=0.5, bias_v=POKE_V)
        ops.go(hx, hy)
        ops.jumps.clear()
        self.log["pokes"] = self.log.get("pokes", 0) + n

    def centre(self, guess, *, within: float = 1.2) -> tuple[float, float]:
        """Where the atom is now: its centroid in a slow frame centred on ``guess`` (a wider one
        when it is not in the first). Raises :class:`Lost` when neither shows it."""
        for size, px in ((SLOW_NM, SLOW_PX), (8.0, 80)):
            f, _ = self.ops.frame(guess[0], guess[1], size, px, SLOW_LINE_S)
            hit = nearest(locate(f), guess[0], guess[1], within)
            if hit is not None:
                return hit
            within = 2.5
        raise Lost("the atom is not in the frame centred on it")

    def tip_check(self, spot) -> dict:
        """A sweep over bare surface at ``spot`` and its reading (:func:`spectral_flaw`)."""
        self.lockin(CHECK_AMP_MV)
        s = self.sweep(spot, CHECK_V[0], CHECK_V[1], CHECK_POINTS, CHECK_INTEG_MS)
        out = spectral_flaw(s["v"], s["didv"])
        self.log.setdefault("tip_checks", []).append({k: (None if isinstance(x, float) and math.isnan(x) else x)
                                                      for k, x in out.items()})
        return out

    def residual_drift(self, at, *, size_nm: float = 20.0) -> tuple[float, float]:
        """What the compensation leaves: the atom at ``at`` in a slow frame, again 30 s later, and
        the velocity between the two (nm/min)."""
        a = self.centre(at)
        t0 = self.ops.now()
        self.v.act("wait", seconds=30.0)
        b = self.centre(a)
        dt = (self.ops.now() - t0) / 60.0
        return (b[0] - a[0]) / dt, (b[1] - a[1]) / dt


def fit_sweeps(atom: list[dict], bare: list[dict], temp_k: float, amp_mv: float) -> dict:
    """Γ (meV), q and ε_K (meV) from the sweeps: the mean over the atom's sweeps against the mean
    over the bare ones (smoothed by a cubic in V: bare Cu(111) has no structure in ±50 mV, and
    the reference's own noise would add to the quotient's), through the instrument's broadening."""
    from ..analysis.kondo import fit_fano

    v = np.asarray(atom[0]["v"], float)
    ya = np.mean([np.asarray(s["didv"], float) for s in atom], axis=0)
    yb = np.mean([np.asarray(s["didv"], float) for s in bare], axis=0)
    ref = np.polyval(np.polyfit(v, yb, 3), v)
    r = fit_fano(v, ya, temp_k=temp_k, vmod_peak_v=amp_mv * 1e-3, reference=ref)
    return {"gamma_mev": float(r.gamma_v) * 1e3, "q_fano": float(r.q), "e_k_mev": float(r.e_k_v) * 1e3,
            "amp": float(r.amp), "r2": float(r.r2), "success": bool(r.success), "note": str(r.note)}


def play(v: Blind, log: dict) -> dict | None:
    """One P6 session through ``v``: the fitted Γ, q and ε_K and the sweeps they cite, or
    ``None`` when no atom could be measured."""
    ops = Ops(v, log)
    bench = Bench(ops, log, float(v.info()["temperature_k"]))
    home = ops.home
    log["temperature_k"] = bench.temp_k
    f, _ = ops.frame(*home, JUDGE_NM, JUDGE_PX, LINE_S)
    diag = diagnose(f)
    log["tip_first"] = diag["verdict"]
    bench.poke_ground(3, 0.4)
    waited = 120.0
    if diag["verdict"] != "ok":
        diag, f = bench.condition()
        waited = 300.0
        if diag["verdict"] != "ok":
            raise RuntimeError("the tip could not be conditioned")
    v.act("wait", seconds=waited)

    # spectroscopy judges what images cannot: a peak that is the apex's, not the sample's
    for k in range(TIP_ROUNDS + 1):
        atoms = locate(f)
        spot = bare_spot(f, atoms, locate(f, compact_frac=1.0), home)
        if spot is None:
            log["tip_check_skipped"] = "no bare terrace in the survey"
            break
        check = bench.tip_check(spot)
        if check["clean"]:
            break
        if k == TIP_ROUNDS:
            raise RuntimeError("the tip's spectrum keeps a peak that is not the sample's")
        bench.poke_ground(3, 0.4 if k == 0 else 0.65)
        v.act("wait", seconds=120.0)
    log["tip_spectrum_rounds"] = k + 1

    comp = ops.drift(*_anchor(f, home))
    lost: list[tuple[float, float]] = []
    for _ in range(4):
        chosen, survey = None, None
        for centre in SEARCH:
            survey, _ = ops.frame(home[0] + centre[0], home[1] + centre[1], JUDGE_NM, JUDGE_PX, LINE_S)
            if diagnose(survey)["verdict"] in ("blunt", "double"):
                break
            why: dict = {}
            cands = candidates(survey, locate(survey), locate(survey, compact_frac=1.0), avoid=lost, why=why)
            log.setdefault("passed_over", []).append(why)
            if cands:
                chosen = cands[0]
                break
        else:
            raise RuntimeError(f"no isolated atom with a bare patch near home ({log['passed_over']})")
        if chosen is None:                     # the survey showed a blunt or double tip
            d, f = bench.condition()
            if d["verdict"] != "ok":
                raise RuntimeError("the tip could not be conditioned")
            v.act("wait", seconds=300.0)
            comp = ops.drift(*_anchor(f, home))
            continue
        atom, bare = chosen
        log["atom_nm"], log["bare_nm"], log["atoms_tried"] = list(atom[:2]), list(bare), len(lost) + 1
        try:
            return _measure(bench, atom[:2], bare, comp, log)
        except Lost as exc:
            log.setdefault("lost", []).append(str(exc))
            lost.append((atom[0], atom[1]))
            bench.poke_ground(3, 0.4)
            v.act("wait", seconds=120.0)
            comp = ops.drift(*_anchor(survey, home, avoid=lost))
    return None


def _anchor(frame: dict, home, *, avoid=()) -> tuple[float, float]:
    """An atom of a frame to measure the drift on: the one nearest home, away from atoms lost.
    Home itself when the frame shows none."""
    atoms = [a for a in locate(frame) if all(math.hypot(a[0] - p[0], a[1] - p[1]) > 2.0 for p in avoid)]
    best = min(atoms, key=lambda a: math.hypot(a[0] - home[0], a[1] - home[1]), default=None)
    return (float(best[0]), float(best[1])) if best is not None else (float(home[0]), float(home[1]))


def _measure(bench: Bench, atom, bare, comp, log: dict) -> dict | None:
    """The drift at the atom, then the sweeps on it and on its bare patch, then the fit."""
    ops, v = bench.ops, bench.v
    ops.go(*atom)
    v.act("wait", seconds=60.0)
    comp = ops.drift(*atom)
    here = bench.centre(atom, within=2.5)
    for _ in range(2):                          # a check, and what it finds taken up
        rx, ry = bench.residual_drift(here)
        log.setdefault("drift_residual_nm_min", []).append([round(rx, 3), round(ry, 3)])
        if math.hypot(rx, ry) <= 0.15:
            break
        comp = (comp[0] + rx, comp[1] + ry)
        v.act("drift_comp", on=True, vx_nm_per_min=comp[0], vy_nm_per_min=comp[1])
    v.act("set_bias", bias_v=SWEEP_BIAS_V)
    bench.lockin(AMP_MV)
    on_atom: list[dict] = []
    for _ in range(N_ATOM + 3):
        if len(on_atom) >= N_ATOM or ops.now() > ops.budget_s - RESERVE_S - 400.0:
            break
        here = bench.centre(here)               # immediately before the sweep
        on_atom.append(bench.sweep(here, -WINDOW_V, WINDOW_V, POINTS, INTEG_MS))
    if len(on_atom) < 3:
        raise RuntimeError("the budget ran out before three sweeps on the atom")
    bare_sw = [bench.sweep(bare, -WINDOW_V, WINDOW_V, POINTS, INTEG_MS) for _ in range(N_BARE)]
    # sweeps whose current stepped (the tip changed inside them) are left out
    vv = on_atom[0]["v"]
    drop_a = set(outlier_sweeps([s["i"] for s in on_atom], vv))
    drop_b = set(outlier_sweeps([s["i"] for s in bare_sw], vv))
    log["dropped_sweeps"] = len(drop_a) + len(drop_b)
    on_atom = [s for k, s in enumerate(on_atom) if k not in drop_a]
    bare_sw = [s for k, s in enumerate(bare_sw) if k not in drop_b]
    if len(on_atom) < 2 or len(bare_sw) < 1:
        raise RuntimeError("too few sweeps without a step in the current")
    fit = fit_sweeps(on_atom, bare_sw, bench.temp_k, AMP_MV)
    log["fit"] = fit
    return {"fit": fit, "evidence": [s["name"] for s in on_atom + bare_sw], "atom_sweeps": [s["name"] for s in on_atom],
            "bare_sweeps": [s["name"] for s in bare_sw]}


def run(seed: int, *, out_dir: str | Path | None = None, scenario: str = SCENARIO) -> dict:
    """One P6 episode by the baseline; what it reported, how it scored, and (from the truth,
    after the episode only) where it went wrong."""
    inst = Instrument(scenario, seed, out_dir)
    log: dict = {}
    got, error = None, None
    try:
        got = play(Blind(inst), log)
    except Exception as exc:  # noqa: BLE001 — a failed session is a result
        error = f"{type(exc).__name__}: {exc}"
    if got is not None:
        fit = got["fit"]
        for cid in ("gamma_mev", "q_fano", "e_k_mev"):
            inst.report(cid, fit[cid], unit="meV" if cid != "q_fano" else "", evidence=got["evidence"],
                        note=f"Fano fit of the mean of {len(got['atom_sweeps'])} sweeps on the atom over "
                             f"{len(got['bare_sweeps'])} on bare surface")
    out = inst.finish()
    det = (inst.verdict.details or {}).get("claims", {})
    err = {}
    for cid, c in det.items():
        t = c.get("truth")
        if got is not None and isinstance(t, (int, float)):
            err[cid] = float(got["fit"][cid]) - float(t)
    # after the episode: where the sweeps were, from the instrument's own records of them
    where = {}
    if got is not None:
        recs = {int(r["idx"]): r for r in inst.world.sts_records}
        for kind in ("atom_sweeps", "bare_sweeps"):
            where[kind] = [[round(float(recs[inst.spectra[n]["index"]]["near_nm"]), 3),
                            str(recs[inst.spectra[n]["index"]]["near_kind"])] for n in got[kind]]
    return {"paper": "P6", "seed": int(seed), "success": bool(out["success"]),
            "partial": float(out.get("partial", 0.0)), "error": error,
            "claims_ok": {k: bool(c.get("value_ok")) and bool(c.get("evidence_ok", True)) for k, c in det.items()},
            "evidence_ok": {k: bool(c.get("evidence_ok")) for k, c in det.items()},
            # the fitted value minus the hidden one, per claim (meV, none, meV)
            "err": err, "sweep_near": where,
            "instrument_s": float(inst.used_s),
            "tip_condition": inst.scenario.draw_hidden(seed).get("tip", {}).get("condition"), **log}
