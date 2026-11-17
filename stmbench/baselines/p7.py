"""P7 scripted baseline: the pull threshold of two isolated atoms with nothing but what the
instrument shows — no truth.

The claim (``r_pull_1_kohm``, ``r_pull_2_kohm``): the panel resistance at which a drag of 1.0 nm at
0.1 nm/s, started with the tip over the atom, carries the atom to the end point in half of the
attempts. What the reference (:mod:`stmbench.reference.p7`) reads from the truth, this reads from
frames and traces:

* **the tip and the drift** as P4's baseline does (:mod:`stmbench.baselines.p4`): a survey judges the
  tip (shallow pokes first, a blunt or double tip pulsed away and judged again), the drift is
  measured with the compensation off and then compensated, a tip that changes during the series
  (row jumps in two frames within 15 minutes) is poked and the series started again;
* **which atoms**: two bumps of the survey with no other bump within 3 nm (checked against every bump,
  compact or not), no step near, away from the survey's edge, at least 5 nm from each other;
* **where an atom is**: a slow 5 nm frame before each drag puts the tip over it (the same imaging apex
  places the tip and shows the atom, so the tip starts within the placement error of the atom); the
  atom lifted onto the tip shows in no frame, is buried with the pokes, and another one is taken;
* **whether a drag carried the atom**: a slow frame after it (:mod:`stmbench.analysis.carry`): the atom
  against where the tip stopped. The trace is the fallback;
* **the threshold**: an adaptive staircase (up while it carries, down while it does not, then one up
  one down in small steps, alternating directions so that the atom stays near where it was) and a
  maximum-likelihood logistic fit in ln R over all drags of the atom. The two values are cited with
  the move traces of their series.

All the settings of the protocol are in :class:`Protocol`: the reference's calibration is where they
come from, and they move together.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..analysis.adatoms import locate
from ..analysis.carry import find_atom, read_carry
from ..analysis.tip import diagnose
from ..instrument import Instrument
from ..reference._manip import nearest
from ._common import JUDGE_NM, JUDGE_PX, Blind, Ops
from .p4 import AVOID_NM, LINE_S, SEARCH, _anchor, _recover, candidates

SCENARIO = "P7_pull_threshold_cu111"


@dataclass(frozen=True)
class Protocol:
    """The drag protocol and the staircase (the reference's, until its calibration says otherwise)."""
    length_nm: float = 1.0
    speed_nm_s: float = 0.1
    #: manipulation bias: at one resistance a lower bias is a lower current, and the tip's change
    #: rate grows with the current (the reference's 2 mV)
    bias_v: float = 0.002
    max_na: float = 80.0
    start_kohm: float = 300.0
    up: float = 1.25
    down: float = 0.8
    #: the staircase after both outcomes were seen: one up, one down, in steps of this factor
    step: float = 1.05
    #: drags per atom (the search for both outcomes included)
    drags_per_atom: int = 32
    min_kohm: float = 20.0
    max_kohm: float = 3000.0
    #: the second atom is at least this far from the first (nm), the frame placing the tip is this big
    apart_nm: float = 5.0
    reserve_s: float = 240.0
    #: the slow frames that place the atom: side (nm), pixels, line time (s) - 0.1 nm/px at 16 nm/s,
    #: where the loop's lag is ~0.01 nm (P4's 5 nm / 50 px / 0.25 s costs 25 s, this 6 s)
    frame_nm: float = 3.2
    frame_px: int = 32
    frame_line_s: float = 0.2
    #: a fresh frame over the atom before every n-th drag (0: only to start, and after a tip event);
    #: otherwise the drag starts where the frame after the last one put the atom
    pre_frame_every: int = 0


PROTO = Protocol()
ISOLATION_NM = 3.0


class Lost(Exception):
    """The atom is no longer where it can be followed: taken by the tip, or gone."""


class Puller:
    """Manipulation and imaging settings, and the frames that place the atom and read the drag."""

    def __init__(self, ops: Ops, proto: Protocol = PROTO):
        self.ops, self.v, self.p = ops, ops.v, proto
        #: instrument seconds spent on the drags, the frames before and after them, and getting onto the atom
        self.spent = {"drag": 0.0, "frame_before": 0.0, "frame_after": 0.0, "move_and_settings": 0.0}

    def manipulate(self, r_ohm: float) -> None:
        """Manipulation conditions: the preamp's range first, then the bias, then the setpoint."""
        v = min(self.p.bias_v, r_ohm * self.p.max_na * 1e-9)
        self.v.act("set_current_range", full_scale_na=100.0)
        self.v.act("set_bias", bias_v=v)
        self.v.act("set_setpoint", current_pa=v / r_ohm * 1e12)

    def image(self) -> None:
        """Back to imaging, in the reverse order: the setpoint comes down before the range does."""
        self.v.act("set_setpoint", current_pa=self.ops.imaging_pa)
        self.v.act("set_bias", bias_v=self.ops.imaging_v)
        self.v.act("set_current_range", full_scale_na=10.0)

    def measure(self, guess, *, within: float = 1.0, wide: bool = False):
        """The atom in a slow frame centred on ``guess``: its place, or ``None``. ``wide``: P4's 5 nm
        frame, for when ``guess`` is where a survey or the creep of a long move put the atom."""
        t0 = self.ops.now()
        size, px, line = (5.0, 50, 0.25) if wide else (self.p.frame_nm, self.p.frame_px, self.p.frame_line_s)
        f, _ = self.ops.frame(guess[0], guess[1], size, px, line)
        self.spent["frame_before"] += self.ops.now() - t0
        return nearest(locate(f), guess[0], guess[1], within)

    def pull(self, pos, sign: float, r_ohm: float) -> dict:
        """One standard drag from ``pos`` (where a frame put the atom), along ±x by ``sign``, at
        ``r_ohm`` (panel resistance). The tip goes onto ``pos`` at the imaging setpoint, the drag is
        made at ``r_ohm``, and a slow frame centred on the path says where the atom is now.
        Returns the drag with its trace, the reading of it (:func:`stmbench.analysis.carry.read_carry`)
        and the atom's place after it (``None``: not found)."""
        end = (pos[0] + sign * self.p.length_nm, pos[1])
        t0 = self.ops.now()
        self.v.act("move_tip", x_nm=pos[0], y_nm=pos[1])
        self.manipulate(r_ohm)
        t1 = self.ops.now()
        r = self.v.act("move_tip", x_nm=end[0], y_nm=end[1], speed_nm_s=self.p.speed_nm_s)
        t2 = self.ops.now()
        self.image()
        t3 = self.ops.now()
        self.spent["move_and_settings"] += (t1 - t0) + (t3 - t2)
        self.spent["drag"] += t2 - t1
        spectrum = self.v.spectrum(r["trace"])
        tip_end = tuple(float(c) for c in r["tip_nm"])
        mid = (0.5 * (pos[0] + tip_end[0]), 0.5 * (pos[1] + tip_end[1]))
        f, _ = self.ops.frame(mid[0], mid[1], self.p.frame_nm, self.p.frame_px, self.p.frame_line_s)
        self.spent["frame_after"] += self.ops.now() - t3
        atom = find_atom(f, pos, tip_end)
        return {"trace": r["trace"], "r_ohm": float(r_ohm), "sign": float(sign), "start": tuple(pos),
                "end": tip_end, "atom": atom, "read": read_carry(spectrum, end=tip_end, atom=atom)}


def fit_logistic(r_ohm, carried) -> dict | None:
    """The maximum-likelihood logistic fit of P(carry) against ln R, location ``mu`` and slope ``k``
    both free (P = 1 / (1 + exp(k (ln R - mu)))): ``{"r_ohm": exp(mu), "k": k}``. ``None`` unless both
    outcomes occurred. When the data separate (every carry below every miss) the likelihood has a
    plateau in ``mu``, and the middle of it is taken; the slope is held to 1-400 per ln unit (as the
    reference's fit does)."""
    x = np.log(np.asarray(r_ohm, float))
    y = np.asarray(carried, float)
    if not (y.any() and (1.0 - y).any()):
        return None
    mus = np.linspace(x.min() - 0.5, x.max() + 0.5, 1201)
    ks = np.exp(np.linspace(0.0, math.log(400.0), 60))
    z = ks[None, :, None] * (x[None, None, :] - mus[:, None, None])          # (mu, k, drag)
    # log P(carry) = -log(1 + e^z); log P(miss) = -log(1 + e^-z)
    ll = (-np.logaddexp(0.0, z) * y - np.logaddexp(0.0, -z) * (1.0 - y)).sum(axis=2)
    best = ll.max(axis=1)
    top = np.flatnonzero(best >= best.max() - 1e-9)
    j = int(top[len(top) // 2])
    k = float(ks[int(np.argmax(ll[j]))])
    return {"r_ohm": float(math.exp(mus[j])), "k": k}


class Series:
    """One atom's drags and the staircase that chooses the next resistance."""

    def __init__(self, proto: Protocol = PROTO):
        self.p, self.drags = proto, []
        self.r = proto.start_kohm * 1e3
        self.seen_carry = self.seen_miss = False

    def readable(self) -> list[dict]:
        return [d for d in self.drags if d["read"]["carried"] is not None]

    def add(self, drag: dict) -> None:
        """Record a drag and choose the next resistance: before both outcomes were seen, up by
        ``up`` after a carry and down by ``down`` after a miss; then one step up after a carry and
        one down after a miss."""
        self.drags.append(drag)
        c = drag["read"]["carried"]
        if c is None:
            return
        self.seen_carry |= bool(c)
        self.seen_miss |= not c
        if not (self.seen_carry and self.seen_miss):
            # looking for both outcomes: a carry means the threshold is above this resistance (go up)
            self.r = drag["r_ohm"] * (self.p.up if c else self.p.down)
            return
        # the fit-guided design of the reference's calibration (2026-11-17, 30 seeds, fork physics): next
        # to the present fit, at most x1.3 from the last resistance, dithered one step up and down
        # alternately so that the slope stays measured. At 12 drags p90 7.2 % against 9.2 % for the
        # plain one-up-one-down staircase
        rd = self.readable()
        fit = fit_logistic([d["r_ohm"] for d in rd], [d["read"]["carried"] for d in rd])
        last = drag["r_ohm"]
        base = min(max(fit["r_ohm"] if fit else last, last / 1.3), last * 1.3)
        self.r = base * (self.p.step if len(self.drags) % 2 else 1.0 / self.p.step)

    def estimate(self) -> dict | None:
        rd = self.readable()
        return fit_logistic([d["r_ohm"] for d in rd], [d["read"]["carried"] for d in rd])


def _lone_atoms(frame: dict, atoms, *, edge_nm: float = 5.0) -> list[tuple[float, float, float]]:
    """The bumps of a survey that can be pulled on: no other bump (compact or not) within
    :data:`ISOLATION_NM`, no step within 1.5 nm of the path, 5 nm inside the frame. The tip's
    ghosts (a double tip) would disqualify every atom: the caller judged the tip first."""
    return candidates(frame, atoms, (PROTO.length_nm, 0.0), obstacles=locate(frame, compact_frac=1.0))


def _prepare(v: Blind, ops: Ops, log: dict):
    """The tip judged and conditioned, the drift measured: the first survey and the compensation."""
    f, _ = ops.frame(*ops.home, JUDGE_NM, JUDGE_PX, LINE_S)
    diag = diagnose(f)
    log["tip_first"] = diag["verdict"]
    ops.poke_near()
    waited = 120.0
    if diag["verdict"] != "ok":
        diag, f = ops.condition()
        waited = 300.0
        if diag["verdict"] != "ok":
            raise RuntimeError("the tip could not be conditioned")
    v.act("wait", seconds=waited)
    return f, ops.drift(*_anchor(f, ops.home))


def _pick_atom(v: Blind, ops: Ops, log: dict, comp, lost: list, done: list) -> tuple[tuple[float, float], tuple]:
    """An isolated atom of the surveys around home: not within 11 nm of one lost, at least 5 nm from
    one measured. Returns (its place, the drift compensation). A survey that shows a blunt or double
    tip has the tip conditioned first."""
    for _ in range(3):
        bad = False
        for centre in SEARCH:
            survey, _ = ops.frame(ops.home[0] + centre[0], ops.home[1] + centre[1], JUDGE_NM, JUDGE_PX, LINE_S)
            if diagnose(survey)["verdict"] in ("blunt", "double"):
                bad = True
                break
            for x, y, _h in _lone_atoms(survey, locate(survey)):
                if any(math.hypot(x - px, y - py) <= AVOID_NM for px, py in lost):
                    continue
                if any(math.hypot(x - px, y - py) < PROTO.apart_nm for px, py in done):
                    continue
                log["survey_atoms"] = len(locate(survey))
                return (float(x), float(y)), comp
        if bad:
            diag, f = ops.condition()
            if diag["verdict"] != "ok":
                raise RuntimeError("the tip could not be conditioned")
            v.act("wait", seconds=300.0)
            comp = ops.drift(*_anchor(f, ops.home))
    raise RuntimeError("no isolated atom was found near home")


def run_series(ops: Ops, pl: Puller, guess, log: dict, *, name: str) -> Series:
    """The staircase on the atom near ``guess``: drags until the budget of drags per atom is spent
    (or the time is). Raises :class:`Lost` when the atom is no longer seen."""
    pos = pl.measure(guess, wide=True)
    if pos is None:
        raise Lost("the atom is not in the frame centred on it")
    s = Series(pl.p)
    k = 0
    while len(s.drags) < pl.p.drags_per_atom:
        if ops.now() > ops.budget_s - pl.p.reserve_s:
            log.setdefault("time_out", []).append(name)
            break
        if not pl.p.min_kohm * 1e3 <= s.r <= pl.p.max_kohm * 1e3:
            raise Lost("the resistance left the range without both outcomes")
        if ops.unstable():
            # the tip keeps changing: the drags so far were made with another grip
            ops.poke_near()
            comp = _recover(ops, pos, log, why="poked")
            log.setdefault("restarts", []).append({"atom": name, "after_drags": len(s.drags)})
            s = Series(pl.p)
            pos = pl.measure(pos, within=2.0, wide=True)
            if pos is None:
                raise Lost("the atom is not where it was after the tip was poked")
        if pl.p.pre_frame_every and k % pl.p.pre_frame_every == 0:
            pos = pl.measure(pos) or pos
        d = pl.pull(pos, 1.0 if k % 2 == 0 else -1.0, s.r)
        k += 1
        s.add(d)
        if d["atom"] is None:
            raise Lost("the atom is in no frame after the drag")
        pos = d["atom"]
    return s


def play(v: Blind, log: dict) -> dict:
    """One P7 session through ``v``: ``{"r_kohm": [R1, R2], "evidence": [[traces], [traces]],
    "series": [Series, Series]}``."""
    ops = Ops(v, log)
    pl = Puller(ops)
    f, comp = _prepare(v, ops, log)
    log["prepare_s"] = ops.now()
    lost: list[tuple[float, float]] = []
    done: list[tuple[tuple[float, float], Series]] = []
    for _ in range(6):
        g, comp = _pick_atom(v, ops, log, comp, lost, [a for a, _ in done])
        try:
            s = run_series(ops, pl, g, log, name=f"atom{len(done) + 1}")
        except Lost as exc:
            log.setdefault("lost", []).append(str(exc))
            lost.append(g)
            ops.poke_near()                              # buries an atom the tip may carry
            v.act("wait", seconds=120.0)
            comp = ops.drift(*_anchor(f, ops.home, avoid=lost))
            continue
        if s.estimate() is None:
            log.setdefault("no_bracket", []).append(len(s.drags))
            lost.append(g)
            continue
        done.append((g, s))
        if len(done) == 2:
            break
    if len(done) < 2:
        raise RuntimeError(f"only {len(done)} atom(s) measured ({log.get('lost')})")
    ests = [s.estimate() for _, s in done]
    log["spent_s"] = {k: round(x) for k, x in pl.spent.items()}
    return {"r_kohm": [e["r_ohm"] / 1e3 for e in ests],
            "evidence": [[d["trace"] for d in s.drags] for _, s in done],
            "series": [s for _, s in done]}


def run(seed: int, *, out_dir: str | Path | None = None, scenario: str = SCENARIO) -> dict:
    """One P7 episode by the baseline; what it reported and how the judge scored it (the judge's
    truth is read after the episode only)."""
    inst = Instrument(scenario, seed, out_dir)
    log: dict = {}
    got, error = None, None
    try:
        got = play(Blind(inst), log)
    except Exception as exc:  # noqa: BLE001 — a failed session is a result
        error = f"{type(exc).__name__}: {exc}"
    if got is not None:
        for cid, val, ev in zip(("r_pull_1_kohm", "r_pull_2_kohm"), got["r_kohm"], got["evidence"]):
            inst.report(cid, val, unit="kΩ", evidence=ev)
    out = inst.finish()
    det = (inst.verdict.details or {}).get("claims", {})
    rows = {}
    for cid in ("r_pull_1_kohm", "r_pull_2_kohm"):
        c = det.get(cid, {})
        rows[cid] = {"value_ok": c.get("value_ok"), "evidence_ok": c.get("evidence_ok"),
                     "truth": c.get("truth"), "evidence": c.get("evidence")}
    series = [] if got is None else [{"drags": len(s.drags), "estimate_kohm": s.estimate()["r_ohm"] / 1e3 if s.estimate() else None,
                                      "r_kohm": [round(d["r_ohm"] / 1e3, 1) for d in s.drags],
                                      "carried": [d["read"]["carried"] for d in s.drags]} for s in got["series"]]
    return {"paper": "P7", "seed": int(seed), "success": bool(out["success"]), "partial": float(out.get("partial", 0.0)),
            "error": error, "claims_ok": {k: bool(c.get("value_ok")) and bool(c.get("evidence_ok", True)) for k, c in det.items()},
            "claims": rows, "reported_kohm": None if got is None else got["r_kohm"], "series": series,
            "instrument_s": float(inst.used_s),
            "tip_condition": inst.scenario.draw_hidden(seed).get("tip", {}).get("condition"), **log}
