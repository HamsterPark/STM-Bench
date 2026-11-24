"""P7 privileged reference solution: the pull threshold of two isolated atoms, as Bartels, Meyer
and Rieder (1997) measured the resistance at which an adatom follows the tip.

The truth navigates - which atoms are isolated and away from steps, where an atom is now, how far
the sample has drifted, which apex holds it, whether the tip needs conditioning, whether a drag
carried the atom (the drag record's ``followed``, as P4's reference counts the world's hop events
rather than reading traces). It never reads a threshold: the resistance of every drag is chosen
from the outcomes of the drags before it, and the reported value is a fit to the outcomes alone.
(After the episode the calibration table compares it with the truth the judge will use, the exact
R½ of the cited drags: :func:`stmsim.physics.drag_threshold.record_threshold`.)

Protocol:

1. two isolated atoms near the origin: no other atom within 3 nm, no step within 1.5 nm, the
   second at least 5 nm from the first;
2. a tip that is blunt, multiple, unstable or carries something is conditioned ~150 nm from the
   atom (pulses spray clusters ~40 nm around), five minutes for the creep of the long move, then
   the lateral drift from two frames 30 s apart, compensated (as P4);
3. drags of exactly 1.0 nm at 0.1 nm/s, each started with the leading apex over the atom (the
   tip put at the atom less that apex's offset, at the imaging setpoint, then the manipulation
   resistance set: the preamplifier's range first, the bias, then the setpoint - the reverse
   afterwards), alternating +x and -x so that the atom stays near its place. The manipulation
   bias is low (2 mV by default): at a given resistance the current, and with it the tip's
   change hazard, falls with the bias;
4. the resistance of the next drag: from 300 kOhm up by x1.25 after a drag that carried the atom,
   down by x0.8 after one that did not, until both have been seen; then (the default ``fit``
   design) next to the present fit, at most x1.3 from the last drag and dithered x1.05 up and down
   so that the slope stays measured, until ``n_drags`` = 32 drags have been made (design
   ``stair``: a one-up-one-down staircase of x1.05 steps instead). Calibrated on seeds 0-29
   (docs/results/p7_reference_seeds_0-29.json): 2 mV and ``fit`` beat ``stair`` at low n and 5 mV
   at every n, and the error falls as 1/sqrt(n) to a p90 of about 5 % at 32 drags;
5. R½ is the location of a maximum-likelihood logistic fit in ln R over every drag of that atom
   under one grip (location and slope free);
6. a drag whose record shows other conditions than the series' (the grip changed: a crash, a
   pick-up, a settle) starts the series again from there; a drag that did not start over the atom
   is not counted; an atom lifted onto the tip is buried with a shallow poke far away and
   another atom taken;
7. both values are reported with the trace names of the drags that make the series as evidence.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ._common import Session, claims_ok, condition_tip, lateral_drift, tip_needs_poke
from ._manip import Mover, MANIP_MAX_NA

SCENARIO = "P7_pull_threshold_cu111"
CLAIMS = ("r_pull_1_kohm", "r_pull_2_kohm")
IMAGING_V, IMAGING_PA = 0.05, 200.0
#: manipulation bias (V): the tip's change hazard is lambda * (1 + |I| / 1 nA) per second, and at a
#: given resistance I = V / R
MANIP_V = 0.002
START_KOHM, BRACKET_UP, BRACKET_DOWN, STAIR_STEP = 300.0, 1.25, 0.8, 1.05
R_MIN_KOHM, R_MAX_KOHM = 30.0, 4000.0
LENGTH_NM, SPEED_NM_S = 1.0, 0.1
#: the judge wants the holding apex this close to the atom when the drag starts (the contract's
#: ``start_within_nm``); aim well inside
START_WITHIN_NM = 0.3
ISOLATION_NM, STEP_CLEAR_NM, APART_NM = 3.0, 1.5, 5.0
#: truncations of the staircase for the precision-cost curve
CURVE_N = (8, 12, 16, 20, 24, 28, 32)
#: attempts per atom (counted drags or not) before the reference gives it up
MAX_DRAGS_PER_ATOM = 80


class PickedUp(Exception):
    """The atom went onto the tip."""


class GaveUp(Exception):
    """The atom could not be measured (it never followed, or always did)."""


def fit_threshold(r_kohm, followed) -> dict:
    """Maximum-likelihood logistic fit of P(carried) = 1 / (1 + exp(b (ln R - mu))) over the
    drags (location mu and slope b free; b bounded to 1-400 per ln unit, with a faint penalty so
    that a perfectly separated series, which has no finite slope, lands in the middle of its gap).
    Returns ``{"r_hat_kohm", "slope"}``; ``r_hat_kohm`` is ``None`` unless both outcomes occur."""
    from scipy.optimize import minimize

    r = np.log(np.asarray(r_kohm, float))
    y = np.asarray(followed, float)
    if y.size == 0 or y.min() == y.max():
        return {"r_hat_kohm": None, "slope": None}

    def nll(p):
        mu, lb = p
        z = np.exp(lb) * (r - mu)
        # log-likelihood of Bernoulli with p = sigmoid(-z), stably
        ll = -(y * np.logaddexp(0.0, z) + (1.0 - y) * np.logaddexp(0.0, -z))
        return -float(ll.sum()) + 1e-3 * lb ** 2

    mid = float(np.mean(r))
    best = None
    for lb0 in (math.log(10.0), math.log(40.0), math.log(150.0)):
        res = minimize(nll, x0=[mid, lb0], method="L-BFGS-B",
                       bounds=[(float(r.min()) - 0.5, float(r.max()) + 0.5), (0.0, math.log(400.0))])
        if best is None or res.fun < best.fun:
            best = res
    return {"r_hat_kohm": float(math.exp(best.x[0])), "slope": float(math.exp(best.x[1]))}


def _key(cond: dict) -> tuple:
    return tuple(sorted((k, float(v)) for k, v in cond.items()))


def evidence_ok(records: list[dict], r_hat_kohm: float, *, min_n: int = 8, min_resistances: int = 3,
                protocol_rel: float = 0.05, start_within_nm: float = START_WITHIN_NM) -> dict:
    """The contract's evidence rule for one cited series, read off the drag records (the judge's
    own check is not on this branch): every drag integrated, on the protocol (length and speed
    within ``protocol_rel``), started with the holding apex within ``start_within_nm`` of one and
    the same atom; at least ``min_n`` drags at ``min_resistances`` distinct resistances (assumed:
    distinct beyond 1 %); and the series brackets its value: it holds a drag that carried the
    atom and one that did not, and R̂ lies within the resistances tried."""
    ok = {"n": len(records) >= min_n,
          "atom": len({r.get("atom_id") for r in records}) == 1 and None not in {r.get("atom_id") for r in records},
          "active": all(r["active"] and r["feedback"] and not r["over_range"] and not r["withdrawn"]
                        for r in records),
          "protocol": all(abs(r["length_nm"] / LENGTH_NM - 1) <= protocol_rel
                          and abs(r["speed_nm_s"] / SPEED_NM_S - 1) <= protocol_rel for r in records),
          "start": all(r.get("start_dist_nm") is not None and r["start_dist_nm"] <= start_within_nm
                       for r in records)}
    rs = sorted({round(math.log(r["r_kohm"]) / math.log1p(0.01)) for r in records if r.get("r_kohm")})
    ok["resistances"] = len(rs) >= min_resistances
    f = [bool(r.get("followed")) for r in records]
    rr = [r["r_kohm"] for r in records if r.get("r_kohm")]
    ok["bracket"] = bool(any(f) and not all(f) and rr and min(rr) <= r_hat_kohm <= max(rr))
    ok["all"] = all(ok.values())
    return ok


class _Runner:
    def __init__(self, s: Session, *, manip_v: float, step: float, n_drags: int, design: str = "fit"):
        self.s, self.inst, self.w = s, s.inst, s.w
        self.reg = s.w.surface.site.adatoms
        self.mv = Mover(s, imaging_v=IMAGING_V, imaging_pa=IMAGING_PA)
        self.manip_v, self.step, self.n_drags = float(manip_v), float(step), int(n_drags)
        self.design = design
        self.log = {"conditionings": 0, "pickups": 0, "restarts": 0, "discarded": 0, "tip_changes": 0,
                    "atoms_tried": 0}

    # ── navigation ──
    def candidates(self, *, avoid: list[tuple[float, float]]) -> list:
        """Isolated atoms nearest the origin first: nobody within the isolation distance, no step
        within the clearance, ``avoid`` (atoms already measured) further than :data:`APART_NM`."""
        reg, w = self.reg, self.w
        on, xy = reg._arrays()
        out = []
        for k, a in enumerate(on):
            if a.status != "on_surface" or a.history:
                continue
            x, y = float(xy[k, 0]), float(xy[k, 1])
            if int(np.count_nonzero(np.hypot(xy[:, 0] - x, xy[:, 1] - y) <= ISOLATION_NM * 1e-9)) > 1:
                continue
            if not w.surface.step_free_window(x, y, 2 * STEP_CLEAR_NM * 1e-9):
                continue
            if any(math.hypot(x - px, y - py) < APART_NM * 1e-9 for px, py in avoid):
                continue
            out.append((math.hypot(x, y), a))
        return [a for _, a in sorted(out, key=lambda t: t[0])]

    def prepare(self, atom) -> None:
        """A good tip, far from the atom, then the drift measured and compensated."""
        s = self.s
        x0, y0 = self.reg.site_xy(atom.i, atom.j, atom.sub)
        if tip_needs_poke(self.w, imaging=True) or self.w.tip.carried:
            self.log["conditionings"] += condition_tip(s, x0, y0, imaging=True,
                                                       clean_ldos=bool(self.w.tip.carried),
                                                       r_min=150e-9, r_max=200e-9, clear=5e-9)
            s.go(x0, y0)
            self.inst.act("wait", seconds=300.0)
        self.mv.drift(x0, y0)

    def manipulate(self, r_ohm: float) -> None:
        """The preamplifier's range first, then the bias, then the setpoint (:meth:`Mover.manipulate`
        at this runner's bias)."""
        v = min(self.manip_v, r_ohm * MANIP_MAX_NA * 1e-9)
        self.inst.act("set_current_range", full_scale_na=100.0)
        self.inst.act("set_bias", bias_v=v)
        self.inst.act("set_setpoint", current_pa=v / r_ohm * 1e12)

    def drag(self, atom, r_ohm: float, sign: int) -> tuple[dict, str]:
        """One standard drag: the leading apex onto the atom at the imaging setpoint, then 1.0 nm
        along +x or -x at 0.1 nm/s at ``r_ohm``. Returns the drag's record and its trace name."""
        w, inst = self.w, self.inst
        ax, ay = self.reg.site_xy(atom.i, atom.j, atom.sub)
        lead = min(w.grip().points, key=lambda p: p.r_factor)
        x, y = self.s.scan_xy(ax - lead.dx, ay - lead.dy)
        inst.act("move_tip", x_nm=x, y_nm=y)
        self.manipulate(r_ohm)
        cx, cy = w.tip_x * 1e9, w.tip_y * 1e9
        out = inst.act("move_tip", x_nm=cx + sign * LENGTH_NM, y_nm=cy, speed_nm_s=SPEED_NM_S)
        self.mv.image()
        rec = w.drag_records[-1]
        assert rec["index"] == inst.spectra[out["trace"]]["index"]
        return rec, out["trace"]

    # ── one atom ──
    def series(self, atom) -> dict:
        """Measure ``atom``: the drags of the series that ends the run (one grip), its fit."""
        inst, w = self.inst, self.w
        self.log["atoms_tried"] += 1
        self.prepare(atom)
        r, sign, key, drags = START_KOHM, 1, None, []
        stuck_low = stuck_high = 0
        for attempt in range(MAX_DRAGS_PER_ATOM):
            if len(drags) >= self.n_drags:
                break
            if inst.max_sim_s is not None and inst.used_s > 0.97 * inst.max_sim_s:
                break
            r = min(max(r, R_MIN_KOHM), R_MAX_KOHM)
            n0 = len(w.tip.events)
            rec, trace = self.drag(atom, r * 1e3, sign)
            sign = -sign
            self.log["tip_changes"] += sum(1 for e in w.tip.events[n0:] if e.kind == "spontaneous_change" or e.kind.startswith("recovery"))
            if rec.get("carrying"):
                raise PickedUp()
            if rec.get("picked"):
                self.log["pickups"] += 1
                raise PickedUp()
            counted = (rec["atom_id"] == atom.id and rec["active"] and rec["feedback"] and not rec["over_range"]
                       and rec.get("start_dist_nm") is not None and rec["start_dist_nm"] <= START_WITHIN_NM)
            if not counted:
                self.log["discarded"] += 1
            else:
                k = _key(rec["conditions"])
                if key is not None and k != key:
                    self.log["restarts"] += 1
                    drags = []
                key = k
                drags.append({"r_kohm": float(r), "followed": bool(rec["followed"]), "trace": trace,
                              "index": rec["index"], "start_dist_nm": float(rec["start_dist_nm"]),
                              "record": rec})
            if tip_needs_poke(w, imaging=True) or w.tip.carried:
                # the manipulation current changed the tip: condition it, then start the series again
                self.prepare(atom)
                key, drags = None, []
                self.log["restarts"] += 1
            f = [d["followed"] for d in drags]
            if counted:
                last = rec["followed"]
                if any(f) and not all(f):
                    if self.design == "fit":
                        # sequential design: next to the present fit (at most x1.3 from here),
                        # dithered by one step up and down so that the slope stays measured
                        fit = fit_threshold([d["r_kohm"] for d in drags], f)
                        base = min(max(fit["r_hat_kohm"] or r, r / 1.3), r * 1.3)
                        r = base * (self.step if len(drags) % 2 else 1.0 / self.step)
                    else:
                        r = r * (self.step if last else 1.0 / self.step)
                else:
                    r = r * (BRACKET_UP if last else BRACKET_DOWN)
                if r <= R_MIN_KOHM:
                    stuck_low += 1
                if r >= R_MAX_KOHM:
                    stuck_high += 1
                if stuck_low > 3 or stuck_high > 3:
                    raise GaveUp("the atom never followed" if stuck_low > 3 else "the atom always followed")
        return {"atom": atom, "drags": drags}

    def pick_two(self) -> list[dict]:
        """Two atoms, each measured; an atom lost onto the tip is replaced."""
        done, lost = [], []
        while len(done) < 2 and self.log["atoms_tried"] < 6:
            avoid = [(a["x0"], a["y0"]) for a in done] + lost
            cands = self.candidates(avoid=avoid)
            if not cands:
                break
            atom = cands[0]
            try:
                res = self.series(atom)
            except PickedUp:
                lost.append((atom.x0, atom.y0))
                x0, y0 = atom.x0, atom.y0
                self.log["conditionings"] += condition_tip(self.s, x0, y0, imaging=True, clean_ldos=True,
                                                           r_min=150e-9, r_max=200e-9, clear=5e-9)
                continue
            except GaveUp:
                lost.append((atom.x0, atom.y0))
                continue
            done.append({"x0": atom.x0, "y0": atom.y0, **res})
        return done


def _truth_kohm(w, records: list[dict]) -> float:
    """Geometric mean over the cited drags of the exact R½ (the judge's truth for a series)."""
    from stmsim.physics.drag_threshold import record_threshold
    info = w.drag_truth()["lattice"]
    vals = [record_threshold(info, r) for r in records]
    vals = [v for v in vals if v]
    return float(math.exp(np.mean(np.log(vals))) / 1e3) if vals else float("nan")


def run(seed: int, *, out_dir: str | Path | None = None, scenario: str = SCENARIO, n_drags: int = 32,
        manip_v: float = MANIP_V, step: float = STAIR_STEP, design: str = "fit",
        finish: bool = True,
        keep_drags: bool = False) -> dict:
    """One P7 episode by the reference protocol; returns what it reported and how it scored.
    ``finish=False`` leaves the judge out (a branch without P7's verdict, or a calibration that
    compares with the truth itself); ``keep_drags`` keeps every drag of every series in the
    row for the precision-cost curve."""
    s = Session(scenario, seed, out_dir)
    inst, w = s.inst, s.w
    run_ = _Runner(s, manip_v=manip_v, step=step, n_drags=n_drags, design=design)
    atoms = run_.pick_two()
    rows, rel_err = [], {}
    for claim, res in zip(CLAIMS, atoms):
        drags = res["drags"]
        fit = fit_threshold([d["r_kohm"] for d in drags], [d["followed"] for d in drags])
        recs = [d["record"] for d in drags]
        row = {"claim": claim, "atom_id": int(res["atom"].id), "n_drags": len(drags),
               "n_followed": sum(d["followed"] for d in drags), "r_hat_kohm": fit["r_hat_kohm"],
               "slope": fit["slope"], "truth_kohm": _truth_kohm(w, recs) if recs else float("nan")}
        if fit["r_hat_kohm"] is not None:
            row["rel_err"] = fit["r_hat_kohm"] / row["truth_kohm"] - 1.0
            row["evidence"] = evidence_ok(recs, fit["r_hat_kohm"])
            rel_err[claim] = row["rel_err"]
            inst.report(claim, fit["r_hat_kohm"], unit="kOhm", evidence=[d["trace"] for d in drags])
        # the precision-cost curve: the same series cut after the first n drags
        row["curve"] = {}
        for n in CURVE_N:
            if n <= len(drags):
                f = fit_threshold([d["r_kohm"] for d in drags[:n]], [d["followed"] for d in drags[:n]])
                row["curve"][str(n)] = (f["r_hat_kohm"] / row["truth_kohm"] - 1.0) if f["r_hat_kohm"] else None
        row["r_series"] = [round(d["r_kohm"], 2) for d in drags]
        row["followed_series"] = [int(d["followed"]) for d in drags]
        row["start_dist_max_nm"] = max((d["start_dist_nm"] for d in drags), default=float("nan"))
        if keep_drags:
            row["drags"] = [{k: v for k, v in d.items() if k != "record"} for d in drags]
        rows.append(row)
    out = inst.finish() if finish else {"success": None, "partial": None}
    return {"paper": "P7", "seed": int(seed), "success": out["success"], "partial": out.get("partial"),
            "claims_ok": claims_ok(inst) if finish else {},
            "rel_err": rel_err, "atoms": rows, "instrument_s": float(inst.used_s), **run_.log,
            "tip_condition": inst.scenario.draw_hidden(seed).get("tip", {}).get("condition")}
