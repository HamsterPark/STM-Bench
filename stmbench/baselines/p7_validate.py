"""Drags for checking P7's carry reader (:mod:`stmbench.analysis.carry`) against the world.

**Validation only.** This module reads the world: it picks the atoms and sets the resistances from
the atom's true threshold. The baseline itself (:mod:`stmbench.baselines.p7`) never imports it.
The drags are made by the baseline's own :class:`~stmbench.baselines.p7.Puller` (the tip put over
the atom from a slow frame, the drag, the slow frame after it, the reading), so what is checked is
what the baseline runs. For each drag it keeps what the reader sees (the trace as
:meth:`Instrument.spectrum` gives it, the atom's place in the frames) and what the judge's record
says happened (``truth()["drags"]["records"]``, ``followed`` above all).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from stmsim.physics.drag_threshold import conditions_of, lattice, pull_threshold

P7 = "P7_pull_threshold_cu111"
#: resistances: this share of the drags is drawn around the atom's own threshold (ln-normal with
#: this width), the rest over the manipulation window, 50-500 kohm, log-uniform
NEAR_SHARE, NEAR_SIGMA_LN = 0.6, 0.30
WINDOW_KOHM = (50.0, 500.0)


def collect(seed: int, n_drags: int, *, out_dir: str | Path | None = None) -> list[dict]:
    """``n_drags`` standard drags on the two isolated atoms nearest the origin of seed ``seed``'s P7
    world, alternating ±x. Each row: ``{"seed", "k", "pull": the Puller's drag, "spectrum", "record":
    the judge's record, "r_half_kohm": the atom's true R½ under the grip as it was, "tip": the tip's
    state at the drag, "tip0": the condition it was drawn with}``. The session ends when the atom
    leaves every frame (lifted onto the tip)."""
    from stmbench.baselines._common import Blind, Ops
    from stmbench.baselines.p7 import Puller, _prepare
    from stmbench.instrument import Instrument

    inst = Instrument(P7, seed, out_dir)
    w = inst.world
    rng = np.random.default_rng([seed, 7077])
    reg = w.surface.site.adatoms
    on, xy = reg._arrays()
    picks = []
    for k in np.argsort(np.hypot(xy[:, 0], xy[:, 1])):
        if int(np.count_nonzero(np.hypot(xy[:, 0] - xy[k, 0], xy[:, 1] - xy[k, 1]) <= 3e-9)) == 1:
            picks.append(on[int(k)])
        if len(picks) == 2:
            break
    geom = lattice(float(w.surface.site.material.nn_m), float(w.surface.site.lattice_angle))
    v = Blind(inst)
    ops = Ops(v, {})
    pl = Puller(ops)
    tip0 = inst.scenario.draw_hidden(seed).get("tip", {}).get("condition")

    def image_xy(atom) -> tuple[float, float]:
        """Where a frame shows the atom now (privileged, the session's start): the sample place
        less the drift, through the leading apex."""
        ax, ay = reg.site_xy(atom.i, atom.j, atom.sub)
        apexes = getattr(w.tip, "apexes", None) or []
        dx = dy = 0.0
        if apexes:
            lead = max(apexes, key=lambda a: a.dz)
            dx, dy = float(lead.dx), float(lead.dy)
        ox, oy, _ = w.drift_offset(w.clock.sim())
        return (ax - dx - ox) * 1e9, (ay - dy - oy) * 1e9

    log: dict = {}
    ops.log = log
    _prepare(v, ops, log)            # the baseline's own opening: the tip judged and conditioned, the drift
    where = [pl.measure(image_xy(a), within=2.0) for a in picks]
    rows: list[dict] = []
    for k in range(n_drags):
        j = k % len(picks)
        atom = picks[j]
        if atom.status != "on_surface" or where[j] is None:
            break
        sign = 1.0 if (k // len(picks)) % 2 == 0 else -1.0
        grip = w.grip()
        cond = conditions_of(reg, atom, grip, grip.points[0], w.surface.terrace_level)
        r_half = pull_threshold(geom, (0, 0), 0.0 if sign > 0 else math.pi, cond)
        if r_half is not None and rng.random() < NEAR_SHARE:
            r = r_half * math.exp(rng.normal(0.0, NEAR_SIGMA_LN))
        else:
            r = 1e3 * math.exp(rng.uniform(math.log(WINDOW_KOHM[0]), math.log(WINDOW_KOHM[1])))
        pos = pl.measure(where[j], within=1.0) or where[j]
        drag = pl.pull(pos, sign, r)
        rec = dict(w.drag_records[-1])
        snap = w.tip.snapshot()
        tip = ("multi" if snap["n_apex"] > 1 else "blunt" if snap["radius_nm"] > 2.0
               else "unstable" if (snap["metastable"] or snap["lambda_per_s"] > 2e-3) else "ok")
        sp = inst.spectrum(drag["trace"])
        rows.append({"seed": int(seed), "k": k, "pull": drag, "record": rec, "r_kohm": r / 1e3,
                     "r_half_kohm": None if r_half is None else r_half / 1e3,
                     "spectrum": {kk: (np.asarray(sp[kk]) if kk != "feedback" else bool(sp[kk]))
                                  for kk in ("z_abs", "dist", "i", "t", "feedback")},
                     "tip": tip, "tip0": tip0, "tip_first": log.get("tip_first")})
        if drag["atom"] is None:
            break
        where[j] = drag["atom"]
    return rows


def main() -> None:
    """``python -m stmbench.baselines.p7_validate OUT.pkl FIRST_SEED N_SEEDS N_DRAGS``: the drags of
    many seeds in one process, pickled as a list of rows (for a cluster job)."""
    import pickle
    import sys
    out, first, n_seeds, n_drags = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    rows: list[dict] = []
    for seed in range(first, first + n_seeds):
        try:
            rows += collect(seed, n_drags)
        except Exception as exc:  # noqa: BLE001 — a failed seed is reported, not fatal
            print(f"seed {seed} failed: {type(exc).__name__}: {exc}", flush=True)
        print(f"seed {seed}: {len(rows)} drags so far", flush=True)
        with open(out, "wb") as fh:
            pickle.dump(rows, fh)


if __name__ == "__main__":
    main()
