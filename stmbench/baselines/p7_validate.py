"""Drags for checking P7's carry reader (:mod:`stmbench.analysis.carry`) against the world.

**Validation only.** This module reads the world: it picks the atoms, puts the tip over them the way
a slow frame would, and sets the resistances from the atom's true threshold. The baseline itself
(:mod:`stmbench.baselines.p7`) never imports it. What it returns for each drag is what the reader
sees (``spectrum``: the move's trace as :meth:`Instrument.spectrum` gives it) and what the judge's
record says happened (``record``: ``truth()["drags"]["records"]``, ``followed`` above all).
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

from stmsim.physics.drag_threshold import conditions_of, pull_threshold, lattice
from stmsim.scenario import Scenario

P7 = "P7_pull_threshold_cu111"
SCENARIO_FILE = Path(__file__).resolve().parents[1] / "trackB" / "scenarios" / f"{P7}.yaml"
#: manipulation at 2 mV as the baseline and the reference do (a lower bias at one resistance means
#: a lower current, and with it a lower tip-change hazard)
MANIP_V = 0.002
#: where a slow frame places an atom against the apex that holds it (nm, rms per axis); the
#: judge only counts drags that started within 0.3 nm
PLACE_SIGMA_NM = 0.08
#: resistances: this share of the drags is drawn around the atom's own threshold (ln-normal with
#: this width), the rest over the manipulation window, 50-500 kohm, log-uniform
NEAR_SHARE, NEAR_SIGMA_LN = 0.6, 0.30
WINDOW_KOHM = (50.0, 500.0)


def collect(seed: int, n_drags: int, *, out_dir: str | Path | None = None, sc: Scenario | None = None,
            drag_nm: float = 1.0, speed_nm_s: float = 0.1) -> list[dict]:
    """``n_drags`` standard drags on the two isolated atoms nearest the origin of seed ``seed``'s P7
    world, alternating ±x, each started with the tip put where a slow frame places the atom. Each
    row: ``{"seed", "k", "spectrum", "record", "r_half_kohm"}`` (``r_half_kohm``: the true R½ of the
    atom under the tip's grip as it was just before the drag, for choosing the resistance). The
    session ends when the atom is lifted onto the tip."""
    sc = sc or Scenario.load(SCENARIO_FILE)
    d = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="p7v_"))
    w = sc.build_world(seed, d / f"s{seed}", virtual_clock=True)
    rng = np.random.default_rng([seed, 7077])
    reg = w.surface.site.adatoms
    on, xy = reg._arrays()
    near = np.argsort(np.hypot(xy[:, 0], xy[:, 1]))
    picks = []
    for k in near:
        if int(np.count_nonzero(np.hypot(xy[:, 0] - xy[k, 0], xy[:, 1] - xy[k, 1]) <= 3e-9)) == 1:
            picks.append(on[int(k)])
        if len(picks) == 2:
            break
    geom = lattice(float(w.surface.site.material.nn_m), float(w.surface.site.lattice_angle))
    w.preamp.full_scale_a = 100e-9
    rows: list[dict] = []
    for k in range(n_drags):
        atom = picks[k % len(picks)]
        if atom.status != "on_surface":
            break
        sign = 1.0 if (k // len(picks)) % 2 == 0 else -1.0
        ax, ay = reg.site_xy(atom.i, atom.j, atom.sub)
        grip = w.grip()
        point = grip.points[0]
        cond = conditions_of(reg, atom, grip, point, w.surface.terrace_level)
        r_half = pull_threshold(geom, (0, 0), 0.0 if sign > 0 else math.pi, cond)
        if r_half is not None and rng.random() < NEAR_SHARE:
            r = r_half * math.exp(rng.normal(0.0, NEAR_SIGMA_LN))
        else:
            r = 1e3 * math.exp(rng.uniform(math.log(WINDOW_KOHM[0]), math.log(WINDOW_KOHM[1])))
        # the tip over the atom as a frame places it, then the manipulation conditions
        off = rng.normal(0.0, PLACE_SIGMA_NM * 1e-9, 2)
        ox, oy, _ = w.drift_offset(w.clock.sim())
        w.move_xy(ax + off[0] - ox - point.dx, ay + off[1] - oy - point.dy)
        v = min(MANIP_V, r * 80e-9)
        w.set_bias(v)
        w.set_setpoint(v / r)
        w.transients.clear()
        ox, oy, _ = w.drift_offset(w.clock.sim())
        w.move_xy_path(w.tip_x + sign * drag_nm * 1e-9, w.tip_y, speed_m_s=speed_nm_s * 1e-9, trace=True)
        tr = w.last_move_trace
        rec = w.drag_records[-1]
        rows.append({"seed": int(seed), "k": k, "r_half_kohm": None if r_half is None else r_half / 1e3,
                     "spectrum": {"z_abs": np.asarray(tr["z_abs"], float), "dist": np.asarray(tr["dist"], float),
                                  "i": np.asarray(tr["i"], float), "t": np.asarray(tr["t"], float),
                                  "feedback": bool(tr["feedback"])},
                     "record": dict(rec), "tip": w.tip.snapshot().get("condition")})
        # a fresh imaging state for the next placement
        w.set_setpoint(0.2e-9)
        w.set_bias(0.05)
        w.transients.clear()
    return rows


def main() -> None:
    """``python -m stmbench.baselines.p7_validate OUT.pkl FIRST_SEED N_SEEDS N_DRAGS``: the drags of
    many seeds in one process, pickled as a list of rows (for a cluster job)."""
    import pickle
    import sys
    out, first, n_seeds, n_drags = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    sc = Scenario.load(SCENARIO_FILE)
    rows: list[dict] = []
    for seed in range(first, first + n_seeds):
        try:
            rows += collect(seed, n_drags, sc=sc)
        except Exception as exc:  # noqa: BLE001 — a failed seed is reported, not fatal
            print(f"seed {seed} failed: {type(exc).__name__}: {exc}", flush=True)
        print(f"seed {seed}: {len(rows)} drags so far", flush=True)
        with open(out, "wb") as fh:
            pickle.dump(rows, fh)


if __name__ == "__main__":
    main()
