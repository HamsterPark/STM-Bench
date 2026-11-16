"""P7's pull threshold (stmsim.physics.drag_threshold) and the judge's record of the drags.

The exact outcome of a drag over an isolated atom must be the one ``AdatomRegistry.drag`` gives,
so it is checked against ``drag()`` itself by Monte Carlo, over the conditions that matter: the
resistance near the threshold, a drag along and across the close-packed rows, a fumbling tip, a
jittery blunt one, a slipping atom, a second apex, and a resistance low enough to lift the atom.
The record the judge reads must only read the instrument: a world that keeps it is the world that
does not, draw for draw.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from stmsim.physics import drag_threshold as DT
from stmsim.physics.adatoms import AdatomParams, AdatomRegistry, Grip, GripPoint

NN = 0.255e-9


def _one_atom(params: AdatomParams, angle: float, r_scale: float, cap_scale: float):
    site = SimpleNamespace(seed=0, lattice_angle=angle, material=SimpleNamespace(nn_m=NN))
    reg = AdatomRegistry(site, params)
    atom = reg.add(0, 0)
    atom.r_scale, atom.cap_scale = r_scale, cap_scale
    return reg, atom


def _mc_full(reg, atom, grip, r_ohm, v, direction, length, n, seed) -> float:
    reg.rng_dyn = np.random.default_rng(seed)
    x0, y0 = reg.site_xy(0, 0)
    end = (x0 + length * math.cos(direction), y0 + length * math.sin(direction))
    full = 0
    for _ in range(n):
        reg.occ.clear()
        atom.i, atom.j, atom.status = 0, 0, "on_surface"
        reg.occ[(0, 0, 0)] = atom.id
        reg.carried = None
        atom.history.clear()
        reg.epoch += 1
        reg.drag((x0, y0), end, r_ohm=r_ohm, v_m_s=v, sim_s=0.0, grip=grip)
        if atom.status == "on_surface":
            ax, ay = reg.site_xy(atom.i, atom.j)
            if math.hypot(ax - end[0], ay - end[1]) <= DT.WITH_TIP_FRACTION * NN:
                full += 1
    return full / n


CASES = [
    # (name, params, lattice angle, drag direction, r_scale, cap_scale, grip, R / law threshold)
    ("row, near threshold", AdatomParams(r_threshold_ohm=200e3, capture_radius_m=0.45e-9), 0.0, 0.0,
     1.0, 1.0, Grip(), 1.05),
    ("across rows, wide grab", AdatomParams(r_threshold_ohm=120e3, capture_radius_m=0.6e-9), 0.2, 0.9,
     1.2, 1.1, Grip(), 1.1),
    ("narrow grab, below threshold", AdatomParams(r_threshold_ohm=300e3, capture_radius_m=0.35e-9), 0.5, 2.0,
     0.85, 0.9, Grip(), 0.95),
    ("flickering tip", AdatomParams(r_threshold_ohm=200e3), 0.1, -0.4, 1.0, 1.0, Grip(fumble=0.35), 0.9),
    ("blunt, jittery", AdatomParams(r_threshold_ohm=250e3), 0.3, 0.3, 1.0, 1.0,
     Grip(threshold_scale=0.6, capture_scale=1.4, jitter=0.3), 1.0),
    ("slipping atom", AdatomParams(r_threshold_ohm=150e3, p_slip=0.1), 0.0, 1.2, 1.0, 1.0, Grip(), 0.9),
    ("second apex holds it", AdatomParams(r_threshold_ohm=200e3), 0.4, 0.0, 1.0, 1.0,
     Grip(points=(GripPoint(0.0, 0.0, 1.7),)), 1.0),
    ("low enough to lift it", AdatomParams(r_threshold_ohm=90e3, r_pick_ohm=40e3), 0.0, 0.5, 1.0, 1.0,
     Grip(), 0.55),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_the_exact_outcome_is_the_one_drag_gives(case):
    _name, params, angle, direction, r_scale, cap_scale, grip, rel = case
    reg, atom = _one_atom(params, angle, r_scale, cap_scale)
    point = grip.points[0]
    cond = DT.conditions_of(reg, atom, grip, point)
    r_ohm = rel * params.r_threshold_ohm * grip.threshold_scale * r_scale / point.r_factor
    x0, y0 = reg.site_xy(0, 0)
    end = (x0 + DT.STANDARD_LENGTH_M * math.cos(direction), y0 + DT.STANDARD_LENGTH_M * math.sin(direction))
    exact = DT.drag_outcome(reg, (0, 0), (x0, y0), end, r_ohm, DT.STANDARD_SPEED_M_S, cond)
    n = 3000
    mc = _mc_full(reg, atom, grip, r_ohm, DT.STANDARD_SPEED_M_S, direction, DT.STANDARD_LENGTH_M, n, seed=7)
    sigma = math.sqrt(max(exact["full"] * (1 - exact["full"]), 1e-4) / n)
    assert 0.02 < exact["full"] < 0.98 or "lift" in _name     # a case that tests something
    assert abs(mc - exact["full"]) <= 4 * sigma, (mc, exact)
    assert abs(sum(exact["sites"].values()) + exact["picked"] - 1.0) < 1e-9


def test_the_threshold_is_where_half_the_drags_carry_the_atom():
    reg = DT.lattice(NN, 0.3)
    cond = DT.DragConditions(r_threshold_ohm=200e3, r_pick_ohm=30e3, v_max_m_s=1e-9, capture_radius_m=0.45e-9,
                             p_slip=0.005, r_scale=1.0, cap_scale=1.0, hold=1.0)
    r_half = DT.pull_threshold(reg, (0, 0), 0.0, cond)
    assert r_half is not None
    p = DT.full_carry_probability(reg, (0, 0), 0.0, r_half, cond)
    assert abs(p - 0.5) < 1e-3
    # a single attempt succeeds half the time at the law's own R_th; a whole drag needs several
    # attempts per site, so its threshold sits a little above it
    assert 0.9 * 200e3 < r_half < 1.3 * 200e3
    # the threshold scales with the atom's hold, the tip's scale and the holding apex's factor
    import dataclasses
    for change, factor in (({"hold": 0.6}, 0.6), ({"threshold_scale": 0.5}, 0.5), ({"r_factor": 2.0}, 0.5)):
        other = DT.pull_threshold(reg, (0, 0), 0.0, dataclasses.replace(cond, **change))
        assert other == pytest.approx(factor * r_half, rel=2e-3), change
    # a wider grab gives more attempts per site: a higher threshold
    wide = DT.pull_threshold(reg, (0, 0), 0.0, dataclasses.replace(cond, capture_radius_m=0.6e-9))
    assert wide > r_half


def _p4_world(seed: int, record: bool, tmp_path):
    from stmsim.scenario import Scenario
    from stmbench.instrument.core import SCENARIO_DIR
    sc = Scenario.load(SCENARIO_DIR / "P4_atom_positioning_cu111.yaml")
    w = sc.build_world(seed, session_dir=tmp_path / f"s{seed}_{int(record)}", virtual_clock=True)
    assert w.record_drags is False                     # P4's claims do not need the record
    w.record_drags = record
    return w


def _drag_series(w) -> None:
    """Go to the scenario's target atom and drag it to and fro at falling resistances."""
    reg = w.surface.site.adatoms
    atom = next(a for a in reg.atoms.values() if a.role == "target")
    ox, oy, _ = w.drift_offset(w.clock.sim())
    ax, ay = reg.site_xy(atom.i, atom.j)
    w.move_xy_path(ax - ox, ay - oy, trace=True)
    w.set_bias(0.005)
    w.set_current_range(2)
    for k, r_kohm in enumerate((500.0, 300.0, 200.0, 150.0, 110.0, 80.0, 60.0)):
        w.set_setpoint(0.005 / (r_kohm * 1e3))
        sign = 1.0 if k % 2 == 0 else -1.0
        w.move_xy_path(w.tip_x + sign * 1e-9, w.tip_y, speed_m_s=0.1e-9, trace=True)
        w.clock.advance_sim(10.0)


def _state(w) -> dict:
    reg = w.surface.site.adatoms
    return {"events": [{k: v for k, v in e.items() if k != "wall_s"} for e in w.events],
            "sites": sorted((a.id, a.i, a.j, a.status, len(a.history)) for a in reg.atoms.values()),
            "tip": w.tip.snapshot(), "n_tip_events": len(w.tip.events),
            "rng": [r.bit_generator.state["state"] for r in (reg.rng_dyn, w.rng, w.tip.rng, w.move_rng,
                                                              w.rng_outscan, w.rng_recovery)],
            "trace": None if w.last_move_trace is None else
            [np.asarray(w.last_move_trace[k]).tolist() for k in ("z_abs", "i")]}


def test_keeping_the_record_changes_nothing(tmp_path):
    a = _p4_world(3, False, tmp_path)
    b = _p4_world(3, True, tmp_path)
    _drag_series(a)
    _drag_series(b)
    assert _state(a) == _state(b)
    assert "drags" not in a.truth()
    recs = b.truth()["drags"]["records"]
    assert [r["index"] for r in recs] == list(range(1, 9))
    dragged = [r for r in recs[1:] if r["atom_id"] is not None]
    assert dragged, "the series should have held the target atom"
    target = next(x for x in b.surface.site.adatoms.atoms.values() if x.role == "target")
    first = dragged[0]
    assert first["atom_id"] == target.id and first["start_dist_nm"] < 0.1
    assert first["conditions"]["r_scale"] == pytest.approx(target.r_scale)
    assert all(r["length_nm"] == pytest.approx(1.0, abs=1e-6) for r in recs[1:])


def test_a_drag_that_carried_the_atom_is_recorded_as_followed(tmp_path):
    w = _p4_world(3, True, tmp_path)
    _drag_series(w)
    recs = [r for r in w.truth()["drags"]["records"] if r["atom_id"] is not None]
    for r in recs:
        if r["followed"]:
            assert r["n_hops"] >= 3 and not r["picked"]
        assert (r["site_after"] is None) == r["picked"]
