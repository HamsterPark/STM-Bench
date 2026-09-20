"""Manipulation behavior: the tip arrives in some condition and that condition
decides how atoms are grabbed; no two atoms are alike; the deposit covers the whole sample;
whatever wrecks the surface takes the atoms with it; and P4 accepts any isolated atom."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from stmsim.physics.adatoms import (CROWD_FACTOR, FIELD_MIN_SEP_M, STEP_FACTOR, AdatomParams,
                                    AdatomRegistry, Grip, GripPoint, grip_of, scatter_field)
from stmsim.physics.rig import RigProfile
from stmsim.physics.surface import Feature, Surface
from stmsim.physics.tip import Apex, Tip
from stmsim.physics.world import World
from stmsim.scenario import Scenario, _draw_leaf, resolve_hidden

SCEN = Path(__file__).resolve().parent.parent / "stmbench" / "trackB" / "scenarios"
PARAMS = AdatomParams(r_threshold_ohm=200e3, r_pick_ohm=30e3, v_max_m_s=1e-9, p_slip=0.0)
KAPPA = 1.09e10


def _world(tmp_path, seed=4, setpoint=1e-9, **layout):
    w = World(rig=RigProfile.load("reference-stm"), seed=seed, material="Cu(111)",
              session_dir=tmp_path / "s", adsorbate_density_per_um2=0.0)
    w.surface.configure_adatoms(PARAMS, {"layout": "single", "target_dx_nm": 4.0, "n_bystanders": 0, **layout})
    w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
    w.withdrawn = False
    w.zctrl_set(True)
    w.drift_v_m_per_s = np.zeros(3)
    w.bias_v = 0.01
    w.zctrl.setpoint_a = setpoint
    return w


def _example(w):
    reg = w.surface.site.adatoms
    return reg, reg.atoms[reg.target["atom_id"]]


# ── the tip arrives in some condition ───────────────────────────────────────
def test_a_weighted_choice_takes_one_draw_and_follows_its_weights():
    rng = np.random.default_rng(1)
    spec = {"choice": ["a", "b", "c"], "weights": [0.6, 0.3, 0.1]}
    got = [_draw_leaf(spec, rng) for _ in range(6000)]
    for k, w in zip("abc", (0.6, 0.3, 0.1)):
        assert got.count(k) / len(got) == pytest.approx(w, abs=0.025)
    with pytest.raises(ValueError):
        _draw_leaf({"choice": ["a", "b"], "weights": [1.0]}, rng)


def test_the_tip_block_draws_last_so_no_earlier_hidden_value_moves():
    sc = Scenario.load(SCEN / "P4_atom_positioning_cu111.yaml")
    without = {k: v for k, v in sc.hidden.items() if k != "tip"}
    for seed in range(5):
        a = resolve_hidden(sc.hidden, seed, sc.id)
        b = resolve_hidden(without, seed, sc.id)
        assert a["adatoms"] == b["adatoms"]


def test_every_paper_starts_with_a_tip_nobody_vouched_for():
    kinds = {}
    for p in sorted(SCEN.glob("P*.yaml")):
        sc = Scenario.load(p)
        assert "condition" in (sc.hidden.get("tip") or {}), p.name
        for word in ("针尖状态", "钝", "双针尖", "分叉", "不稳", "脏"):
            assert word not in sc.task, f"{p.name}: the task must not hint at the tip's condition ({word})"
    sc = Scenario.load(SCEN / "P4_atom_positioning_cu111.yaml")
    for seed in range(40):
        kind = resolve_hidden(sc.hidden, seed, sc.id)["tip"]["condition"]
        kinds.setdefault(kind, seed)
    assert set(kinds) == set(Tip.CONDITIONS)


@pytest.mark.parametrize("kind", Tip.CONDITIONS)
def test_each_condition_is_what_it_says(kind):
    tip = Tip(rng=np.random.default_rng(3))
    tip.radius_m = 1e-9
    state0 = tip.rng.bit_generator.state
    tip.apply_condition(kind, 0.5, 30.0)
    assert tip.rng.bit_generator.state == state0, "applying a condition must not draw"
    s = tip.snapshot()
    if kind == "good":
        assert s["radius_nm"] == pytest.approx(1.0) and s["n_apex"] == 1 and s["ldos"] == "flat"
    elif kind == "blunt":
        assert 3.0 <= s["radius_nm"] <= 8.0 and s["apex_sigma_nm"] > 0.15
    elif kind == "double":
        assert s["multi"] and s["n_apex"] == 2
    elif kind == "unstable":
        assert s["metastable"] and s["lambda_per_s"] >= 3e-3
    elif kind == "dirty":
        assert s["ldos"] == "featured" and s["phi_ev"] < 4.2
    with pytest.raises(ValueError):
        tip.apply_condition("sticky")


def test_a_shallow_poke_buries_whatever_was_adsorbed_on_the_apex():
    tip = Tip(rng=np.random.default_rng(3))
    tip.apply_condition("dirty", 0.5)
    for _ in range(10):
        out = tip.poke(1.0, 0.5e-9, 0.01)                   # into the copper, low bias (no qPlus ring-up)
        if out["outcome"] in ("cluster", "pit"):
            break
    assert out["outcome"] in ("cluster", "pit")
    assert tip.snapshot()["ldos"] == "flat" and tip.phi_ev >= 4.0


# ── the tip decides how atoms are grabbed ───────────────────────────────────
def test_the_grip_follows_the_tip():
    good = Tip(rng=np.random.default_rng(0))
    good.radius_m = 1e-9
    g = grip_of(good, KAPPA)
    assert g == Grip(points=(GripPoint(0.0, 0.0, 1.0),), threshold_scale=1.0, capture_scale=1.0, fumble=0.0, jitter=0.0)
    blunt = Tip(rng=np.random.default_rng(0))
    blunt.apply_condition("blunt", 0.8)
    gb = grip_of(blunt, KAPPA)
    assert gb.threshold_scale < 0.5 and gb.capture_scale > 1.3 and gb.jitter > 0.2
    double = Tip(rng=np.random.default_rng(0))
    double.apply_condition("double", 0.0, 0.0)
    gd = grip_of(double, KAPPA)
    assert len(gd.points) == 2 and 1.0 < gd.points[1].r_factor < 3.0
    assert gd.points[1].dx == pytest.approx(0.8e-9)
    step_off = Tip(rng=np.random.default_rng(0))
    step_off.apexes = [Apex(0, 0, 0, 1.0), Apex(2.5e-9, -1.5e-9, -0.2354e-9, 0.8)]   # a whole step shorter
    assert len(grip_of(step_off, KAPPA).points) == 1
    crashed = Tip(rng=np.random.default_rng(0))
    crashed.crash(0.0)
    assert grip_of(crashed, KAPPA).fumble == pytest.approx(0.35)
    unsteady = Tip(rng=np.random.default_rng(0))
    unsteady.radius_m = 1e-9
    unsteady.metastable = True
    assert grip_of(unsteady, KAPPA).fumble == pytest.approx(0.12)
    carrying = Tip(rng=np.random.default_rng(0))
    carrying.radius_m = 1e-9
    carrying.pick_up(0.0, "Fe")
    assert grip_of(carrying, KAPPA).threshold_scale == pytest.approx(0.7)


def _moves(w, grip, r_ohm, trials=30):
    reg, atom = _example(w)
    home = (atom.i, atom.j)
    x0, y0 = reg.site_xy(*home)
    ok = 0
    for _ in range(trials):
        reg.occ.pop((atom.i, atom.j, atom.sub), None)
        atom.i, atom.j, atom.status = home[0], home[1], "on_surface"
        reg.occ[(atom.i, atom.j, atom.sub)] = atom.id
        reg.carried = None
        reg.epoch += 1
        reg.drag((x0, y0), (x0 + 4e-9, y0), r_ohm=r_ohm, v_m_s=0.3e-9, sim_s=0.0,
                 terrace=w.surface.terrace_level, grip=grip)
        ok += reg.site_xy(atom.i, atom.j)[0] - x0 > 3e-9
    return ok


def test_a_blunt_tip_has_to_press_much_closer(tmp_path):
    w = _world(tmp_path)
    _, atom = _example(w)
    blunt = Tip(rng=np.random.default_rng(0))
    blunt.apply_condition("blunt", 0.8)
    r = 0.55 * PARAMS.r_threshold_ohm * atom.r_scale
    assert _moves(w, Grip(), r) >= 25
    assert _moves(w, grip_of(blunt, KAPPA), r) <= 3
    assert _moves(w, grip_of(blunt, KAPPA), 0.2 * PARAMS.r_threshold_ohm * atom.r_scale) >= 15


def test_a_double_tip_drags_the_atom_under_its_other_apex(tmp_path):
    w = _world(tmp_path)
    reg, atom = _example(w)
    x0, y0 = reg.site_xy(atom.i, atom.j)
    tip = Tip(rng=np.random.default_rng(0))
    tip.apply_condition("double", 0.4, 90.0)                  # second apex ~2.1 nm along +y
    g = grip_of(tip, KAPPA)
    off = g.points[1].dy
# the lead apex is positioned 2 nm *below* the atom — at the atom's ghost. The second
# apex, ~3× further in resistance, sits on the target atom: low enough R and it pulls
    r = 0.15 * PARAMS.r_threshold_ohm * atom.r_scale
    ev = reg.drag((x0, y0 - off), (x0 + 4e-9, y0 - off), r_ohm=r,
                  v_m_s=0.3e-9, sim_s=0.0, terrace=w.surface.terrace_level, grip=g)
    assert any(e["kind"] == "adatom_hop" and e["atom_id"] == atom.id for e in ev)
    assert reg.site_xy(atom.i, atom.j)[0] - x0 > 2e-9
    # a single sharp apex on the same path touches nothing
    w2 = _world(tmp_path / "b")
    reg2, atom2 = _example(w2)
    x2, y2 = reg2.site_xy(atom2.i, atom2.j)
    assert reg2.drag((x2, y2 - off), (x2 + 4e-9, y2 - off), r_ohm=r, v_m_s=0.3e-9, sim_s=0.0, grip=Grip()) == []


def test_dragging_at_a_manipulation_current_can_change_the_tip(tmp_path):
    w = _world(tmp_path, setpoint=85e-9)
    w.tip.lambda_per_s = 5e-3                                 # an unstable tip
    reg, atom = _example(w)
    x0, y0 = reg.site_xy(atom.i, atom.j)
    w.move_xy(x0, y0)
    n0 = len(w.tip.events)
    for k in range(10):
        w.move_xy_path(x0 + (k + 1) * 0.4e-9, y0, speed_m_s=0.3e-9)
    assert any(e.kind == "spontaneous_change" for e in w.tip.events[n0:])


# ── no two atoms alike ──────────────────────────────────────────────────────
def test_atoms_differ_and_a_step_or_a_neighbour_holds_them_harder():
    surf = Surface("Cu(111)", seed=3, adsorbate_density_per_um2=0.0)
    reg = AdatomRegistry(surf.site, PARAMS)
    surf.site.adatoms = reg
    placed = scatter_field(reg, density_per_nm2=0.02, half_m=60e-9)
    logs = np.log([a.r_scale for a in reg.atoms.values()])
    assert placed > 200 and float(np.std(logs)) == pytest.approx(0.18, abs=0.03)
    again = AdatomRegistry(Surface("Cu(111)", seed=3, adsorbate_density_per_um2=0.0).site, PARAMS)
    scatter_field(again, density_per_nm2=0.02, half_m=60e-9)
    assert [a.r_scale for a in again.atoms.values()] == [a.r_scale for a in reg.atoms.values()]
    lone = reg.add(*reg.site_of(200e-9, 200e-9))
    assert reg.hold_factor(lone) == pytest.approx(lone.r_scale)
    pal = reg.add(lone.i + 1, lone.j)                          # a neighbour one hollow away
    assert reg.hold_factor(lone) == pytest.approx(lone.r_scale * CROWD_FACTOR)

    def always_a_step(x, y):
        return (np.asarray(x) > float(np.asarray(x).flat[0])).astype(int)

    reg.occ.pop((pal.i, pal.j, pal.sub))
    pal.status = "lost"
    reg.epoch += 1
    assert reg.hold_factor(lone, always_a_step) == pytest.approx(lone.r_scale * STEP_FACTOR)


# ── the deposit covers the sample ───────────────────────────────────────────
def test_the_field_covers_the_sample_and_leaves_the_layout_its_room(tmp_path):
    sc = Scenario.load(SCEN / "P4_atom_positioning_cu111.yaml")
    w = sc.build_world(0, session_dir=tmp_path / "s")
    reg = w.surface.site.adatoms
    field = [a for a in reg.atoms.values() if a.role == "field"]
    assert 0.75 * 0.004 * 300 ** 2 < len(field) < 1.25 * 0.004 * 300 ** 2
    xy = reg.positions()
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1]) + np.eye(len(xy))
    assert float(d.min()) >= FIELD_MIN_SEP_M - 1e-12
    ex = reg.atoms[reg.target["atom_id"]]
    for f in field:
        for frac in (0.0, 0.5, 1.0):
            assert math.hypot(f.x0 - (ex.x0 + frac * 4e-9), f.y0 - ex.y0) >= 3.5e-9 - 1e-12
    # the placed atoms are the ones this seed placed without a field
    bare = copy.deepcopy(sc)
    bare.initial = {**sc.initial, "adatoms": {k: v for k, v in sc.initial["adatoms"].items()
                                               if not k.startswith("field_")}}
    w0 = bare.build_world(0, session_dir=tmp_path / "t")
    placed = [(a.i, a.j, a.role, a.r_scale) for a in w0.surface.site.adatoms.atoms.values()]
    assert placed == [(a.i, a.j, a.role, a.r_scale) for a in reg.atoms.values() if a.role != "field"]
    corral = Scenario.load(SCEN / "P3_corral_cu111.yaml").build_world(0, session_dir=tmp_path / "c")
    creg = corral.surface.site.adatoms
    cx, cy = (v * 1e-9 for v in creg.ring["centre_nm"])
    room = creg.ring["radius_nm"] * 1e-9 + 8e-9
    assert all(math.hypot(a.x0 - cx, a.y0 - cy) >= room for a in creg.atoms.values() if a.role == "field")
    assert any(a.role == "field" for a in creg.atoms.values())


# ── whatever wrecks the surface takes the atoms ─────────────────────────────
def test_pokes_and_pulses_take_the_atoms_they_hit(tmp_path):
    w = _world(tmp_path)
    reg, atom = _example(w)
    w.move_xy(*reg.site_xy(atom.i, atom.j))
    w.tip.contact_depth_m = 0.1e-9
    w.tip_shaper_start({"tip_lift_m": -0.6e-9, "bias_v": 0.01, "change_bias": True})
    ev = [e for e in w.events if e["kind"] == "poke"][-1]
    if ev["outcome"] in ("cluster", "pit"):
        assert atom.status == "lost"
        assert any(e["kind"] == "adatom_lost" and e["cause"] == "poke" for e in w.events)
    w2 = _world(tmp_path / "b")
    reg2, atom2 = _example(w2)
    w2.move_xy(*reg2.site_xy(atom2.i, atom2.j))
    w2.bias_pulse(0.05, 1.0, True, 1)                        # under the threshold: nothing happens
    assert atom2.status == "on_surface"
    w2.bias_pulse(0.05, 4.0, True, 1)                        # a reshaping pulse right over it
    assert atom2.status == "lost" and any(e["kind"] == "adatom_lost" and e["cause"] == "pulse" for e in w2.events)


def test_debris_on_a_site_stops_the_drag(tmp_path):
    w = _world(tmp_path)
    reg, atom = _example(w)
    x0, y0 = reg.site_xy(atom.i, atom.j)
    w.surface.add_feature(Feature(x0 + 2.2e-9, y0, 0.4e-9, 2.4e-9, kind="cluster"))   # blocks 1.2 nm round
    ev = reg.drag((x0, y0), (x0 + 4e-9, y0), r_ohm=0.4 * PARAMS.r_threshold_ohm * atom.r_scale, v_m_s=0.3e-9,
                  sim_s=0.0, terrace=w.surface.terrace_level, grip=Grip(), blocked=w.site_blocked)
    x1, _ = reg.site_xy(atom.i, atom.j)
    assert ev and x1 - x0 < 1.2e-9                            # it stopped short of the cluster


# ── P4: any isolated atom ───────────────────────────────────────────────────
def _p4_verdict(w, rep_xy_nm, sc):
    from stmbench.trackB.truth_criteria import claims_verified

    truth = w.truth()
    frame = {"kind": "scan_saved", "complete": True, "idx": 1, "sim_s": 1e9,
             "cx_m": rep_xy_nm[0] * 1e-9, "cy_m": rep_xy_nm[1] * 1e-9, "w_m": 8e-9, "h_m": 8e-9,
             "angle_deg": 0.0, "nx": 128, "ny": 128, "drift_m": [0.0, 0.0]}
    events = list(w.events) + [frame]
    report = {"claims": {"target_site": {"value": rep_xy_nm[0], "x_nm": rep_xy_nm[0], "y_nm": rep_xy_nm[1]}}}
    return claims_verified(truth, claims=sc.claims, report=report, events=events).details["claims"]


def _drag_by_d(w, atom):
    """What MoveAtomTo does, by hand: 8 mV and a current that puts R at 0.4 of this atom's
    threshold, 0.2 nm/s, retried until the atom sits on its goal."""
    reg = w.surface.site.adatoms
    bias, setpoint = w.bias_v, w.zctrl.setpoint_a
    w.bias_v = 0.008
    w.zctrl.setpoint_a = w.bias_v / (0.4 * reg.params.r_threshold_ohm * atom.r_scale)
    goal = reg.site_of(atom.x0 + 4e-9, atom.y0)
    for _ in range(4):
        if (atom.i, atom.j) == goal:
            break
        w.move_xy(*reg.site_xy(atom.i, atom.j))
        w.move_xy_path(*reg.site_xy(*goal), speed_m_s=0.2e-9)
    w.bias_v, w.zctrl.setpoint_a = bias, setpoint


def test_p4_takes_any_isolated_atom_and_judges_the_one_the_report_points_at(tmp_path):
    sc = Scenario.load(SCEN / "P4_atom_positioning_cu111.yaml")
    w = sc.build_world(7, session_dir=tmp_path / "s")         # seed 7 arrives with a good tip
    assert w.hidden["draws"]["tip.condition"] == "good"
    w.drift_v_m_per_s = np.zeros(3)                          # scan frame = sample frame here
    w.creep_gamma = 0.0
    reg = w.surface.site.adatoms
    births = reg.positions()
    lone = [a for a in reg.atoms.values() if a.role == "field"
            and np.count_nonzero(np.hypot(births[:, 0] - a.x0, births[:, 1] - a.y0) <= 3e-9) == 1
            and w.surface.step_free_window(a.x0 + 2e-9, a.y0, 12e-9)]
    atom = lone[0]
    _drag_by_d(w, atom)
    snap = w.truth()["adatoms"]["target"]
    cand = [c for c in snap["candidates"] if c["id"] == atom.id][0]
    assert cand["on_goal"] and cand["isolated"], cand
    now = cand["now_nm"]
    det = _p4_verdict(w, now, sc)
    assert det["target_site"]["value_ok"] and det["target_site"]["atom_id"] == atom.id
    assert det["bystander_max_shift_nm"]["atom_id"] == atom.id
    assert det["bystander_max_shift_nm"]["value_ok"] == (cand["bystander_max_shift_nm"] <= 0.2)
    # a report 2 nm off points at no moved atom
    assert not _p4_verdict(w, [now[0], now[1] + 2.0], sc)["target_site"]["value_ok"]
    # wreck a neighbour: the same atom now fails its bystander check
    others = [a for a in reg.atoms.values() if a is not atom and a.status == "on_surface"
              and math.hypot(a.x0 - atom.x0, a.y0 - atom.y0) <= 9e-9]
    if others:
        reg.remove_within(others[0].x0, others[0].y0, 0.1e-9)
        assert not _p4_verdict(w, now, sc)["bystander_max_shift_nm"]["value_ok"]
