"""Adatoms on lattice sites, and moving them with the tip."""
from __future__ import annotations

import math

import numpy as np
import pytest

from stmsim.physics.adatoms import AdatomParams, AdatomRegistry
from stmsim.physics.rig import RigProfile
from stmsim.physics.scanner import ScanSettings
from stmsim.physics.surface import Surface
from stmsim.physics.world import World

PARAMS = AdatomParams(r_threshold_ohm=200e3, r_pick_ohm=30e3, v_max_m_s=1e-9)


def _reg(seed: int = 3) -> AdatomRegistry:
    return AdatomRegistry(Surface("Cu(111)", seed=seed, adsorbate_density_per_um2=0.0).site, PARAMS)


def test_site_index_round_trips():
    reg = _reg()
    rng = np.random.default_rng(0)
    for _ in range(200):
        x, y = rng.uniform(-20e-9, 20e-9, 2)
        i, j = reg.site_of(x, y)
        px, py = reg.site_xy(i, j)
        # the nearest site by brute force over the surrounding cells
        best = min(((reg.site_xy(i + di, j + dj), (di, dj))
                    for di in (-1, 0, 1) for dj in (-1, 0, 1)),
                   key=lambda t: (t[0][0] - x) ** 2 + (t[0][1] - y) ** 2)
        assert best[1] == (0, 0), (x, y, best[1])
        assert math.hypot(px - x, py - y) <= reg.site.material.nn_m


def test_sites_sit_in_the_hollows_of_the_lattice():
    """An adatom binds in a hollow, not on top of a substrate atom: the three-cosine lattice
    is at its minimum there, which pins the basis convention."""
    surf = Surface("Cu(111)", seed=3, adsorbate_density_per_um2=0.0)
    reg = AdatomRegistry(surf.site, PARAMS)
    for i, j in [(0, 0), (2, -1), (-3, 4)]:
        x, y = reg.site_xy(i, j)
        h = float(surf.atomic_height(np.array([x]), np.array([y]))[0])
        assert h == pytest.approx(-surf.material.corrugation_m / 2, rel=1e-6)


def test_adatoms_render_as_bumps_and_cost_nothing_when_absent():
    surf = Surface("Cu(111)", seed=3, adsorbate_density_per_um2=0.0)
    g = np.linspace(-3e-9, 3e-9, 61)
    x, y = np.meshgrid(g, g)
    before = surf.height_smooth(x, y)
    reg = AdatomRegistry(surf.site, PARAMS)
    surf.site.adatoms = reg
    assert np.array_equal(surf.height_smooth(x, y), before)      # empty registry: no change
    reg.add(0, 0)
    after = surf.height_smooth(x, y)
    px, py = reg.site_xy(0, 0)
    peak = np.unravel_index(np.argmax(after - before), after.shape)
    assert (after - before).max() == pytest.approx(PARAMS.height_m, rel=0.02)
    assert math.hypot(x[peak] - px, y[peak] - py) < 0.25e-9


def test_adatoms_are_not_counted_as_operator_damage():
    surf = Surface("Cu(111)", seed=3, adsorbate_density_per_um2=0.0)
    reg = AdatomRegistry(surf.site, PARAMS)
    surf.site.adatoms = reg
    reg.add(0, 0)
    assert surf.damage_area_nm2() == 0.0
    assert surf.damage_in_window(0.0, 0.0, 5e-9) == 0


def _dragging_world(tmp_path, seed=4, bias=0.01, setpoint=50e-9):
    w = World(rig=RigProfile.load("reference-stm"), seed=seed, material="Cu(111)",
              session_dir=tmp_path / "s", adsorbate_density_per_um2=0.0)
    w.surface.configure_adatoms(PARAMS, {"layout": "single", "target_dx_nm": 4.0,
                                         "n_bystanders": 0})
    w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
    w.withdrawn = False
    w.zctrl_set(True)
    w.drift_v_m_per_s = np.zeros(3)
    w.bias_v = bias
    w.zctrl.setpoint_a = setpoint
    return w


def _target(w):
    reg = w.surface.site.adatoms
    return [a for a in reg.atoms.values() if a.role == "target"][0]


def test_low_resistance_drags_the_atom_and_high_resistance_does_not(tmp_path):
    """The 1990 protocol in one assertion: the same move either carries the atom or images
    over it, and which one it does is set by |V| / I."""
    w = _dragging_world(tmp_path, setpoint=50e-9)        # 10 mV / 50 nA = 200 kΩ ≈ R_th
    reg = w.surface.site.adatoms
    atom = _target(w)
    start = reg.site_xy(atom.i, atom.j)
    w.move_xy(start[0], start[1])
    w.zctrl.setpoint_a = 200e-9                          # 50 kΩ, well under the threshold
    events = w.move_xy_path(start[0] + 4e-9, start[1], speed_m_s=0.5e-9)
    moved = reg.site_xy(atom.i, atom.j)
    assert any(e["kind"] == "adatom_hop" for e in events)
    assert moved[0] - start[0] > 3e-9
    assert abs(moved[1] - start[1]) < 1e-9

    imaging = _dragging_world(tmp_path / "b", setpoint=1e-9)   # 10 mV / 1 nA = 10 MΩ
    reg2 = imaging.surface.site.adatoms
    a2 = _target(imaging)
    s2 = reg2.site_xy(a2.i, a2.j)
    imaging.move_xy(*s2)
    assert imaging.move_xy_path(s2[0] + 4e-9, s2[1], speed_m_s=0.5e-9) == []
    assert reg2.site_xy(a2.i, a2.j) == s2


def test_dragging_too_fast_loses_the_atom(tmp_path):
    w = _dragging_world(tmp_path, setpoint=200e-9)
    reg = w.surface.site.adatoms
    atom = _target(w)
    start = reg.site_xy(atom.i, atom.j)
    w.move_xy(*start)
    w.move_xy_path(start[0] + 4e-9, start[1], speed_m_s=50e-9)   # 50× v_max
    assert reg.site_xy(atom.i, atom.j)[0] - start[0] < 1e-9


def test_pick_up_changes_the_tip_and_dropping_restores_it(tmp_path):
    w = _dragging_world(tmp_path, setpoint=2e-6)          # 10 mV / 2 µA = 5 kΩ, below R_pick
    reg = w.surface.site.adatoms
    atom = _target(w)
    start = reg.site_xy(atom.i, atom.j)
    before = w.tip.snapshot()
    w.move_xy(start[0] - 1e-9, start[1])
    events = w.move_xy_path(start[0] + 1e-9, start[1], speed_m_s=0.5e-9)
    assert any(e["kind"] == "adatom_picked" for e in events)
    assert reg.carried is atom and atom.status == "on_tip"
    after = w.tip.snapshot()
    assert after["carried"] == "Fe"
    assert after["apex_sigma_nm"] < before["apex_sigma_nm"]
    assert after["phi_ev"] < before["phi_ev"]
    assert after["ldos"] == "featured"
    assert reg.n_on_surface == 0
    assert w.drop_carried()
    assert reg.carried is None and atom.status == "on_surface"
    restored = w.tip.snapshot()
    for key in ("apex_sigma_nm", "phi_ev", "lambda_per_s", "ldos", "carried"):
        assert restored[key] == before[key], key


def test_scanning_at_an_imaging_resistance_leaves_the_atoms_alone(tmp_path):
    w = _dragging_world(tmp_path, setpoint=1e-9)
    reg = w.surface.site.adatoms
    before = {a.id: (a.i, a.j) for a in reg.atoms.values()}
    st = ScanSettings(cx=0, cy=0, w=20e-9, h=20e-9, nx=64, ny=64,
                      line_time_fwd_s=0.586, line_time_bwd_s=0.586)
    w.renderer.render(st, 0.0)
    assert {a.id: (a.i, a.j) for a in reg.atoms.values()} == before


def test_scanning_at_a_manipulation_resistance_moves_them(tmp_path):
    """Imaging at 10 mV / 500 nA is a manipulation, whether the operator meant it or not."""
    w = _dragging_world(tmp_path, setpoint=500e-9)
    reg = w.surface.site.adatoms
    before = {a.id: (a.i, a.j) for a in reg.atoms.values()}
    st = ScanSettings(cx=0, cy=0, w=20e-9, h=20e-9, nx=64, ny=64,
                      line_time_fwd_s=5.0, line_time_bwd_s=5.0)   # 4 nm/s, slow enough to drag
    w.renderer.render(st, 0.0)
    after = {a.id: (a.i, a.j) for a in reg.atoms.values()}
    assert after != before or reg.carried is not None
    assert any(e["kind"] in ("adatom_hop", "adatom_picked") for e in w.events)


def test_moving_the_scan_frame_never_drags_an_atom(tmp_path):
    """``Scan.FrameSet`` and the snap at the start of a scan are teleports, not FolMe moves."""
    w = _dragging_world(tmp_path, setpoint=500e-9)
    reg = w.surface.site.adatoms
    atom = _target(w)
    before = (atom.i, atom.j)
    epoch = reg.epoch
    w.scan = ScanSettings(cx=6e-9, cy=6e-9, w=5e-9, h=5e-9, nx=32, ny=32,
                          line_time_fwd_s=0.1, line_time_bwd_s=0.1)
    w.move_xy(6e-9, 6e-9)
    assert (atom.i, atom.j) == before and reg.epoch == epoch


def test_a_crash_takes_the_nearby_atoms_with_it(tmp_path):
    w = _dragging_world(tmp_path, setpoint=1e-9)
    reg = w.surface.site.adatoms
    atom = _target(w)
    w.move_xy(*reg.site_xy(atom.i, atom.j))
    w._crash("test", severity=1.0)
    assert atom.status == "lost"
    assert any(e["kind"] == "adatom_lost" for e in w.events)


def test_the_registry_snapshot_carries_what_the_judge_needs(tmp_path):
    w = _dragging_world(tmp_path, setpoint=1e-9)
    snap = w.truth()["adatoms"]
    assert snap["n"] >= 1 and snap["carried"] is None
    assert snap["target"]["site"] and snap["target"]["site_nm"]
    assert snap["target"]["on_target"] is False
    assert "bystander_max_shift_nm" in snap


def test_streams_are_independent_of_the_terrain(tmp_path):
    """Configuring adatoms must not move the terrain or the tip of any other scenario."""
    plain = Surface("Cu(111)", seed=9)
    ref = (plain.site.step_angle, plain.site.lattice_angle, plain.site.features[0].x)
    surf = Surface("Cu(111)", seed=9)
    surf.configure_adatoms(PARAMS, {"layout": "single"})
    assert (surf.site.step_angle, surf.site.lattice_angle) == ref[:2]
    # _clear_area removes the adsorbates inside the working window on purpose; outside it the
    # population is untouched
    assert len(surf.site.features) <= len(plain.site.features)


def test_a_folme_move_over_the_wire_drags_the_atom(tmp_path):
    """``FolMe.XYPosSet`` is the path MoveAtomTo drags an atom along, so the wire handler
    must integrate the move through the registry at the FolMe speed. It used to teleport
    (``move_xy``): no manipulation ever reached the physics from the wire, and every
    MoveAtomTo in every mode reported "moved" over an atom that had not moved
    (2026-09-11, P4 / P3-repair trials)."""
    from stmsim.modules import build_dispatcher

    w = _dragging_world(tmp_path, setpoint=200e-9)        # 10 mV / 200 nA = 50 kΩ, under R_th
    d = build_dispatcher(w)
    atom = _target(w)
    reg = w.surface.site.adatoms
    start = reg.site_xy(atom.i, atom.j, atom.sub)
    w.move_xy(start[0], start[1])                          # park on the atom: a teleport, no drag
    d.handler_for("FolMe_SpeedSet")(0.5e-9, 1)             # 0.5 nm/s, under v_max
    d.handler_for("FolMe_XYPosSet")(start[0] + 4e-9, start[1], 0)
    hops = [e for e in w.events if e["kind"] == "adatom_hop"]
    assert hops and hops[-1]["cause"] == "folme", w.events[-3:]
    end = reg.site_xy(atom.i, atom.j, atom.sub)
    assert (end[0] - start[0]) > 3e-9 and abs(end[1] - start[1]) < 1e-9
    # the same move at an imaging resistance is a plain move
    imaging = _dragging_world(tmp_path / "b", setpoint=1e-9)   # 10 MΩ
    d2 = build_dispatcher(imaging)
    a2 = _target(imaging)
    s2 = imaging.surface.site.adatoms.site_xy(a2.i, a2.j, a2.sub)
    imaging.move_xy(s2[0], s2[1])
    d2.handler_for("FolMe_SpeedSet")(0.5e-9, 1)
    d2.handler_for("FolMe_XYPosSet")(s2[0] + 4e-9, s2[1], 0)
    assert not [e for e in imaging.events if e["kind"] == "adatom_hop"]
    assert imaging.tip_x == pytest.approx(s2[0] + 4e-9)


def test_ring_occupancy_is_geometric_so_a_gap_closed_on_the_next_hollow_counts(tmp_path):
    """A repaired ring is judged by where the atoms sit, not by which lattice index they
    landed on: an atom within half a site spacing of a designed site holds it. A corral
    closed with every atom on the hollow next to the designed one (2026-09-11 P3-repair
    trial: right resonances, one exact index of six) is closed."""
    from stmsim.physics.rig import RigProfile
    from stmsim.physics.world import World

    w = World(rig=RigProfile.load("reference-stm"), seed=9, material="Cu(111)",
              session_dir=tmp_path / "s", adsorbate_density_per_um2=0.0)
    w.surface.configure_adatoms(PARAMS, {"layout": "corral", "ring_radius_nm": 7.13, "ring_n": 48,
                                         "gap_atoms": 3})
    reg = w.surface.site.adatoms
    gaps = [tuple(g) for g in reg.ring["gap_sites"]]
    n = reg.ring["n_sites"]
    assert len(gaps) == 3 and reg.ring_occupancy() == pytest.approx((n - 3) / n)
    spacing = 2 * math.pi * reg.ring["radius_nm"] * 1e-9 / n
    # close the gap on hollows displaced ~0.2 nm from the designed sites (a different index)
    for (i, j) in gaps:
        sx, sy = reg.site_xy(i, j)
        ni, nj = reg.site_of(sx + 0.2e-9, sy + 0.15e-9)
        if (ni, nj) == (i, j):
            ni, nj = reg.site_of(sx - 0.25e-9, sy + 0.1e-9)
        assert (ni, nj) != (i, j)
        assert reg.add(ni, nj, role="ring") is not None
    assert reg.ring_occupancy() == pytest.approx(1.0)
    # an atom a whole site spacing away from a designed site holds nothing
    w2 = World(rig=RigProfile.load("reference-stm"), seed=9, material="Cu(111)",
               session_dir=tmp_path / "t", adsorbate_density_per_um2=0.0)
    w2.surface.configure_adatoms(PARAMS, {"layout": "corral", "ring_radius_nm": 7.13, "ring_n": 48,
                                          "gap_atoms": 3})
    reg2 = w2.surface.site.adatoms
    (i, j) = tuple(reg2.ring["gap_sites"][0])
    sx, sy = reg2.site_xy(i, j)
    cx, cy = (c * 1e-9 for c in reg2.ring["centre_nm"])
    ux, uy = (sx - cx) / math.hypot(sx - cx, sy - cy), (sy - cy) / math.hypot(sx - cx, sy - cy)
    far = reg2.site_of(sx + ux * 1.2 * spacing, sy + uy * 1.2 * spacing)   # radially outward
    assert reg2.add(*far, role="spare") is not None
    assert reg2.ring_occupancy() == pytest.approx((n - 3) / n)
