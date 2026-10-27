"""Scenario physics switches (``initial.physics``): parsed strictly, all off unless asked for,
and every new stochastic mechanism on a generator of its own."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from stmbench.instrument import Instrument, verify
from stmsim.physics.switches import (TIP_ANISO_STREAM, TIP_OUTSCAN_STREAM, TIP_RECOVERY_STREAM,
                                     PhysicsSwitches)
from stmsim.physics.world import World
from stmsim.scenario import Scenario, resolve_hidden

SCEN = Path(__file__).resolve().parent.parent / "stmbench" / "trackB" / "scenarios"


def _scenario_with(tmp_path, physics, name="x"):
    d = yaml.safe_load((SCEN / "P4_atom_positioning_cu111.yaml").read_text(encoding="utf-8"))
    d["initial"] = {k: v for k, v in d["initial"].items() if k != "physics"}
    if physics is not None:
        d["initial"]["physics"] = physics
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    return p


def test_every_switch_is_off_by_default():
    sw = PhysicsSwitches()
    assert not sw.any()
    assert sw.as_dict() == {"tip_changes_outside_scans": False, "tip_recovery": False,
                            "tip_anisotropy": False, "crash_over_range": False}
    assert PhysicsSwitches.from_config(None) == sw
    assert PhysicsSwitches.from_config({}) == sw


def test_a_missing_key_is_off_and_a_named_one_is_on():
    sw = PhysicsSwitches.from_config({"tip_recovery": True, "crash_over_range": False})
    assert sw.tip_recovery and not sw.crash_over_range and not sw.tip_anisotropy
    assert sw.any()


@pytest.mark.parametrize("cfg", [{"tip_recovry": True}, {"tip_recovery": True, "bogus": False}])
def test_an_unknown_switch_raises(cfg):
    with pytest.raises(ValueError, match="unknown physics switch"):
        PhysicsSwitches.from_config(cfg)


@pytest.mark.parametrize("cfg", [{"tip_recovery": "false"}, {"tip_recovery": 1}, {"tip_recovery": None}])
def test_a_switch_must_be_true_or_false(cfg):
    with pytest.raises(ValueError, match="true or false"):
        PhysicsSwitches.from_config(cfg)


def test_the_block_must_be_a_mapping():
    with pytest.raises(ValueError, match="mapping"):
        PhysicsSwitches.from_config(["tip_recovery"])


def test_a_typo_in_a_scenario_file_raises_when_it_is_loaded(tmp_path):
    p = _scenario_with(tmp_path, {"tip_anisotropi": True})
    with pytest.raises(ValueError, match="unknown physics switch"):
        Scenario.load(p)


def test_a_scenario_without_the_block_builds_a_world_with_every_switch_off(tmp_path):
    sc = Scenario.load(_scenario_with(tmp_path, None))
    assert not sc.physics_switches().any()
    w = sc.build_world(0, tmp_path / "s", virtual_clock=True)
    assert not w.physics.any()
    assert "physics" not in w.truth()                 # a ledger of such a world reads as it always did


def test_the_switches_of_a_scenario_reach_its_world(tmp_path):
    sc = Scenario.load(_scenario_with(tmp_path, {"tip_changes_outside_scans": True, "crash_over_range": True}))
    w = sc.build_world(0, tmp_path / "s", virtual_clock=True)
    assert w.physics == PhysicsSwitches(tip_changes_outside_scans=True, crash_over_range=True)
    assert w.truth()["physics"] == {"tip_changes_outside_scans": True, "tip_recovery": False,
                                    "tip_anisotropy": False, "crash_over_range": True}


def test_only_p4_switches_physics_on_and_it_switches_all_four():
    on = {}
    for p in sorted(SCEN.glob("*.yaml")):
        sw = Scenario.load(p).physics_switches()
        if sw.any():
            on[p.stem] = sw
    assert on == {"P4_atom_positioning_cu111": PhysicsSwitches(
        tip_changes_outside_scans=True, tip_recovery=True, tip_anisotropy=True, crash_over_range=True)}


def test_the_p4_world_has_its_mechanisms_wired(tmp_path):
    w = Scenario.load(SCEN / "P4_atom_positioning_cu111.yaml").build_world(0, tmp_path / "s", virtual_clock=True)
    assert w.physics.as_dict() == {k: True for k in PhysicsSwitches.keys()}
    assert w.tip.recovery_rng is w.rng_recovery and w.tip.aniso_rng is w.rng_aniso
    assert w.truth()["physics"] == w.physics.as_dict()
    assert w.tip.lambda_rest_per_s == pytest.approx(1e-4)
    assert not w.over_range()                          # the scenario does not start pressed into the sample


def test_a_scenario_that_starts_over_the_range_is_refused(tmp_path):
    p = _scenario_with(tmp_path, {"crash_over_range": True})
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    d["initial"]["setpoint_a"] = 50.0e-9                # 50 nA on the rig's 10 nA range
    p.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="above the preamplifier range"):
        Scenario.load(p).build_world(0, tmp_path / "s", virtual_clock=True)


def test_the_new_streams_are_seeded_by_seed_and_key_and_start_untouched(tmp_path):
    w = World(seed=7, session_dir=tmp_path / "s")
    for rng, key in ((w.rng_outscan, TIP_OUTSCAN_STREAM), (w.rng_recovery, TIP_RECOVERY_STREAM),
                     (w.rng_aniso, TIP_ANISO_STREAM)):
        assert rng.bit_generator.state == np.random.default_rng([7, key]).bit_generator.state
    assert (TIP_OUTSCAN_STREAM, TIP_RECOVERY_STREAM, TIP_ANISO_STREAM) == (0x71D1, 0x71D2, 0x71D3)


def test_a_p4_episode_with_every_mechanism_at_work_replays_to_the_same_measurements(tmp_path):
    """P4 as shipped (all four on) with a tip that changes: a long wait, a spectrum, a setpoint over
    the range (a crash, then a plough), a pulse and a deep poke (an anisotropic tip) - re-checked
    from the ledger by `verify`."""
    sc = Scenario.load(SCEN / "P4_atom_positioning_cu111.yaml")
    seed = next(s for s in range(60) if resolve_hidden(sc.hidden, s, sc.id)["tip"]["condition"] == "unstable")
    inst = Instrument(sc.id, seed, tmp_path / "run")
    w = inst.world
    assert w.physics == PhysicsSwitches(tip_changes_outside_scans=True, tip_recovery=True,
                                        tip_anisotropy=True, crash_over_range=True)
    inst.act("wait", seconds=900.0)
    inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=10.0, pixels=48, line_time_s=0.1)
    inst.act("sts", v_start=-0.3, v_end=0.3, points=32)
    inst.act("set_setpoint", current_pa=20000.0)                  # over the range: a crash, then a plough
    inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=8.0, pixels=32, line_time_s=0.05)
    inst.act("set_current_range", full_scale_na=100.0)
    inst.act("set_setpoint", current_pa=200.0)
    inst.act("pulse", bias_v=4.0, width_ms=50.0)
    inst.act("poke", depth_nm=1.2, dwell_s=0.5)
    inst.act("wait", seconds=1200.0)
    inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=10.0, pixels=48, line_time_s=0.1)
    kinds = {e.kind for e in w.tip.events}
    assert {"spontaneous_change", "crash", "pulse"} <= kinds
    inst.finish()
    assert verify(tmp_path / "run")["ok"]
    assert '"angle_deg"' in (tmp_path / "run" / "tip_timeline.json").read_text(encoding="utf-8")   # anisotropy is in the history
