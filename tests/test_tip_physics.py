"""The P4 round's tip physics, one mechanism at a time: the tip changing outside scans, the tip
recovering on its own, the tip's anisotropy reaching the images, and the crash when the setpoint
is over the preamplifier's range. Every mechanism is a switch (``initial.physics``); these tests
turn them on one by one on the P4 world, and test_physics_switches.py checks the switches."""
from __future__ import annotations

import numpy as np
import pytest

from stmbench.instrument import Instrument
from stmsim.physics.switches import PhysicsSwitches

P4 = "P4_atom_positioning_cu111"


def _inst(tmp_path, seed=0, scenario=P4, **switches) -> Instrument:
    """An episode on ``scenario`` with exactly the given switches on (the scenario's own are
    replaced), and the tip made a good one so a test sees only what it set up."""
    inst = Instrument(scenario, seed, tmp_path / f"run{seed}")
    w = inst.world
    w.set_physics(PhysicsSwitches(**switches))
    w.tip.lambda_per_s = 1e-4
    w.tip.metastable = False
    return inst


def _states(w) -> dict:
    """The state of every random generator the world and its tip own."""
    gens = {"world": w.rng, "tip": w.tip.rng, "noise": w.noise.rng, "df": w.df_rng,
            "outscan": w.rng_outscan, "recovery": w.rng_recovery, "aniso": w.rng_aniso}
    return {k: g.bit_generator.state for k, g in gens.items()}


def _changes(w, since=0):
    return [e for e in w.tip.events[since:] if e.kind == "spontaneous_change"]


# ── tip_changes_outside_scans ────────────────────────────────────────────────
def test_a_long_wait_at_a_high_hazard_changes_the_tip_inside_the_wait(tmp_path):
    inst = _inst(tmp_path, tip_changes_outside_scans=True)
    w = inst.world
    w.tip.lambda_per_s = 0.01                        # one change per ~80 s at 200 pA / 50 mV
    t0, n0 = w.clock.sim(), len(w.tip.events)
    inst.act("wait", seconds=600.0)
    ch = _changes(w, n0)
    assert len(ch) >= 3                              # 600 s × 0.012 /s = 7.2 expected
    times = [e.sim_s for e in ch]
    assert times == sorted(times)
    assert all(t0 <= t <= t0 + 600.0 for t in times)
    assert len(set(times)) == len(times)             # each at a time of its own, not at the start
    assert all(e.state is not None and e.state["n_apex"] >= 1 for e in ch)
    assert w.clock.sim() == pytest.approx(t0 + 600.0)


def test_no_change_at_a_zero_hazard_and_no_draw_either(tmp_path):
    inst = _inst(tmp_path, tip_changes_outside_scans=True)
    w = inst.world
    w.tip.lambda_per_s = 0.0
    before, n0 = _states(w), len(w.tip.events)
    inst.act("wait", seconds=3600.0)
    assert _changes(w, n0) == []
    assert _states(w) == before


def test_with_the_switch_off_a_wait_is_as_safe_as_it_was(tmp_path):
    inst = _inst(tmp_path)                           # nothing on
    w = inst.world
    w.tip.lambda_per_s = 0.05
    before, n0 = _states(w), len(w.tip.events)
    inst.act("wait", seconds=3600.0)
    assert _changes(w, n0) == []
    assert _states(w) == before


def test_the_changes_draw_from_their_own_stream_and_no_other(tmp_path):
    inst = _inst(tmp_path, tip_changes_outside_scans=True)
    w = inst.world
    w.tip.lambda_per_s = 0.01
    before = _states(w)
    inst.act("wait", seconds=600.0)
    after = _states(w)
    assert after["outscan"] != before["outscan"]
    assert {k: v for k, v in after.items() if k != "outscan"} == {k: v for k, v in before.items() if k != "outscan"}


def test_the_number_of_changes_follows_the_hazard_law(tmp_path):
    """λ·(1 + |I|/1 nA + (|V|/2 V)²) per second, whatever the length of the interval."""
    inst = _inst(tmp_path, tip_changes_outside_scans=True)
    w = inst.world
    w.tip.lambda_per_s = 1e-3
    for current, volts in ((200e-12, 0.05), (5e-9, 0.05), (200e-12, 3.0)):
        n0 = len(w.tip.events)
        for _ in range(500):
            w.tunnel_interval(100.0, current_a=current, bias_v=volts)
        got = len(w.tip.events) - n0
        want = 500 * 100.0 * 1e-3 * (1 + current / 1e-9 + (volts / 2.0) ** 2)
        assert want * 0.8 < got < want * 1.2, (current, volts, got, want)


def test_a_withdrawn_tip_is_not_hazarded(tmp_path):
    inst = _inst(tmp_path, tip_changes_outside_scans=True)
    w = inst.world
    w.tip.lambda_per_s = 0.05
    inst.act("withdraw")
    before, n0 = _states(w), len(w.tip.events)
    inst.act("wait", seconds=3600.0)
    assert _changes(w, n0) == [] and _states(w) == before


def test_a_tip_held_far_from_the_surface_is_not_hazarded_either(tmp_path):
    inst = _inst(tmp_path, tip_changes_outside_scans=True)
    w = inst.world
    w.tip.lambda_per_s = 0.05
    inst.act("feedback", on=False)
    w.zctrl.z_n = w.zctrl.limit_high_m - 100e-9       # 100 nm up: no junction
    n0 = len(w.tip.events)
    inst.act("wait", seconds=3600.0)
    assert _changes(w, n0) == []
    w.zctrl.z_n = w.z_ctrl_from_tip(w.surface_height_here() + 0.5e-9)   # back in range
    assert w.tunnelling_now()
    inst.act("wait", seconds=3600.0)
    assert len(_changes(w, n0)) > 0


class _Spy:
    """Records the calls a world method gets, then runs it."""

    def __init__(self, w, name="tunnel_interval"):
        self.calls, self._real = [], getattr(w, name)
        setattr(w, name, self)

    def __call__(self, dt, **kw):
        self.calls.append((float(dt), kw))
        return self._real(dt, **kw)


def test_a_sweep_passes_its_largest_field_and_its_largest_current(tmp_path):
    inst = _inst(tmp_path, tip_changes_outside_scans=True)
    w = inst.world
    spy = _Spy(w)
    r = inst.act("sts", v_start=-1.0, v_end=0.4, points=64, settle_ms=2.0, integration_ms=3.0)
    (dt, kw), = spy.calls
    assert dt == pytest.approx(r["duration_s"], abs=1e-3)
    assert kw["bias_v"] == pytest.approx(1.0)
    got = inst.spectrum(r["spectrum"])
    assert kw["current_a"] == pytest.approx(float(np.max(np.abs(got["i"]))))
    spy.calls.clear()
    r = inst.act("zspec", sweep_nm=0.1, points=32)
    (dt, kw), = spy.calls
    assert dt >= r["duration_s"] - 1e-3
    assert kw["current_a"] == pytest.approx(float(np.nanmax(np.abs(inst.spectrum(r["spectrum"])["i"]))))


def test_waiting_navigating_and_settling_count_and_the_rest_does_not(tmp_path):
    inst = _inst(tmp_path, tip_changes_outside_scans=True)
    w = inst.world
    spy = _Spy(w)
    inst.act("wait", seconds=12.0)
    assert [c[0] for c in spy.calls] == [12.0]
    spy.calls.clear()
    for name, args in (("set_bias", {"bias_v": 0.06}), ("set_setpoint", {"current_pa": 220.0}),
                       ("feedback", {"on": False}), ("feedback", {"on": True}),
                       ("set_current_range", {"full_scale_na": 10.0}), ("lockin", {"on": True}),
                       ("drift_comp", {"on": False})):
        inst.act(name, **args)
    assert len(spy.calls) == 7 and all(c[0] >= 0.05 for c in spy.calls)
    spy.calls.clear()
    x0, y0 = w.tip_x * 1e9, w.tip_y * 1e9
    inst.act("move_tip", x_nm=x0 + 9.0, y_nm=y0, speed_nm_s=300.0)    # a navigation move: 9 nm at 300 nm/s
    assert len(spy.calls) == 1 and spy.calls[0][0] == pytest.approx(9.0 / 300.0)
    spy.calls.clear()
    # pulses and pokes have their own outcomes; coarse steps and the approach are not tunnelling
    inst.act("pulse", bias_v=1.0, width_ms=20.0)
    inst.act("poke", depth_nm=0.1, dwell_s=0.1)
    inst.act("withdraw")
    inst.act("coarse_move", direction="X+", steps=2)
    inst.act("approach", timeout_s=60.0)
    assert spy.calls == []


def test_a_drag_keeps_its_own_hazard_and_does_not_count_twice(tmp_path):
    inst = _inst(tmp_path, tip_changes_outside_scans=True)
    w = inst.world
    reg = w.surface.site.adatoms
    # a manipulation resistance low enough for the move to drag: the drag's own hazard, not the interval's
    inst.act("set_current_range", full_scale_na=100.0)
    inst.act("set_bias", bias_v=0.005)
    inst.act("set_setpoint", current_pa=5e4)
    assert w.junction_resistance_ohm() < w.DRAG_CEILING * reg.params.r_threshold_ohm
    spy = _Spy(w)
    inst.act("move_tip", x_nm=5.0, y_nm=5.0, speed_nm_s=200.0)
    assert spy.calls == []


def test_a_scan_does_not_notice_the_switch(tmp_path):
    runs = []
    for on in (False, True):
        inst = _inst(tmp_path / str(on), tip_changes_outside_scans=on)
        inst.world.tip.lambda_per_s = 0.01
        r = inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=20.0, pixels=64, line_time_s=0.1)
        runs.append((inst.frame_digest(r["frame"]), [(e.sim_s, e.kind) for e in inst.world.tip.events]))
    assert runs[0] == runs[1]


def test_the_same_episode_replays_to_the_same_changes(tmp_path):
    out = []
    for k in range(2):
        inst = _inst(tmp_path / str(k), seed=3, tip_changes_outside_scans=True)
        inst.world.tip.lambda_per_s = 0.01
        inst.act("wait", seconds=300.0)
        inst.act("sts", v_start=-0.5, v_end=0.5, points=32)
        inst.act("move_tip", x_nm=6.0, y_nm=1.0, speed_nm_s=100.0)
        out.append(([(e.sim_s, e.kind, dict(e.detail)) for e in inst.world.tip.events], inst.digest()))
    assert out[0] == out[1]
    assert any(k == "spontaneous_change" for _, k, _ in out[0][0])
