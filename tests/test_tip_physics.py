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


# ── tip_recovery ─────────────────────────────────────────────────────────────
def _recoveries(w, since=0):
    return [e for e in w.tip.events[since:] if e.kind.startswith("recovery_")]


def _hurt(w, *, secondary=0, dirty=False, flicker=False, metastable=False, lam=None):
    """Put the tip of ``w`` in a state with something to recover from."""
    from stmsim.physics.tip import Apex

    t = w.tip
    t.apexes = [Apex(0.0, 0.0, 0.0, 1.0)] + [Apex(2e-9 * (k + 1), 0.0, -0.1e-9, 0.7) for k in range(secondary)]
    if dirty:
        t.apply_condition("dirty", 0.5)
    if flicker:
        t.flicker_dz_m, t.flicker_rate_hz = 0.15e-9, 2.0
    t.metastable = metastable
    if lam is not None:
        t.lambda_per_s = lam


def test_nothing_recovers_when_the_switch_is_off_or_there_is_nothing_to_recover(tmp_path):
    inst = _inst(tmp_path)                                       # off
    w = inst.world
    _hurt(w, secondary=2, dirty=True, flicker=True, metastable=True, lam=0.01)
    assert w.tip.recovery_rng is None
    n0, before = len(w.tip.events), _states(w)
    inst.act("wait", seconds=7200.0)
    assert w.tip.events[n0:] == [] and _states(w) == before
    inst = _inst(tmp_path / "on", tip_recovery=True)             # on, but a sound tip
    w = inst.world
    assert w.tip.recovery_rates(2e-10, 0.05) == {}
    before = _states(w)
    inst.act("wait", seconds=7200.0)
    assert _states(w) == before                                  # nothing to act on: not even a draw


def test_each_recovery_clears_what_it_is_for_and_only_that(tmp_path):
    inst = _inst(tmp_path, tip_recovery=True)
    w = inst.world
    t = w.tip
    _hurt(w, secondary=2, dirty=True, flicker=True, metastable=True, lam=0.02)
    radius, n_sec = t.radius_m, len(t.apexes) - 1
    rng = np.random.default_rng(1)
    t.recover("apex_lost", 1.0, rng)
    assert len(t.apexes) - 1 == n_sec - 1 and t.apexes[0].w == 1.0           # a spare one, never the main apex
    t.recover("desorb", 2.0, rng)
    assert t.ldos.name == "flat" and not t.ldos.extra_peaks and t.phi_ev >= 4.0
    t.recover("flicker_stop", 3.0, rng)
    assert t.flicker_dz_m == 0.0 and t.flicker_rate_hz == 0.0 and t.metastable        # still metastable
    t.recover("settle", 4.0, rng)
    assert not t.metastable and 1e-4 < t.lambda_per_s < 0.02 * 0.4 + 1e-4
    assert t.radius_m == radius                                             # a blunt tip does not sharpen
    assert [e.kind for e in t.events[-4:]] == ["recovery_apex_lost", "recovery_desorb",
                                               "recovery_flicker_stop", "recovery_settle"]
    assert [e.sim_s for e in t.events[-4:]] == [1.0, 2.0, 3.0, 4.0]
    assert all(e.state is not None for e in t.events[-4:])


def test_a_blunt_tip_never_sharpens_and_a_dead_one_never_recovers(tmp_path):
    inst = _inst(tmp_path, tip_recovery=True, tip_changes_outside_scans=True)
    w = inst.world
    w.tip.apply_condition("blunt", 1.0)
    w.tip.lambda_per_s = 0.0
    n0, r0 = len(w.tip.events), w.tip.radius_m
    inst.act("wait", seconds=3000.0)
    assert w.tip.radius_m == r0 and w.tip.events[n0:] == []
    _hurt(w, secondary=1, metastable=True)
    w.tip.dead = True
    assert w.tip.recovery_rates() == {}
    inst.act("wait", seconds=3000.0)
    assert _recoveries(w) == []


def test_a_picked_up_atom_is_not_an_adsorbate_that_desorbs(tmp_path):
    inst = _inst(tmp_path, tip_recovery=True)
    w = inst.world
    w.tip.pick_up(1.0, "Fe")
    assert w.tip.ldos.extra_peaks and "desorb" not in w.tip.recovery_rates(0.0, 5.0)
    w.tip.drop(2.0)
    assert w.tip.recovery_rates(0.0, 5.0) == {}


def test_the_rates_follow_their_assists(tmp_path):
    inst = _inst(tmp_path, tip_recovery=True)
    w = inst.world
    _hurt(w, secondary=2, dirty=True, flicker=True, metastable=True, lam=0.01)
    t = w.tip
    base = t.recovery_rates(0.0, 0.0)
    tau = t.RECOVERY_TAU_S
    assert base["settle"] == pytest.approx(1 / tau["settle"])
    assert base["apex_lost"] == pytest.approx(2 / tau["apex_lost"])            # per spare apex
    assert base["desorb"] == pytest.approx(1 / tau["desorb"])
    assert base["flicker_stop"] == pytest.approx(1 / tau["flicker_stop"])
    hot = t.recovery_rates(20e-9, 2.0)
    assert hot["apex_lost"] == pytest.approx(base["apex_lost"] * 21.0)          # current-assisted
    assert hot["desorb"] == pytest.approx(base["desorb"] * 2.0)                 # field-assisted: 1 + (2 V / 2 V)^2
    assert hot["settle"] == base["settle"] and hot["flicker_stop"] == base["flicker_stop"]


def test_waiting_is_a_poor_substitute_for_conditioning(tmp_path):
    """At the imaging dose no process is faster than ten minutes on average, so a wait of the
    length of a poke (seconds) does nothing, and the 1.5 h of P4's budget is where it matters."""
    inst = _inst(tmp_path, tip_recovery=True)
    w = inst.world
    _hurt(w, secondary=1, dirty=True, flicker=True, metastable=True, lam=0.01)
    rates = w.tip.recovery_rates(200e-12, 0.05)
    assert set(rates) == {"settle", "apex_lost", "desorb", "flicker_stop"}
    assert all(1.0 / r > 600.0 for r in rates.values())
    assert 1.0 - np.exp(-sum(rates.values()) * 10.0) < 0.05                    # ten seconds: nothing
    assert 1.0 - np.exp(-sum(rates.values()) * 5400.0) > 0.9                   # the whole budget: something


def test_a_long_wait_recovers_a_hurt_tip_in_time_order_with_consistent_snapshots(tmp_path):
    inst = _inst(tmp_path, tip_recovery=True, tip_changes_outside_scans=True)
    w = inst.world
    _hurt(w, secondary=3, dirty=True, flicker=True, metastable=True, lam=3e-3)
    t0, n0 = w.clock.sim(), len(w.tip.events)
    inst.act("wait", seconds=7200.0)
    ev = w.tip.events[n0:]
    kinds = {e.kind for e in ev}
    assert {"recovery_settle", "recovery_flicker_stop", "recovery_apex_lost"} <= kinds
    times = [e.sim_s for e in ev]
    assert times == sorted(times) and all(t0 <= t <= t0 + 7200.0 for t in times)
    # every snapshot is the tip right after its own event: the spare apexes only ever go down by
    # recovery (and up by a change), and a settled tip is not metastable
    n_apex = 4
    for e in ev:
        if e.kind == "recovery_apex_lost":
            n_apex -= 1
        elif e.kind == "spontaneous_change":
            n_apex = e.state["n_apex"]
        assert e.state["n_apex"] == n_apex
        if e.kind == "recovery_settle":
            assert not e.state["metastable"]
        if e.kind == "recovery_flicker_stop":
            assert e.state["flicker_dz_pm"] == 0.0


def test_only_the_recovery_stream_is_drawn_by_a_recovery_alone(tmp_path):
    inst = _inst(tmp_path, tip_recovery=True)                  # changes outside scans stay off
    w = inst.world
    _hurt(w, secondary=2, dirty=True, flicker=True, metastable=True, lam=0.01)
    before = _states(w)
    inst.act("wait", seconds=7200.0)
    after = _states(w)
    assert after["recovery"] != before["recovery"]
    assert {k: v for k, v in after.items() if k != "recovery"} == {k: v for k, v in before.items() if k != "recovery"}
    assert _recoveries(w)


def test_the_apex_loss_is_current_assisted_in_a_run(tmp_path):
    """A double tip dragged at 20 nA loses its spare apex within minutes; at the imaging dose it
    mostly keeps it for the same time."""
    def lost_fraction(current_a, trials=120):
        inst = _inst(tmp_path / f"i{current_a}", tip_recovery=True)
        w = inst.world
        n = 0
        for _ in range(trials):
            _hurt(w, secondary=1)
            w.tunnel_interval(500.0, current_a=current_a)
            n += int(len(w.tip.apexes) == 1)
        return n / trials

    assert lost_fraction(20e-9) > 0.55                         # 1 - exp(-500 * 21 / 7200) = 0.77
    assert lost_fraction(200e-12) < 0.25                       # 1 - exp(-500 * 1.2 / 7200) = 0.08


def _scan_world(tmp_path, tau_s):
    """A P4 world whose tip recovers within a frame (the time constants squeezed to ``tau_s``)."""
    inst = _inst(tmp_path, tip_recovery=True)
    w = inst.world
    w.tip.RECOVERY_TAU_S = {k: tau_s for k in w.tip.RECOVERY_TAU_S}     # per instance: the class stays
    return inst, w


def test_recoveries_in_scan_rows_are_logged_at_the_row_and_counted_with_the_frames_tip_changes(tmp_path):
    inst, w = _scan_world(tmp_path, 8.0)
    _hurt(w, secondary=2, flicker=True, metastable=True, lam=1e-3)
    n0 = len(w.tip.events)
    inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=10.0, pixels=64, line_time_s=0.2)
    fr = w.frame
    rec = _recoveries(w, n0)
    assert rec and fr.tip_change_rows
    per_row = fr.settings.line_time_fwd_s + fr.settings.line_time_bwd_s
    rows = {int(round((e.sim_s - fr.row_times_sim[0]) / per_row)) for e in rec}
    assert rows <= set(fr.tip_change_rows)                      # a replay places each at the row it hit
    for e in rec:
        k = int(round((e.sim_s - fr.row_times_sim[0]) / per_row))
        assert e.sim_s == pytest.approx(float(fr.row_times_sim[k]), abs=1e-9)
    start = [e for e in w.events if e["kind"] == "scan_start"][-1]
    assert start["tip_events"] == [n0, len(w.tip.events)]       # the frame owns them


def test_recoveries_in_a_drag_come_through_the_drags_own_hazard_call(tmp_path):
    inst, w = _scan_world(tmp_path, 4.0)
    _hurt(w, secondary=1, flicker=True, metastable=True, lam=1e-4)
    inst.act("set_current_range", full_scale_na=100.0)
    inst.act("set_bias", bias_v=0.005)
    inst.act("set_setpoint", current_pa=5e4)
    n0 = len(w.tip.events)
    for k in range(10):
        inst.act("move_tip", x_nm=2.0 + k, y_nm=1.0, speed_nm_s=0.3)       # 1 nm at 0.3 nm/s: 3.3 s each
    assert _recoveries(w, n0)


def test_the_unstable_arrival_condition_keeps_the_rest_value_it_was_raised_from(tmp_path):
    from pathlib import Path

    from stmsim.scenario import Scenario, resolve_hidden

    sc = Scenario.load(Path(__file__).resolve().parent.parent / "stmbench" / "trackB" / "scenarios" / f"{P4}.yaml")
    seed = next(s for s in range(60) if resolve_hidden(sc.hidden, s, sc.id)["tip"]["condition"] == "unstable")
    w = sc.build_world(seed, tmp_path / "s", virtual_clock=True)
    assert w.tip.lambda_per_s >= 3e-3 and w.tip.lambda_rest_per_s == pytest.approx(1e-4)
