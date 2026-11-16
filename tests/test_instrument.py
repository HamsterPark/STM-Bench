"""The benchmark instrument: primitive actions on a virtual clock, read-only
observations, the version-1 result channel, a ledger, and re-checking by replay."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from stmbench.instrument import ACTIONS, API_VERSION, ActionError, Instrument, replay, verify
from stmbench.instrument.core import Instrument as _I, validate
from stmbench.trackB.rules import RULES_VERSION
from stmsim.io.sxm_reader import read_dat
from stmsim.physics.adatoms import path_points

P4 = "P4_atom_positioning_cu111"
P5 = "P5_sader_jarvis_qplus"


def test_a_trace_reads_the_current_the_status_reads_in_the_manipulation_regime():
    """Screen trial 3 (seed 451): the traces of drags at 100 nA read a flat 34.78 nA at 10 mV and
    6.96 nA at 2 mV, a fixed 287 kohm, while the status bar read the setpoint; the agent concluded
    that the junction current saturates near 35 nA. The status and the trace now agree in both
    regimes: the gap floor binds at 100 and 20 kohm and not at 500."""
    inst = Instrument(P4, 451)
    inst.act("set_current_range", full_scale_na=100.0)
    x = -6.0
    for bias, setpoint_pa in ((0.010, 1e5), (0.002, 1e5), (0.010, 2e4)):
        inst.act("set_bias", bias_v=bias)
        inst.act("set_setpoint", current_pa=setpoint_pa)
        status = inst.status()["current_pa"]
        r = inst.act("move_tip", x_nm=x + 1.5, y_nm=-6.0, speed_nm_s=0.25)
        x += 1.5
        i = np.asarray(inst.spectrum(r["trace"])["i"]) * 1e12
        assert status == pytest.approx(setpoint_pa, rel=1e-3)
        assert np.mean(i) == pytest.approx(status, rel=0.01) and np.std(i) < 0.01 * status, (bias, setpoint_pa)


def test_a_scenario_the_instrument_cannot_judge_is_refused_at_the_start():
    """The skill scenarios judge on evidence the instrument does not record; an agent must
    not play a whole episode only for the judge to fail at the end."""
    with pytest.raises(ValueError, match="success kind is 'tip_repaired'"):
        Instrument("B5_repair_blunt", 0)
    assert Instrument(P4, 0).scenario.id == P4                 # the paper scenarios open


def _script(inst: Instrument, *, look: bool = False) -> None:
    """A short session touching most actions; ``look`` interleaves observations."""
    def peek():
        if look:
            inst.status()
            inst.list_data()
            for f in list(inst.frames):
                inst.frame(f)
    r = inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=20.0, pixels=64)
    peek()
    inst.act("set_bias", bias_v=0.02)
    peek()
    inst.act("set_setpoint", current_pa=500.0)
    inst.act("move_tip", x_nm=2.0, y_nm=-1.0, speed_nm_s=5.0)
    peek()
    inst.act("sts", v_start=-0.4, v_end=0.4, points=64)
    peek()
    inst.act("wait", seconds=120.0)
    inst.act("scan", center_x_nm=1.0, center_y_nm=1.0, size_nm=12.0, pixels=48, direction="up")
    peek()
    inst.report("target_site", 2.0, x_nm=2.0, y_nm=-1.0, frame=r["frame"], evidence=[r["frame"]])


def test_every_action_has_an_implementation_and_a_schema():
    for name, spec in ACTIONS.items():
        assert callable(getattr(_I, f"_do_{name}", None)), name
        schema = spec.schema()
        assert schema["type"] == "object" and set(schema["properties"]) == {p.name for p in spec.params}
    assert "scan" in Instrument.manual() and "move_tip" in Instrument.manual()


@pytest.mark.parametrize("args,needle", [({"center_x_nm": 0, "center_y_nm": 0, "size_nm": 10, "pixels": 7}, "outside"),
                                         ({"center_x_nm": 0, "center_y_nm": 0}, "required"),
                                         ({"center_x_nm": 0, "center_y_nm": 0, "size_nm": 10, "zoom": 2}, "unknown"),
                                         ({"center_x_nm": "0", "center_y_nm": 0, "size_nm": 10}, "number"),
                                         ({"center_x_nm": 0, "center_y_nm": 0, "size_nm": 10, "direction": "left"}, "one of")])
def test_a_bad_call_is_refused_and_costs_nothing(args, needle):
    inst = Instrument(P4, 0)
    t = inst.used_s
    with pytest.raises(ActionError, match=needle):
        inst.act("scan", **args)
    assert inst.used_s == t and inst.n_actions == 0
    assert inst.log[-1]["kind"] == "refused"


def test_a_refusal_by_the_instruments_own_checks_is_on_the_record_and_costs_nothing():
    """Range and state checks inside an action refuse before anything moves; the ledger (and an
    export, which a grader replays) holds them as it holds a refused bad call."""
    inst = Instrument(P4, 0)
    t = inst.used_s
    with pytest.raises(ActionError, match="outside the piezo range"):
        inst.act("move_tip", x_nm=1.0e6, y_nm=0.0)
    with pytest.raises(ActionError, match="exceeds the piezo range"):
        inst.act("scan", center_x_nm=1.0e6, center_y_nm=0.0, size_nm=10.0)
    inst.act("withdraw")
    t_withdrawn = inst.used_s
    with pytest.raises(ActionError, match="withdrawn"):
        inst.act("pulse", bias_v=3.0, width_ms=50.0)
    assert inst.used_s == t_withdrawn > t and inst.n_actions == 1
    assert [r["kind"] for r in inst.log] == ["refused", "refused", "act", "refused"]
    assert inst.log[0]["error"] == "move_tip: outside the piezo range"
    assert inst.log[0]["args"] == {"x_nm": 1.0e6, "y_nm": 0.0}


def test_the_same_actions_give_the_same_measurements():
    a, b = Instrument(P4, 3), Instrument(P4, 3)
    _script(a)
    _script(b)
    assert a.digest() == b.digest()
    assert a.used_s == b.used_s


def test_looking_does_not_change_what_is_measured():
    """Observations spend no instrument time and draw no random number."""
    quiet, curious = Instrument(P4, 3), Instrument(P4, 3)
    _script(quiet)
    _script(curious, look=True)
    assert quiet.digest() == curious.digest() and quiet.used_s == curious.used_s


def test_instrument_time_is_the_physical_duration_of_the_actions():
    inst = Instrument(P4, 0)
    r = inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=20.0, pixels=64, line_time_s=0.5)
    assert r["duration_s"] == pytest.approx(64 * 2 * 0.5)
    assert inst.used_s == pytest.approx(64.0, abs=0.01)
    inst.act("wait", seconds=30.0)
    assert inst.used_s == pytest.approx(94.0, abs=0.01)


@pytest.mark.parametrize("pixels,line_time_s", [(40, 0.586), (64, 0.5), (97, 0.333), (128, 0.586)])
def test_every_saved_frame_is_complete(pixels, line_time_s):
    """Regression: the frame counts rows as floor(elapsed / row time); advancing exactly the
    nominal duration could leave the last row unacquired, and the judge only accepts complete
    frames as evidence."""
    inst = Instrument(P4, 0)
    r = inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=10.0, pixels=pixels, line_time_s=line_time_s)
    ev = [e for e in inst.world.events if e["kind"] == "scan_saved"][-1]
    assert ev["complete"] and ev["rows"] == pixels
    assert np.isfinite(inst.frame(r["frame"])["array"]).all()


def test_a_frame_comes_back_in_image_orientation():
    """Row 0 is the top of the frame whichever way the slow axis ran."""
    inst = Instrument(P4, 7)
    inst.world.drift_v_m_per_s = np.zeros(3)          # the same surface twice, nothing moving
    inst.world._creep.clear()
    r = inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=20.0, pixels=64)
    f = inst.frame(r["frame"])
    assert f["array"].shape == (64, 64) and f["unit"] == "m" and f["nm_per_px"] == pytest.approx(20 / 64)
    up = inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=20.0, pixels=64, direction="up")
    a = f["array"] - np.median(f["array"], axis=1, keepdims=True)
    b = inst.frame(up["frame"])["array"]
    b = b - np.median(b, axis=1, keepdims=True)
    same = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    flipped = np.corrcoef(a.ravel(), b[::-1].ravel())[0, 1]
    assert same > 0.6 and same > flipped
    with pytest.raises(ActionError):
        inst.frame("nope.sxm")


def test_the_budget_ends_the_episode():
    inst = Instrument(P4, 0)
    inst.act("wait", seconds=inst.max_sim_s * 1.06)
    with pytest.raises(ActionError, match="budget"):
        inst.act("set_bias", bias_v=0.1)
    out = inst.finish()
    assert out["success"] is False and out["budget_overrun"]


def test_reports_follow_the_result_channel():
    inst = Instrument(P4, 0)
    with pytest.raises(ActionError, match="unknown claim"):
        inst.report("e0_mev", 1.0)
    msg = inst.report("target_site", 1.0, x_nm=1.0, y_nm=1.0, frame="x.sxm", evidence=["x.sxm"])
    assert "still to report: none" in msg            # the constraint is not reported
    out = inst.finish()
    assert out["success"] is False and out["claims_total"] == 2
    with pytest.raises(ActionError, match="over"):
        inst.act("wait", seconds=1.0)


def test_an_episode_rechecks_by_replay(tmp_path):
    inst = Instrument(P4, 5, out_dir=tmp_path / "run")
    _script(inst, look=True)
    inst.finish()
    ep = json.loads((tmp_path / "run" / "episode.json").read_text(encoding="utf-8"))
    assert ep["rules_version"] == RULES_VERSION and ep["api_version"] == API_VERSION
    assert (tmp_path / "run" / "actions.jsonl").exists() and (tmp_path / "run" / "events.json").exists()
    assert verify(tmp_path / "run")["ok"]
    # a ledger whose measurements were edited no longer re-checks. Without fingerprints (a ledger of
    # an older run) the check is exact: another hash is a difference
    fps = ep.pop("fingerprints")
    ep["digest"]["frames"]["sim001.sxm"]["z_fwd"] = "0" * 64
    (tmp_path / "run" / "episode.json").write_text(json.dumps(ep), encoding="utf-8")
    bad = verify(tmp_path / "run")
    assert not bad["ok"] and "frames" in bad["differences"]
    assert replay(tmp_path / "run").digest()["frames"]["sim001.sxm"]["z_fwd"] != "0" * 64
    # with the fingerprints another hash alone is a rounding difference: the values are the same
    ep["fingerprints"] = fps
    (tmp_path / "run" / "episode.json").write_text(json.dumps(ep), encoding="utf-8")
    again = verify(tmp_path / "run")
    assert again["ok"] and again["rounding"] == [
        {"frame": "sim001.sxm", "array": "z_fwd", "max_rel_diff": 0.0}]
    # ...and a fingerprint that moved beyond rounding is a difference, whatever the hash says
    ep["fingerprints"]["frames"]["sim001.sxm"]["z_fwd"]["max"] *= 1.0 + 1e-6
    (tmp_path / "run" / "episode.json").write_text(json.dumps(ep), encoding="utf-8")
    assert "frames" in verify(tmp_path / "run")["differences"]
    # events are compared by value: a float off in its last bits passes, a changed integer does not
    ev_path = tmp_path / "run" / "events.json"
    events = json.loads(ev_path.read_text(encoding="utf-8"))
    if events:
        ep["fingerprints"]["frames"]["sim001.sxm"]["z_fwd"]["max"] /= 1.0 + 1e-6
        ep["digest"]["events"] = "0" * 64
        (tmp_path / "run" / "episode.json").write_text(json.dumps(ep), encoding="utf-8")
        assert verify(tmp_path / "run")["ok"]                     # the replay's events equal events.json
        events.pop()
        ev_path.write_text(json.dumps(events), encoding="utf-8")
        assert "events" in verify(tmp_path / "run")["differences"]


def test_fingerprints_tell_rounding_from_a_different_measurement():
    from stmbench.instrument.core import RTOL, array_fingerprint, compare_fingerprints, events_differ

    rng = np.random.default_rng(0)
    a = rng.normal(size=(20, 30)) * 1e-10
    a[3, 4] = np.nan
    fa = array_fingerprint(a)
    assert fa["shape"] == [20, 30] and fa["nonfinite"] == 1 and len(fa["blocks"]) == 64
    assert json.loads(json.dumps(fa)) == fa                      # exact round trip through JSON
    assert compare_fingerprints(fa, array_fingerprint(a)) == ("rounding", 0.0)
    kind, rel = compare_fingerprints(fa, array_fingerprint(a * (1 + 1e-13)))
    assert kind == "rounding" and 0 < rel < RTOL
    assert compare_fingerprints(fa, array_fingerprint(a * (1 + 1e-6)))[0] == "beyond"
    b = a.copy()
    b[3, 4] = 0.0
    assert compare_fingerprints(fa, array_fingerprint(b))[0] == "structure"      # a non-finite value moved
    assert compare_fingerprints(fa, array_fingerprint(a[:, :29]))[0] == "structure"
    assert compare_fingerprints(None, fa)[0] == "structure"
    # any shape: a curve, a tiny frame, an array with no finite value, an all-zero one (exact)
    assert len(array_fingerprint(np.arange(5.0))["blocks"]) == 5
    assert array_fingerprint(np.full((2, 2), np.nan))["min"] is None
    z = array_fingerprint(np.zeros(4))
    assert compare_fingerprints(z, array_fingerprint(np.full(4, 1e-300)))[0] == "beyond"
    # events: floats within rounding, everything else exact
    ev = [{"kind": "hop", "t": 12.5, "n": 3, "ok": True, "x": [1.0, 2.0]}]
    assert events_differ(ev, [{"kind": "hop", "t": 12.5 * (1 + 1e-13), "n": 3, "ok": True, "x": [1.0, 2.0]}]) is None
    assert events_differ(ev, [{"kind": "hop", "t": 12.5 * (1 + 1e-6), "n": 3, "ok": True, "x": [1.0, 2.0]}])
    assert events_differ(ev, [{"kind": "hop", "t": 12.5, "n": 4, "ok": True, "x": [1.0, 2.0]}])
    assert events_differ(ev, [{"kind": "hop", "t": 12.5, "n": 3, "ok": False, "x": [1.0, 2.0]}])
    assert events_differ(ev, ev + ev)


def test_force_spectroscopy_through_the_instrument_reproduces_p5():
    """A privileged reference run: the truth says where the atom is, but every measurement is
    an instrument action, and the report cites the curves those actions saved."""
    inst = Instrument(P5, 0)
    w = inst.world
    reg = w.surface.site.adatoms
    atom = min(reg.atoms.values(), key=lambda a: math.hypot(*reg.site_xy(a.i, a.j, a.sub)))
    inst.act("pll", on=True, amplitude_pm=50.0)

    def go(sx, sy):
        ox, oy, _ = w.drift_offset(w.clock.sim())
        inst.act("move_tip", x_nm=(sx - ox) * 1e9, y_nm=(sy - oy) * 1e9)

    ax, ay = reg.site_xy(atom.i, atom.j, atom.sub)
    go(ax, ay)
    on_atom = inst.act("zspec", sweep_nm=0.5, z_offset_nm=0.3, points=256)["spectrum"]
    go(ax + 6e-9, ay)
    clean = inst.act("zspec", sweep_nm=0.5, z_offset_nm=0.3, points=256)["spectrum"]
    truth = w.truth()
    for c in inst.scenario.claims:
        inst.report(c["id"], float(truth["force"][c["truth"].split(".", 1)[1]]), evidence=[on_atom, clean])
    out = inst.finish()
    det = inst.verdict.details["claims"]
    assert out["success"], {k: (d["value_ok"], d["evidence"]) for k, d in det.items()}


def test_validate_fills_defaults_and_keeps_types():
    p = validate(ACTIONS["scan"], {"center_x_nm": 1, "center_y_nm": 2, "size_nm": 10})
    assert p["pixels"] == 256 and isinstance(p["center_x_nm"], float) and p["direction"] == "down"


def test_z_spectra_carry_the_loops_absolute_z_and_frames_their_start_time():
    """What lines a curve over an atom up with one over bare surface (the loop's Z, as the .dat
    file has it), and what turns a shift between two frames into a drift velocity."""
    inst = Instrument(P5, 0)
    r = inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=5.0, pixels=32, line_time_s=0.05)
    f = inst.frame(r["frame"])
    assert f["t_start_s"] == pytest.approx(inst.status()["instrument_time_s"] - r["duration_s"], abs=0.01)
    inst.act("pll", on=True, amplitude_pm=50.0)
    z0 = inst.status()["z_nm"] * 1e-9
    s = inst.spectrum(inst.act("zspec", sweep_nm=0.4, z_offset_nm=0.3, points=64, backward=False)["spectrum"])
    # the loop's Z at every point: the start offset plus the sweep coordinate, from where it held
    assert np.allclose(s["z_abs"], z0 + 0.3e-9 + s["z_rel"], rtol=0, atol=2e-14)   # status rounds to 1e-5 nm



def test_a_frame_records_the_state_it_was_scanned_in_and_keeps_its_start_when_saved_again():
    """What the judge needs to tell a topograph from a sheet of noise, and a fresh frame from an
    old one saved again (Codex review, 2026-10-14)."""
    inst = Instrument(P4, 0)
    inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=8.0, pixels=32, line_time_s=0.05)
    inst.act("withdraw")
    inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=8.0, pixels=32, line_time_s=0.05)
    w = inst.world
    w.save_frame()                                            # the controller can save a frame again
    saved = [e for e in w.events if e["kind"] == "scan_saved"]
    assert saved[0]["feedback"] is True and saved[0]["withdrawn"] is False
    assert saved[1]["feedback"] is False and saved[1]["withdrawn"] is True
    z = np.asarray(inst.frame(Path(saved[1]["path"]).name)["array"], float)
    assert np.nanstd(z) < 10e-12                              # what such a frame shows
    assert saved[2]["t_start_sim"] == saved[1]["t_start_sim"] < saved[2]["sim_s"]


def test_the_judge_maps_an_atom_through_the_apex_that_imaged_it():
    """An operator moves an atom by what the images show and reports where a frame shows it;
    the tip images (and drags) through its leading apex, wherever that sits. On P4 seed 15
    of the scripted baseline a new apex 2.5 nm off took over from a receded one: the atom was
    on its site and the report 2.5 nm from it through the drift alone."""
    from stmbench.analysis.adatoms import locate
    from stmbench.trackB.claims import to_sample_frame
    from stmsim.physics.tip import Apex

    inst = Instrument(P4, 7)                                   # a good tip
    w = inst.world
    reg = w.surface.site.adatoms
    atom = reg.atoms[reg.target["atom_id"]]
    ax, ay = (v * 1e9 for v in reg.site_xy(atom.i, atom.j, atom.sub))
    w.tip.apexes = [Apex(0.0, 0.0, -0.3e-9, 1.0), Apex(0.0, 2.0e-9, 0.0, 0.8)]   # the second one leads
    ox, oy, _ = w.drift_offset(w.clock.sim())
    sx, sy = ax - ox * 1e9, ay - oy * 1e9 - 2.0                # where it shows: drift and apex
    r = inst.act("scan", center_x_nm=sx, center_y_nm=sy, size_nm=8.0, pixels=80, line_time_s=0.2)
    hit = min(locate(inst.frame(r["frame"])), key=lambda a: math.hypot(a[0] - sx, a[1] - sy))
    ev = [e for e in w.events if e["kind"] == "scan_saved"][-1]
    assert ev["apex_runs_nm"] == [[0, 0.0, 2.0]]
    px, py = to_sample_frame(ev, hit[0], hit[1])
    assert math.hypot(px - ax, py - ay) < 0.15
    bare = {k: v for k, v in ev.items() if k != "apex_runs_nm"}
    qx, qy = to_sample_frame(bare, hit[0], hit[1])
    assert abs(math.hypot(qx - ax, qy - ay) - 2.0) < 0.2


def test_status_reports_the_programmed_drift_compensation_and_reading_it_changes_nothing():
    """The drift compensation an operator programmed is a reading like the lock-in's state: on or off, and
    the velocities it was given, in the units the action takes (what was set, never the drift itself)."""
    inst = Instrument(P4, 0)
    assert inst.status()["drift_comp"] == {"on": False, "v_nm_per_min": [0.0, 0.0], "vz_pm_per_min": 0.0}
    inst.act("drift_comp", on=True, vx_nm_per_min=1.5, vy_nm_per_min=-2.25, vz_pm_per_min=30.0)
    t = inst.used_s
    st = inst.status()
    assert st["drift_comp"] == {"on": True, "v_nm_per_min": [1.5, -2.25], "vz_pm_per_min": 30.0}
    assert inst.status() == st and inst.used_s == t                    # reading takes no time and changes no reading
    inst.act("drift_comp", on=False, vx_nm_per_min=0.5)
    assert inst.status()["drift_comp"] == {"on": False, "v_nm_per_min": [0.5, 0.0], "vz_pm_per_min": 0.0}
    # the reading is the programmed value in the units the action took, and reads no random number: an
    # instrument that was looked at after every action measures what one nobody looked at does
    quiet, curious = Instrument(P4, 3), Instrument(P4, 3)
    for inst_, look in ((quiet, False), (curious, True)):
        inst_.act("drift_comp", on=True, vx_nm_per_min=-0.75, vy_nm_per_min=0.5)
        if look:
            assert inst_.status()["drift_comp"]["v_nm_per_min"] == [-0.75, 0.5]
        inst_.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=12.0, pixels=32)
        if look:
            inst_.status()
    assert quiet.digest() == curious.digest()


# ── the trace of a move ─────────────────────────────────────────────────────
def _session(inst: Instrument) -> str:
    """A tip onto an atom, a move at the imaging setpoint and a drag below the pull threshold."""
    w = inst.world
    reg = w.surface.site.adatoms
    atom = reg.atoms[reg.target["atom_id"]]
    ax, ay = reg.site_xy(atom.i, atom.j)
    ox, oy, _ = w.drift_offset(w.clock.sim())
    x, y = (ax - ox) * 1e9, (ay - oy) * 1e9
    inst.act("move_tip", x_nm=x, y_nm=y)
    inst.act("set_current_range", full_scale_na=100.0)
    inst.act("set_bias", bias_v=0.005)
    inst.act("set_setpoint", current_pa=0.005 / 80e3 * 1e12)
    return inst.act("move_tip", x_nm=x + 4.0, y_nm=y, speed_nm_s=0.25)["trace"]


def test_a_move_replies_with_its_trace_and_keeps_it_like_a_spectrum(tmp_path):
    inst = Instrument(P4, 24, tmp_path / "run")
    r = inst.act("move_tip", x_nm=3.0, y_nm=1.0, speed_nm_s=100.0)
    assert set(r) == {"tip_nm", "distance_nm", "trace", "duration_s", "instrument_time_s", "budget_s"}
    assert r["tip_nm"] == [3.0, 1.0] and r["distance_nm"] == pytest.approx(math.hypot(3, 1), abs=1e-3)
    assert r["duration_s"] == pytest.approx(math.hypot(3, 1) / 100.0, abs=1e-3)
    assert inst.used_s == pytest.approx(math.hypot(3, 1) / 100.0 + 0.05, abs=1e-3)       # time spent as before
    listed = inst.list_data()["spectra"]
    assert listed == [{"spectrum": r["trace"], "kind": "move", "tip_nm": [3.0, 1.0],
                       "points": path_points(math.hypot(3, 1) * 1e-9), "t_s": 0.0}]
    s = inst.spectrum(r["trace"])
    assert set(s) == {"name", "kind", "index", "tip_nm", "from_nm", "to_nm", "speed_nm_s", "bias_v", "setpoint_pa",
                      "feedback", "t_s", "t", "dist", "z_abs", "i"}
    assert s["kind"] == "move" and s["index"] == 1 and s["from_nm"] == [0.0, 0.0] and s["to_nm"] == [3.0, 1.0]
    assert s["speed_nm_s"] == 100.0 and s["feedback"] is True and s["setpoint_pa"] == pytest.approx(200.0)
    assert len(s["t"]) == len(s["dist"]) == len(s["z_abs"]) == len(s["i"]) >= 2
    r2 = inst.act("move_tip", x_nm=0.0, y_nm=0.0, speed_nm_s=100.0)
    assert inst.spectrum(r2["trace"])["index"] == 2 and r2["trace"] != r["trace"]
    assert "current_spikes" not in r and "current_spikes" not in json.dumps(inst.log)
    # the same file is in the session directory, with the arrays and their units
    d = read_dat(Path(inst.world.dats_saved[0]))
    assert d["header"]["Experiment"] == "move trace"
    assert list(d["columns"]) == ["Distance (m)", "Time (s)", "Z (m)", "Current (A)"]
    assert d["columns"]["Z (m)"] == pytest.approx(s["z_abs"], rel=1e-5, abs=1e-12)
    ev = [e for e in inst.world.events if e["kind"] == "dat_saved"]
    assert [e["experiment"] for e in ev] == ["move trace", "move trace"] and ev[0]["path"] == inst.world.dats_saved[0]


def test_an_observation_of_a_trace_costs_nothing_and_changes_nothing(tmp_path):
    quiet, curious = Instrument(P4, 24), Instrument(P4, 24)
    for inst, look in ((quiet, False), (curious, True)):
        t = _session(inst)
        if look:
            inst.list_data()
            inst.spectrum(t)
    assert quiet.digest() == curious.digest() and quiet.used_s == curious.used_s
    assert set(quiet.digest()["spectra"]) and all(set(h) == {"t", "dist", "z_abs", "i"} for h in quiet.digest()["spectra"].values())


def test_an_episode_with_moves_replays_to_the_same_traces(tmp_path):
    """``stmbench verify``: the same seed and actions give the same arrays, byte for byte."""
    inst = Instrument(P4, 24, tmp_path / "run")
    _session(inst)
    inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=8.0, pixels=48, line_time_s=0.2)
    inst.act("move_tip", x_nm=-5.0, y_nm=5.0, speed_nm_s=300.0)
    inst.finish()
    assert any(s["kind"] == "move" for s in inst.spectra.values()) and len(inst.spectra) == 3
    res = verify(tmp_path / "run")
    assert res["ok"], res
    again = replay(tmp_path / "run")
    assert again.digest()["spectra"] == inst.digest()["spectra"]


def test_a_trace_cited_as_evidence_is_not_a_frame_and_judging_goes_on(tmp_path):
    """The name of a move's trace in a report's evidence is data the judge does not know as a frame,
    an STS or a Z spectrum: it resolves to nothing, and the claim is judged on the rest."""
    from stmbench.trackB import claims as C

    inst = Instrument(P4, 24, tmp_path / "run")
    t = _session(inst)
    assert C.resolve_citation(inst.world.events, t) is None
    inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=8.0, pixels=48, line_time_s=0.2)
    frame = next(iter(inst.frames))
    inst.report("target_site", 0.0, x_nm=0.0, y_nm=0.0, frame=frame, evidence=[frame, t])
    out = inst.finish()
    claim = inst.verdict.details["claims"]["target_site"]
    assert claim["unresolved_citations"] == [t] and claim["cited"] == [frame, t, frame]
    assert out["claims_total"] == 2 and claim["evidence"]["kind"] == "frame_covers_after"


def test_the_trace_of_an_oscillating_tip_carries_the_setpoint_current():
    """The loop holds the *averaged* current of a qPlus tip at the setpoint, so the current its
    trace reads is the setpoint, not the junction's current at the mean height (a third of it
    at 100 pm of amplitude)."""
    inst = Instrument(P5, 0)
    inst.act("pll", on=True, amplitude_pm=100.0)
    s = inst.spectrum(inst.act("move_tip", x_nm=5.0, y_nm=0.0, speed_nm_s=50.0)["trace"])
    assert inst.world.osc_factor() > 2.0
    assert np.mean(s["i"]) == pytest.approx(s["setpoint_pa"] * 1e-12, rel=0.05)


def test_event_paths_reduce_to_file_names_whatever_host_wrote_them():
    """A ledger written on Windows is verified on Linux: its events carry ``D:\\run\\x.sxm`` there."""
    from stmbench.instrument.core import _portable_events

    got = _portable_events([{"path": "D:\\run\\private\\session\\sim001.sxm"}, {"path": "/a/b/Spec1.dat"},
                            {"path": "plain.dat"}, {"kind": "x"}])
    assert [e.get("path") for e in got] == ["sim001.sxm", "Spec1.dat", "plain.dat", None]
