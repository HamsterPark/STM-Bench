"""The MAST-free benchmark instrument: primitive actions on a virtual clock, read-only
observations, the version-1 result channel, a ledger, and re-checking by replay."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from stmbench.instrument import ACTIONS, ActionError, Instrument, replay, verify
from stmbench.instrument.core import Instrument as _I, validate
from stmbench.trackB.rules import RULES_VERSION

P4 = "P4_atom_positioning_cu111"
P5 = "P5_sader_jarvis_qplus"


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
    assert ep["rules_version"] == RULES_VERSION and ep["api_version"] == 1
    assert (tmp_path / "run" / "actions.jsonl").exists() and (tmp_path / "run" / "events.json").exists()
    assert verify(tmp_path / "run")["ok"]
    # a ledger whose measurements were edited no longer re-checks
    ep["digest"]["frames"]["sim001.sxm"]["z_fwd"] = "0" * 64
    (tmp_path / "run" / "episode.json").write_text(json.dumps(ep), encoding="utf-8")
    bad = verify(tmp_path / "run")
    assert not bad["ok"] and "frames" in bad["differences"]
    assert replay(tmp_path / "run").digest()["frames"]["sim001.sxm"]["z_fwd"] != "0" * 64


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

