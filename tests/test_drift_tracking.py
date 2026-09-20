"""Thermal drift, and the instrument's own way of dealing with it.

A real STM drifts: the lab rig walks about a nanometre a minute at 4 K, and any measurement
that takes an hour — a spectroscopy line, a corral repair, a force curve on a chosen atom —
has the sample slide out from under the tip while it runs. Turning that down in the scenarios
would delete the capability worth testing; what the benchmark asks instead is that the agent
notice it and track it.

The instrument offers the means (``Piezo.DriftCompSet``) and MAST drives it
(``SetDriftCompensation`` after ``MeasureFrameDrift``). These tests pin that the loop closes:
compensation programmed from a measurement stops the motion, the wrong sign makes it worse
rather than doing nothing, and the ramp runs out of piezo eventually.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stmsim.modules import build_dispatcher
from stmsim.physics.rig import RigProfile
from stmsim.physics.world import World


def _world(tmp_path: Path, seed: int = 0) -> World:
    w = World(rig=RigProfile.load("reference-stm"), seed=seed,
              session_dir=tmp_path / "session", time_scale=1.0)
    w._creep = []                       # creep is a separate effect; these tests are about drift
    return w


def _offset(w: World, t: float) -> np.ndarray:
    return np.array(w.drift_offset(t))


def _apparent_velocity(w: World, t0: float, span: float = 600.0) -> np.ndarray:
    """What a two-frame drift measurement reports: how fast features move in the scan frame.

    A feature fixed on the sample sits at scan coordinates ``sample − offset``, so its apparent
    velocity is the negative of the offset's."""
    return -(_offset(w, t0 + span) - _offset(w, t0)) / span


def test_the_rig_drifts_about_a_nanometre_a_minute(tmp_path):
    """The number the scenarios are asked to cope with, from the profile's own calibration."""
    speeds = []
    for seed in range(8):
        v = _world(tmp_path / f"s{seed}", seed).drift_v_m_per_s
        speeds.append(float(np.hypot(v[0], v[1])) * 60 * 1e9)
    assert 0.3 < float(np.median(speeds)) < 3.0, speeds


def test_compensation_programmed_from_a_measurement_stops_the_motion(tmp_path):
    """Type in what was measured and the sample stops moving. This is the whole contract:
    if it did not hold, an agent that did everything right would gain nothing."""
    w = _world(tmp_path)
    t0 = w.clock.sim()
    v = _apparent_velocity(w, t0)
    assert np.hypot(v[0], v[1]) > 0

    build_dispatcher(w).handler_for("Piezo_DriftCompSet")(1, float(v[0]), float(v[1]),
                                                          float(v[2]), 0.9)
    frozen = _offset(w, w.clock.sim())
    for later in (600.0, 3600.0):
        moved = np.abs(_offset(w, w.clock.sim() + later) - frozen)
        assert moved.max() < 1e-12, (later, moved)


def test_compensation_freezes_the_drift_it_does_not_undo_it(tmp_path):
    """An hour of drift that already happened stays happened. Anything else would let an agent
    recover a measurement it took while the sample was walking."""
    w = _world(tmp_path)
    t0 = w.clock.sim()
    w.clock.advance_sim(1800.0)
    before = _offset(w, w.clock.sim())
    assert np.hypot(before[0], before[1]) > 1e-9      # it really did move

    v = _apparent_velocity(w, t0)
    w.set_drift_comp(True, float(v[0]), float(v[1]), float(v[2]))
    after = _offset(w, w.clock.sim() + 1800.0)
    assert np.allclose(after, before, atol=1e-12)


def test_the_wrong_sign_doubles_the_drift_instead_of_doing_nothing(tmp_path):
    """A sign error has to be visible. If it merely did nothing, "I set the compensation" and
    "the compensation works" would look the same, and the verification step would be pointless."""
    w = _world(tmp_path)
    t0 = w.clock.sim()
    v = _apparent_velocity(w, t0)
    plain = np.hypot(*(_offset(w, t0 + 600.0) - _offset(w, t0))[:2])

    w.set_drift_comp(True, -float(v[0]), -float(v[1]), -float(v[2]))
    t1 = w.clock.sim()
    wrong = np.hypot(*(_offset(w, t1 + 600.0) - _offset(w, t1))[:2])
    assert wrong == pytest.approx(2 * plain, rel=0.02), (wrong, plain)


def test_switching_it_off_leaves_the_piezo_where_the_ramp_took_it(tmp_path):
    w = _world(tmp_path)
    t0 = w.clock.sim()
    v = _apparent_velocity(w, t0)
    w.set_drift_comp(True, float(v[0]), float(v[1]), float(v[2]))
    w.clock.advance_sim(1200.0)
    held = _offset(w, w.clock.sim())

    w.set_drift_comp(False, 0.0, 0.0, 0.0)
    assert np.allclose(_offset(w, w.clock.sim()), held, atol=1e-12)
    # and the drift resumes from there, rather than jumping back
    resumed = _offset(w, w.clock.sim() + 600.0) - held
    assert np.hypot(resumed[0], resumed[1]) == pytest.approx(
        np.hypot(*w.drift_v_m_per_s[:2]) * 600.0, rel=0.02)


def test_the_ramp_runs_out_of_piezo(tmp_path):
    """The compensation is a piezo offset, and the scanner is finite. Past the saturation limit
    the sample walks away again — which is why a long campaign still needs re-registration."""
    w = _world(tmp_path)
    rx, ry = w.rig.xy_range_m
    w.set_drift_comp(True, 1e-6, 0.0, 0.0, sat=0.5)        # 1 um/s: saturates almost at once
    accum = w._comp_accum(w.clock.slow() + 100.0)
    assert np.hypot(accum[0], accum[1]) == pytest.approx(0.5 * min(rx, ry) / 2, rel=1e-6)


def test_the_panel_reads_back_what_was_set(tmp_path):
    w = _world(tmp_path)
    d = build_dispatcher(w)
    d.handler_for("Piezo_DriftCompSet")(1, 1.5e-11, -2.5e-11, 0.0, 0.8)
    on, vx, vy, vz, *_rest = d.handler_for("Piezo_DriftCompGet")()
    assert on == 1
    assert (vx, vy, vz) == pytest.approx((1.5e-11, -2.5e-11, 0.0))
    assert w.drift_comp_on and w.drift_comp_v[0] == pytest.approx(1.5e-11)


def test_no_scenario_turns_the_drift_down_to_make_itself_passable(tmp_path):
    """``drift_scale`` exists for scenarios that model a different machine, not as a way to
    make a hard measurement easy. A paper that needs an hour of stability has to earn it with
    drift compensation, because that is the skill being tested."""
    import yaml

    scen = Path(__file__).resolve().parent.parent / "stmbench" / "trackB" / "scenarios"
    offenders = []
    for f in sorted(scen.glob("P*.yaml")):
        ini = (yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get("initial") or {}
        scale = ini.get("drift_scale")
        if scale is not None and float(scale) < 1.0:
            offenders.append(f"{f.name}: drift_scale={scale}")
    assert not offenders, offenders


# ── the baselines' side: a reading has to be credible before it is acted on ──────
class _FakeBaseline:
    """Enough of ``papers._runner.Baseline`` to drive ``null_the_drift`` without a microscope."""

    def __init__(self, replies: dict, frames: int = 9):
        self.replies = replies
        self.calls: list[tuple[str, dict]] = []
        self._frames = frames

    def run(self, skill, params=None, *, optional=False):
        self.calls.append((skill, dict(params or {})))
        got = self.replies.get(skill)
        if isinstance(got, list):                 # a sequence: one reply per call, last repeats
            n = sum(1 for c in self.calls if c[0] == skill) - 1
            return got[min(n, len(got) - 1)]
        return got

    @staticmethod
    def data(res, key, default=None):
        if res is None:
            return default
        return res.get(key, default)

    def latest_frame(self):
        self._frames -= 1
        return Path(f"frame_{self._frames}.sxm")


def _reply(**kw) -> dict:
    return kw


def _null(replies, **kw):
    from stmbench.papers._drift import null_the_drift

    b = _FakeBaseline(replies)
    out = null_the_drift(b, (0.0, 0.0), frame_nm=40.0, pixels=128, **kw)
    return b, out


def test_a_measurement_whose_pairs_disagree_is_not_acted_on(monkeypatch):
    """``MeasureFrameDrift`` says so itself when the pairs contradict each other. Programming a
    compensation from that median points the piezo somewhere nobody measured."""
    import stmbench.papers._drift as drift_mod

    monkeypatch.setattr(drift_mod, "acq_seconds", lambda p: 80.0)
    b, out = _null({"MeasureFrameDrift": _reply(
        verdict="measured", frame_seconds=80.0, dx_median_nm=1.0, dy_median_nm=-3.0,
        n_measured=2, consistency_warning="各对帧给出的 Δy 彼此对不上")})
    assert out["applied"] is False and out["trusted"] is False
    assert not any(c[0] == "SetDriftCompensation" for c in b.calls)
    assert "disagree" in out["note"]


def test_a_window_with_nothing_in_it_reports_why_rather_than_zero(monkeypatch):
    """Two frames of a bare patch correlate to about nothing, which reads exactly like a stable
    machine. The difference has to survive into the ledger."""
    import stmbench.papers._drift as drift_mod

    monkeypatch.setattr(drift_mod, "acq_seconds", lambda p: 80.0)
    b, out = _null({"MeasureFrameDrift": _reply(verdict="undetermined")})
    assert out["applied"] is False
    assert "no feature to track" in out["note"]
    assert not any(c[0] == "SetDriftCompensation" for c in b.calls)


def test_a_good_measurement_is_set_and_then_checked(monkeypatch):
    """What a working compensation looks like: the second reading is a fraction of the first."""
    import stmbench.papers._drift as drift_mod

    monkeypatch.setattr(drift_mod, "acq_seconds", lambda p: 80.0)
    b, out = _null({"MeasureFrameDrift": [
        _reply(verdict="measured", frame_seconds=80.0, dx_median_nm=-1.3, dy_median_nm=0.2,
               feature_dx_median_nm=1.3, feature_dy_median_nm=0.2, n_measured=2),
        _reply(verdict="measured", frame_seconds=80.0, dx_median_nm=-0.08, dy_median_nm=0.01,
               feature_dx_median_nm=0.08, feature_dy_median_nm=0.01, n_measured=2)]})
    sets = [c for c in b.calls if c[0] == "SetDriftCompensation"]
    assert out["applied"] is True and len(sets) == 1
    assert sets[0][1]["enable"] is True
    # what was measured is what was typed in: the features moved 1.3 nm (+x) over 80 s, and
    # Piezo.DriftCompSet takes the velocity at which features are seen to move
    assert sets[0][1]["vx"] == pytest.approx(1.3e-9 / 80.0, rel=1e-9)
    assert sets[0][1]["vy"] == pytest.approx(0.2e-9 / 80.0, rel=1e-9)
    # and it measured again afterwards rather than assuming
    assert sum(1 for c in b.calls if c[0] == "MeasureFrameDrift") >= 2


def test_an_older_skill_without_feature_fields_gets_its_x_sign_turned_round(monkeypatch):
    """``dx_median_nm`` is the register-shift in array axes (x reversed); without the
    feature_* fields the baseline turns x round itself rather than typing the wrong sign in."""
    import stmbench.papers._drift as drift_mod

    monkeypatch.setattr(drift_mod, "acq_seconds", lambda p: 80.0)
    b, out = _null({"MeasureFrameDrift": [
        _reply(verdict="measured", frame_seconds=80.0, dx_median_nm=-1.3, dy_median_nm=0.2,
               n_measured=2),
        _reply(verdict="measured", frame_seconds=80.0, dx_median_nm=-0.08, dy_median_nm=0.01,
               n_measured=2)]})
    sets = [c for c in b.calls if c[0] == "SetDriftCompensation"]
    assert out["applied"] is True and sets[0][1]["vx"] == pytest.approx(1.3e-9 / 80.0, rel=1e-9)


def test_the_registration_frames_are_the_cheap_ones(monkeypatch):
    """A drift measurement that spends measurement-grade frames eats the scenario's budget —
    which is how four P1 seeds verified every claim and still failed, over budget."""
    import stmbench.papers._drift as drift_mod

    monkeypatch.setattr(drift_mod, "acq_seconds", lambda p: 80.0)
    b, _ = _null({"MeasureFrameDrift": _reply(
        verdict="measured", frame_seconds=80.0, dx_median_nm=1.3, dy_median_nm=0.2,
        n_measured=2)})
    scans = [c[1] for c in b.calls if c[0] == "ScanAt"]
    assert scans and all(s["size_m"] == pytest.approx(40e-9) and s["pixels"] == 128
                         for s in scans)


def test_a_compensation_that_did_not_help_is_switched_back_off(monkeypatch):
    """The residual staying where it was means the reading was not something the window could
    support — a herringbone grating pins only the across-stripe component, a straight edge only
    its normal. Leaving that compensation on points the piezo somewhere nobody measured."""
    import stmbench.papers._drift as drift_mod

    monkeypatch.setattr(drift_mod, "acq_seconds", lambda p: 80.0)
    steady = _reply(verdict="measured", frame_seconds=80.0, dx_median_nm=1.3,
                    dy_median_nm=0.2, n_measured=2)
    b, out = _null({"MeasureFrameDrift": steady})
    assert out["applied"] is False
    assert "switched off" in out["note"]
    last = [c for c in b.calls if c[0] == "SetDriftCompensation"][-1]
    assert last[1]["enable"] is False
