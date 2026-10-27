"""The trace of a move: Z and the current the controller records while the tip crosses its path.

``move_tip`` used to count the atom hops for the agent (``current_spikes``). The instrument now
records what a controller records, drawn from its own state (``stmsim.physics.trace``), and the hops
are read from it: a sawtooth in Z for an atom that follows the tip, a fall for one that is left
behind. These tests pin what the trace is made of; ``test_analysis_trace`` pins how it is read."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from stmbench.instrument.core import _json_default
from stmsim.physics import trace as movetrace
from stmsim.physics.adatoms import AdatomParams, AdatomRegistry, path_points
from stmsim.physics.rig import RigProfile
from stmsim.physics.scanner import SIG_Z, ScanSettings
from stmsim.physics.switches import PhysicsSwitches
from stmsim.physics.tip import Apex, Tip
from stmsim.physics.world import World
from stmsim.scenario import Scenario

P4 = "P4_atom_positioning_cu111"
PARAMS = AdatomParams(r_threshold_ohm=200e3, r_pick_ohm=30e3, v_max_m_s=1e-9)


def _world(tmp_path, *, seed: int = 4, bias: float = 0.01, setpoint: float = 200e-9, quiet: bool = False,
           tip: str = "good") -> World:
    """One Fe atom on Cu(111), 4 nm from its target, the tip in tunnelling (on the 1 uA range, so
    that a manipulation current does not saturate) and no drift; ``quiet`` switches the Z noise off
    and gives the apex no lattice transfer, so that the trace is the atom's bump alone."""
    w = World(rig=RigProfile.load("reference-stm"), seed=seed, material="Cu(111)",
              session_dir=tmp_path / "s", adsorbate_density_per_um2=0.0)
    w.surface.configure_adatoms(PARAMS, {"layout": "single", "target_dx_nm": 4.0, "n_bystanders": 0})
    w.tip.radius_m = 1.0e-9
    w.tip.apex_radius_ref_m = 1.0e-9
    w.preamp.full_scale_a = 1.0e-6
    if tip != "good":
        w.tip.apply_condition(tip, 0.5, 0.0)
    w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
    w.withdrawn = False
    w.zctrl_set(True)
    w.drift_v_m_per_s = np.zeros(3)
    w.bias_v = bias
    w.zctrl.setpoint_a = setpoint
    w.transients.clear()
    if quiet:
        w.noise.z_floor_m = 0.0
        w.tip.apex_sigma_m = 0.3e-9
    return w


def _atom(w):
    reg = w.surface.site.adatoms
    return reg, [a for a in reg.atoms.values() if a.role == "target"][0]


def _drag(w, *, speed: float = 0.25e-9, dx: float = 4e-9, trace: bool = True):
    reg, atom = _atom(w)
    ax, ay = reg.site_xy(atom.i, atom.j)
    w.move_xy(ax, ay)
    events = w.move_xy_path(ax + dx, ay, speed_m_s=speed, trace=trace)
    return events, (ax, ay)


def _bump(w, params=PARAMS):
    """The atom's apparent height and width after the tip's blur, as the renderer filters it."""
    s = w.tip.sigma_eff_m(w.phi_junction())
    s2 = params.sigma_m ** 2
    return params.height_m * s2 / (s2 + s * s), math.sqrt(s2 + s * s)


# ── what the trace is ───────────────────────────────────────────────────────
def test_a_good_tip_dragging_an_atom_writes_a_sawtooth_whose_steps_are_the_hops(tmp_path):
    """The height under the tip is the atom's bump where the atom is at each sample: it falls as
    the tip moves on and jumps when the atom hops after it. With the noise off, the trace's steps
    are the bump's, hop for hop, to within what the lattice and the loop add (a picometre)."""
    w = _world(tmp_path, quiet=True)
    events, (ax, ay) = _drag(w)
    tr = w.last_move_trace
    hops = [e for e in tr["events"] if e["kind"] == "hop"]
    assert sum(e["n_hops"] for e in events if e["kind"] == "adatom_hop") == len(hops) >= 12
    assert len(tr["t"]) == len(tr["dist"]) == len(tr["z_abs"]) == len(tr["i"]) == path_points(4e-9)
    # the bump of the atom, where the atom is at every sample
    reg, atom = _atom(w)
    amp, width = _bump(w)
    site = {0: reg.site_xy(*hops[0]["site"])}                       # filled below from the hops
    start = (ax, ay)
    pos = np.tile(start, (len(tr["t"]), 1))
    for e in hops:
        pos[e["sample"]:] = reg.site_xy(*e["site"])
    tip_xy = np.column_stack([ax + tr["dist"], np.full(len(tr["dist"]), ay)])
    h = amp * np.exp(-((tip_xy - pos) ** 2).sum(axis=1) / (2 * width ** 2))
    dz = np.diff(tr["z_abs"])
    assert np.max(np.abs(dz - np.diff(h))) < 1.0e-12
    # a tooth: up at the hop (the atom jumps toward the tip, not always straight at it, so not
    # at every hop), down in between; the largest steps up are hops
    at = np.array([e["sample"] - 1 for e in hops])
    rest = np.setdiff1d(np.arange(dz.size), at)
    assert np.mean(dz[rest]) < 0.0 < np.mean(dz[at])
    assert set(np.argsort(dz)[-4:]) <= set(at)
    # the atom was on the surface the whole way, so Z did not fall away like a bump left behind
    assert tr["z_abs"][-1] - tr["z_abs"][0] == pytest.approx(h[-1] - h[0], abs=1.0e-12)


def test_an_atom_that_stays_gives_no_step_up(tmp_path):
    """At an imaging resistance the atom is not moved, and the trace is the bump falling away
    behind the tip: no hop, no step up beyond the noise, and a fall of the bump's height."""
    w = _world(tmp_path, setpoint=1e-9)
    events, _ = _drag(w)
    tr = w.last_move_trace
    assert events == [] and tr["events"] == []
    z = tr["z_abs"] * 1e12
    noise = 2.0 * math.sqrt(2.0)                                   # per-sample noise 2 pm, of a difference
    assert np.diff(z).max() < 4.0 * noise
    amp, _ = _bump(w)
    assert np.mean(z[:4]) - np.mean(z[-12:]) == pytest.approx(amp * 1e12, rel=0.25)


def test_the_trace_follows_every_apex_of_the_tip(tmp_path):
    """A double tip images an atom twice: a second apex that passes over the atom a nanometre
    behind the first shows a second bump in the trace."""
    w = _world(tmp_path, setpoint=1e-9, quiet=True)
    w.tip.apexes = [Apex(0.0, 0.0, 0.0, 1.0), Apex(-1.0e-9, 0.0, -0.02e-9, 0.9)]
    _drag(w)
    z = w.last_move_trace["z_abs"] * 1e12
    d = np.array(w.last_move_trace["dist"]) * 1e9
    first, second = z[d < 0.3].max(), z[(d > 0.7) & (d < 1.3)].max()
    assert second > z[d > 2.5].mean() + 5.0 and second > 0.4 * (first - z[d > 2.5].mean()) + z[d > 2.5].mean()


def test_a_move_over_bare_surface_is_noise_on_a_level(tmp_path):
    w = _world(tmp_path, bias=0.05, setpoint=200e-12)
    w.move_xy(-40e-9, -40e-9)                                      # far from the one atom
    w.move_xy_path(-40e-9, -30e-9, speed_m_s=50e-9, trace=True)
    tr = w.last_move_trace
    z = tr["z_abs"] * 1e12
    assert np.std(np.diff(z)) < 6.0
    assert np.mean(tr["i"]) == pytest.approx(w.zctrl.setpoint_a, rel=0.05)


def test_feedback_off_holds_z_and_the_current_follows_the_atom(tmp_path):
    w = _world(tmp_path, setpoint=1e-9, bias=0.1)
    reg, atom = _atom(w)
    ax, ay = reg.site_xy(atom.i, atom.j)
    w.move_xy(ax - 2e-9, ay)
    w.zctrl_set(False)
    w.move_xy_path(ax + 2e-9, ay, speed_m_s=2e-9, trace=True)
    tr = w.last_move_trace
    assert tr["feedback"] is False
    assert np.std(tr["z_abs"]) < 4e-12                              # held (the noise floor only)
    i = tr["i"]
    amp, _ = _bump(w)
    assert i.max() / np.median(i[:10]) == pytest.approx(math.exp(2 * w.kappa_m() * amp), rel=0.35)


def test_a_withdrawn_tip_records_the_retract_limit_and_no_current(tmp_path):
    w = _world(tmp_path)
    w.withdraw()
    w.move_xy_path(5e-9, 0.0, speed_m_s=50e-9, trace=True)
    tr = w.last_move_trace
    assert tr["withdrawn"] is True
    assert np.all(tr["z_abs"] == w.zctrl.limit_high_m)
    assert np.max(np.abs(tr["i"])) < 5e-12


def test_a_manipulation_current_saturates_a_small_range_and_not_a_wide_one(tmp_path):
    """17 nA is what 33 nA asked for at 5 mV can carry here; the 10 nA range reads its limit all
    along the move, the 100 nA range reads the current."""
    for full_scale, saturated in ((10e-9, True), (100e-9, False)):
        w = _world(tmp_path / str(full_scale), bias=0.005, setpoint=33e-9)
        w.preamp.full_scale_a = full_scale
        w.move_xy_path(3e-9, 0.0, speed_m_s=10e-9, trace=True)
        i = w.last_move_trace["i"]
        assert bool(np.all(i == w.preamp.clamp_a)) is saturated
        if not saturated:
            assert 10e-9 < np.median(i) < 20e-9


def test_a_pickup_takes_the_bump_off_the_surface_at_its_sample(tmp_path):
    w = _world(tmp_path, setpoint=2e-6, bias=0.01, quiet=True)      # 5 kohm: below the pick-up threshold
    reg, atom = _atom(w)
    ax, ay = reg.site_xy(atom.i, atom.j)
    w.move_xy(ax - 1e-9, ay)
    events = w.move_xy_path(ax + 1e-9, ay, speed_m_s=0.5e-9, trace=True)
    assert any(e["kind"] == "adatom_picked" for e in events) and reg.carried is atom
    tr = w.last_move_trace
    (pick,) = [e for e in tr["events"] if e["kind"] == "pickup"]
    z = tr["z_abs"]
    drop = z[pick["sample"] - 1] - z[pick["sample"]]
    assert drop > 0.1 * _bump(w)[0]                               # the bump's height at that offset, gone at once
    assert np.ptp(z[pick["sample"]:]) < 1e-12                       # and nothing under the tip after it


def test_a_tip_change_during_the_move_is_a_step_at_a_time_of_the_traces_own_choosing(tmp_path):
    w = _world(tmp_path, quiet=True)                                # 50 kohm: a drag, which dosed the tip
    w.tip.lambda_per_s = 1e3                                       # certain to change within the move
    state_before = w.rng.bit_generator.state
    _drag(w)
    tr = w.last_move_trace
    k = tr["tip_change_sample"]
    assert k is not None and 1 <= k < len(tr["t"])
    dz = np.diff(tr["z_abs"])
    assert abs(dz[k - 1]) > 15e-12 and abs(dz[k - 1]) == np.max(np.abs(dz))
    assert w.rng.bit_generator.state == state_before


def test_a_long_move_is_sampled_coarser_and_a_zero_move_still_has_two_points(tmp_path):
    w = _world(tmp_path)
    w.move_xy_path(200e-9, 0.0, speed_m_s=300e-9, trace=True)
    tr = w.last_move_trace
    assert len(tr["t"]) == movetrace.MAX_POINTS < path_points(200e-9)
    assert tr["dist"][-1] == pytest.approx(200e-9) and tr["t"][-1] == pytest.approx(200 / 300)
    assert np.all(np.isfinite(tr["z_abs"])) and np.all(np.isfinite(tr["i"]))
    w.move_xy_path(200e-9, 0.0, speed_m_s=300e-9, trace=True)
    z = w.last_move_trace
    assert len(z["t"]) == 2 and np.all(np.isfinite(z["z_abs"]))


def test_the_loop_settles_within_a_slow_sample_as_it_does_in_reality(tmp_path):
    """A target held for 0.2 s is a few thousand loop time constants: the output has reached it,
    and does not ring from sample to sample (the bilinear transform at that step would)."""
    w = _world(tmp_path, quiet=True, bias=0.05, setpoint=200e-12)
    w.noise.floor_a = 0.0
    w.noise.pink_amp_a = w.noise.mains_amp_a = 0.0
    w.noise.spectral_lines = {}
    w.move_xy(-40e-9, -40e-9)                                      # far from the one atom
    w.move_xy_path(-30e-9, -30e-9, speed_m_s=2e-9, trace=True)     # 14 nm of bare surface in 7 s
    z = w.last_move_trace["z_abs"]
    i = w.last_move_trace["i"]
    assert np.max(np.abs(np.diff(z, 2))) < 1e-12                    # a ramp (the sample's tilt), no ringing
    assert np.ptp(i) / np.mean(i) < 0.01


# ── the scenario's physics: what the trace has to follow ────────────────────
def _frame_profile(w, ax, ay, theta, n=120, size=6e-9):
    """The z channel of a slow frame centred on the atom (its tilt plane taken out), along the line
    through the atom at angle ``theta``, at 0.05 nm: what the same atom looks like in an image."""
    from scipy import ndimage

    st = ScanSettings(cx=ax, cy=ay, w=size, h=size, nx=n, ny=n, line_time_fwd_s=40.0, line_time_bwd_s=40.0)
    z = np.asarray(w.renderer.render(st, 0.0).data[SIG_Z][0])
    yy, xx = np.mgrid[0:n, 0:n]
    px = size / n
    far = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2) * px > 2.0e-9
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(far.sum()), xx[far], yy[far]]), z[far], rcond=None)
    z = z - (coef[0] + coef[1] * xx + coef[2] * yy)
    s = np.linspace(-2e-9, 2e-9, 81)
    return ndimage.map_coordinates(z, [(n - 1) / 2 - s * math.sin(theta) / px, (n - 1) / 2 + s * math.cos(theta) / px], order=3)


@pytest.mark.parametrize("axis_ratio,tip_deg,path_deg", [(1.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.5, 0.0, 90.0), (0.3, 40.0, 130.0),
                                                         (0.3, 40.0, 10.0), (0.5, 25.0, 205.0)])
def test_an_atom_looks_the_same_in_a_trace_as_in_a_frame_for_a_round_tip_and_an_elliptical_one(
        tmp_path, axis_ratio, tip_deg, path_deg):
    """The trace blurs an atom as the frame does, the tip's ellipse included (``tip_anisotropy``):
    the trace of a move across an atom is the frame's profile along the same line, to a fraction of
    a picometre in 80, and the width along the long axis is the wider one. The paths run along the
    long axis, across it, obliquely to it (30 deg off; 25 deg off in the other sense) and one of them
    backwards."""
    w = _world(tmp_path, setpoint=1e-9, quiet=True)                  # imaging: nothing is dragged
    w.tip.lambda_per_s = 0.0                                         # the frame lasts hours: no change in it
    w.set_physics(PhysicsSwitches(tip_anisotropy=True))
    w.tip.axis_ratio, w.tip.angle = axis_ratio, math.radians(tip_deg)
    reg, atom = _atom(w)
    ax, ay = reg.site_xy(atom.i, atom.j)
    th = math.radians(path_deg)
    prof = _frame_profile(w, ax, ay, th)
    w.move_xy(ax - 2e-9 * math.cos(th), ay - 2e-9 * math.sin(th))
    w.move_xy_path(ax + 2e-9 * math.cos(th), ay + 2e-9 * math.sin(th), speed_m_s=0.25e-9, trace=True)
    tr = w.last_move_trace
    d = tr["dist"]
    ends = (d < 0.6e-9) | (d > 3.4e-9)
    z = tr["z_abs"] - np.polyval(np.polyfit(d[ends], tr["z_abs"][ends], 1), d)         # the tilt of the sample out
    assert np.sqrt(np.mean((z - prof) ** 2)) < 1.0e-12 and z.max() == pytest.approx(prof.max(), rel=0.03)
    half = lambda y: float(np.sum(y > y.max() / 2) * 0.05)                              # noqa: E731
    if axis_ratio < 1.0 and path_deg in (0.0, 90.0):
        assert half(z) == pytest.approx(0.8 if path_deg == 0.0 else 0.7, abs=0.051)   # long axis along x: wider along x


@pytest.mark.parametrize("axis_ratio,tip_deg,scan_deg,scan_dir", [(0.5, 0.0, 0.0, "down"), (0.5, 0.0, 30.0, "up"),
                                                                  (0.3, 40.0, 75.0, "down"), (0.35, -20.0, 200.0, "up")])
def test_the_trace_blurs_with_the_covariance_of_the_renderer(tmp_path, axis_ratio, tip_deg, scan_deg, scan_dir):
    """Whatever frame an image is taken in, the renderer's blur (px², row by column, in a scan frame
    turned by ``scan_deg``, rows counted either way) is the same ellipse in the sample frame as the
    one a trace blurs an atom with: taken back to the sample frame it is the trace's covariance."""
    w = _world(tmp_path, quiet=True)
    w.set_physics(PhysicsSwitches(tip_anisotropy=True))
    w.tip.axis_ratio, w.tip.angle = axis_ratio, math.radians(tip_deg)
    phi = w.phi_junction()
    st = ScanSettings(w=8e-9, h=6e-9, nx=64, ny=48, angle_deg=scan_deg, scan_dir=scan_dir)   # pixels not square
    cov = w.renderer._anisotropic_blur_cov(st, w.tip.sigma_eff_m(phi))
    px_u, px_v = st.w / st.nx, st.h / st.ny
    sgn = -1.0 if scan_dir == "down" else 1.0
    c_uu = (cov[1, 1] - w.renderer.MIN_SIGMA_PX ** 2) * px_u ** 2          # the line's bandwidth is not the tip's
    c_vv = cov[0, 0] * px_v ** 2
    c_uv = sgn * cov[0, 1] * px_u * px_v
    th = math.radians(scan_deg)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    in_sample_frame = rot @ np.array([[c_uu, c_uv], [c_uv, c_vv]]) @ rot.T
    mine = movetrace.blur_covariance(w.tip, phi, True)
    assert in_sample_frame == pytest.approx(mine, rel=1e-9, abs=1e-30)
    short, long_ = np.linalg.eigvalsh(mine)                                 # an ellipse: its axes are as the ratio says
    assert short / long_ == pytest.approx(axis_ratio ** 2, rel=1e-9) and long_ == pytest.approx(w.tip.sigma_eff_m(phi) ** 2)
    assert movetrace.blur_covariance(w.tip, phi, False) == pytest.approx(np.eye(2) * w.tip.sigma_eff_m(phi) ** 2)


def test_a_tip_pressed_into_the_sample_reads_the_clamp_and_follows_contact(tmp_path, monkeypatch):
    """With ``crash_over_range`` and a setpoint above the range, the loop's target is contact and the
    current reads the clamp (:meth:`World.equilibrium_gap`, :meth:`World._dc_current`); the move
    ploughs, which is a change of the tip at the start. The same move once the range allows the
    setpoint, over the same surface with the same tip, sits one tunnelling gap higher (the crashed
    apex's flicker is left out: two traces draw it differently)."""
    monkeypatch.setattr(Tip, "CRASH_FLICKER_DZ_M", (0.0, 0.0))
    w = _world(tmp_path, bias=0.05, setpoint=200e-12, quiet=True)
    w.set_physics(PhysicsSwitches(crash_over_range=True))
    w.move_xy(-40e-9, -40e-9)                                       # away from the one atom
    w.preamp.full_scale_a = 10e-9
    w.set_setpoint(20e-9)                                           # 20 nA on the 10 nA range
    assert w.over_range()
    n_crash = sum(1 for e in w.events if e["kind"] == "crash")
    w.move_xy_path(-40e-9, 0.0, speed_m_s=100e-9, trace=True)
    pressed = w.last_move_trace
    assert pressed["over_range"] and pressed["tip_change_samples"] == [0]
    assert sum(1 for e in w.events if e["kind"] == "crash") == n_crash + 1            # and the plough is a crash
    assert np.all(pressed["i"] >= 0.999 * w.preamp.clamp_a) and np.mean(pressed["i"]) == pytest.approx(w.preamp.clamp_a, rel=1e-3)
    w.preamp.full_scale_a = 1e-6                                    # the range reaches the setpoint again: the loop pulls back
    w.transients.clear()
    w.move_xy(-40e-9, -40e-9)
    w.move_xy_path(-40e-9, 0.0, speed_m_s=100e-9, trace=True)
    free = w.last_move_trace
    assert not free["over_range"] and np.mean(free["i"]) == pytest.approx(20e-9, rel=0.15)   # the loop lags the pit's flank
    assert np.median(free["z_abs"] - pressed["z_abs"]) == pytest.approx(w.equilibrium_gap(), abs=10e-12)


def _long_move(w, *, length_nm=20.0, speed_nm_s=5.0):
    w.move_xy(-40e-9, -40e-9)
    t0 = w.clock.sim()
    n0 = len(w.tip.events)
    w.move_xy_path(-40e-9 + length_nm * 1e-9, -40e-9, speed_m_s=speed_nm_s * 1e-9, trace=True)
    return w.last_move_trace, t0, length_nm / speed_nm_s, w.tip.events[n0:]


def test_a_change_of_the_tip_during_a_navigation_move_is_a_step_where_the_tips_log_puts_it(tmp_path):
    """``tip_changes_outside_scans``: the interval a move tunnels through draws the tip's changes at
    times of their own, and the trace has each as the step it is, at the sample of its time."""
    w = _world(tmp_path, bias=0.05, setpoint=200e-12, quiet=True)
    w.set_physics(PhysicsSwitches(tip_changes_outside_scans=True))
    w.tip.lambda_per_s = 0.5                                        # a change every couple of seconds
    tr, t0, dur, events = _long_move(w)
    changes = [e for e in events if e.kind == "spontaneous_change"]
    assert len(changes) >= 2
    want = [int(math.ceil((e.sim_s - t0) / dur * (len(tr["t"]) - 1))) for e in changes]
    assert tr["tip_change_samples"] == want and all(0 < k < len(tr["t"]) for k in want)
    dz = np.diff(tr["z_abs"])
    for k, e in zip(want, changes):
        assert abs(dz[k - 1]) == pytest.approx(abs(e.detail["dz_pm"]) * 1e-12, rel=0.25, abs=3e-12)
    assert all(abs(dz[k - 1]) > 15e-12 for k in want)


def test_a_recovery_during_a_navigation_move_is_a_step_at_its_time_too(tmp_path, monkeypatch):
    """``tip_recovery``: a double tip that loses its second apex in the middle of a move shows it
    where the tip's log says, and the heights after it are the single apex's."""
    monkeypatch.setitem(Tip.RECOVERY_TAU_S, "apex_lost", 4.0)
    w = _world(tmp_path, bias=0.05, setpoint=200e-12, quiet=True, tip="double")
    w.set_physics(PhysicsSwitches(tip_recovery=True))
    w.tip.lambda_per_s = 0.0
    tr, t0, dur, events = _long_move(w)
    lost = [e for e in events if e.kind == "recovery_apex_lost"]
    assert len(lost) == 1 and w.tip.n_apex == 1
    k = int(math.ceil((lost[0].sim_s - t0) / dur * (len(tr["t"]) - 1)))
    assert tr["tip_change_samples"] == [k] and 0 < k < len(tr["t"])
    assert abs(np.diff(tr["z_abs"])[k - 1]) > 5e-12                  # the ghost apex's share of the height goes at once


def test_the_change_a_drag_stamps_at_its_start_is_placed_by_the_traces_own_stream(tmp_path):
    """A drag's change is logged at the move's start (``Tip.maybe_change``, as before): where in the
    move it happened is drawn from the trace's stream, the same for the same world."""
    samples = []
    for k in range(2):
        w = _world(tmp_path / str(k), setpoint=200e-9, quiet=True)  # a drag
        w.tip.lambda_per_s = 1e3
        _drag(w)
        samples.append(w.last_move_trace["tip_change_samples"])
    assert samples[0] == samples[1] and len(samples[0]) == 1 and 0 < samples[0][0] < path_points(4e-9)


# ── what the trace may not touch ────────────────────────────────────────────
def _states(w) -> dict:
    """The state of every random stream the world holds, by name."""
    reg = w.surface.site.adatoms
    gens = {"rng": w.rng, "noise": w.noise.rng, "df": w.df_rng, "tip": w.tip.rng, "layout": reg.rng_layout,
            "dyn": reg.rng_dyn, "traits": reg.rng_traits, "outscan": w.rng_outscan, "recovery": w.rng_recovery,
            "aniso": w.rng_aniso}
    return {k: json.dumps(g.bit_generator.state, sort_keys=True, default=str) for k, g in gens.items()}


@pytest.mark.parametrize("seed", [3, 24, 28])
def test_a_trace_changes_no_stream_and_nothing_the_move_did(tmp_path, seed):
    """Same world, same moves, with and without the trace: the same events, the same atoms, the
    same tip, every stream of the world in the same state. Only the trace's own stream moves."""
    sc = Scenario.load(Path("stmbench/trackB/scenarios") / f"{P4}.yaml")
    worlds = []
    for k, trace in enumerate((False, True)):
        w = sc.build_world(seed, tmp_path / f"s{k}", virtual_clock=True)
        reg = w.surface.site.adatoms
        atom = reg.atoms[reg.target["atom_id"]]
        ax, ay = reg.site_xy(atom.i, atom.j)
        w.preamp.full_scale_a = 100e-9
        w.set_bias(0.005)
        for r_k in (400.0, 200.0, 100.0, 50.0):
            w.set_setpoint(0.005 / (r_k * 1e3))
            w.clock.advance(1.0)
            w.move_xy(ax, ay)
            w.move_xy_path(ax + 4e-9, ay, speed_m_s=0.25e-9, trace=trace)
            w.clock.advance(20.0)
        w.move_xy_path(ax - 60e-9, ay - 60e-9, speed_m_s=300e-9, trace=trace)
        worlds.append(w)
    plain, traced = worlds
    assert plain.last_move_trace is None and traced.last_move_trace is not None
    assert _states(plain) == _states(traced)
    strip = lambda ev: json.dumps(ev, default=_json_default, sort_keys=True)          # noqa: E731
    assert strip(plain.events) == strip(traced.events)
    assert strip(plain.tip.snapshot()) == strip(traced.tip.snapshot())
    ra, rb = plain.surface.site.adatoms, traced.surface.site.adatoms
    assert [(a.id, a.i, a.j, a.status, a.history) for a in ra.atoms.values()] == \
           [(a.id, a.i, a.j, a.status, a.history) for a in rb.atoms.values()]
    assert (plain.tip_x, plain.tip_y, plain.zctrl.z_n) == (traced.tip_x, traced.tip_y, traced.zctrl.z_n)
    assert plain.move_rng.bit_generator.state != traced.move_rng.bit_generator.state


@pytest.mark.parametrize("seed", [0, 3, 5, 15, 24])                    # blunt, blunt, double, unstable, good
def test_recording_a_drag_changes_neither_its_outcome_nor_its_draws(seed):
    """``AdatomRegistry.drag(log=...)`` only writes down what happened."""
    from stmsim.physics.adatoms import grip_of
    from stmsim.physics.surface import Surface

    sc = Scenario.load(Path("stmbench/trackB/scenarios") / f"{P4}.yaml")
    drawn = sc.draw_hidden(seed)
    out = []
    for log in (None, []):
        reg = AdatomRegistry(Surface("Cu(111)", seed=seed, adsorbate_density_per_um2=0.0).site, PARAMS)
        reg.site.adatoms = reg
        a = reg.add(0, 0)
        reg.add(8, -3, role="bystander")
        tip = Tip(rng=np.random.default_rng(seed + 1))
        tip.radius_m = 1e-9
        tip.apply_condition(drawn["tip"]["condition"], drawn["tip"]["condition_scale"], drawn["tip"]["condition_angle_deg"])
        x0, y0 = reg.site_xy(a.i, a.j)
        events = []
        for r in (150e3, 90e3, 40e3, 20e3):
            events += reg.drag((x0, y0), (x0 + 4e-9, y0), r_ohm=r, v_m_s=0.25e-9, sim_s=1.0, grip=grip_of(tip, 1.05e10), log=log)
        out.append((events, [(q.id, q.i, q.j, q.status) for q in reg.atoms.values()],
                    reg.rng_dyn.bit_generator.state, reg.carried is not None))
    assert out[0] == out[1]
