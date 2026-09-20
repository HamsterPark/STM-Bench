"""qPlus: the PLL module, the Δf(z) sweep, and the ``.dat`` an analysis skill will read."""
from __future__ import annotations

import math

import numpy as np
import pytest

from stmsim.calibrate import iz_templates as izt
from stmsim.modules import build_dispatcher
from stmsim.modules.signals import SIGNAL_NAMES
from stmsim.physics.forces import ForceParams
from stmsim.physics.rig import RigProfile
from stmsim.physics.world import World
from stmsim.scenario import Scenario
from stmsim.wire.errors import ModuleNotRunning

FORCE = ForceParams.from_yaml({"D_e_mev": 150.0, "a_per_nm": 14.0, "z_e_nm": 0.32,
                               "hamaker_zj": 250.0, "z0_nm": 0.30, "bg_frac": 0.0})


def _qplus_world(tmp_path, *, with_atom=True, seed=5):
    from stmsim.physics.adatoms import AdatomParams

    w = World(rig=RigProfile.load("reference-stm-qplus"), seed=seed, material="Cu(111)",
              session_dir=tmp_path / "s", adsorbate_density_per_um2=0.0)
    if with_atom:
        w.surface.configure_adatoms(AdatomParams(), {"layout": "single", "n_bystanders": 0})
    w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
    w.withdrawn = False
    w.zctrl_set(True)
    w.transients.clear()                 # start settled, not mid-landing (Scenario does this too)
    w.achievable_z_tip()
    w.drift_v_m_per_s = np.zeros(3)
    w.bias_v = 0.02
    w.force = FORCE
    return w


def _over_atom(w):
    reg = w.surface.site.adatoms
    atom = [a for a in reg.atoms.values() if a.role == "target"][0]
    ax, ay = reg.site_xy(atom.i, atom.j)
    ox, oy, _ = w.drift_offset(w.clock.sim())
    w.move_xy(ax - ox, ay - oy)
    w.achievable_z_tip()
    return ax, ay


# ── profile ──────────────────────────────────────────────────────────────────
def test_qplus_profile_inherits_the_stm_rig():
    base = RigProfile.load("reference-stm")
    qp = RigProfile.load("reference-stm-qplus")
    assert qp.name == "reference-stm-qplus"
    assert qp.get("drift") == base.get("drift")          # every calibrated number stays shared
    assert qp.preamp_full_scale_a == base.preamp_full_scale_a
    assert qp.z_range_m == base.z_range_m
    assert "PLL" in qp.modules_loaded and "PLL" not in base.modules_loaded
    # the analysis extensions stay unlicensed, as on the real rack
    for m in ("PLLSignalAnlzr", "PLLZoomFFT", "PLLPhasSwp", "OCSync"):
        assert m in qp.modules_not_loaded


def test_pll_is_needmodule_on_the_stm_rig(tmp_path):
    w = World(rig=RigProfile.load("reference-stm"), seed=1, session_dir=tmp_path / "s")
    d = build_dispatcher(w)
    assert w.pll is None
    with pytest.raises(ModuleNotRunning):
        d.handler_for("PLL.CenterFreqGet")
    q = _qplus_world(tmp_path / "q")
    dq = build_dispatcher(q)
    assert dq.handler_for("PLL.CenterFreqGet") is not None
    with pytest.raises(ModuleNotRunning):
        dq.handler_for("PLLSignalAnlzr.Open")


# ── the PLL panel ────────────────────────────────────────────────────────────
def test_amplitude_rings_up_and_the_excitation_follows(tmp_path):
    w = _qplus_world(tmp_path)
    assert w.amplitude_now() == pytest.approx(w.pll.undriven_floor_m, abs=5e-12)
    assert w.pll.excitation_v == 0.0
    w.pll.set_output(True, w.clock.wall())
    assert w.pll.excitation_v > 0                     # MAST reads this to decide "drive is on"
    assert w.pll.amplitude_now(w.clock.wall()) < 1e-13         # not ready yet
    ready = w.pll.amplitude_now(w.clock.wall() + 5 * w.pll.tau_s)
    assert ready == pytest.approx(w.pll.amp_setpoint_m, rel=0.02)


def test_frequency_shift_is_the_force_gradient(tmp_path):
    w = _qplus_world(tmp_path)
    assert abs(w.df_now()) < 0.5                      # drive off: just noise
    w.pll.set_output(True, w.clock.wall() - 5.0)
    _over_atom(w)
    assert w.df_now() < -0.5                          # attractive: Δf goes negative
    w.withdrawn = True
    assert abs(w.df_now()) < 0.5


def test_auto_center_zeroes_the_reading_without_touching_the_physics(tmp_path):
    w = _qplus_world(tmp_path)
    w.pll.set_output(True, w.clock.wall() - 5.0)
    _over_atom(w)
    d = build_dispatcher(w)
    before = w.df_now()
    d.handler_for("PLL.FreqShiftAutoCenter")(1)
    assert abs(w.df_now()) < 1.0
    assert w.pll.center_freq_hz != pytest.approx(w.tip.qplus_f0_hz)
    assert w.pll.center_freq_hz - w.tip.qplus_f0_hz == pytest.approx(before, abs=1.0)


def test_pll_verbs_round_trip(tmp_path):
    w = _qplus_world(tmp_path)
    d = build_dispatcher(w)
    d.handler_for("PLL.AmpCtrlSetpntSet")(1, 80e-12)
    assert d.handler_for("PLL.AmpCtrlSetpntGet")(1) == pytest.approx(80e-12)
    d.handler_for("PLL.CenterFreqSet")(1, 31234.0)
    assert d.handler_for("PLL.CenterFreqGet")(1) == pytest.approx(31234.0)
    d.handler_for("PLL.OutOnOffSet")(1, 1)
    assert d.handler_for("PLL.OutOnOffGet")(1) == 1
    assert d.handler_for("PLL.ExcitationGet")(1) > 0
    d.handler_for("PLL.PhasCtrlGainSet")(1, 7.0, 2e-3)
    assert d.handler_for("PLL.PhasCtrlGainGet")(1) == [7.0, 2e-3]


def test_every_registered_pll_verb_is_implemented(tmp_path):
    """Every locally registered PLL command has a simulator handler."""
    from stmsim.wire.spec import commands

    w = _qplus_world(tmp_path)
    d = build_dispatcher(w)
    names = [name for name, spec in commands().items()
             if spec.module in {"PLL", "PLLFreqSwp"}]
    assert len(names) >= 50
    missing = [name for name in names if d.handler_for(name) is None]
    assert missing == []


def test_frequency_sweep_measures_f0_and_q(tmp_path):
    w = _qplus_world(tmp_path)
    d = build_dispatcher(w)
    d.handler_for("PLL.OutOnOffSet")(1, 1)
    out = d.handler_for("PLLFreqSwp.Start")(1, 1, 0)
    assert out[6] == pytest.approx(w.tip.qplus_f0_hz, rel=5e-3)
    assert out[7] == pytest.approx(w.tip.qplus_q, rel=0.01)


def test_oscillating_tip_sits_further_out_for_the_same_setpoint(tmp_path):
    """The feedback holds the *averaged* current, so switching the drive on retracts the tip."""
    w = _qplus_world(tmp_path)
    gap_static = w.equilibrium_gap()
    w.pll.set_output(True, w.clock.wall() - 5.0)
    gap_osc = w.equilibrium_gap()
    from stmsim.physics.forces import averaged_current_factor
    expected = math.log(averaged_current_factor(w.kappa_m(), w.pll.amp_setpoint_m)) / (2 * w.kappa_m())
    assert gap_osc - gap_static == pytest.approx(expected, rel=0.05)


def test_signal_names_are_the_rig_abbreviations():
    """MAST finds the amplitude and Δf channels by name; these are the rig's own strings."""
    assert SIGNAL_NAMES[16] == "OC D1 Amplitude (m)"
    assert SIGNAL_NAMES[17] == "OC M1 Freq. Shift (Hz)"
    assert "amplitude" in SIGNAL_NAMES[16].lower() and "oc " in SIGNAL_NAMES[16].lower()
    assert "freq. shift" in SIGNAL_NAMES[17].lower()


# ── the sweep ────────────────────────────────────────────────────────────────
def test_delta_f_curve_has_its_minimum_where_the_force_law_says(tmp_path):
    from stmsim.physics.forces import force_truth

    w = _qplus_world(tmp_path)
    w.pll.set_output(True, w.clock.wall() - 5.0)
    _over_atom(w)
    cur = w.zspec_curve(256, 0.5e-9, z_offset_m=0.3e-9)
    assert cur["z_rel"][0] == 0.0 and cur["z_rel"][-1] == pytest.approx(-0.5e-9)
    assert cur["df_min_hz"] < -1.0
    assert not cur["contact"]
    t = force_truth(FORCE, radius_m=w.tip.radius_m, amplitude_m=cur["amplitude_m"],
                    f0_hz=w.tip.qplus_f0_hz, k_n_per_m=w.tip.qplus_k_n_per_m)
    # the sweep's Δf minimum and the analytic one agree to a few tens of pm
    z_at_min = cur["z_rel"][int(np.nanargmin(cur["df"]))]
    depth_from_start = -(z_at_min)
    assert 0 < depth_from_start < 0.5e-9
    assert cur["df_min_hz"] == pytest.approx(t["df_min_hz"], rel=0.35)


def test_the_event_says_where_the_curve_was_taken(tmp_path):
    w = _qplus_world(tmp_path)
    w.pll.set_output(True, w.clock.wall() - 5.0)
    _over_atom(w)
    w.zspec_curve(128, 0.4e-9, z_offset_m=0.3e-9)
    w.move_xy(w.tip_x + 8e-9, w.tip_y)
    w.achievable_z_tip()
    w.zspec_curve(128, 0.4e-9, z_offset_m=0.3e-9)
    evs = [e for e in w.events if e["kind"] == "zspec"]
    assert evs[0]["near_kind"] == "adatom" and evs[0]["near_nm"] < 0.15
    assert evs[1]["near_nm"] > 2.0
    assert w.truth()["n_zspec"] == 2
    assert w.truth()["zspec_best_adatom"]["idx"] == evs[0]["idx"]


def test_sweeping_into_contact_damages_the_tip(tmp_path):
    w = _qplus_world(tmp_path)
    w.pll.set_output(True, w.clock.wall() - 5.0)
    _over_atom(w)
    before = w.tip.snapshot()
    cur = w.zspec_curve(128, 2.0e-9)             # far past the surface
    assert cur["contact"]
    assert w.tip.snapshot() != before


def test_the_retract_condition_stops_the_sweep(tmp_path):
    w = _qplus_world(tmp_path)
    w.pll.set_output(True, w.clock.wall() - 5.0)
    _over_atom(w)
    before = w.tip.snapshot()
    cur = w.zspec_curve(128, 2.0e-9, retract=(1, 5e-10, 0, 0))
    assert cur["retract_abort"] and not cur["contact"]
    assert w.tip.snapshot() == before


# ── the file ─────────────────────────────────────────────────────────────────
def test_the_dat_is_readable_and_carries_the_sensor_parameters(tmp_path):
    w = _qplus_world(tmp_path)
    w.pll.set_output(True, w.clock.wall() - 5.0)
    _over_atom(w)
    d = build_dispatcher(w)
    d.handler_for("ZSpectr.ChsSet")([17, 0, 16])
    d.handler_for("ZSpectr.RangeSet")(0.3e-9, 0.5e-9)
    d.handler_for("ZSpectr.PropsSet")(1, 128, 1, 1, 2, 1)
    d.handler_for("ZSpectr.Start")(1, "P5_atom")
    paths = sorted((tmp_path / "s").glob("*.dat"))
    assert [p.name for p in paths] == ["P5_atom00001.dat"]
    got = izt.read_dat(paths[0])
    cols = got["columns"]
    assert izt._pick(cols, izt.Z_PREFIXES) == "Z rel (m)"
    assert izt._pick(cols, izt.CURRENT_PREFIXES) is not None
    df_col = next(c for c in cols if "freq. shift" in c.lower())
    assert float(np.nanmin(cols[df_col])) < -1.0
    hdr = got["header"]
    assert hdr["Experiment"] == "Z spectroscopy"
    assert float(hdr["Oscillation Control>Center Frequency (Hz)"]) == pytest.approx(
        w.pll.center_freq_hz, rel=1e-4)
    assert float(hdr["Oscillation Control>Amplitude Setpoint (m)"]) == pytest.approx(
        w.pll.amp_setpoint_m, rel=1e-4)
    assert float(hdr["Z Spectroscopy>Sweep End (m)"]) < 0        # negative = toward the surface
    assert abs(float(hdr["X (m)"])) < 1e-3 and abs(float(hdr["Y (m)"])) < 1e-3
    assert [e for e in w.events if e["kind"] == "dat_saved"]


def test_bias_spectroscopy_also_leaves_a_dat(tmp_path):
    sc = Scenario.load(__import__("pathlib").Path(__file__).resolve().parent.parent
                       / "stmbench" / "trackB" / "scenarios" / "B6_sts_clean.yaml")
    w = sc.build_world(1, session_dir=tmp_path / "s")
    d = build_dispatcher(w)
    d.handler_for("BiasSpectr.ChsSet")([0, 20])
    d.handler_for("BiasSpectr.LimitsSet")(-1.0, 1.0)
    d.handler_for("BiasSpectr.Start")(1, "sw_p001")
    p = sorted((tmp_path / "s").glob("*.dat"))
    assert [q.name for q in p] == ["sw_p00100001.dat"]
    got = izt.read_dat(p[0])
    assert izt._pick(got["columns"], izt.BIAS_PREFIXES) == "Bias calc (V)"
    assert izt._pick(got["columns"], izt.LOCKIN_PREFIXES) is not None
    assert got["header"]["Experiment"] == "bias spectroscopy"
    assert w.sts_records[-1]["path"].endswith("sw_p00100001.dat")
