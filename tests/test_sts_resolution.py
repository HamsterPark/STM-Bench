"""A bias sweep with the ``sts_resolution`` switch on: the lock-in's resolution, the Kondo resonance of
magnetic adatoms through a real tip, the noise, the drift along the sweep, the random streams, the
ledger - and, with the switch off, the sweep exactly as it was (digests computed at 14bdc55)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.optimize import curve_fit

from stmbench.instrument import Instrument, verify
from stmsim.physics.switches import STS_LOCKIN_STREAM
from stmsim.physics.tip import Apex
from stmsim.scenario import Scenario
from tests._p6_world import FIXED, p6_like
from tests._sts_pin import sweep_digest

#: noise-free: at this integration time per point the junction noise is ~1e-4 of the peak
QUIET_S = 1.0e5


def _build(tmp_path, seed=0, name="w", **kw):
    sc = Scenario.load(p6_like(tmp_path, name=name, **kw))
    w = sc.build_world(seed, tmp_path / f"s_{name}", virtual_clock=True)
    w.drift_v_m_per_s[:] = 0.0
    return w


def _isolated_atom(w, clear_nm=3.0) -> tuple[float, float]:
    xy = w.surface.site.adatoms._arrays()[1]
    for k in range(len(xy)):
        d = np.hypot(xy[:, 0] - xy[k, 0], xy[:, 1] - xy[k, 1])
        d[k] = np.inf
        if d.min() > clear_nm * 1e-9:
            return float(xy[k, 0]), float(xy[k, 1])
    raise AssertionError("no isolated atom")


def _on_atom(w, dx_nm=0.0):
    x, y = _isolated_atom(w)
    w.tip_x, w.tip_y = x + dx_nm * 1e-9, y
    return x, y


def _fano(v, g, q, ek, a, c0, c1):
    e = (v * 1e3 - ek) / g
    return (c0 + c1 * v * 1e3) * (1 + a * ((q + e) ** 2 / (1 + e * e) - 1))


def _fit(v, d):
    p, _ = curve_fit(_fano, v, d, p0=[5.0, 0.2, 1.0, 0.2, d[len(d) // 2], 0.0], maxfev=20000)
    return p


def _sweep(w, amp_mv, *, lockin=True, n=201, integ=QUIET_S, v0=-0.025, v1=0.025, settle=0.0):
    w.lockin_on = lockin
    w.lockin_amp_v = amp_mv * 1e-3
    return w.sts_curve(v0, v1, n, settle_s=settle, integ_s=integ)


# ───────────────────────── the resonance, seen through the lock-in ─────────────────────────

def test_a_sweep_on_an_atom_with_a_small_modulation_recovers_the_line(tmp_path):
    w = _build(tmp_path, kondo=FIXED)
    _on_atom(w)
    r = _sweep(w, 0.2)
    assert r["near_nm"] < 0.01
    g, q, ek, a = _fit(r["v"], r["didv"])[:4]
    assert g == pytest.approx(6.0, rel=0.03)               # the thermal kernel adds ~1.5 %
    assert q == pytest.approx(0.3, abs=0.03)
    assert ek == pytest.approx(1.5, abs=0.2)
    assert 0.2 < a < 0.26                                  # a0 = 0.25, blurred by the 0.08 nm apex


def test_a_large_modulation_widens_the_line(tmp_path):
    w = _build(tmp_path, kondo=FIXED)
    _on_atom(w)
    g_small = _fit(*[(r := _sweep(w, 0.2))["v"], r["didv"]])[0]
    g_big = _fit(*[(r := _sweep(w, 10.0))["v"], r["didv"]])[0]
    assert g_big > 1.6 * g_small                           # 10 mV zero-to-peak: FWHM 17 mV against 12 mV
    assert g_big == pytest.approx(11.8, rel=0.1)


def test_the_hidden_attenuation_narrows_what_reaches_the_junction(tmp_path):
    widths = []
    for att in (1.0, 0.8):
        w = _build(tmp_path, kondo=FIXED, lockin_attenuation=att, name=f"att{att}")
        _on_atom(w)
        r = _sweep(w, 10.0)
        widths.append(_fit(r["v"], r["didv"])[0])
        assert "attenuation" not in json.dumps({k: v for k, v in r.items() if k not in ("v", "i", "didv")})
        assert "attenuation" not in json.dumps(w.events[-1])
    assert widths[1] < 0.95 * widths[0]


def test_two_nanometres_from_any_atom_there_is_no_resonance(tmp_path):
    w = _build(tmp_path, kondo=FIXED)
    _on_atom(w, dx_nm=2.5)
    r = _sweep(w, 0.2)
    assert r["near_nm"] > 2.0
    w.kondo = None
    bare = _sweep(w, 0.2)
    assert np.allclose(r["didv"] / bare["didv"], 1.0, atol=2e-3)


def test_a_world_without_a_kondo_resonance_still_has_the_resolution(tmp_path):
    """The switch alone: a smooth, noise-limited curve, and a lock-in whose noise is the design's."""
    w = _build(tmp_path, kondo=FIXED)
    w.kondo = None
    r = _sweep(w, 1.0)
    assert np.ptp(r["didv"] / r["didv"].mean()) < 0.05


# ───────────────────────── across the tip ─────────────────────────

def _kondo_coefficients_at(w, r_nm, sigma_t):
    """C0, C1, C2 from the formulas of the design, for one apex, one atom at r."""
    k = w.kondo
    r2 = (r_nm * 1e-9) ** 2

    def g(s):
        s2 = s * s + sigma_t ** 2
        return s * s / s2 * math.exp(-r2 / (2 * s2))
    s1 = (1 / k.sigma_m ** 2 + 1 / k.q_sigma_m ** 2) ** -0.5
    s2_ = (1 / k.sigma_m ** 2 + 2 / k.q_sigma_m ** 2) ** -0.5
    return k.amplitude * g(k.sigma_m), k.amplitude * k.q * g(s1), k.amplitude * k.q ** 2 * g(s2_)


def _fano_of(c0, c1, c2):
    """The Fano amplitude and q a fit would find for coefficients C: Aq = C1, A(q²-1) = C2 - C0."""
    a = 0.5 * (-(c2 - c0) + math.sqrt((c2 - c0) ** 2 + 4 * c1 * c1))
    return a, c1 / a


@pytest.mark.parametrize("offset_nm", [0.0, 0.25, 0.5, 0.8])
def test_amplitude_and_q_fall_off_with_the_offset_as_specified(tmp_path, offset_nm):
    w = _build(tmp_path, kondo=FIXED)
    _on_atom(w, dx_nm=offset_nm)
    r = _sweep(w, 0.2)
    p = _fit(r["v"], r["didv"])
    a_want, q_want = _fano_of(*_kondo_coefficients_at(w, offset_nm, w.tip.apex_sigma_m))
    assert p[3] == pytest.approx(a_want, abs=0.006)
    assert p[1] == pytest.approx(q_want, abs=0.03)


def test_the_resonance_is_weaker_and_q_smaller_off_the_atom(tmp_path):
    amps, qs = [], []
    for off in (0.0, 0.4, 0.8):
        w = _build(tmp_path, kondo=FIXED, name=f"off{off}")
        _on_atom(w, dx_nm=off)
        r = _sweep(w, 0.2)
        p = _fit(r["v"], r["didv"])
        amps.append(p[3])
        qs.append(p[1])
    assert amps[0] > amps[1] > amps[2] > 0 and qs[0] > qs[1] > qs[2]


def _far_apex_position(w, ax, ay, dist_nm=5.0):
    """An offset from (ax, ay) at which the second apex of a double tip has no Co atom within 3 nm."""
    xy = w.surface.site.adatoms._arrays()[1]
    for ang in np.linspace(0.0, 2 * np.pi, 36, endpoint=False):
        px, py = ax + dist_nm * 1e-9 * math.cos(ang), ay + dist_nm * 1e-9 * math.sin(ang)
        if np.hypot(xy[:, 0] - px, xy[:, 1] - py).min() > 3e-9:
            return dist_nm * 1e-9 * math.cos(ang), dist_nm * 1e-9 * math.sin(ang)
    raise AssertionError("no free spot for a second apex")


def test_a_double_tip_lowers_the_amplitude_on_the_atom_by_its_weight(tmp_path):
    single = _build(tmp_path, kondo=FIXED, name="single")
    x, y = _on_atom(single)
    a_single = _fit(*[(r := _sweep(single, 0.2))["v"], r["didv"]])[3]
    dbl = _build(tmp_path, kondo=FIXED, name="double")
    _on_atom(dbl)
    dx, dy = _far_apex_position(dbl, x, y)
    dbl.tip.apexes = [Apex(0.0, 0.0, 0.0, 1.0), Apex(dx, dy, 0.15e-9, 0.6)]
    a_double = _fit(*[(r := _sweep(dbl, 0.2))["v"], r["didv"]])[3]
    kappa = dbl.kappa_m()
    h_at = float(dbl.surface.height_smooth(np.array([x]), np.array([y]))[0])
    h_far = float(dbl.surface.height_smooth(np.array([x + dx]), np.array([y + dy]))[0])
    w0 = 1.0 / (1.0 + 0.6 * math.exp(2 * kappa * (h_far + 0.15e-9 - h_at)))
    assert a_double < 0.7 * a_single
    assert a_double / a_single == pytest.approx(w0, rel=0.06)


def test_a_blunt_tip_lowers_the_amplitude_and_q(tmp_path):
    out = {}
    for label, sigma in (("sharp", 0.08e-9), ("blunt", 0.4e-9)):
        w = _build(tmp_path, kondo=FIXED, name=label)
        _on_atom(w)
        w.tip.apex_sigma_m = sigma
        r = _sweep(w, 0.2)
        out[label] = _fit(r["v"], r["didv"])
        a_want, q_want = _fano_of(*_kondo_coefficients_at(w, 0.0, sigma))
        assert out[label][3] == pytest.approx(a_want, abs=0.006)
        assert out[label][1] == pytest.approx(q_want, abs=0.03)
    assert out["blunt"][3] < 0.6 * out["sharp"][3] and out["blunt"][1] < out["sharp"][1]


# ───────────────────────── noise ─────────────────────────

def _noise_std(w, amp_mv, integ, *, n=2000, lockin=True):
    """Standard deviation of the lock-in's dI/dV about the noise-free curve, and that curve's peak."""
    clean = _sweep(w, amp_mv, lockin=lockin, n=n, integ=QUIET_S, v0=0.05, v1=0.15)
    noisy = _sweep(w, amp_mv, lockin=lockin, n=n, integ=integ, v0=0.05, v1=0.15)
    return float(np.std(noisy["didv"] - clean["didv"])), float(np.max(np.abs(clean["didv"]))), noisy, clean


def test_the_lockin_noise_is_3_percent_of_the_peak_at_10_mv_and_5_ms(tmp_path):
    w = _build(tmp_path, kondo=FIXED)
    _on_atom(w, dx_nm=3.0)                                 # a clean spot: a smooth background
    std, peak, *_ = _noise_std(w, 10.0, 0.005)
    assert std / peak == pytest.approx(0.03, rel=0.08)


def test_the_lockin_noise_falls_as_the_modulation_grows_and_as_the_integration_lengthens(tmp_path):
    w = _build(tmp_path, kondo=FIXED)
    _on_atom(w, dx_nm=3.0)
    s1, *_ = _noise_std(w, 1.0, 0.005)
    s2, *_ = _noise_std(w, 2.0, 0.005)
    s4, *_ = _noise_std(w, 4.0, 0.005)
    assert s1 / s2 == pytest.approx(2.0, rel=0.1) and s2 / s4 == pytest.approx(2.0, rel=0.1)
    t5, *_ = _noise_std(w, 2.0, 0.005)
    t20, *_ = _noise_std(w, 2.0, 0.020)
    assert t5 / t20 == pytest.approx(2.0, rel=0.1)


def test_without_the_lockin_the_derivative_of_the_current_carries_the_same_current_noise(tmp_path):
    w = _build(tmp_path, kondo=FIXED)
    _on_atom(w, dx_nm=3.0)
    n = 1000
    quiet = _sweep(w, 5.0, lockin=False, n=n, integ=QUIET_S, v0=0.05, v1=0.15)
    off = _sweep(w, 5.0, lockin=False, n=n, integ=0.005, v0=0.05, v1=0.15)
    assert np.allclose(off["didv"], np.gradient(off["i"], off["v"]))     # no other dI/dV noise
    sigma_i = float(np.std(off["i"] - quiet["i"]))
    on = _sweep(w, 5.0, lockin=True, n=n, integ=0.005, v0=0.05, v1=0.15)
    quiet_on = _sweep(w, 5.0, lockin=True, n=n, integ=QUIET_S, v0=0.05, v1=0.15)
    sigma_lock = float(np.std(on["didv"] - quiet_on["didv"])) * 5e-3     # × V_p: the same current figure
    assert sigma_lock == pytest.approx(sigma_i, rel=0.2)
    assert not np.allclose(off["didv"], 0.0)


def test_a_time_trace_at_one_bias_still_gives_the_lockin_values_and_zeros_without_it(tmp_path):
    w = _build(tmp_path, kondo=FIXED)
    _on_atom(w)
    on = _sweep(w, 1.0, n=50, integ=0.05, v0=0.003, v1=0.003)
    assert np.all(on["didv"] != 0.0) and np.all(on["v"] == 0.003)
    off = _sweep(w, 1.0, lockin=False, n=50, integ=0.05, v0=0.003, v1=0.003)
    assert np.all(off["didv"] == 0.0)


def test_a_4096_point_sweep_is_fast(tmp_path):
    import time
    w = _build(tmp_path, kondo=FIXED)
    _on_atom(w)
    t = time.perf_counter()
    _sweep(w, 1.0, n=4096, integ=0.005, v0=-1.0, v1=1.0, settle=0.002)
    assert time.perf_counter() - t < 1.0


# ───────────────────────── drift along the sweep ─────────────────────────

def _drifting(w, nm_over_s_sweep: float, t_sweep_s: float):
    """Give the sample the drift that moves it ``nm_over_s_sweep`` nanometres in ``t_sweep_s`` seconds
    of instrument time (along +x)."""
    w.drift_v_m_per_s[:] = [1e-11, 0.0, 0.0]
    t0 = w.clock.sim()
    moved = w.drift_offset(t0 + t_sweep_s)[0] - w.drift_offset(t0)[0]
    w.drift_v_m_per_s[:] = [1e-11 * nm_over_s_sweep * 1e-9 / moved, 0.0, 0.0]


def test_a_long_sweep_drifts_off_the_atom_and_the_resonance_fades(tmp_path):
    w = _build(tmp_path, kondo=FIXED)
    _on_atom(w)
    n, integ = 400, 0.5
    _drifting(w, 1.2, n * integ)
    bias = FIXED["e_k_mev"] * 1e-3 - FIXED["q"] * FIXED["gamma_mev"] * 1e-3      # the dip's floor
    with_kondo = _sweep(w, 1.0, n=n, integ=integ, v0=bias, v1=bias)
    kon, w.kondo = w.kondo, None
    bare = _sweep(w, 1.0, n=n, integ=integ, v0=bias, v1=bias)                    # same drift, no resonance
    w.kondo = kon
    k = with_kondo["didv"] / bare["didv"]                                         # the Kondo factor along the trace
    start, end = k[:20].mean(), k[-20:].mean()
    assert start < 0.85 and end > start + 0.1          # a 25 % dip on the atom, filled in 1.2 nm away
    assert with_kondo["drift_nm"] == pytest.approx(1.2, rel=0.02)
    assert with_kondo["near_nm"] < 0.01 and with_kondo["near_nm_end"] > 0.5


def test_the_tip_sees_a_larger_gap_and_less_current_as_it_drifts_off_the_atom(tmp_path):
    w = _build(tmp_path, kondo=FIXED)
    w.kondo = None
    _on_atom(w)
    _drifting(w, 1.0, 100 * 0.05)
    drifting = _sweep(w, 1.0, n=100, integ=0.05, v0=0.05, v1=0.05)
    w.drift_v_m_per_s[:] = 0.0
    still = _sweep(w, 1.0, n=100, integ=0.05, v0=0.05, v1=0.05)
    assert drifting["i"][-20:].mean() < 0.8 * drifting["i"][:20].mean()
    assert np.ptp(still["i"]) < 0.2 * still["i"].mean()


def test_the_record_and_event_carry_the_timing_and_the_drift_and_nothing_hidden(tmp_path):
    w = _build(tmp_path, kondo=FIXED, lockin_attenuation=0.85)
    _on_atom(w)
    _drifting(w, 0.5, 60 * 0.02)
    r = _sweep(w, 1.0, n=60, integ=0.015, settle=0.005)
    rec, ev = w.sts_records[-1], [e for e in w.events if e["kind"] == "sts"][-1]
    for d in (rec, ev):
        assert d["settle_s"] == 0.005 and d["integ_s"] == 0.015
        assert d["drift_nm"] == pytest.approx(0.5, rel=0.05) and d["near_nm_end"] > 0.0
    assert rec["near_nm"] == r["near_nm"] and rec["near_kind"] == "adatom"
    for old in ("idx", "gap_m", "jump", "phi_ev", "sim_s", "sx_nm", "sy_nm", "corral_centre_dist_nm"):
        assert old in rec
    blob = json.dumps(ev, default=float).lower() + json.dumps({k: v for k, v in rec.items() if k not in ("v", "i", "didv")},
                                                              default=float).lower()
    assert "atten" not in blob and "gamma" not in blob and "kondo" not in blob and "0.85" not in blob


# ───────────────────────── random streams ─────────────────────────

def _states(w):
    return [g.bit_generator.state for g in (w.rng, w.noise.rng, w.tip.rng, w.move_rng, w.df_rng, w.rng_outscan,
                                            w.rng_recovery, w.rng_aniso)]


@pytest.mark.parametrize("tip", [None, {"lambda_per_s": 0.02}])
def test_the_lockin_noise_has_a_stream_of_its_own(tmp_path, tip):
    """The same sweep with the lock-in on and off leaves every generator of the world in the same
    state: the new noise draws from default_rng([seed, key, sweep index]) and nothing else. (The
    unstable tip's jump draws from the world's stream, as before.)"""
    states = []
    for lockin in (True, False):
        w = _build(tmp_path, kondo=FIXED, tip=tip, name=f"s{lockin}")
        _on_atom(w)
        for _ in range(3):
            _sweep(w, 2.0, lockin=lockin, n=64, integ=0.01)
        states.append(_states(w))
    assert states[0] == states[1]
    # and the stream is the documented one
    w = _build(tmp_path, kondo=FIXED, name="docs")
    _on_atom(w)
    a = _sweep(w, 2.0, n=64, integ=0.01)
    w2 = _build(tmp_path, kondo=FIXED, name="docs2")
    _on_atom(w2)
    b = _sweep(w2, 2.0, n=64, integ=0.01)
    assert np.array_equal(a["didv"], b["didv"]) and STS_LOCKIN_STREAM == 0x57A1
    c = _sweep(w, 2.0, n=64, integ=0.01)                    # the next sweep draws from the next index
    assert not np.allclose(a["didv"], c["didv"], rtol=1e-3, atol=0.0)


# ───────────────────────── the ledger ─────────────────────────

def test_a_p6_like_episode_replays_to_the_same_measurements(tmp_path, monkeypatch):
    import stmbench.instrument.core as core
    sc_dir = tmp_path / "scen"
    sc_dir.mkdir()
    p = p6_like(sc_dir, draw=True, name="p6like")
    monkeypatch.setattr(core, "SCENARIO_DIR", sc_dir)
    inst = Instrument(str(p), 3, tmp_path / "run")
    w = inst.world
    assert w.physics.sts_resolution and w.kondo is not None and 0.8 <= w.lockin_attenuation <= 1.0
    xy = w.surface.site.adatoms._arrays()[1]
    inst.act("move_tip", x_nm=float(xy[0, 0] * 1e9), y_nm=float(xy[0, 1] * 1e9), speed_nm_s=5.0)
    inst.act("lockin", on=True, amplitude_mv=1.0)
    inst.act("sts", v_start=-0.02, v_end=0.02, points=64, settle_ms=2.0, integration_ms=40.0)
    inst.act("lockin", on=False, amplitude_mv=1.0)
    inst.act("sts", v_start=-0.02, v_end=0.02, points=64, settle_ms=2.0, integration_ms=40.0)
    inst.finish()
    assert len(w.sts_records) == 2 and "drift_nm" in w.sts_records[0]
    assert verify(tmp_path / "run")["ok"]


# ───────────────────────── switch off = unchanged ─────────────────────────

#: Digests of two sweeps (lock-in off, then on at 5 mV) through the Instrument and of the
#: generators' end states, computed with the code of commit 14bdc55 (tests/_sts_pin.py).
PINNED = {
    "P2_standing_waves_cu111": {
        "off": {"didv": "d4e9b04da25cec05", "event": "037de49c02e61763", "i": "a41e3b18b98a93d7",
                "record": "527ceb5c61e69a5b", "v": "7542854c8e76672e"},
        "on": {"didv": "776a9c64899f0892", "event": "db9055aa47c50d2b", "i": "9644924449f96830",
               "record": "8f22e61a69e5ddcd", "v": "7542854c8e76672e"},
        "rng": {"aniso": "9417d2cbcde41de8", "df": "75fda30782837583", "move": "a79b8004744ecb82",
                "noise": "76fcc44aed0df2b5", "outscan": "f07f2abf296d57e0", "recovery": "b7239032f0d6dfaf",
                "tip": "7e62aff6a05eb04c", "world": "48a3a845b9917274"}},
    "P4_atom_positioning_cu111": {
        "off": {"didv": "1e613c61ca15ca93", "event": "29cda97e712c0c70", "i": "9c9665111afeb83e",
                "record": "645df452fea99b40", "v": "7542854c8e76672e"},
        "on": {"didv": "92b986bfc6bcaf72", "event": "05ad886f1cc9be68", "i": "53b43829d4de7d9c",
               "record": "756e4b5b3d01014c", "v": "7542854c8e76672e"},
        "rng": {"aniso": "9417d2cbcde41de8", "df": "75fda30782837583", "move": "a79b8004744ecb82",
                "noise": "76fcc44aed0df2b5", "outscan": "567e517da91f9516", "recovery": "b7239032f0d6dfaf",
                "tip": "7e62aff6a05eb04c", "world": "48a3a845b9917274"}},
}


@pytest.mark.parametrize("scenario_id", sorted(PINNED))
def test_with_the_switch_off_a_sweep_is_what_it_was_at_14bdc55(scenario_id):
    assert sweep_digest(scenario_id) == PINNED[scenario_id]


def test_the_switch_off_sweep_has_none_of_the_new_keys(tmp_path):
    w = _build(tmp_path, kondo=FIXED, sts_resolution=False, name="off")
    _on_atom(w)
    r = _sweep(w, 1.0, n=40, integ=0.005)
    assert not {"settle_s", "integ_s", "near_nm_end", "drift_nm"} & set(r)
    assert not {"settle_s", "integ_s", "near_nm_end", "drift_nm"} & set(w.events[-1])


def test_the_record_gives_the_leading_apex_and_its_distance_to_the_nearest_atom(tmp_path):
    """A tip whose single apex sits 0.6 nm off the tip centre, centred so that the APEX is over the
    atom: the apex fields read 0, the centre fields 0.6, and the resonance is seen."""
    w = _build(tmp_path, kondo=FIXED)
    x, y = _isolated_atom(w)
    w.tip.apexes = [Apex(0.6e-9, 0.0, 0.0, 1.0)]
    w.tip_x, w.tip_y = x - 0.6e-9, y
    r = _sweep(w, 0.2)
    rec, ev = w.sts_records[-1], [e for e in w.events if e["kind"] == "sts"][-1]
    for d in (r, rec, ev):
        assert d["apex_offset_nm"] == pytest.approx([0.6, 0.0])
        assert d["near_kind_apex"] == "adatom" and d["near_nm_apex"] < 0.01 and d["near_nm_apex_end"] < 0.01
        assert d["near_nm"] == pytest.approx(0.6, abs=0.01)
    p = _fit(r["v"], r["didv"])
    assert p[3] > 0.2 and p[1] == pytest.approx(0.3, abs=0.04)
    # a single apex at the centre: offset [0, 0] and the apex fields equal the centre's
    w2 = _build(tmp_path, kondo=FIXED, name="centred")
    _on_atom(w2)
    r2 = _sweep(w2, 0.2)
    assert r2["apex_offset_nm"] == [0.0, 0.0] and r2["near_nm_apex"] == r2["near_nm"]
