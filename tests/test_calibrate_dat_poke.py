"""Calibration rows with no owner (DESIGN §4.5): ``.dat`` I–Z / dI/dV templates and poke timing.

Everything here runs on synthetic ``.dat`` text and synthetic readback traces written into
``tmp_path`` (< 1 s). The two real-data smoke tests carry ``requires_data`` and are skipped
*visibly* at collection when the inputs are absent.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from stmsim.calibrate import iz_templates as izt
from stmsim.calibrate import index_dat as idt
from stmsim.calibrate import poke_timing as pkt
from stmsim.physics.junction import (LDOSTemplate, MATERIAL_ONSET_EV, current_metal,
                                     gap_for_current, load_ldos_template)
from tests.conftest import requires_data

# ── synthetic .dat ───────────────────────────────────────────────────────────


def _dat_text(header: dict, names: list[str], rows: list[list[float]], ragged_tail: bool = False) -> str:
    lines = [f"{k}\t{v}\t" for k, v in header.items()]
    lines += ["", "[DATA]", "\t".join(names)]
    for r in rows:
        lines.append("\t".join(f"{x:.6E}" for x in r))
    if ragged_tail:
        lines.append("1.0E-12")           # truncated last row (file cut mid-write)
    return "\n".join(lines) + "\n"


def _iz_dat(phi: float, n: int = 101, sweep_m: float = 0.3e-9, noise: float = 0.0, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    z_rel = np.linspace(-sweep_m, 0.0, n)                    # controller: 0 = start height, negative = closer
    gap0 = gap_for_current(100e-12, 0.1, phi)
    i = current_metal(gap0 + z_rel, 0.1, phi) * (1 + noise * rng.standard_normal(n))
    header = {"Experiment": "Z spectroscopy", "Saved Date": "13.02.2025 02:38:55", "Bias>Bias (V)": "100E-3",
              "Z Spectroscopy>Sweep Start (m)": "0E+0", "Z Spectroscopy>Sweep End (m)": f"{-sweep_m:.3E}"}
    return _dat_text(header, ["Z rel (m)", "Current (A)", "Bias (V)"],
                     [[z, c, 0.1] for z, c in zip(z_rel, i)])


def _iv_rows(onset_v: float, v_lo: float, v_hi: float, n: int = 401, gain: float = 1.0,
             lockin_ok: bool = True) -> tuple[list[str], list[list[float]]]:
    v = np.linspace(v_hi, v_lo, n)                            # controller sweeps from start to end
    tmpl = LDOSTemplate("x", onset_ev=onset_v, step_height=0.8, broadening_ev=0.015)
    order = np.argsort(v)
    vs = v[order]
    rho = tmpl.rho(vs)
    i_sorted = np.concatenate([[0.0], np.cumsum(0.5 * (rho[1:] + rho[:-1]) * np.diff(vs))])
    i_sorted -= np.interp(0.0, vs, i_sorted)                  # I(0) = 0
    i_sorted *= 1e-10
    i = np.empty_like(i_sorted)
    i[order] = i_sorted
    li = gain * 1e-10 * tmpl.rho(v) if lockin_ok else np.sin(40 * v) * 1e-11
    return (["Bias calc (V)", "Current (A)", "Bias (V)", "LI Demod 1 X (A)"],
            [[a, b, a, c] for a, b, c in zip(v, i, li)])


# ── parser ───────────────────────────────────────────────────────────────────


def test_parse_dat_header_columns_and_ragged_rows(tmp_path):
    names, rows = _iv_rows(-0.49, -1.0, 1.0, n=50)
    p = tmp_path / "Bias Spectroscopy001.dat"
    p.write_text(_dat_text({"Experiment": "bias spectroscopy", "Comment01": "Au(111)", "User": ""},
                           names, rows, ragged_tail=True), encoding="utf-8")
    d = izt.read_dat(p)
    assert d["header"]["Experiment"] == "bias spectroscopy"
    assert d["header"]["Comment01"] == "Au(111)"
    assert d["header"]["User"] == ""
    assert list(d["columns"]) == names
    col = d["columns"]["Current (A)"]
    assert col.shape == (51,)                                  # ragged row kept, NaN-padded
    assert np.isnan(col[-1]) and np.isfinite(col[:-1]).all()
    assert d["columns"]["Bias calc (V)"][-1] == pytest.approx(1e-12)


def test_parse_dat_without_data_block_gives_empty_columns():
    d = izt.parse_dat_text("Experiment\tSpectrum\t\nSaved Date\tx\t\n")
    assert d["header"]["Experiment"] == "Spectrum" and d["columns"] == {}


@pytest.mark.parametrize("comment, expected", [
    ("Ag(111)", "Ag"), ("Ag", "Ag"), ("Ag(111)/mica", "Ag"), ("Au(111) sharp tip", "Au"),
    ("Au(111) image1,2", "Au"), ("Cu(111)", "Cu"), ("cu (111)", "Cu"),
    ("FeO/Au(111) 1.5A 40s", None), ("B/Ag(111)", None), ("BCN/Cu", None), ("S/Cu(111)", None),
    ("B/Ag(111) new phase", None), ("TaPdTe", None), ("200mV 10pA", None), ("", None), (None, None),
    ("Aug 2019 sample", None), ("Cu2O", None),
])
def test_material_from_comment_accepts_clean_substrates_only(comment, expected):
    assert izt.material_from_comment(comment) == expected


# ── I–Z fit ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("phi", [1.0, 4.0, 5.5])
def test_fit_iz_recovers_phi_from_synthetic_dat(tmp_path, phi):
    p = tmp_path / "Z-Spectroscopy00001.dat"
    p.write_text(_iz_dat(phi, noise=0.01, seed=3), encoding="utf-8")
    r = izt.fit_iz_file(izt.read_dat(p))
    assert r is not None and r["clean"], r
    assert r["phi_ev"] == pytest.approx(phi, rel=0.05)
    assert r["r2"] > 0.99 and r["n_jumps"] == 0
    assert r["sweep_pm"] == pytest.approx(300.0, rel=1e-6)
    assert r["bias_v"] == pytest.approx(0.1)


def test_fit_iz_flags_jumps_and_bad_r2_instead_of_reporting_a_number_as_clean():
    z = np.linspace(-0.3e-9, 0, 101)
    i = current_metal(0.6e-9 + z, 0.1, 4.0)
    i_jump = i.copy()
    i_jump[50:] *= 30.0                                        # tip switch mid-ramp
    r = izt.fit_iz(z, i_jump)
    assert not r["clean"] and r["why"] in ("jumps", "r2")
    rng = np.random.default_rng(0)
    r2 = izt.fit_iz(z, np.abs(rng.standard_normal(101)) * 1e-12)
    assert not r2["clean"] and r2["why"] in ("r2", "phi_range", "jumps")
    assert izt.fit_iz(z[:3], i[:3])["why"] == "too_few"


def test_fit_iz_file_takes_bias_from_the_column_when_the_header_is_blank():
    text = _iz_dat(4.0, noise=0.01, seed=7)                 # noiseless + 6-digit text = false "jumps"
    text = text.replace("Bias>Bias (V)\t100E-3\t\n", "Bias>Bias (V)\t\t\n")
    d = izt.parse_dat_text(text)
    assert d["header"]["Bias>Bias (V)"] == ""
    r = izt.fit_iz_file(d)
    assert r["clean"] and r["bias_v"] == pytest.approx(0.1)
    assert izt.bias_bin(r["bias_v"]) == "50-500mV" and izt.bias_bin(float("nan")) == "unknown"
    assert izt.bias_bin(-0.001) == "<5mV" and izt.bias_bin(0.0) == "~0" and izt.bias_bin(-2.0) == ">=0.5V"


@pytest.mark.parametrize("path, year", [
    ("E:/m/corpus/Rig_A SPM data/2016/201605/0517/Bias Spectroscopy001.dat", "2016"),
    ("E:\\m\\archive\\session_a\\201706\\0605\\Bias Spectroscopy014.dat", "2017"),
    ("E:/m/archive/session_b/20191101/x.dat", "2019"),
    ("E:/m/archive/sample_unknown/sample_unknown-STM data/x.dat", "?"),
])
def test_year_of_reads_the_directory_component(path, year):
    assert izt._year_of(path) == year


def test_dat_index_groups_relative_to_configured_root(tmp_path):
    root = tmp_path / "raw"
    assert idt._group(str(root / "Rig_A" / "session" / "x.dat"), str(root)) == "Rig_A"
    assert idt._group(str(root / "archive" / "nested" / "x.dat"), str(root)) == "archive"
    assert idt._group(str(root / "root-level.dat"), str(root)) == "other"
    assert idt._group(str(tmp_path / "outside" / "x.dat"), str(root)) == "other"


def test_fit_iz_uses_only_points_above_the_floor():
    z = np.linspace(-0.3e-9, 0, 101)
    i = current_metal(0.6e-9 + z, 0.1, 4.0)
    i[-20:] = 1e-16                                            # noise floor: 1e-4 × max drops these
    r = izt.fit_iz(z, i)
    assert r["n_ok"] == 81 and r["clean"] and r["phi_ev"] == pytest.approx(4.0, rel=0.02)


# ── dI/dV templates ──────────────────────────────────────────────────────────


def test_didv_prefers_lockin_only_when_it_agrees_with_the_numeric_derivative():
    names, rows = _iv_rows(-0.49, -1.0, 1.0, lockin_ok=True)
    cols = {n: np.asarray([r[k] for r in rows]) for k, n in enumerate(names)}
    v, y, src = izt.didv_from_columns(cols)
    assert src == "lockin" and np.all(np.diff(v) > 0)
    names, rows = _iv_rows(-0.49, -1.0, 1.0, lockin_ok=False)
    cols = {n: np.asarray([r[k] for r in rows]) for k, n in enumerate(names)}
    v2, y2, src2 = izt.didv_from_columns(cols)
    assert src2 == "numeric"
    # the numeric derivative still shows the step
    assert y2[(v2 > -0.35) & (v2 < -0.2)].mean() > 1.3 * y2[(v2 > -0.8) & (v2 < -0.6)].mean()


def test_template_median_and_onset_check_recover_the_onset_they_were_built_with():
    curves = []
    for k, (lo, hi) in enumerate([(-1.0, 1.0), (-0.8, 0.8), (-1.0, 0.5), (-0.2, 0.2), (-0.7, 1.0), (-1.0, 1.0)]):
        names, rows = _iv_rows(-0.49, lo, hi, gain=3.0 * (k + 1), n=301)
        cols = {n: np.asarray([r[j] for r in rows]) for j, n in enumerate(names)}
        v, y, _ = izt.didv_from_columns(cols)
        yn = izt.normalise_to_window(v, y)
        assert yn is not None
        curves.append((v, yn))
    t = izt.build_template(curves, izt.V_GRID, min_n=4)
    grid = np.asarray(t["v_grid"])
    didv = np.asarray([np.nan if x is None else x for x in t["didv"]], float)
    n = np.asarray(t["n_per_point"])
    assert n[np.isclose(grid, 0.0)] == 6 and n[np.isclose(grid, -0.9)] == 3 and n[np.isclose(grid, 0.9)] == 3
    assert np.isnan(didv[np.isclose(grid, 0.9)]) and np.isnan(didv[np.isclose(grid, -0.9)])   # < min_n → NaN
    assert np.isfinite(didv[np.isclose(grid, -0.75)]) and t["v_covered"][0] == pytest.approx(-0.8)
    win = (grid >= 0.05) & (grid <= 0.2)
    assert didv[win].mean() == pytest.approx(1.0, abs=0.02)   # normalised in the window
    chk = izt.onset_check(grid, didv, -0.49)
    assert chk["ok"], chk
    assert abs(chk["detected_v"] + 0.49) <= 0.03
    assert chk["step_ratio"] > 1.5
    # a flat template does not pass the check
    flat = izt.onset_check(grid, np.ones_like(grid), -0.49)
    assert not flat["ok"] and flat["why"] == "no_step"
    # a template that does not cover the window says so
    short = izt.onset_check(grid, np.where(np.abs(grid) <= 0.2, 1.0, np.nan), -0.49)
    assert short["why"] == "window_not_covered"


def test_curve_is_usable_rejects_saturation_and_spikes():
    v = np.linspace(-1, 1, 201)
    i = 1e-10 * v
    assert izt.curve_is_usable(v, i) == (True, "")
    assert izt.curve_is_usable(v, i * 200)[1] == "saturated"
    spiky = i.copy()
    spiky[[20, 60, 100, 140]] += 5e-11
    assert izt.curve_is_usable(v, spiky)[1] == "spikes"


def test_select_rows_dedupes_mirror_trees_and_bounds_each_class():
    pd = pytest.importorskip("pandas")
    rows = []
    for k in range(30):
        rows.append({"path": f"E:/m/corpus/Rig_A SPM data/2016/x/Bias Spectroscopy{k:03d}.dat", "size": 1000 + k,
                     "h:Saved Date": f"d{k}", "h:Experiment": "bias spectroscopy", "h:Comment01": "Ag(111)",
                     "group": "corpus:Rig_A", "err": ""})
        rows.append({"path": f"E:/m/archive/Rig_A/2016/x/Bias Spectroscopy{k:03d}.dat", "size": 1000 + k,
                     "h:Saved Date": f"d{k}", "h:Experiment": "bias spectroscopy", "h:Comment01": "Ag(111)",
                     "group": "archive", "err": ""})
    for k in range(7):
        rows.append({"path": f"E:/m/corpus/Rig_B SPM data/2024/04/25/Z-Spectroscopy{k:05d}.dat", "size": k,
                     "h:Saved Date": f"z{k}", "h:Experiment": "Z spectroscopy", "h:Comment01": "",
                     "group": "corpus:Rig_B", "err": ""})
    rows.append({"path": "E:/m/x/Bias Spectroscopy999.dat", "size": 1, "h:Saved Date": "e", "h:Experiment": "bias spectroscopy",
                 "h:Comment01": "FeO/Au(111) 1.5A 40s", "group": "archive", "err": ""})
    rows.append({"path": "E:/m/x/bad.dat", "size": 1, "h:Saved Date": "e", "h:Experiment": "bias spectroscopy",
                 "h:Comment01": "Au(111)", "group": "archive", "err": "ValueError: x"})
    iz, iv = izt.select_rows(pd.DataFrame(rows), max_per_class=10)
    assert len(iz) == 7 and set(iz["material"].fillna("u")) == {"u"}
    assert len(iv) == 10 + 1                                   # 30 dedup'd Ag → 10; the FeO one is unlabelled
    assert (iv["material"] == "Ag").sum() == 10 and iv["material"].isna().sum() == 1
    assert "bad.dat" not in "".join(iv["path"])


def test_run_iz_and_run_iv_on_synthetic_files(tmp_path):
    pd = pytest.importorskip("pandas")
    rows = []
    for k in range(6):
        p = tmp_path / f"Z-Spectroscopy{k:05d}.dat"
        p.write_text(_iz_dat(4.0 + 0.1 * k, noise=0.005, seed=k), encoding="utf-8")
        rows.append({"path": str(p), "group": "corpus:Rig_B", "material": None})
    bad = tmp_path / "Z-Spectroscopy00099.dat"
    bad.write_text(_iz_dat(4.0).replace("Current (A)", "Input 2 (V)"), encoding="utf-8")
    rows.append({"path": str(bad), "group": "corpus:Rig_B", "material": None})
    # a zero-bias Δf(z) sweep: 1 pA of preamp noise passes the RELATIVE floor, so the absolute one must catch it
    rng = np.random.default_rng(5)
    quiet = tmp_path / "Z-Spectroscopy00098.dat"
    quiet.write_text(_dat_text({"Experiment": "Z spectroscopy", "Bias>Bias (V)": "0E+0"},
                               ["Z rel (m)", "Current (A)", "OC M1 Freq. Shift (Hz)"],
                               [[z, c, 0.0] for z, c in zip(np.linspace(-1e-10, 0, 51), 1e-12 * rng.standard_normal(51))]),
                     encoding="utf-8")
    rows.append({"path": str(quiet), "group": "corpus:Rig_B", "material": None})
    iz = izt.run_iz(pd.DataFrame(rows))
    assert iz["unlabelled"]["n_files"] == 8 and iz["unlabelled"]["n_clean"] == 6
    assert iz["unlabelled"]["why_failed"] == {"no_z_or_current_column": 1, "no_signal": 1}
    assert 3.9 < iz["unlabelled"]["phi_ev"]["p50"] < 4.6
    assert set(iz["unlabelled"]["by_bias_bin"]) == {"50-500mV", "~0", "unknown"}
    assert iz["unlabelled"]["by_bias_bin"]["~0"]["why_failed"] == {"no_signal": 1}
    assert iz["unlabelled"]["by_bias_bin"]["50-500mV"]["n_clean"] == 6
    assert iz["materials"]["Au"] == {"n_files": 0, "n_clean": 0}
    assert "corpus:Rig_B/?" in iz["by_group_year"]

    rows = []
    for k in range(6):
        names, drows = _iv_rows(-0.065, -0.3, 0.3, n=201, gain=1.0 + k)
        p = tmp_path / f"Bias Spectroscopy{k:03d}.dat"
        p.write_text(_dat_text({"Experiment": "bias spectroscopy", "Comment01": "Ag(111)"}, names, drows), encoding="utf-8")
        rows.append({"path": str(p), "group": "archive", "material": "Ag"})
    sts = izt.run_iv(pd.DataFrame(rows), min_n=3)
    ag = sts["materials"]["Ag"]
    assert ag["n_curves"] == 6 and ag["tally"] == {"lockin": 6}
    assert ag["onset"]["ok"], ag["onset"]
    assert ag["v_covered"][0] == pytest.approx(-0.3) and ag["v_covered"][1] == pytest.approx(0.3)
    assert sts["materials"]["Au"]["n_curves"] == 0 and sts["materials"]["Au"]["onset"]["why"] == "window_not_covered"
    json.dumps(sts), json.dumps(iz)                            # serialisable as written by main()


# ── junction.load_ldos_template ──────────────────────────────────────────────


def _sts_file(tmp_path, mats: dict[str, tuple[float, float, float]]) -> "Path":
    """Write a minimal sts_templates.json; mats = {key: (v_lo, v_hi, onset)} with a real step."""
    grid = izt.V_GRID
    out = {"v_grid": [float(x) for x in grid], "materials": {}}
    for key, (lo, hi, onset) in mats.items():
        tmpl = LDOSTemplate(key, onset_ev=onset, step_height=0.8)
        vals = tmpl.rho(grid)
        vals = np.where((grid >= lo) & (grid <= hi), vals, np.nan)
        out["materials"][key] = {"didv": [None if not np.isfinite(x) else float(x) for x in vals],
                                 "onset": {"ok": True}}
    p = tmp_path / "sts_templates.json"
    p.write_text(json.dumps(out), encoding="utf-8")
    return p


def test_load_ldos_template_prefers_calibrated_table_and_falls_back_honestly(tmp_path):
    p = _sts_file(tmp_path, {"Au": (-1.0, 1.0, -0.49), "Ag": (-0.2, 0.2, -0.065)})
    au = load_ldos_template("Au(111)", path=p)
    assert au.source.startswith("calibrated:sts_templates.json") and au.table_e_ev
    e = np.array([-0.7, -0.3, 0.1])
    ref = LDOSTemplate("Au", onset_ev=-0.49, step_height=0.8).rho(e)
    assert au.rho(e) == pytest.approx(ref, rel=1e-6)          # the table, not the analytic default
    assert au.rho(np.array([1.5]))[0] == pytest.approx(au.rho(np.array([1.0]))[0])   # edge held
    ag = load_ldos_template("Ag(111)", path=p)
    assert ag.source.startswith("calibrated:") and min(ag.table_e_ev) == pytest.approx(-0.2)
    cu = load_ldos_template("Cu(111)", path=p)
    assert cu.source == "analytic:fallback(no_Cu)" and cu.onset_ev == MATERIAL_ONSET_EV["Cu"] and not cu.table_e_ev
    none = load_ldos_template("Au(111)", path=tmp_path / "missing.json")
    assert none.source == "analytic:fallback(no_file)" and none.onset_ev == -0.49
    hopg = load_ldos_template("HOPG", path=p)
    assert hopg.source == "analytic" and hopg.onset_ev is None
    # a table too narrow for the onset is refused (Au needs −0.49 ± 0.1 V)
    p2 = _sts_file(tmp_path / "n", {"Au": (-0.2, 0.2, -0.49)}) if (tmp_path / "n").mkdir() is None else None
    narrow = load_ldos_template("Au", path=p2)
    assert narrow.source.startswith("analytic:fallback(onset_not_covered") and not narrow.table_e_ev


def test_load_ldos_template_clips_nonpositive_values(tmp_path):
    grid = izt.V_GRID
    vals = np.where(grid < -0.5, -0.3, 1.0)
    p = tmp_path / "sts_templates.json"
    p.write_text(json.dumps({"v_grid": grid.tolist(), "materials": {"Cu": {"didv": vals.tolist()}}}), encoding="utf-8")
    cu = load_ldos_template("Cu", path=p)
    assert cu.source.startswith("calibrated:") and cu.source.endswith("onset_ok=None")
    assert cu.rho(np.array([-0.8]))[0] == pytest.approx(0.05)


# ── poke timing ──────────────────────────────────────────────────────────────


def _trace(depth_m: float = 0.5e-9, lag_s: float = 0.2, hold_s: float = 0.5, dz4_m: float = 0.25e-9,
           capture_s: float = 3.0, retract_lag_s: float | None = None, i0: float = 100e-12,
           i_sat: float = 10.004e-9, noise_pm: float = 2.0, seed: int = 0, event_t: float = 0.05,
           poll_hz: float = 2000.0) -> dict:
    """A four-segment readback trace with known hardware lags (capture_s short ⇒ no ④)."""
    rng = np.random.default_rng(seed)
    switch_off, t1, settle, t2, end_wait = 0.1, 0.1, hold_s, 0.1, 0.1
    d_plunge = event_t + switch_off
    d_retract = d_plunge + t1 + settle
    stages = [{"stage": "pre_roll", "t_start": 0.0, "t_end": event_t},
              {"stage": "switch_off", "t_start": event_t, "t_end": d_plunge},
              {"stage": "z_ramp_1_plunge", "t_start": d_plunge, "t_end": d_plunge + t1},
              {"stage": "bias_settle", "t_start": d_plunge + t1, "t_end": d_retract},
              {"stage": "z_ramp_2_retract", "t_start": d_retract, "t_end": d_retract + t2},
              {"stage": "end_wait", "t_start": d_retract + t2, "t_end": d_retract + t2 + end_wait},
              {"stage": "post_roll", "t_start": d_retract + t2 + end_wait, "t_end": None}]
    a_plunge = d_plunge + lag_s
    a_bottom = a_plunge + t1
    a_retract = a_bottom + hold_s if retract_lag_s is None else d_retract + retract_lag_s
    a_back = a_retract + t2
    t_fb = a_back + 0.4                                        # feedback restored (segment ④ starts)
    t = np.arange(0.0, capture_s, 1.0 / poll_hz)
    z1 = -100e-9
    z = np.full(t.size, z1)
    m = (t >= a_plunge) & (t < a_bottom)
    z[m] = z1 - depth_m * (t[m] - a_plunge) / t1
    z[(t >= a_bottom) & (t < a_retract)] = z1 - depth_m
    m = (t >= a_retract) & (t < a_back)
    z[m] = z1 - depth_m + depth_m * (t[m] - a_retract) / t2
    m = t >= t_fb
    z[m] = z1 + dz4_m * (1 - np.exp(-(t[m] - t_fb) / 0.05))
    z += rng.normal(0, noise_pm * 1e-12, t.size)
    # current readback: setpoint until the tip is in, saturated until the feedback is back
    cur = np.full(t.size, i0)
    cur[(t >= a_bottom + 0.03) & (t < t_fb + 0.15)] = i_sat
    cur[(t >= t_fb + 0.15) & (t < t_fb + 0.3)] = 5.5e-9         # preamp leaves saturation in steps
    q = int(round(poll_hz * 0.1))                               # readback updates every 100 ms
    cur = np.repeat(cur[::q], q)[: t.size]
    return {"schema": pkt.SCHEMA, "skill": "TipShapeWithReadback", "event_t_s": event_t, "capture_s": capture_s,
            "fired": True, "aborted": False, "stages": stages,
            "meta": {"tip_lift_m": -depth_m, "lift_height_m": depth_m, "bias_v": 0.02, "switch_off_delay_s": switch_off,
                     "lift_time_1_s": t1, "bias_settling_s": settle, "lift_time_2_s": t2, "end_wait_s": end_wait,
                     "poll_hz": poll_hz, "pre_roll_s": event_t, "post_roll_s": 0.1, "verdict": "cluster"},
            "channels": {"z": {"unit": "m", "n": t.size, "t_s": t.tolist(), "samples": z.tolist()},
                         "current": {"unit": "A", "n": t.size, "t_s": t.tolist(), "samples": cur.tolist()}},
            "_expected": {"plunge_lag_s": lag_s, "press_hold_s": hold_s, "retract_lag_s": a_retract - d_retract,
                          "dz4_pm": dz4_m * 1e12, "t4_s": t_fb + 0.3}}


def test_measure_segments_recovers_known_lags_hold_and_dz4():
    tr = _trace(depth_m=0.5e-9, lag_s=0.22, hold_s=0.5, dz4_m=0.27e-9)
    m = pkt.measure_segments(tr)
    exp = tr["_expected"]
    assert m["why"] == "", m
    assert m["plunge_lag_s"] == pytest.approx(exp["plunge_lag_s"], abs=0.004)
    assert m["plunge_ramp_s"] == pytest.approx(0.1, abs=0.006)
    assert m["press_hold_s"] == pytest.approx(exp["press_hold_s"], abs=0.008)
    assert m["retract_lag_s"] == pytest.approx(exp["retract_lag_s"], abs=0.004)
    assert m["retract_end_lag_s"] == pytest.approx(exp["retract_lag_s"], abs=0.006)
    assert m["depth_reached_frac"] == pytest.approx(1.0, abs=0.02)
    assert abs(m["dz3_pm"]) < 3.0                              # segment ③ is back at baseline by construction
    assert m["dz4_pm"] == pytest.approx(exp["dz4_pm"], abs=3.0)
    assert m["t4_s"] == pytest.approx(exp["t4_s"], abs=0.11)   # current readback quantum is 100 ms
    assert 0.0 < m["current_rise_lag_s"] < 0.3
    assert m["i_sat_over_i0"] == pytest.approx(100.04, rel=1e-3)


def test_measure_segments_reports_no_return_instead_of_a_zero_dz4():
    tr = _trace(capture_s=1.2)                                 # ends while the current is still saturated
    m = pkt.measure_segments(tr)
    assert m["why"] == "no_return"
    assert "dz4_pm" not in m and m.get("dz3_pm") is not None
    assert m["plunge_lag_s"] == pytest.approx(0.2, abs=0.004)


def test_measure_segments_reports_no_plunge_when_z_never_moves():
    tr = _trace()
    z1 = -100e-9
    tr["channels"]["z"]["samples"] = (z1 + np.random.default_rng(1).normal(0, 2e-12, len(tr["channels"]["z"]["t_s"]))).tolist()
    m = pkt.measure_segments(tr)
    assert m["why"] == "no_plunge" and "plunge_lag_s" not in m


def test_measure_segments_scales_edges_to_the_achieved_depth():
    tr = _trace(depth_m=2.0e-9, lag_s=0.3, hold_s=0.5)
    z = np.asarray(tr["channels"]["z"]["samples"])
    z1 = -100e-9
    z = z1 + (z - z1) * 0.6                                    # the stage only reached 60 %
    tr["channels"]["z"]["samples"] = z.tolist()
    m = pkt.measure_segments(tr)
    assert m["depth_reached_frac"] == pytest.approx(0.6, abs=0.02)
    assert m["plunge_lag_s"] == pytest.approx(0.3, abs=0.004)


def test_run_aggregates_traces_and_writes_json(tmp_path):
    for k, (lag, hold, cap, depth) in enumerate([(0.2, 0.5, 3.0, 0.5e-9), (0.25, 0.5, 3.0, 0.5e-9),
                                                 (0.3, 0.5, 1.2, 1.0e-9), (0.22, 0.5, 3.0, 1.0e-9)]):
        (tmp_path / f"TipShapeWithReadback_{k}.json").write_text(
            json.dumps(_trace(depth_m=depth, lag_s=lag, hold_s=hold, capture_s=cap, seed=k)), encoding="utf-8")
    (tmp_path / "TipShapeWithReadback_bad.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "TipShapeWithReadback_other.json").write_text(json.dumps({"schema": "x"}), encoding="utf-8")
    out = tmp_path / "calib" / "poke_timing.json"
    rc = pkt.main(["--traces", str(tmp_path), "--out", str(out), "--per-trace", str(tmp_path / "per.json")])
    assert rc == 0 and out.exists()
    agg = json.loads(out.read_text(encoding="utf-8"))
    assert agg["n_traces"] == 6 and agg["n_reached_segment4"] == 3
    assert agg["why"] == {"ok": 3, "no_return": 1, "load:JSONDecodeError": 1, "load:ValueError": 1}
    q = agg["overall"]["plunge_lag_s"]
    assert q["n"] == 4 and 0.2 <= q["p50"] <= 0.3 and q["min"] == pytest.approx(0.2, abs=0.004)
    assert agg["overall"]["dz4_pm"]["n"] == 3
    assert set(agg["by_depth_pm"]) == {"0", "500", "1000"}
    assert agg["by_depth_pm"]["1000"]["dz4_pm"]["n"] == 1     # the 1.2 s capture has no ④
    assert agg["declared_params"]["bias_settling_s=0.5"] == 4
    assert len(json.loads((tmp_path / "per.json").read_text(encoding="utf-8"))) == 6


def test_main_without_traces_dir_fails_loudly(tmp_path):
    assert pkt.main(["--traces", str(tmp_path / "nowhere"), "--out", str(tmp_path / "o.json")]) == 2


# ── real data (CLI-only work; bounded smoke) ─────────────────────────────────


@requires_data
def test_real_dat_index_bounded_run_writes_both_tables(tmp_path):
    """40 files per class off the real index into tmp_path — the CLI's shape, not its numbers."""
    from stmsim.paths import index_dir
    idx = index_dir() / "dat_index.parquet"
    if not idx.exists():
        pytest.fail(f"{idx} missing — run `python -m stmsim.calibrate.index_dat` first")
    rc = izt.main(["--index", str(idx), "--out-dir", str(tmp_path), "--max-per-class", "40"])
    assert rc == 0
    iz = json.loads((tmp_path / "iz_phi.json").read_text(encoding="utf-8"))
    sts = json.loads((tmp_path / "sts_templates.json").read_text(encoding="utf-8"))
    assert iz["unlabelled"]["n_files"] > 0 and iz["meta"]["contract"]["r2_min"] == 0.9
    assert set(sts["materials"]) == {"Au", "Ag", "Cu"} and len(sts["v_grid"]) == 201
    for mat, t in sts["materials"].items():
        assert t["n_curves"] + sum(t["tally"].values()) - t["tally"].get("lockin", 0) - t["tally"].get("numeric", 0) \
            == sum(v for k, v in t["tally"].items() if k not in ("lockin", "numeric")) + t["n_curves"]


@requires_data
def test_real_traces_measure_the_lag_the_world_hardcodes(tmp_path):
    traces = pkt.default_traces_dir()
    if traces is None or not traces.is_dir():
        pytest.fail(f"calibration traces not found under data root: {traces}")
    agg, measures = pkt.run(traces)
    assert agg["n_traces"] == 168
    lag = agg["overall"]["plunge_lag_s"]
    assert lag["n"] >= 150 and 0.1 <= lag["p50"] <= 0.4       # world.py: hw_lag = 0.22 + U(0, 0.11)
    assert agg["why"].get("no_return", 0) >= 100              # the memory note: 159/168 never reached ④
