"""Hidden per-seed parameters: drawn reproducibly, and inert when a scenario has none."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stmsim.scenario import HIDDEN_SCHEMA, Scenario, flatten_draws, resolve_hidden

SCEN = Path(__file__).resolve().parent.parent / "stmbench" / "trackB" / "scenarios"


def test_draws_are_reproducible_and_seed_dependent():
    hidden = {"herringbone": {"period_nm": [6.0, 6.6], "domain_spacing_nm": [120.0, 300.0]}}
    a = resolve_hidden(hidden, 3, "P1")
    assert a == resolve_hidden(hidden, 3, "P1")
    assert a != resolve_hidden(hidden, 4, "P1")
    assert a != resolve_hidden(hidden, 3, "P2")          # the scenario id is part of the stream
    assert 6.0 <= a["herringbone"]["period_nm"] <= 6.6


def test_draw_forms():
    hidden = {"adatoms": {"ring_n": [36, 60],
                          "r_threshold_kohm": {"range": [80.0, 400.0], "log": True},
                          "gap_atoms": 3}}
    got = resolve_hidden(hidden, 1, "P3")["adatoms"]
    assert isinstance(got["ring_n"], int) and 36 <= got["ring_n"] <= 60
    assert 80.0 <= got["r_threshold_kohm"] <= 400.0
    assert got["gap_atoms"] == 3                          # a scalar passes through untouched


def test_key_order_does_not_change_the_draw():
    a = {"herringbone": {"period_nm": [6.0, 6.6], "amp_pm": [8.0, 15.0]}}
    b = {"herringbone": {"amp_pm": [8.0, 15.0], "period_nm": [6.0, 6.6]}}
    assert resolve_hidden(a, 2, "X") == resolve_hidden(b, 2, "X")


def test_unknown_keys_are_refused():
    """A silent miss looks exactly like the model getting it right, so a typo must raise."""
    with pytest.raises(ValueError, match="HIDDEN_SCHEMA"):
        resolve_hidden({"herringbone": {"periodd_nm": [6.0, 6.6]}}, 0, "X")
    with pytest.raises(ValueError, match="HIDDEN_SCHEMA"):
        resolve_hidden({"nonsense": {"a": [0, 1]}}, 0, "X")


def test_flatten_draws():
    assert flatten_draws({"tip": {"radius_nm": 2.0}}) == {"tip.radius_nm": 2.0}


def test_no_hidden_block_leaves_the_world_bit_identical(tmp_path):
    """Every B scenario predates the hidden mechanism: adding it must not move their worlds."""
    sc = Scenario.load(SCEN / "B5_repair_blunt.yaml")
    assert sc.hidden == {}
    a = sc.build_world(4, session_dir=tmp_path / "a")
    b = sc.build_world(4, session_dir=tmp_path / "b")
    assert a.tip.apex_sigma_m == b.tip.apex_sigma_m
    assert np.array_equal(a.drift_v_m_per_s, b.drift_v_m_per_s)
    assert a.surface.site.features[0].x == b.surface.site.features[0].x
    assert a.truth()["hidden"] == {}


def test_paper_scenario_publishes_its_draws(tmp_path):
    sc = Scenario.load(SCEN / "P1_barth1990_au111.yaml")
    w = sc.build_world(0, session_dir=tmp_path / "s")
    t = w.truth()
    assert t["hidden"]["draws"]["herringbone.period_nm"] == pytest.approx(
        t["herringbone"]["period_nm"])
    assert t["herringbone"]["single_domain"] is False


def test_scan_saved_carries_geometry_and_drift(tmp_path):
    """The judge maps a reported scan-frame position through these, so they must be there."""
    from stmsim.physics.scanner import ScanSettings

    sc = Scenario.load(SCEN / "P1_barth1990_au111.yaml")
    w = sc.build_world(0, session_dir=tmp_path / "s")
    w.scan = ScanSettings(cx=1e-9, cy=-2e-9, w=50e-9, h=50e-9, nx=128, ny=128,
                          line_time_fwd_s=0.01, line_time_bwd_s=0.01)
    w.scan_start()
    w.clock.advance_sim(10.0)
    w.save_frame()
    ev = [e for e in w.events if e["kind"] == "scan_saved"][-1]
    for key in ("cx_m", "cy_m", "w_m", "h_m", "angle_deg", "nx", "ny", "drift_m", "complete",
                "bias_v", "setpoint_a", "idx"):
        assert key in ev, key
    assert ev["cx_m"] == pytest.approx(1e-9)
    assert len(ev["drift_m"]) == 2


def test_sts_event_carries_position_and_scatterer(tmp_path):
    sc = Scenario.load(SCEN / "B6_sts_clean.yaml")
    w = sc.build_world(1, session_dir=tmp_path / "s")
    w.sts_curve(-1.0, 1.0, 64)
    ev = [e for e in w.events if e["kind"] == "sts"][-1]
    for key in ("idx", "x_m", "y_m", "sx_m", "sy_m", "near_kind", "near_nm", "bias_v"):
        assert key in ev, key
    assert w.truth()["n_sts"] == 1
    assert len(w.sts_records) == 1


def test_every_hidden_key_used_by_a_scenario_is_in_the_schema():
    for path in sorted(SCEN.glob("*.yaml")):
        sc = Scenario.load(path)                          # load() validates, so this is the check
        for block, body in (sc.hidden or {}).items():
            assert block in HIDDEN_SCHEMA
            assert set(body) <= HIDDEN_SCHEMA[block], (path.name, block)
