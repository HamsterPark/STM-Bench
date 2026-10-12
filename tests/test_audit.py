"""The chance audit: what a fixed answer sheet earns, and which claims are held to the gate."""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from stmbench.trackB import audit as A
from stmsim.scenario import Scenario

SCEN = Path(__file__).resolve().parent.parent / "stmbench" / "trackB" / "scenarios"
PAPERS = sorted(SCEN.glob("P*.yaml"))


def test_material_constants_are_the_only_knowledge_claims():
    """Decision of 2026-10-12: knowing a material constant is part of the skill, so the
    herringbone period and the Cu(111) band bottom / effective mass are not held to the
    guessing gate. Every other reported value must be measured."""
    got = {(Scenario.load(p).id, c["id"]) for p in PAPERS for c in Scenario.load(p).claims
           if A.claim_policy(c) == "knowledge"}
    assert got == {("P1_barth1990_au111", "stripe_period_nm"),
                   ("P2_standing_waves_cu111", "e0_mev"),
                   ("P2_standing_waves_cu111", "m_eff")}


def test_claims_judged_on_the_instrument_state_are_not_audited():
    p4 = Scenario.load(SCEN / "P4_atom_positioning_cu111.yaml")
    assert {A.claim_policy(c) for c in p4.claims} == {"state"}
    repair = Scenario.load(SCEN / "P3_corral_repair_cu111.yaml")
    assert {c["id"]: A.claim_policy(c) for c in repair.claims}["ring_occupancy"] == "state"


@pytest.mark.parametrize("flag,kind", [("yes", "scalar"), ("allowed", "constraint")])
def test_the_knowledge_flag_is_validated(tmp_path, flag, kind):
    src = (SCEN / "P5_sader_jarvis_qplus.yaml").read_text(encoding="utf-8")
    claim = ("  - id: extra\n    kind: constraint\n    truth: force.f_min_pn\n    max: 0\n"
             if kind == "constraint" else
             "  - id: extra\n    kind: scalar\n    tol: {rel: 0.1}\n    truth: force.f_min_pn\n")
    claim += f"    knowledge: {flag}\n    evidence: {{kind: events, event: zspec}}\n"
    bad = tmp_path / "bad.yaml"
    bad.write_text(src.replace("claims:\n", "claims:\n" + claim, 1), encoding="utf-8")
    with pytest.raises(ValueError, match="knowledge"):
        Scenario.load(bad)


# ── the search itself, on synthetic pass matrices ──
def _grid(cid: str, passes: np.ndarray, **claim) -> A.ClaimGrid:
    return A.ClaimGrid({"id": cid, "kind": "scalar", **claim},
                       np.arange(passes.shape[0], dtype=float), passes)


def test_the_two_claim_search_is_exact():
    rng = np.random.default_rng(3)
    for _ in range(20):
        p1 = rng.random((9, 40)) < 0.4
        p2 = rng.random((7, 40)) < 0.5
        g1, g2 = _grid("a", p1), _grid("b", p2)
        brute = max((p1[i] & p2[j]).mean() for i in range(9) for j in range(7))
        choice, val = A.best_joint([g1, g2])
        assert val == pytest.approx(brute)
        assert A.joint_pass([g1, g2], choice).mean() == pytest.approx(val)


def test_the_pair_search_keeps_the_judges_peak_rules():
    """Two peak answers must match different peaks, and the higher claim must be higher."""
    rng = np.random.default_rng(5)
    ranks_lo = rng.integers(-1, 3, size=(8, 30))
    ranks_hi = rng.integers(-1, 3, size=(8, 30))
    lo = A.ClaimGrid({"id": "lo", "kind": "peak_in_list"}, np.arange(8.0), ranks_lo >= 0, ranks=ranks_lo)
    hi = A.ClaimGrid({"id": "hi", "kind": "peak_in_list", "distinct_from": {"claim": "lo"},
                      "greater_than": "lo"}, np.arange(8.0), ranks_hi >= 0, ranks=ranks_hi)
    brute = max(A.joint_pass([lo, hi], [i, j]).mean() for i in range(8) for j in range(8))
    choice, val = A.best_joint([lo, hi])
    assert val == pytest.approx(brute)
    assert choice[1] > choice[0]


def test_three_unlinked_claims_are_searched_exactly():
    """P5's three numbers: coordinate ascent had reported 13.7 % where the best sheet passes
    14.0 % (seeds 0-299), and ranked two tolerances the wrong way round."""
    rng = np.random.default_rng(11)
    for _ in range(10):
        ps = [rng.random((6, 50)) < 0.5 for _ in range(3)]
        grids = [_grid(f"c{k}", p) for k, p in enumerate(ps)]
        brute = max(A.joint_pass(grids, list(ch)).mean() for ch in itertools.product(range(6), repeat=3))
        choice, val = A.best_joint(grids)
        assert val == pytest.approx(brute)
        assert A.joint_pass(grids, choice).mean() == pytest.approx(val)


def test_a_linked_pair_and_a_free_claim_are_searched_exactly():
    """P3's shape: two peak answers tied by the judge's rules (different peaks, ascending) and
    a third claim (the radius) free of them."""
    rng = np.random.default_rng(7)
    for _ in range(5):
        ranks_lo = rng.integers(-1, 3, size=(6, 40))
        ranks_hi = rng.integers(-1, 3, size=(6, 40))
        lo = A.ClaimGrid({"id": "lo", "kind": "peak_in_list"}, np.arange(6.0), ranks_lo >= 0, ranks=ranks_lo)
        hi = A.ClaimGrid({"id": "hi", "kind": "peak_in_list", "distinct_from": {"claim": "lo"},
                          "greater_than": "lo"}, np.arange(6.0), ranks_hi >= 0, ranks=ranks_hi)
        rad = _grid("radius", rng.random((7, 40)) < 0.4)
        grids = [lo, hi, rad]
        brute = max(A.joint_pass(grids, [i, j, k]).mean()
                    for i in range(6) for j in range(6) for k in range(7))
        choice, val = A.best_joint(grids)
        assert val == pytest.approx(brute)
        assert A.joint_pass(grids, choice).mean() == pytest.approx(val)


def test_four_claims_give_a_lower_bound_on_the_best_sheet():
    rng = np.random.default_rng(12)
    for _ in range(5):
        ps = [rng.random((4, 50)) < 0.5 for _ in range(4)]
        grids = [_grid(f"c{k}", p) for k, p in enumerate(ps)]
        brute = max(A.joint_pass(grids, list(ch)).mean() for ch in itertools.product(range(4), repeat=4))
        start = [int(np.argmax(p.mean(axis=1))) for p in ps]
        choice, val = A.best_joint(grids)
        assert A.joint_pass(grids, start).mean() <= val + 1e-12 <= brute + 1e-9
        assert A.joint_pass(grids, choice).mean() == pytest.approx(val)


# ── on the real scenarios (few seeds: the machinery, not the numbers) ──
def test_p2_has_nothing_to_gate_and_its_constants_are_recorded():
    res = A.audit_scenario(SCEN / "P2_standing_waves_cu111.yaml", 30)
    rows = {r["id"]: r for r in res["claims"]}
    assert {r["policy"] for r in rows.values()} == {"knowledge"}
    assert res["joint_measure"] is None
    for r in rows.values():
        assert 0.0 < r["pass_frac"] <= 1.0
        assert r["truth_min"] - 1e-9 <= r["best_constant"] <= r["truth_max"] + 1e-9 or r["pass_frac"] == 0
    assert res["joint_all"]["pass_frac"] <= min(r["pass_frac"] for r in rows.values()) + 1e-12
    assert "joint, measure claims" in A.format_report(res)


def test_p1_gates_its_orientations_and_not_its_period():
    res = A.audit_scenario(SCEN / "P1_barth1990_au111.yaml", 12)
    pol = {r["id"]: r["policy"] for r in res["claims"]}
    assert pol == {"stripe_period_nm": "knowledge", "stripe_orientation_deg": "measure",
                   "domain_a_orientation_deg": "measure", "domain_b_orientation_deg": "measure"}
    assert set(res["joint_measure"]["answer"]) == {"stripe_orientation_deg", "domain_a_orientation_deg",
                                                   "domain_b_orientation_deg"}
    assert 0.0 <= res["joint_measure"]["pass_frac"] <= 1.0
