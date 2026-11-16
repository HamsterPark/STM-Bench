"""P3's repair variant as a calibrated task: what the card says, what the radius claim's evidence
rejects, and that a fixed answer sheet does not carry over."""
from __future__ import annotations

from pathlib import Path

import pytest

from stmbench.trackB import rules as R
from stmbench.trackB.claims import check_evidence
from stmbench.trackB.truth_criteria import judge
from stmsim.scenario import Scenario

SCEN = Path(__file__).resolve().parent.parent / "stmbench" / "trackB" / "scenarios"
REPAIR = SCEN / "P3_corral_repair_cu111.yaml"
WHOLE = SCEN / "P3_corral_cu111.yaml"


def _claims(sc):
    return {c["id"]: c for c in sc.claims}


def test_the_repair_asks_for_the_measuring_variants_numbers_and_the_ring():
    rep, whole = _claims(Scenario.load(REPAIR)), _claims(Scenario.load(WHOLE))
    assert set(rep) == {"ring_occupancy", "peak_lo_mev", "peak_hi_mev", "radius_nm"}
    for cid in ("peak_lo_mev", "peak_hi_mev", "radius_nm"):
        assert rep[cid]["tol"] == whole[cid]["tol"], cid          # 10 meV, 0.15 nm: as P3 calibrated
    assert rep["ring_occupancy"]["min"] == 0.97


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_both_cards_state_the_same_numbers_and_the_spares_as_many_as_the_gap(lang):
    text = R.task_text(Scenario.load(REPAIR), lang)
    for number in ("10", "0.15", "22", "0.97"):
        assert number in text, (lang, number)
    # the ring's atoms cannot be counted one by one; the spares can, and the task says how many there are
    # (only the count: how the ring looks is the agent's to see); it asks for the radius, now a claim
    if lang == "en":
        assert "as many spare atoms as atoms missing" in text
        assert "measure the radius of the repaired corral" in text
        assert "cannot be counted" not in text
    else:
        assert "备用原子的个数与环上缺的个数相同" in text
        assert "量出围栏的半径" in text
        assert "数不出" not in text
    assert "after" in text if lang == "en" else "之后" in text


def _frame(idx: int, *, t0: float, fov_nm: float = 30.0, nm_per_px: float = 0.1) -> dict:
    n = int(fov_nm / nm_per_px)
    return {"kind": "scan_saved", "idx": idx, "sim_s": t0 + 30.0, "t_start_sim": t0, "complete": True,
            "feedback": True, "withdrawn": False, "cx_m": 8e-9, "cy_m": -6e-9, "w_m": fov_nm * 1e-9,
            "h_m": fov_nm * 1e-9, "angle_deg": 0.0, "nx": n, "ny": n, "drift_m": [0.0, 0.0]}


def test_the_radius_needs_a_big_fine_frame_scanned_after_the_last_hop():
    rule = _claims(Scenario.load(REPAIR))["radius_nm"]["evidence"]
    truth = {"corral": {"centre_nm": [8.0, -6.0]}}
    hop = {"kind": "adatom_hop", "cause": "folme", "sim_s": 100.0}
    good = _frame(2, t0=120.0)
    assert check_evidence(rule, events=[hop, good], report={}, truth=truth)[0]
    before = _frame(1, t0=50.0)                              # the ring as it was: not a repaired corral
    assert not check_evidence(rule, events=[hop, before], report={}, truth=truth)[0]
    assert not check_evidence(rule, events=[before], report={}, truth=truth)[0]          # nothing was moved
    assert not check_evidence(rule, events=[hop, _frame(3, t0=120.0, fov_nm=20.0)], report={}, truth=truth)[0]
    assert not check_evidence(rule, events=[hop, _frame(4, t0=120.0, nm_per_px=0.2)], report={}, truth=truth)[0]


def test_the_seed_0_sheet_does_not_pass_the_radius_across_seeds(tmp_path):
    """Radius ±0.15 nm over a ring drawn at 5.5-9 nm: the sheet that matches seed 0 matches few others."""
    sc = Scenario.load(REPAIR)
    r0 = sc.build_world(0, session_dir=tmp_path / "s0").truth()["corral"]["radius_nm"]
    close = sum(abs(sc.build_world(s, session_dir=tmp_path / f"s{s}").truth()["corral"]["radius_nm"] - r0) <= 0.15
                for s in range(1, 13))
    assert close <= 3, close


def test_a_broken_ring_measured_by_a_perfect_sheet_still_fails_the_state_claim(tmp_path):
    """The values right, the radius frame and spectra after a hop, but the ring never closed."""
    sc = Scenario.load(REPAIR)
    w = sc.build_world(0, session_dir=tmp_path / "s")
    truth = w.truth()
    c = truth["corral"]
    report = {"claims": {"peak_lo_mev": {"value": c["peaks_mev"][0]}, "peak_hi_mev": {"value": c["peaks_mev"][1]},
                         "radius_nm": {"value": c["radius_nm"]}}}
    hop = {"kind": "adatom_hop", "cause": "folme", "sim_s": 5.0}
    sts = [{"kind": "sts", "sim_s": 200.0, "n": 200, "v0": -0.6, "v1": 0.4, "jump": False, "sx_m": 0.0,
            "sy_m": 0.0, "near_kind": "none", "near_nm": 9.0, "corral_centre_dist_nm": 0.2}]
    frame = _frame(1, t0=100.0)
    v = judge("claims_verified", truth, claims=sc.claims, report=report, events=[hop, frame, *sts])
    assert not v.success
    assert not v.details["claims"]["ring_occupancy"]["value_ok"]
