"""Judge conformance on the real paper scenarios: the truth, reported with the data the rule
card asks for, reproduces every paper; a value just inside a tolerance passes and one just
outside fails. These run the version-1 rules (rules_version=1) end to end on built worlds."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from stmbench.harness.results import ResultSink
from stmbench.trackB import claims as C
from stmbench.trackB.rules import RULES_VERSION
from stmbench.trackB.truth_criteria import judge
from stmsim.scenario import Scenario

SCEN = Path(__file__).resolve().parent.parent / "stmbench" / "trackB" / "scenarios"


def _frame(idx: int, cx_nm: float, cy_nm: float, size_nm: float, px: int, sim_s: float) -> dict:
    return {"kind": "scan_saved", "sim_s": sim_s, "idx": idx, "complete": True, "path": f"s/sim{idx:03d}.sxm",
            "cx_m": cx_nm * 1e-9, "cy_m": cy_nm * 1e-9, "w_m": size_nm * 1e-9, "h_m": size_nm * 1e-9,
            "angle_deg": 0.0, "nx": px, "ny": px, "drift_m": [0.0, 0.0], "scan_dir": "down",
            "row_offsets_nm": [[0.0, 0.0]] * px}


def _two_domains(world) -> tuple[tuple[float, float], tuple[float, float]]:
    """Two places (nm) whose stripes differ by at least 30° — two rotational domains."""
    surf = world.surface
    grid = [(x, y) for x in range(-900, 901, 150) for y in range(-900, 901, 150)]
    first = (0.0, 0.0)
    a = surf.orientation_at(0.0, 0.0)["stripe_deg"]
    for x, y in grid:
        b = surf.orientation_at(x * 1e-9, y * 1e-9)["stripe_deg"]
        if C.angle_diff(a, b) >= 35.0:
            return first, (float(x), float(y))
    raise AssertionError("no second domain within ±900 nm")


def _oracle(name: str, tmp_path):
    """(scenario, truth, report, events, orientation_at) with the truth reported as the rule card asks."""
    sc = Scenario.load(SCEN / name)
    w = sc.build_world(0, session_dir=tmp_path / "s")
    truth = w.truth()
    sink = ResultSink()
    events: list[dict] = []
    orient = None
    if sc.family == "P1":
        (ax, ay), (bx, by) = _two_domains(w)
        big = _frame(1, 0.0, 0.0, 120.0, 512, 10.0)
        fa, fb = _frame(2, ax, ay, 50.0, 256, 20.0), _frame(3, bx, by, 50.0, 256, 30.0)
        events += [big, fa, fb]
        surf = w.surface
        orient = (lambda x, y: surf.orientation_at(x * 1e-9, y * 1e-9))
        sink.report("stripe_period_nm", C.truth_at_path(truth, "herringbone.period_along_arm_nm"),
                    evidence=["sim001.sxm"])
        for cid, (x, y), f in (("stripe_orientation_deg", (ax, ay), "sim002.sxm"),
                               ("domain_a_orientation_deg", (ax, ay), "sim002.sxm"),
                               ("domain_b_orientation_deg", (bx, by), "sim003.sxm")):
            sink.report(cid, orient(x, y)["stripe_deg"], x_nm=x, y_nm=y, frame=f, evidence=[f])
    elif sc.family == "P2":
        for i in range(8):
            events.append({"kind": "sts", "sim_s": 10.0 + i, "idx": i + 1, "n": 200, "v0": -0.6, "v1": 0.4,
                           "jump": False, "sx_m": i * 1e-9, "sy_m": 0.0, "near_kind": "step", "near_nm": 1.0 + i})
        cited = [f"sts:{i + 1}" for i in range(8)]
        for cid in ("e0_mev", "m_eff"):
            claim = next(c for c in sc.claims if c["id"] == cid)
            sink.report(cid, C.truth_at_path(truth, claim["truth"]), evidence=cited)
    elif sc.family == "P3":
        peaks = C.truth_at_path(truth, "corral.peaks_mev")
        events.append({"kind": "sts", "sim_s": 10.0, "idx": 1, "n": 220, "v0": -0.6, "v1": 0.4, "jump": False,
                       "sx_m": 0.0, "sy_m": 0.0, "near_kind": "adatom", "near_nm": 6.0,
                       "corral_centre_dist_nm": 0.1})
        sink.report("peak_lo_mev", peaks[0], evidence=["sts:1"])
        sink.report("peak_hi_mev", peaks[1], evidence=["sts:1"])
    elif sc.family == "P5":
        events += [{"kind": "zspec", "sim_s": 10.0, "idx": 1, "near_kind": "adatom", "near_nm": 0.05,
                    "jump": False, "contact": False, "df_span_beyond_min_pm": 120.0},
                   {"kind": "zspec", "sim_s": 20.0, "idx": 2, "near_kind": "none", "near_nm": 9.0,
                    "jump": False, "contact": False, "df_span_beyond_min_pm": 120.0}]
        for c in sc.claims:
            sink.report(c["id"], C.truth_at_path(truth, c["truth"]), evidence=["zspec:1", "zspec:2"])
    return sc, truth, sink.folded(), events, orient


ORACLE_PAPERS = ["P1_barth1990_au111.yaml", "P2_standing_waves_cu111.yaml", "P3_corral_cu111.yaml",
                 "P5_sader_jarvis_qplus.yaml"]


@pytest.mark.parametrize("name", ORACLE_PAPERS)
def test_the_truth_reported_as_the_card_asks_reproduces_the_paper(name, tmp_path):
    sc, truth, report, events, orient = _oracle(name, tmp_path)
    v = judge("claims_verified", truth, claims=sc.claims, report=report, events=events,
              orientation_at=orient, rules_version=RULES_VERSION)
    bad = {cid: d for cid, d in v.details["claims"].items() if not (d["value_ok"] and d["evidence_ok"])}
    assert v.success and not bad, bad


@pytest.mark.parametrize("name", ORACLE_PAPERS)
def test_the_same_report_without_its_citations_fails_every_cited_claim(name, tmp_path):
    sc, truth, report, events, orient = _oracle(name, tmp_path)
    for entry in report["claims"].values():
        entry.pop("evidence", None)
    v = judge("claims_verified", truth, claims=sc.claims, report=report, events=events,
              orientation_at=orient, rules_version=RULES_VERSION)
    for c in sc.claims:
        if (c.get("evidence") or {}).get("kind") in ("frame", "spectra_near_scatterer",
                                                      "sts_at_corral_centre", "zspec_pair"):
            assert not v.details["claims"][c["id"]]["evidence_ok"], c["id"]


@pytest.mark.parametrize("name", ["P1_barth1990_au111.yaml", "P2_standing_waves_cu111.yaml",
                                  "P5_sader_jarvis_qplus.yaml"])
def test_scalar_tolerances_hold_at_their_edges(name, tmp_path):
    sc, truth, report, events, orient = _oracle(name, tmp_path)
    for c in [c for c in sc.claims if c["kind"] == "scalar"]:
        t = float(C.truth_at_path(truth, c["truth"]))
        tol = c["tol"]
        lim = max(float(tol.get("abs", 0.0) or 0.0), float(tol.get("rel", 0.0) or 0.0) * abs(t))
        for sign in (1, -1):
            for factor, ok in ((0.999, True), (1.001, False)):
                rep = {"claims": {**report["claims"], c["id"]: {**report["claims"][c["id"]],
                                                                 "value": t + sign * factor * lim}}}
                v = judge("claims_verified", truth, claims=sc.claims, report=rep, events=events,
                          orientation_at=orient, rules_version=RULES_VERSION)
                assert v.details["claims"][c["id"]]["value_ok"] is ok, (c["id"], sign, factor)


def test_a_peak_just_outside_its_window_matches_nothing(tmp_path):
    sc, truth, report, events, orient = _oracle("P3_corral_cu111.yaml", tmp_path)
    peaks = C.truth_at_path(truth, "corral.peaks_mev")
    lim = next(c for c in sc.claims if c["id"] == "peak_lo_mev")["tol"]["abs"]
    assert peaks[1] - peaks[0] > 2.5 * lim            # the next peak is out of reach of the probe
    for value, ok in ((peaks[0] - 0.999 * lim, True), (peaks[0] - 1.001 * lim, False)):
        rep = {"claims": {**report["claims"], "peak_lo_mev": {**report["claims"]["peak_lo_mev"], "value": value}}}
        v = judge("claims_verified", truth, claims=sc.claims, report=rep, events=events,
                  orientation_at=orient, rules_version=RULES_VERSION)
        assert v.details["claims"]["peak_lo_mev"]["value_ok"] is ok, value
    assert not math.isnan(peaks[0])
