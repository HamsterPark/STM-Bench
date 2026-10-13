"""Version-1 judging rules: a report names the frame its position was read in and the data its
value came from, and the evidence rule is checked within that data."""
from __future__ import annotations


import pytest

from stmbench.harness.results import ResultSink, fold_results
from stmbench.trackB import claims as C
from stmbench.trackB.truth_criteria import judge
FRAME = {"kind": "scan_saved", "sim_s": 10.0, "idx": 1, "complete": True, "path": "session/sim001.sxm",
         "cx_m": 0.0, "cy_m": 0.0, "w_m": 60e-9, "h_m": 60e-9, "angle_deg": 0.0,
         "nx": 256, "ny": 256, "drift_m": [0.0, 0.0]}


# ── citations ─────────────────────────────────────────────────────────────────
def _episode() -> list[dict]:
    frame = {**FRAME, "path": "C:\\runs\\x\\session\\Sim007.sxm", "idx": 7}
    sts = {"kind": "sts", "sim_s": 20.0, "idx": 3, "n": 200, "v0": -0.6, "v1": 0.4}
    zs = {"kind": "zspec", "sim_s": 30.0, "idx": 2, "path": None}
    return [frame, sts,
            {"kind": "dat_saved", "sim_s": 20.1, "path": "/data/s/BiasSpec003.dat", "sts_idx": 3},
            zs, {"kind": "dat_saved", "sim_s": 30.1, "path": "/data/s/ZSpec002.dat", "zspec_idx": 2}]


@pytest.mark.parametrize("ref,kind,idx", [("Sim007.sxm", "scan_saved", 7), ("sim007.SXM", "scan_saved", 7),
                                          ("frame:7", "scan_saved", 7), ("BiasSpec003.dat", "sts", 3),
                                          ("sts:3", "sts", 3), ("ZSpec002.dat", "zspec", 2),
                                          ("zspec:2", "zspec", 2)])
def test_citations_resolve_by_file_name_or_index(ref, kind, idx):
    ev = C.resolve_citation(_episode(), ref)
    assert ev is not None and ev["kind"] == kind and ev["idx"] == idx


@pytest.mark.parametrize("ref", ["sim008.sxm", "frame:8", "sts:x", "", "Sim007"])
def test_an_unknown_citation_resolves_to_nothing(ref):
    assert C.resolve_citation(_episode(), ref) is None


def test_the_result_channel_keeps_frame_and_evidence():
    folded = fold_results([{"args": {"claim_id": "a", "value": 1.0, "frame": "sim001.sxm",
                                     "evidence": ["sim001.sxm", "sts:4"]}},
                           {"args": {"claim_id": "b", "value": 2.0, "evidence": "sim002.sxm"}}])
    assert folded["claims"]["a"]["frame"] == "sim001.sxm"
    assert folded["claims"]["a"]["evidence"] == ["sim001.sxm", "sts:4"]
    assert folded["claims"]["b"]["evidence"] == ["sim002.sxm"]
    sink = ResultSink()
    sink.report("a", 1.0, x_nm=1.0, y_nm=2.0, frame="f.sxm", evidence=["f.sxm"])
    assert sink.folded()["claims"]["a"]["frame"] == "f.sxm"


# ── claims_verified under the version-1 rules ────────────────────────────────
def _angle_claim():
    return {"id": "stripe_orientation_deg", "kind": "angle", "tol": {"abs": 6, "mod": 180},
            "truth": "orientation_at", "position": "required",
            "evidence": {"kind": "frame_covers", "min_fov_nm": 40, "max_nm_per_px": 0.5}}


def _report(cid, value, **kw):
    sink = ResultSink()
    sink.report(cid, value, **kw)
    return sink.folded()


def test_v1_reads_a_position_in_the_frame_the_report_names():
    near = {**FRAME, "idx": 1, "path": "s/sim001.sxm"}
    far = {**FRAME, "idx": 2, "sim_s": 20.0, "path": "s/sim002.sxm", "cx_m": 200e-9}
    seen = []

    def orient(x, y):
        seen.append((x, y))
        return 130.0

    common = dict(claims=[_angle_claim()], events=[near, far], orientation_at=orient, rules_version=1)
    # no frame named: the position has no reading, so the value cannot be judged
    v = judge("claims_verified", {}, report=_report("stripe_orientation_deg", 130.0, x_nm=5.0, y_nm=2.0),
              **common)
    assert not v.success and v.details["claims"]["stripe_orientation_deg"]["frame_reason"] == "no_reference_frame"
    # a frame that does not contain the place
    v = judge("claims_verified", {}, report=_report("stripe_orientation_deg", 130.0, x_nm=5.0, y_nm=2.0,
                                                    frame="sim002.sxm"), **common)
    assert not v.success
    # the right frame
    v = judge("claims_verified", {}, report=_report("stripe_orientation_deg", 130.0, x_nm=5.0, y_nm=2.0,
                                                    frame="sim001.sxm"), **common)
    assert v.success and seen
    # an episode judged before the rule card keeps the earlier rule
    v = judge("claims_verified", {}, claims=[_angle_claim()], events=[near, far], orientation_at=orient,
              report=_report("stripe_orientation_deg", 130.0, x_nm=5.0, y_nm=2.0))
    assert v.success and v.details["rules_version"] == 0


def _any_target_truth():
    return {"adatoms": {"target": {"rule": "any", "d_nm": [4.0, 0.0], "example": {"id": 5},
                                   "candidates": [{"id": 5, "now_nm": [10.0, 0.0],
                                                   "on_goal": True, "isolated": True}]}}}


def _target_claim():
    return {"id": "target_site", "kind": "position", "tol": {"abs_nm": 0.16, "report_nm": 0.6},
            "truth": "adatoms.target", "position": "required",
            "evidence": {"kind": "frame_covers_after", "min_fov_nm": 6, "max_nm_per_px": 0.2,
                         "after_event": "adatom_hop"}}


def test_v1_the_verification_frame_must_follow_the_last_hop():
    before = {**FRAME, "idx": 1, "sim_s": 40.0, "path": "s/a001.sxm", "cx_m": 10e-9,
              "w_m": 10e-9, "h_m": 10e-9, "nx": 64, "ny": 64}
    hop = {"kind": "adatom_hop", "cause": "folme", "sim_s": 50.0}
    after = {**before, "idx": 2, "sim_s": 60.0, "path": "s/a002.sxm"}
    events = [before, hop, after]
    for frame, ok in (("a001.sxm", False), ("a002.sxm", True)):
        v = judge("claims_verified", _any_target_truth(), claims=[_target_claim()], events=events,
                  report=_report("target_site", 10.0, x_nm=10.0, y_nm=0.0, frame=frame), rules_version=1)
        assert v.success is ok, (frame, v.details["claims"]["target_site"])


def test_v1_a_site_all_but_as_near_as_the_nearest_counts():
    """Which site is nearest to birth + d can turn on the lattice constant's last digit (P4
    notes, 2026-10-13): a site at most ``site_tie_nm`` farther from the point than the
    nearest one counts too; a site a spacing off still does not."""
    hop = {"kind": "adatom_hop", "cause": "folme", "sim_s": 50.0}
    after = {**FRAME, "idx": 2, "sim_s": 60.0, "path": "s/a002.sxm", "cx_m": 10e-9,
             "w_m": 10e-9, "h_m": 10e-9, "nx": 64, "ny": 64}
    claim = {**_target_claim(), "tol": {"site_tie_nm": 0.02, "report_nm": 0.6}}
    for d_now, ok in ((0.130, True), (0.145, True), (0.160, False), (0.300, False)):
        truth = _any_target_truth()
        truth["adatoms"]["target"]["candidates"][0].update(on_goal=False, goal_dist_nm=d_now,
                                                           goal_nearest_nm=0.130)
        v = judge("claims_verified", truth, claims=[claim], events=[hop, after],
                  report=_report("target_site", 10.0, x_nm=10.0, y_nm=0.0, frame="a002.sxm"), rules_version=1)
        assert v.success is ok, (d_now, v.details["claims"]["target_site"])
    # without the field the rule is the nearest site only
    truth["adatoms"]["target"]["candidates"][0].update(goal_dist_nm=0.131)
    v = judge("claims_verified", truth, claims=[_target_claim()], events=[hop, after],
              report=_report("target_site", 10.0, x_nm=10.0, y_nm=0.0, frame="a002.sxm"), rules_version=1)
    assert not v.success


def _period_claim():
    return {"id": "period_nm", "kind": "scalar", "tol": {"abs": 0.4}, "truth": "herringbone.period_nm",
            "evidence": {"kind": "frame", "min_fov_nm": 40, "max_nm_per_px": 0.5}}


def test_v1_evidence_is_checked_in_the_cited_data_only():
    good = {**FRAME, "idx": 1, "path": "s/good001.sxm"}
    coarse = {**FRAME, "idx": 2, "path": "s/coarse002.sxm", "nx": 64, "ny": 64}
    truth = {"herringbone": {"period_nm": 6.4}}
    cases = [([], False), (["coarse002.sxm"], False), (["good001.sxm"], True), (["nope.sxm"], False)]
    for cited, ok in cases:
        v = judge("claims_verified", truth, claims=[_period_claim()], events=[good, coarse],
                  report=_report("period_nm", 6.5, evidence=cited), rules_version=1)
        assert v.success is ok, cited
    v = judge("claims_verified", truth, claims=[_period_claim()], events=[good, coarse],
              report=_report("period_nm", 6.5, evidence=["nope.sxm"]), rules_version=1)
    assert v.details["claims"]["period_nm"]["unresolved_citations"] == ["nope.sxm"]


def test_v1_spectra_must_be_among_the_cited_ones():
    rule = {"kind": "spectra_near_scatterer", "min_n": 2, "min_positions": 2, "near_nm": 8.0,
            "min_points": 64, "v_lo_max": -0.5, "v_hi_min": 0.1}
    claim = {"id": "e0_mev", "kind": "scalar", "tol": {"abs": 15}, "truth": "surface_state.e0_mev",
             "evidence": rule}
    sts = [{"kind": "sts", "sim_s": 5.0 + i, "idx": i + 1, "n": 200, "v0": -0.6, "v1": 0.4, "jump": False,
            "sx_m": i * 1e-9, "sy_m": 0.0, "near_kind": "step", "near_nm": 2.0 + i} for i in range(3)]
    truth = {"surface_state": {"e0_mev": -440.0}}
    for cited, ok in ((["sts:1"], False), (["sts:1", "sts:2"], True)):
        v = judge("claims_verified", truth, claims=[claim], events=sts,
                  report=_report("e0_mev", -438.0, evidence=cited), rules_version=1)
        assert v.success is ok, cited
