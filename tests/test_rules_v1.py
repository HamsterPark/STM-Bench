"""Version-1 judging rules: a report names the frame its position was read in and the data its
value came from, and the evidence rule is checked within that data. They apply to every rules
version from 1 on: version 2 reworded P4's card and left the judge alone (the last tests)."""
from __future__ import annotations


import pytest

from stmbench.trackB.results import ResultSink, fold_results
from stmbench.trackB import claims as C
from stmbench.trackB.rules import RULES_VERSION
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


def _verification(**kw) -> dict:
    return {**FRAME, "idx": 2, "sim_s": 60.0, "t_start_sim": 55.0, "path": "s/a002.sxm", "cx_m": 10e-9,
            "w_m": 10e-9, "h_m": 10e-9, "nx": 64, "ny": 64, **kw}


def _verdict(events, *, frame="a002.sxm", version=1) -> bool:
    return judge("claims_verified", _any_target_truth(), claims=[_target_claim()], events=events,
                 report=_report("target_site", 10.0, x_nm=10.0, y_nm=0.0, frame=frame),
                 rules_version=version).success


def test_v1_a_frame_scanned_without_tunnelling_verifies_nothing():
    """Withdrawn, or with the Z loop off, a scan is a flat sheet of noise (~2 pm) that covers
    the right place at the right size: a run that withdrew before its verification frame and
    reported the right place from earlier passed (Codex review, 2026-10-14)."""
    hop = {"kind": "adatom_hop", "cause": "folme", "sim_s": 50.0}
    for state, ok in (({"feedback": True, "withdrawn": False}, True),
                      ({"feedback": False, "withdrawn": True}, False),
                      ({"feedback": False, "withdrawn": False}, False),
                      ({}, True)):                      # a ledger from before the state was recorded
        assert _verdict([hop, _verification(**state)]) is ok, state


def test_v1_the_verification_frame_is_timed_by_when_it_was_scanned():
    """A frame shows the surface as it was when it was scanned. One taken before the atom
    moved and saved (again) after it shows the old place (Codex review, 2026-10-14), and one
    during which imaging moved the atom is not after that hop either."""
    hop = {"kind": "adatom_hop", "cause": "folme", "sim_s": 50.0}
    assert _verdict([hop, _verification(t_start_sim=52.0)])
    assert not _verdict([hop, _verification(t_start_sim=40.0)])          # scanned before, saved after
    # imaging at a low resistance moved the atom in a row of the frame itself: the event carries
    # the frame's start as its time and the row's as ``row_sim_s``
    in_frame = {"kind": "adatom_hop", "cause": "scan", "sim_s": 55.0, "row_sim_s": 57.5}
    assert not _verdict([hop, in_frame, _verification(t_start_sim=55.0)])
    assert _verdict([hop, in_frame, _verification(idx=3, path="s/a002.sxm", t_start_sim=58.0, sim_s=70.0)])


def test_v1_a_site_exactly_site_tie_nm_farther_than_the_nearest_counts():
    """The card says "by at most": the judge accepts a margin ``<= site_tie_nm``, not ``<``. Binary
    fractions, so the margin is exactly the tie (0.375 - 0.125 = 0.25)."""
    hop = {"kind": "adatom_hop", "cause": "folme", "sim_s": 50.0}
    claim = {**_target_claim(), "tol": {"site_tie_nm": 0.25, "report_nm": 0.6}}
    for d_now, ok in ((0.375, True), (0.3751, False)):
        truth = _any_target_truth()
        truth["adatoms"]["target"]["candidates"][0].update(on_goal=False, goal_dist_nm=d_now,
                                                           goal_nearest_nm=0.125)
        v = judge("claims_verified", truth, claims=[claim], events=[hop, _verification()],
                  report=_report("target_site", 10.0, x_nm=10.0, y_nm=0.0, frame="a002.sxm"), rules_version=1)
        assert v.success is ok, (d_now, v.details["claims"]["target_site"])


def test_v1_the_report_is_matched_to_the_moved_atom_nearest_to_it():
    """The card says so: a moved atom that is farther from the reported place, but would pass, does not
    rescue a nearer one that does not (both are within ``report_nm``)."""
    hop = {"kind": "adatom_hop", "cause": "folme", "sim_s": 50.0}
    truth = _any_target_truth()
    truth["adatoms"]["target"]["candidates"] = [
        {"id": 5, "now_nm": [10.0, 0.0], "on_goal": False, "isolated": True},
        {"id": 6, "now_nm": [10.4, 0.0], "on_goal": True, "isolated": True}]
    for x_nm, ok, matched in ((10.1, False, 5), (10.3, True, 6)):
        v = judge("claims_verified", truth, claims=[_target_claim()], events=[hop, _verification()],
                  report=_report("target_site", x_nm, x_nm=x_nm, y_nm=0.0, frame="a002.sxm"), rules_version=1)
        assert v.success is ok and v.details["claims"]["target_site"]["atom_id"] == matched, \
            v.details["claims"]["target_site"]


def _battery(version: int) -> list[bool]:
    """Four verdicts that only the strict rules decide the way they do: a position without the frame
    it was read in, a frame scanned before the last hop, the right frame, and a value whose evidence
    is not cited."""
    hop = {"kind": "adatom_hop", "cause": "folme", "sim_s": 50.0}
    early = _verification(idx=1, path="s/a001.sxm", sim_s=45.0, t_start_sim=40.0)
    good = {**FRAME, "idx": 1, "path": "s/good001.sxm"}
    return [_verdict([hop, _verification()], frame=None, version=version),
            _verdict([early, hop, _verification()], frame="a001.sxm", version=version),
            _verdict([hop, _verification()], version=version),
            judge("claims_verified", {"herringbone": {"period_nm": 6.4}}, claims=[_period_claim()],
                  events=[good], report=_report("period_nm", 6.5, evidence=[]),
                  rules_version=version).success]


def test_every_rules_version_from_1_on_is_judged_by_the_same_rules():
    """Version 2 of the rule card reworded P4's card and left the judge alone (``rules_version >= 1`` is
    the strict rules), so one episode gets one verdict under every version from 1 on. The battery tells
    the rules apart: an episode from before the card is still judged as it was (any covering frame, any
    data of the episode), and passes three cases the strict rules fail."""
    versions = sorted({1, 2, RULES_VERSION})
    assert [_battery(v) for v in versions] == [[False, False, True, False]] * len(versions)
    assert _battery(0) == [True, True, True, True]


def test_one_acquisition_named_several_times_counts_once():
    """``min_n`` counts acquisitions: ``sts:1``, its .dat name and a repeat are one spectrum
    (external review of 39fcd13: one spectrum cited eight times passed ``min_n`` 8). Naming a
    frame both as ``frame`` and in ``evidence`` is still one frame, and still fine."""
    rule = {"kind": "spectra_near_scatterer", "min_n": 2, "min_positions": 1, "near_nm": 8.0,
            "min_points": 64, "v_lo_max": -0.5, "v_hi_min": 0.1}
    claim = {"id": "e0_mev", "kind": "scalar", "tol": {"abs": 15}, "truth": "surface_state.e0_mev",
             "evidence": rule}
    sts = [{"kind": "sts", "sim_s": 5.0 + i, "idx": i + 1, "n": 200, "v0": -0.6, "v1": 0.4, "jump": False,
            "sx_m": 0.0, "sy_m": 0.0, "near_kind": "step", "near_nm": 2.0} for i in range(2)]
    events = sts + [{"kind": "dat_saved", "sim_s": 5.1, "path": "/d/BiasSpec001.dat", "sts_idx": 1}]
    truth = {"surface_state": {"e0_mev": -440.0}}
    for cited, ok, n in ((["sts:1", "sts:1", "BiasSpec001.dat"], False, 1),
                         (["sts:1", "BiasSpec001.dat", "sts:2"], True, 2)):
        v = judge("claims_verified", truth, claims=[claim], events=events,
                  report=_report("e0_mev", -438.0, evidence=cited), rules_version=1)
        info = v.details["claims"]["e0_mev"]
        assert v.success is ok, cited
        assert info["cited"] == cited and info["n_distinct_cited"] == n
    # the same frame as ``frame`` and in ``evidence``: one frame, not a failure
    v = judge("claims_verified", {"herringbone": {"period_nm": 6.4}}, claims=[_period_claim()], events=[FRAME],
              report=_report("period_nm", 6.5, evidence=["sim001.sxm"], frame="sim001.sxm"), rules_version=1)
    assert v.success and v.details["claims"]["period_nm"]["n_distinct_cited"] == 1
