"""``claims_verified``: reported numbers, hidden truth, and the evidence behind them."""
from __future__ import annotations

import math

import pytest

from stmbench.harness.results import ResultSink, fold_results, missing_claims
from stmbench.trackB.claims import (angle_diff, check_evidence, frame_covering,
                                    frame_covering_sample, to_sample_frame)
from stmbench.trackB.truth_criteria import CRITERIA, judge

FRAME = {"kind": "scan_saved", "sim_s": 10.0, "idx": 1, "complete": True,
         "cx_m": 0.0, "cy_m": 0.0, "w_m": 60e-9, "h_m": 60e-9, "angle_deg": 0.0,
         "nx": 256, "ny": 256, "drift_m": [0.0, 0.0]}


def _scalar_claim(**kw):
    base = {"id": "period_nm", "kind": "scalar", "tol": {"abs": 0.4},
            "truth": "herringbone.period_nm",
            "evidence": {"kind": "frame", "min_fov_nm": 40, "max_nm_per_px": 0.5}}
    base.update(kw)
    return base


def _truth(**kw):
    t = {"herringbone": {"period_nm": 6.4, "chevron_period_nm": 28.0}}
    t.update(kw)
    return t


def _report(**vals):
    sink = ResultSink()
    for cid, v in vals.items():
        if isinstance(v, tuple):
            sink.report(cid, v[0], x_nm=v[1], y_nm=v[2])
        else:
            sink.report(cid, v)
    return sink.folded()


def test_registered():
    assert "claims_verified" in CRITERIA


def test_scalar_within_tolerance_with_evidence():
    v = judge("claims_verified", _truth(), claims=[_scalar_claim()],
              report=_report(period_nm=6.55), events=[FRAME])
    assert v.success and v.partial == 1.0
    assert v.details["claims"]["period_nm"]["value_ok"]


def test_scalar_outside_tolerance_fails():
    v = judge("claims_verified", _truth(), claims=[_scalar_claim()],
              report=_report(period_nm=7.2), events=[FRAME])
    assert not v.success and v.partial == 0.0


def test_right_number_without_the_measurement_is_not_a_reproduction():
    """The literature value with no frame behind it is a recital, not a result."""
    v = judge("claims_verified", _truth(), claims=[_scalar_claim()],
              report=_report(period_nm=6.4), events=[])
    assert not v.success
    assert v.details["claims"]["period_nm"]["value_ok"]
    assert not v.details["claims"]["period_nm"]["evidence_ok"]


def test_a_frame_too_coarse_or_too_small_is_not_evidence():
    small = {**FRAME, "w_m": 20e-9, "h_m": 20e-9}
    coarse = {**FRAME, "nx": 32}
    partial_frame = {**FRAME, "complete": False}
    for bad in (small, coarse, partial_frame):
        v = judge("claims_verified", _truth(), claims=[_scalar_claim()],
                  report=_report(period_nm=6.4), events=[bad])
        assert not v.success


def test_no_report_is_a_failure_not_an_abstention():
    v = judge("claims_verified", _truth(), claims=[_scalar_claim()], report={}, events=[FRAME])
    assert not v.success and v.partial == 0.0 and v.details["no_report"]


def test_partial_is_the_fraction_of_claims_reproduced():
    claims = [_scalar_claim(), _scalar_claim(id="chevron_period_nm", tol={"rel": 0.2},
                                             truth="herringbone.chevron_period_nm")]
    v = judge("claims_verified", _truth(), claims=claims,
              report=_report(period_nm=6.4, chevron_period_nm=99.0), events=[FRAME])
    assert v.partial == pytest.approx(0.5) and not v.success


def test_thresholds_can_override_a_claim_tolerance():
    v = judge("claims_verified", _truth(), {"claim_tol": {"period_nm": {"abs": 1.5}}},
              claims=[_scalar_claim()], report=_report(period_nm=7.2), events=[FRAME])
    assert v.success


# ── angle + position ─────────────────────────────────────────────────────────
def _angle_claim(cid="stripe_orientation_deg", **kw):
    base = {"id": cid, "kind": "angle", "tol": {"abs": 6, "mod": 180},
            "truth": "orientation_at", "position": "required",
            "evidence": {"kind": "frame_covers", "min_fov_nm": 40, "max_nm_per_px": 0.5}}
    base.update(kw)
    return base


def test_angle_uses_the_orientation_at_the_reported_place():
    calls = []

    def orientation_at(x, y):
        calls.append((x, y))
        return 130.0

    v = judge("claims_verified", _truth(), claims=[_angle_claim()],
              report=_report(stripe_orientation_deg=(128.0, 5.0, -3.0)), events=[FRAME],
              orientation_at=orientation_at)
    assert v.success and calls == [(5.0, -3.0)]


def test_angle_wraps_modulo_180():
    v = judge("claims_verified", _truth(), claims=[_angle_claim()],
              report=_report(stripe_orientation_deg=(2.0, 0.0, 0.0)), events=[FRAME],
              orientation_at=lambda x, y: 179.0)
    assert v.success                                     # 2° and 179° are the same line
    assert angle_diff(2.0, 179.0) == pytest.approx(3.0)


def test_position_outside_every_frame_is_not_evidence():
    v = judge("claims_verified", _truth(), claims=[_angle_claim()],
              report=_report(stripe_orientation_deg=(130.0, 500.0, 0.0)), events=[FRAME],
              orientation_at=lambda x, y: 130.0)
    assert not v.success
    assert not v.details["claims"]["stripe_orientation_deg"]["evidence_ok"]


def test_a_position_claim_without_coordinates_is_not_reported():
    v = judge("claims_verified", _truth(), claims=[_angle_claim()],
              report=_report(stripe_orientation_deg=130.0), events=[FRAME],
              orientation_at=lambda x, y: 130.0)
    assert not v.details["checks"]["stripe_orientation_deg.reported"]


def test_two_domains_must_really_be_two_domains():
    """Both answers right, both in the same domain: that is one domain, reported twice."""
    claims = [_angle_claim("domain_a_orientation_deg"),
              _angle_claim("domain_b_orientation_deg",
                           distinct_from={"claim": "domain_a_orientation_deg", "min_sep_deg": 30})]
    same = judge("claims_verified", _truth(), claims=claims,
                 report=_report(domain_a_orientation_deg=(70.0, -10.0, 0.0),
                                domain_b_orientation_deg=(70.0, 10.0, 0.0)),
                 events=[FRAME], orientation_at=lambda x, y: 70.0)
    assert not same.success
    assert same.details["claims"]["domain_b_orientation_deg"]["distinct_from_failed"]
    two = judge("claims_verified", _truth(), claims=claims,
                report=_report(domain_a_orientation_deg=(70.0, -10.0, 0.0),
                               domain_b_orientation_deg=(130.0, 10.0, 0.0)),
                events=[FRAME], orientation_at=lambda x, y: 70.0 if x < 0 else 130.0)
    assert two.success


# ── drift mapping ────────────────────────────────────────────────────────────
def test_reported_position_is_mapped_through_the_frame_drift():
    drifted = {**FRAME, "drift_m": [7e-9, -3e-9]}
    fr = frame_covering([drifted], 5.0, 2.0, {"min_fov_nm": 40})
    assert fr is not None
    assert to_sample_frame(fr, 5.0, 2.0) == pytest.approx((12.0, -1.0))
    seen = []
    judge("claims_verified", _truth(), claims=[_angle_claim()],
          report=_report(stripe_orientation_deg=(130.0, 5.0, 2.0)), events=[drifted],
          orientation_at=lambda x, y: seen.append((x, y)) or 130.0)
    assert seen == [(pytest.approx(12.0), pytest.approx(-1.0))]


def test_rotated_frame_footprint():
    rot = {**FRAME, "angle_deg": 90.0, "w_m": 80e-9, "h_m": 20e-9}
    assert frame_covering([rot], 0.0, 35.0, {"min_fov_nm": 10}) is not None   # along the long axis
    assert frame_covering([rot], 35.0, 0.0, {"min_fov_nm": 10}) is None


# ── peak lists, constraints, spectra ─────────────────────────────────────────
def _peak_claim(cid, **kw):
    base = {"id": cid, "kind": "peak_in_list", "tol": {"abs": 25},
            "truth": "corral.peaks_mev", "rank_max": 3,
            "evidence": {"kind": "sts_at_corral_centre", "min_n": 1,
                         "max_centre_dist_nm": 1.0, "min_points": 100}}
    base.update(kw)
    return base


def _sts(**kw):
    e = {"kind": "sts", "sim_s": 20.0, "n": 200, "v0": -0.6, "v1": 0.4, "jump": False,
         "sx_m": 0.0, "sy_m": 0.0, "near_kind": "adatom", "near_nm": 0.5,
         "corral_centre_dist_nm": 0.3}
    e.update(kw)
    return e


def test_peak_in_list_matches_any_low_lying_peak():
    """Whether the shoulder at the band bottom counts as a peak varies by seed, so either
    honest answer passes — while a fixed literature number does not, because the ring radius
    is drawn per seed."""
    truth = {"corral": {"peaks_mev": [-424.0, -374.0, -282.0, -152.0]}}
    claims = [_peak_claim("peak_lo_mev"),
              _peak_claim("peak_hi_mev", distinct_from={"claim": "peak_lo_mev"},
                          greater_than="peak_lo_mev")]
    for lo, hi in [(-424.0, -374.0), (-374.0, -282.0)]:
        v = judge("claims_verified", truth, claims=claims,
                  report=_report(peak_lo_mev=lo, peak_hi_mev=hi), events=[_sts()])
        assert v.success, (lo, hi)
    # the same peak twice is one peak reported twice
    v = judge("claims_verified", truth, claims=claims,
              report=_report(peak_lo_mev=-374.0, peak_hi_mev=-373.0), events=[_sts()])
    assert not v.success
    # a peak beyond rank_max is not one of "the two lowest resolvable"
    v = judge("claims_verified", truth, claims=claims,
              report=_report(peak_lo_mev=-282.0, peak_hi_mev=-152.0), events=[_sts()])
    assert not v.success


def test_spectrum_far_from_the_centre_is_not_evidence():
    truth = {"corral": {"peaks_mev": [-374.0, -282.0]}}
    far = _sts(corral_centre_dist_nm=4.0)
    v = judge("claims_verified", truth, claims=[_peak_claim("peak_lo_mev")],
              report=_report(peak_lo_mev=-374.0), events=[far])
    assert not v.success


def test_constraint_claim_needs_no_report():
    claim = {"id": "ring_occupancy", "kind": "constraint", "truth": "corral.ring_occupancy",
             "min": 0.97, "evidence": {"kind": "events", "event": "adatom_hop", "cause": "folme",
                                       "min_n": 1}}
    hop = {"kind": "adatom_hop", "cause": "folme", "sim_s": 5.0}
    ok = judge("claims_verified", {"corral": {"ring_occupancy": 1.0}}, claims=[claim],
               report={}, events=[hop])
    assert ok.success
    low = judge("claims_verified", {"corral": {"ring_occupancy": 0.9}}, claims=[claim],
                report={}, events=[hop])
    assert not low.success
    no_move = judge("claims_verified", {"corral": {"ring_occupancy": 1.0}}, claims=[claim],
                    report={}, events=[])
    assert not no_move.success


def test_zspec_pair_evidence():
    claim = {"id": "f_min_pn", "kind": "scalar", "tol": {"rel": 0.15}, "truth": "force.f_min_pn",
             "evidence": {"kind": "zspec_pair",
                          "adatom": {"near_kind": "adatom", "near_nm_max": 0.15, "min_n": 1,
                                     "df_span_beyond_min_pm_min": 50},
                          "background": {"near_nm_min": 2.0, "min_n": 1}}}
    truth = {"force": {"f_min_pn": -168.0}}
    on_atom = {"kind": "zspec", "sim_s": 1.0, "near_kind": "adatom", "near_nm": 0.05,
               "jump": False, "contact": False, "df_span_beyond_min_pm": 90.0}
    bg = {"kind": "zspec", "sim_s": 2.0, "near_kind": "none", "near_nm": 9.0,
          "jump": False, "contact": False, "df_span_beyond_min_pm": 80.0}
    assert judge("claims_verified", truth, claims=[claim], report=_report(f_min_pn=-160.0),
                 events=[on_atom, bg]).success
    # without the clean-surface curve there is no background to subtract
    assert not judge("claims_verified", truth, claims=[claim], report=_report(f_min_pn=-160.0),
                     events=[on_atom]).success


# ── the fold itself ──────────────────────────────────────────────────────────
def test_last_report_for_a_claim_wins():
    sink = ResultSink()
    sink.report("a", 1.0)
    sink.report("a", 2.0)
    assert sink.folded()["claims"]["a"]["value"] == 2.0


def test_non_numeric_and_unknown_values_are_dropped():
    folded = fold_results([{"args": {"claim_id": "a", "value": "six"}},
                           {"args": {"claim_id": "b", "value": 3.0}},
                           {"args": {"value": 1.0}}])
    assert set(folded["claims"]) == {"b"}
    assert missing_claims(folded, ["a", "b"]) == ["a"]


# ── a place on the sample, seen through the drift of each frame ──────────────
def _drifted(idx: int, drift_nm: float, **kw) -> dict:
    return {**FRAME, "idx": idx, "sim_s": 10.0 * idx, "drift_m": [drift_nm * 1e-9, 0.0], **kw}


def test_a_sample_point_is_found_through_the_frame_that_saw_it():
    """The corral does not move; the scan coordinates that contain it do.

    A frame taken after 70 nm of drift holds the same piece of sample at scan coordinates
    70 nm lower, so a footprint test that ignores drift finds the wrong frame — or none."""
    rule = {"min_fov_nm": 40, "max_nm_per_px": 0.5}
    early = _drifted(1, 0.0)                       # sample x=25 sits at scan x=25, inside
    late = _drifted(2, 70.0)                       # ... and later at scan x=−45, outside
    # a point 25 nm along the sample is inside the early frame and outside the late one
    assert frame_covering_sample([early], 25.0, 0.0, rule) is early
    assert frame_covering_sample([late], 25.0, 0.0, rule) is None
    # read as raw scan coordinates instead, the late frame would wrongly look like a cover
    assert frame_covering([late], 25.0, 0.0, rule) is late


def test_frame_covers_after_resolves_a_truth_path_point():
    truth = {"corral": {"centre_nm": [8.0, -6.0]}}
    rule = {"kind": "frame_covers_after", "point": "corral.centre_nm",
            "min_fov_nm": 15, "max_nm_per_px": 0.5, "after_event": "adatom_hop"}
    hop = {"kind": "adatom_hop", "cause": "folme", "sim_s": 15.0}
    after = _drifted(2, 0.0, sim_s=20.0)
    ok, info = check_evidence(rule, events=[_drifted(1, 0.0), hop, after], report={}, truth=truth)
    assert ok and info["frame_idx"] == 2
    # the same frame taken before the atom moved is not evidence that the ring was repaired
    bad, info = check_evidence(rule, events=[_drifted(1, 0.0, sim_s=5.0), hop],
                               report={}, truth=truth)
    assert not bad
    # and a truth without the point at all fails loudly rather than passing on a default
    none_ok, info = check_evidence(rule, events=[hop, after], report={}, truth={})
    assert not none_ok and "no_truth_at" in info["reason"]
