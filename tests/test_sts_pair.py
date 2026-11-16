"""P6's evidence rule ``sts_pair``: a bias sweep on top of an atom and one on the bare surface.

The rule is exercised on synthetic ``sts`` events (as the claims tests do), then the whole
scenario end to end through :class:`stmbench.instrument.Instrument`."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from stmbench.trackB import claims as C
from stmbench.trackB.rules import RULES_VERSION, rule_card
from stmbench.trackB.truth_criteria import judge
from stmsim.scenario import Scenario

SCEN = Path(__file__).resolve().parent.parent / "stmbench" / "trackB" / "scenarios"
P6 = SCEN / "P6_kondo_co_cu111.yaml"

RULE = {"kind": "sts_pair", "min_points": 41, "v_lo_max": -0.02, "v_hi_min": 0.02,
        "adatom": {"near_nm_max": 0.15, "min_n": 1}, "background": {"near_nm_min": 2.0, "min_n": 1}}


def _sts(idx: int, near_kind: str, near_nm: float, *, n: int = 81, v0: float = -0.05, v1: float = 0.05,
         jump: bool = False) -> dict:
    return {"kind": "sts", "idx": idx, "sim_s": 10.0 * idx, "n": n, "v0": v0, "v1": v1, "jump": jump,
            "sx_m": idx * 1e-8, "sy_m": 0.0, "near_kind": near_kind, "near_nm": near_nm}


def _check(events, cited=None):
    return C.check_evidence(RULE, events=events, report={}, truth={}, cited=cited)


def test_one_sweep_on_the_atom_and_one_on_bare_surface_pass():
    ok, info = _check([_sts(1, "adatom", 0.05), _sts(2, "none", 9.0)])
    assert ok and info["n_on_atom"] == 1 and info["n_background"] == 1


def test_the_bare_sweep_may_be_beside_a_step_or_beside_a_far_atom():
    assert _check([_sts(1, "adatom", 0.0), _sts(2, "step", 0.4)])[0]
    assert _check([_sts(1, "adatom", 0.0), _sts(2, "adatom", 2.0)])[0]


@pytest.mark.parametrize("on_atom", [
    _sts(1, "adatom", 0.3),                        # 0.3 nm off the centre
    _sts(1, "none", 0.05),                         # no atom is the nearest thing
    _sts(1, "adatom", 0.05, v0=-0.015),            # window misses -20 mV
    _sts(1, "adatom", 0.05, v1=0.015),             # window misses +20 mV
    _sts(1, "adatom", 0.05, n=40),                 # too few points
    _sts(1, "adatom", 0.05, jump=True),            # the tip jumped during the sweep
], ids=["off-atom", "not-an-atom", "misses-minus-20mV", "misses-plus-20mV", "40-points", "jump"])
def test_a_flawed_sweep_on_the_atom_does_not_count(on_atom):
    ok, info = _check([on_atom, _sts(2, "none", 9.0)])
    assert not ok and info["n_on_atom"] == 0


def test_a_sweep_from_20mV_to_minus_20mV_covers_the_window_and_41_points_suffice():
    assert _check([_sts(1, "adatom", 0.0, n=41, v0=0.02, v1=-0.02), _sts(2, "none", 9.0)])[0]


@pytest.mark.parametrize("bare", [
    _sts(2, "adatom", 1.0),                        # 1 nm from an atom
    _sts(2, "none", 9.0, n=40),
    _sts(2, "none", 9.0, jump=True),
    _sts(2, "none", 9.0, v1=0.015),
], ids=["1nm-from-atom", "40-points", "jump", "window"])
def test_a_flawed_background_sweep_does_not_count(bare):
    ok, info = _check([_sts(1, "adatom", 0.05), bare])
    assert not ok and info["n_background"] == 0


def test_without_a_background_sweep_the_pair_fails():
    ok, info = _check([_sts(1, "adatom", 0.05), _sts(2, "adatom", 0.05)])
    assert not ok and info == {"kind": "sts_pair", "n_on_atom": 2, "n_background": 0}


def test_without_a_sweep_on_the_atom_the_pair_fails():
    ok, info = _check([_sts(1, "none", 9.0), _sts(2, "none", 12.0)])
    assert not ok and info["n_on_atom"] == 0 and info["n_background"] == 2


def test_an_undelivered_sweep_does_not_count():
    atom = {**_sts(1, "adatom", 0.05), "delivered": False}
    assert not _check([atom, _sts(2, "none", 9.0)])[0]


def test_under_rules_v1_only_the_cited_sweeps_count():
    events = [_sts(1, "adatom", 0.05), _sts(2, "none", 9.0)]
    assert _check(events)[0]                                  # the earlier rule: anywhere in the episode
    ok, info = _check(events, cited=[events[0]])
    assert not ok and info["cited"] == 1 and info["n_background"] == 0
    ok, info = _check(events, cited=events)
    assert ok and info["cited"] == 2


def _apex(e: dict, kind: str, nm: float) -> dict:
    return {**e, "near_kind_apex": kind, "near_nm_apex": nm}


def test_the_distance_is_the_leading_apexs_when_the_event_records_it():
    # the tip centre is 0.6 nm off the atom but the apex tunnels on it: passes
    assert _check([_apex(_sts(1, "adatom", 0.6), "adatom", 0.05), _sts(2, "none", 9.0)])[0]
    # the centre is on the atom but the apex is 0.6 nm off: fails
    ok, info = _check([_apex(_sts(1, "adatom", 0.02), "adatom", 0.6), _sts(2, "none", 9.0)])
    assert not ok and info["n_on_atom"] == 0
    # a "background" sweep whose centre is clear but whose apex is 1 nm from an atom: fails
    ok, info = _check([_sts(1, "adatom", 0.05), _apex(_sts(2, "none", 9.0), "adatom", 1.0)])
    assert not ok and info["n_background"] == 0
    # and one whose centre is 1 nm from an atom but whose apex is clear: passes
    assert _check([_sts(1, "adatom", 0.05), _apex(_sts(2, "adatom", 1.0), "none", 9.0)])[0]


def test_an_event_without_apex_fields_falls_back_and_one_without_a_distance_counts_for_neither():
    assert _check([_sts(1, "adatom", 0.05), _sts(2, "none", 9.0)])[0]
    unknown = {k: v for k, v in _sts(1, "adatom", 0.05).items() if k != "near_nm"}
    ok, info = _check([unknown, _sts(2, "adatom", 0.0)])
    assert info["n_on_atom"] == 1 and info["n_background"] == 0 and not ok


def test_the_claims_of_p6_read_the_same_rule_and_cite_it_by_name():
    """Cited by ``sts:N``, through the judge: values are the truth's, so only the evidence can fail."""
    sc = Scenario.load(P6)
    assert [c["id"] for c in sc.claims] == ["gamma_mev", "q_fano", "e_k_mev"]
    assert all(c["evidence"] == RULE for c in sc.claims)
    truth = {"kondo": {"gamma_mev": 5.0, "q": 0.5, "e_k_mev": 1.0}}
    events = [_sts(1, "adatom", 0.05), _sts(2, "none", 9.0)]

    def verdict(cites):
        rep = {"claims": {c["id"]: {"value": C.truth_at_path(truth, c["truth"]), "evidence": cites}
                          for c in sc.claims}}
        return judge("claims_verified", truth, claims=sc.claims, report=rep, events=events,
                     rules_version=RULES_VERSION)

    assert verdict(["sts:1", "sts:2"]).success
    v = verdict(["sts:1"])
    assert not v.success
    assert all(d["value_ok"] and not d["evidence_ok"] for d in v.details["claims"].values())
    assert not verdict([]).success


def test_the_rule_card_states_the_pair_in_both_languages_and_no_hidden_value():
    sc = Scenario.load(P6)
    zh, en = rule_card(sc), rule_card(sc, "en")
    assert "在吸附原子正上方" in zh and "0.15 nm" in zh and "2 nm" in zh and "41 个点" in zh
    assert "-0.02 V 到 0.02 V" in zh and "没有突变" in zh
    assert "directly above an adatom" in en and "0.15 nm" in en and "at least 2 nm" in en
    assert "at least 41 points" in en and "-0.02 V to 0.02 V" in en
    assert "最靠近样品的针尖尖端" in zh and "leading apex" in en
    assert "{" not in zh and "{" not in en
    # a unitless claim's tolerance has no dangling unit
    assert "相差不超过 0.15\n" in zh and "by at most 0.15\n" in en
    for seed in range(4):
        for block, body in sc.draw_hidden(seed).items():
            for key, val in body.items():
                if isinstance(val, float):
                    assert f"{val:.4f}" not in zh and f"{val:.4f}" not in en, (seed, block, key)


# -- the whole scenario through the instrument --------------------------------
def test_p6_end_to_end_the_evidence_passes_and_made_up_values_are_judged(tmp_path):
    """Survey, go to an isolated atom (navigating with the truth, as the reference solutions do), a
    sweep on it and one on bare surface, then report made-up numbers citing them: the evidence is
    accepted and the values are judged, and fail."""
    from stmbench.reference._common import Session, bare_spot

    s = Session("P6_kondo_co_cu111", 0, tmp_path / "ep")
    inst, w = s.inst, s.w
    reg = w.surface.site.adatoms
    xy = reg.positions()
    best = None
    for a in reg.atoms.values():
        x, y = reg.site_xy(a.i, a.j, a.sub)
        near = sorted(math.hypot(px - x, py - y) for px, py in xy)[1]
        if near >= 3e-9 and w.surface.step_free_window(x, y, 3e-9):
            d = math.hypot(x, y)
            if best is None or d < best[0]:
                best = (d, x, y)
    assert best is not None, "P6 seed 0 has an isolated atom to measure"
    _, ax, ay = best
    assert math.hypot(ax, ay) < 30e-9, "the task promises one within 30 nm of the origin"
    bx, by = bare_spot(w, ax, ay)

    inst.act("scan", center_x_nm=ax * 1e9, center_y_nm=ay * 1e9, size_nm=20.0, pixels=64, line_time_s=0.05)
    s.go(ax, ay)
    on = inst.act("sts", v_start=-0.05, v_end=0.05, points=81, backward=False)["spectrum"]
    s.go(bx, by)
    off = inst.act("sts", v_start=-0.05, v_end=0.05, points=81, backward=False)["spectrum"]
    ev = [e for e in w.events if e.get("kind") == "sts"]
    assert [e["near_kind"] for e in ev] == ["adatom", ev[1]["near_kind"]]
    assert ev[0]["near_nm"] <= 0.15 and (ev[1]["near_kind"] != "adatom" or ev[1]["near_nm"] >= 2.0)

    for cid, val in (("gamma_mev", 50.0), ("q_fano", 5.0), ("e_k_mev", 50.0)):
        inst.report(cid, val, evidence=[on, off])
    out = inst.finish()
    claims = inst.verdict.details["claims"]
    assert set(claims) == {"gamma_mev", "q_fano", "e_k_mev"}
    for cid, d in claims.items():
        assert d["evidence_ok"], (cid, d)
        assert not d["value_ok"], (cid, d)                 # made-up numbers
    assert out["success"] is False and out["claims_verified"] == 0
