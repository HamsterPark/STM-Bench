"""P7 (the pull threshold): the scenario, how a drag series is judged, and an episode on the instrument.

The evidence rule, the truth and the pairing rule are read from hand-made ``truth["drags"]``
records and ``dat_saved`` events, so each rule is pinned alone; one episode through the benchmark
instrument shows the same rules applied to what the instrument records.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from stmbench.trackB import claims as C
from stmbench.trackB import rules as R
from stmbench.trackB.truth_criteria import judge
from stmsim.scenario import Scenario

SCEN = Path(__file__).resolve().parent.parent / "stmbench" / "trackB" / "scenarios"
P7 = SCEN / "P7_pull_threshold_cu111.yaml"
SC = Scenario.load(P7)
C1, C2 = "r_pull_1_kohm", "r_pull_2_kohm"
LATTICE = {"nn_m": 0.255e-9, "angle_rad": 0.0}


# ── hand-made records and events ──────────────────────────────────────────────────────────────────
def rec(index, r_kohm, followed, *, atom=7, th=200.0, **kw) -> dict:
    """One record of a traced move that started over atom ``atom`` and ran the standard drag. ``th`` is
    the threshold the (patched) truth function returns for it."""
    out = {"index": index, "sim_s": 10.0 * index, "r_kohm": r_kohm, "speed_nm_s": 0.1, "length_nm": 1.0,
           "direction_rad": 0.0, "start_nm": [0.0, 0.0], "end_nm": [1.0, 0.0], "active": True,
           "feedback": True, "withdrawn": False, "over_range": False, "carrying": False,
           "atom_id": atom, "grab_idx": 0, "point": 0, "n_points": 1, "start_dist_nm": 0.05,
           "site_before": [0, 0], "site_after": [4, 0] if followed else [0, 0], "n_hops": 4 if followed else 0,
           "picked": False, "end_dist_nm": 0.02 if followed else 1.0, "followed": followed,
           "conditions": {"th": th}}
    out.update(kw)
    return out


def event(index) -> dict:
    return {"kind": "dat_saved", "path": f"/run/session/Move-Trace{index:05d}.dat", "experiment": "move trace",
            "idx": index + 1, "move_idx": index, "sim_s": 10.0 * index}


def series(first, atom=7, th=200.0, rs=(400, 300, 230, 170, 130, 100, 80, 60), carried_at_or_below=170):
    """Eight drags at falling resistances: carried at ``carried_at_or_below`` and under, not above."""
    return [rec(first + k, float(r), r <= carried_at_or_below, atom=atom, th=th) for k, r in enumerate(rs)]


@pytest.fixture(autouse=True)
def _threshold_from_the_record(monkeypatch):
    """The truth of a record is ``conditions["th"]`` kohm here: what is under test is which drags count
    and how they are combined; the threshold function has its own tests (test_drag_threshold.py)."""
    from stmsim.physics import drag_threshold as D
    real = D.record_threshold

    def patched(lat, r, **kw):
        cond = r.get("conditions")
        if cond is not None and "th" not in cond:            # a record the instrument wrote: the real function
            return real(lat, r, **kw)
        return None if cond is None else float(cond["th"]) * 1e3

    monkeypatch.setattr(D, "record_threshold", patched)


def verdict(records, reports, *, extra_events=(), rules_version=2, claims=None):
    """``reports``: ``{claim id: (value, [cited move indexes or other citations])}``."""
    events = [event(r["index"]) for r in records] + list(extra_events)
    folded = {cid: {"value": v, "evidence": [f"move:{c}" if isinstance(c, int) else c for c in cites]}
              for cid, (v, cites) in reports.items()}
    truth = {"drags": {"lattice": LATTICE, "records": records}}
    return judge("claims_verified", truth, claims=claims or SC.claims, report={"claims": folded},
                 events=events, rules_version=rules_version)


def two_atoms(**kw):
    a, b = series(0, atom=7, th=200.0, **kw), series(100, atom=9, th=140.0, **kw)
    return a + b, {C1: (200.0, [r["index"] for r in a]), C2: (140.0, [r["index"] for r in b])}


# ── the scenario ──────────────────────────────────────────────────────────────────────────────────
def test_the_scenario_loads_and_asks_for_two_pull_thresholds():
    assert SC.family == "P7" and SC.paper_id == "P7"
    assert SC.claim_ids == [C1, C2]
    for c in SC.claims:
        assert c["truth"] == "drag_threshold" and c["evidence"]["kind"] == "drag_series"
        assert c["protocol"] == {"length_nm": 1.0, "speed_nm_s": 0.1}
    assert SC.claims[1]["evidence"]["distinct_atom_from"] == C1


def test_only_a_scenario_with_a_drag_series_keeps_the_drag_record(tmp_path):
    """``record_drags`` is on for P7 and off for every other scenario (the record only reads the
    instrument, but a world that does not need it does not keep it)."""
    wants = {}
    for p in sorted(SCEN.glob("*.yaml")):
        sc = Scenario.load(p)
        wants[sc.id] = any((c.get("evidence") or {}).get("kind") == "drag_series" for c in sc.claims)
    assert [k for k, v in wants.items() if v] == [SC.id]
    assert SC.build_world(0, session_dir=tmp_path / "p7").record_drags
    p4 = Scenario.load(SCEN / "P4_atom_positioning_cu111.yaml")
    w4 = p4.build_world(0, session_dir=tmp_path / "p4")
    assert not w4.record_drags and "drags" not in w4.truth()


# ── citing a move trace ───────────────────────────────────────────────────────────────────────────
def test_a_move_trace_is_cited_by_its_file_name_or_by_move_n_alike():
    events = [event(3), event(12), {"kind": "dat_saved", "path": "/x/Spec00001.dat", "sts_idx": 5, "idx": 1}]
    by_name = C.resolve_citation(events, "Move-Trace00012.dat")
    assert by_name is events[1] and C.resolve_citation(events, "move:12") is events[1]
    assert C.resolve_citation(events, "C:\\run\\session\\MOVE-TRACE00003.DAT") is events[0]     # any directory part
    assert C.resolve_citation(events, "move:7") is None and C.resolve_citation(events, "move:x") is None
    assert C.resolve_citation(events, "Move-Trace00099.dat") is None


def test_the_p4_verdict_does_not_change_when_move_traces_are_citable(tmp_path):
    """P4's rules never look at cited move traces: a P4 report that cites a move trace is judged as
    one that cites nothing, and only ``unresolved_citations`` differs."""
    p4 = Scenario.load(SCEN / "P4_atom_positioning_cu111.yaml")
    w = p4.build_world(0, session_dir=tmp_path / "s")
    base = {"claims": {"target_site": {"value": 0.0, "x_nm": 1.0, "y_nm": 1.0, "evidence": []}}}
    cited = copy.deepcopy(base)
    cited["claims"]["target_site"]["evidence"] = ["Move-Trace00001.dat", "move:1"]
    events = [event(1)]
    kw = dict(claims=p4.claims, events=events, rules_version=3)
    a = judge("claims_verified", w.truth(), report=base, **kw)
    b = judge("claims_verified", w.truth(), report=cited, **kw)
    assert (a.success, a.partial) == (b.success, b.partial)
    for cid, da in a.details["claims"].items():
        db = b.details["claims"][cid]
        assert da["value_ok"] == db["value_ok"] and da["evidence_ok"] == db["evidence_ok"]
        assert not db.get("unresolved_citations")


# ── the evidence ──────────────────────────────────────────────────────────────────────────────────
def test_a_series_that_meets_every_rule_is_accepted():
    records, reports = two_atoms()
    v = verdict(records, reports)
    for cid in (C1, C2):
        d = v.details["claims"][cid]
        assert d["evidence_ok"], d["evidence"]
        assert d["evidence"]["n_qualifying"] == 8 and d["evidence"]["n_resistances"] == 8
        assert d["evidence"]["bracket"] == {"carried_at_or_below": True, "not_carried_at_or_above": True}
    assert v.details["claims"][C1]["evidence"]["atom_id"] == 7
    assert v.details["claims"][C2]["evidence"]["atom_id"] == 9
    assert v.success and v.partial == 1.0                  # value: the (patched) truth is exactly the report


def _reason(records, reports, cid=C1):
    d = verdict(records, reports).details["claims"][cid]
    assert not d["evidence_ok"], d["evidence"]
    return d["evidence"]["reason"], d["evidence"]


def test_too_few_drags_are_refused():
    records, reports = two_atoms()
    reports[C1] = (200.0, reports[C1][1][:6])
    assert _reason(records, reports)[0] == "too_few_drags"


def test_too_few_distinct_resistances_are_refused():
    """Eight drags at three resistances are three values; at two, two; and values within 2 % of the first
    of their group are one value."""
    a = [rec(k, r, r <= 170.0) for k, r in enumerate((400, 400, 400, 400, 170, 170, 170, 170))]
    assert _reason(a, {C1: (200.0, [r["index"] for r in a])})[0] == "too_few_resistances"
    b = [rec(k, r, r <= 170.0) for k, r in enumerate((400, 401, 405, 408, 170, 171, 172, 173))]    # 400-408: one
    why, info = _reason(b, {C1: (200.0, [r["index"] for r in b])})
    assert why == "too_few_resistances" and info["n_resistances"] == 2


def test_distinct_resistances_start_a_new_value_beyond_two_percent_of_the_first_of_the_group():
    assert C.distinct_resistances([100.0, 101.0, 102.0, 102.1]) == 2       # 102.1 > 100 * 1.02
    assert C.distinct_resistances([100.0, 101.0, 102.0]) == 1
    assert C.distinct_resistances([300.0, 100.0, 200.0]) == 3
    assert C.distinct_resistances([]) == 0


def test_a_value_with_no_carry_at_or_below_it_is_refused():
    records, reports = two_atoms()
    reports[C1] = (50.0, reports[C1][1])                   # nothing carried the atom at 50 kohm or less
    assert _reason(records, reports)[0] == "no_carry_below_value"


def test_a_value_with_no_failure_at_or_above_it_is_refused():
    records, reports = two_atoms()
    reports[C1] = (450.0, reports[C1][1])                  # nothing failed at 450 kohm or more
    assert _reason(records, reports)[0] == "no_failure_above_value"


def test_a_series_of_two_atoms_is_refused():
    a = series(0, atom=7)
    b = [rec(50 + k, 90.0 + 30 * k, k < 3, atom=8) for k in range(4)]
    records = a[:5] + b
    why, info = _reason(records, {C1: (200.0, [r["index"] for r in records])})
    assert why == "more_than_one_atom" and info["atom_id"] == 7


@pytest.mark.parametrize("change,reason", [
    ({"length_nm": 1.2}, "length_off"), ({"length_nm": 0.9}, "length_off"),
    ({"speed_nm_s": 0.2}, "speed_off"), ({"speed_nm_s": 0.0}, "speed_off"),
    ({"start_dist_nm": 0.35}, "start_too_far"), ({"over_range": True}, "over_range"),
    ({"feedback": False}, "feedback_off"), ({"withdrawn": True}, "withdrawn"),
    ({"carrying": True}, "carrying"), ({"r_kohm": None}, "no_resistance"),
    ({"r_kohm": float("inf")}, "no_resistance"), ({"atom_id": None}, "no_atom"),
])
def test_a_drag_that_does_not_qualify_does_not_count(change, reason):
    """One drag of the standard series is spoiled: it is listed with its reason and the series, now
    seven drags, is too short; the others still count."""
    records, reports = two_atoms()
    records[3] = {**records[3], **change}
    why, info = _reason(records, reports)
    assert why == "too_few_drags" and info["n_qualifying"] == 7
    assert info["rejected"] == [{"index": 3, "reason": reason}]


def test_the_protocol_may_differ_by_the_relative_tolerance_and_a_start_at_the_limit_counts():
    records, reports = two_atoms()
    records[0] = {**records[0], "length_nm": 1.05, "speed_nm_s": 0.095, "start_dist_nm": 0.3}
    d = verdict(records, reports).details["claims"][C1]
    assert d["evidence_ok"], d["evidence"]


def test_a_citation_that_is_not_a_move_trace_is_ignored_by_the_rule():
    records, reports = two_atoms()
    frame = {"kind": "scan_saved", "path": "/run/sim001.sxm", "idx": 1, "sim_s": 1.0, "complete": True}
    reports[C1] = (200.0, [*reports[C1][1], "sim001.sxm", "frame:1", "sts:4"])
    v = verdict(records, reports, extra_events=[frame])
    d = v.details["claims"][C1]
    assert d["evidence_ok"] and d["evidence"]["rejected"] == []
    assert d["unresolved_citations"] == ["sts:4"]


def test_a_report_that_cites_only_a_frame_has_no_series():
    records, reports = two_atoms()
    frame = {"kind": "scan_saved", "path": "/run/sim001.sxm", "idx": 1, "sim_s": 1.0, "complete": True}
    reports[C1] = (200.0, ["sim001.sxm"])
    d = verdict(records, reports, extra_events=[frame]).details["claims"][C1]
    assert not d["evidence_ok"] and d["evidence"]["reason"] == "too_few_drags" and d["evidence"]["n_qualifying"] == 0


def test_the_series_is_read_from_the_cited_drags_only():
    """Eight more qualifying drags in the episode do not help a report that cites three."""
    records, reports = two_atoms()
    reports[C1] = (200.0, reports[C1][1][:3])
    assert _reason(records, reports)[0] == "too_few_drags"


def test_a_cited_move_without_a_record_does_not_count():
    records, reports = two_atoms()
    reports[C1] = (200.0, [*reports[C1][1], 555])
    events = [event(555)]
    d = verdict(records, reports, extra_events=events).details["claims"][C1]
    assert d["evidence_ok"] and d["evidence"]["rejected"] == [{"index": 555, "reason": "no_record"}]


def test_the_second_claim_must_use_another_atom():
    a = series(0, atom=7, th=200.0)
    b = series(100, atom=7, th=200.0)
    records = a + b
    v = verdict(records, {C1: (200.0, [r["index"] for r in a]), C2: (200.0, [r["index"] for r in b])})
    assert v.details["claims"][C1]["evidence_ok"]
    d2 = v.details["claims"][C2]
    assert not d2["evidence_ok"] and d2["evidence"]["reason"] == "same_atom_as"
    assert d2["evidence"]["same_atom_as"] == C1 and not v.success and v.partial == 0.5


def test_the_old_rules_have_no_citations_to_check_a_series_against():
    records, reports = two_atoms()
    v = verdict(records, reports, rules_version=0)
    assert not v.details["claims"][C1]["evidence_ok"]


# ── the truth ─────────────────────────────────────────────────────────────────────────────────────
def test_the_truth_is_the_geometric_mean_over_the_qualifying_drags():
    rs = (400, 300, 230, 170, 130, 100, 80, 60)
    ths = (100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 800.0)
    records = [rec(k, float(r), r <= 170, th=t) for k, (r, t) in enumerate(zip(rs, ths))]
    want = math.exp(sum(math.log(t) for t in ths) / len(ths))
    reports = {C1: (want, [r["index"] for r in records])}
    d = verdict(records, reports, claims=[SC.claims[0]]).details["claims"][C1]
    assert d["truth"] == pytest.approx(want, rel=1e-9) and d["value_ok"] and d["evidence_ok"]
    assert d["truth_drags"] == {"atom_id": 7, "n_qualifying": 8, "n_thresholds": 8}


def test_a_drag_that_does_not_qualify_does_not_move_the_truth():
    records, reports = two_atoms()
    base = verdict(records, reports).details["claims"][C1]["truth"]
    spoiled = records[:]
    spoiled.append(rec(900, 150.0, True, th=5000.0, start_dist_nm=0.8))        # started too far from the atom
    spoiled.append(rec(901, 150.0, True, th=5000.0, length_nm=2.0))
    reports[C1] = (200.0, [*reports[C1][1], 900, 901])
    d = verdict(spoiled, reports).details["claims"][C1]
    assert d["truth"] == pytest.approx(base)
    assert d["truth"] == pytest.approx(200.0)
    assert {x["reason"] for x in d["evidence"]["rejected"]} == {"start_too_far", "length_off"}


def test_a_value_outside_the_tolerance_of_the_truth_fails_whatever_the_evidence():
    records, reports = two_atoms()
    reports[C1] = (230.0, reports[C1][1])                   # evidence is fine at 230 (it brackets it); truth is 200
    d = verdict(records, reports).details["claims"][C1]
    assert d["evidence_ok"] and not d["value_ok"]


def test_with_no_qualifying_drag_there_is_no_truth_and_the_value_fails():
    records, reports = two_atoms()
    records = [{**r, "over_range": True} for r in records]
    d = verdict(records, reports).details["claims"][C1]
    assert d["truth"] is None and d["truth_reason"] == "no_qualifying_drag" and not d["value_ok"]


def test_records_without_a_threshold_are_left_out_of_the_truth():
    records, reports = two_atoms()
    records[0] = {**records[0], "conditions": None}
    d = verdict(records, reports).details["claims"][C1]
    assert d["truth_drags"]["n_qualifying"] == 8 and d["truth_drags"]["n_thresholds"] == 7
    assert d["truth"] == pytest.approx(200.0)


def test_with_two_atoms_in_the_series_the_truth_is_the_atom_with_most_drags():
    a = series(0, atom=7, th=200.0)
    b = [rec(50 + k, 60.0 + 30 * k, k < 3, atom=8, th=999.0) for k in range(3)]
    records = a + b
    d = verdict(records, {C1: (200.0, [r["index"] for r in records])}, claims=[SC.claims[0]]).details["claims"][C1]
    assert d["truth_drags"]["atom_id"] == 7 and d["truth"] == pytest.approx(200.0) and not d["evidence_ok"]


def test_evidence_and_truth_select_the_same_drags():
    """One function selects them: its qualifying drags are the ones the evidence counts and the truth reads."""
    records, reports = two_atoms()
    records[2] = {**records[2], "start_dist_nm": 0.9}
    cited = [event(i) for i in reports[C1][1]]
    sel = C.select_drags({**SC.claims[0]["evidence"], "protocol": SC.claims[0]["protocol"]},
                         {"drags": {"lattice": LATTICE, "records": records}}, cited)
    d = verdict(records, reports).details["claims"][C1]
    assert [r["index"] for r in sel["series"]] == [0, 1, 3, 4, 5, 6, 7]
    assert d["evidence"]["n_qualifying"] == d["truth_drags"]["n_qualifying"] == len(sel["series"]) == 7


# ── a frozen answer sheet ─────────────────────────────────────────────────────────────────────────
def test_a_frozen_answer_sheet_fails_most_seeds(tmp_path):
    """Seed 0's two thresholds, as the chance audit reads them, against seeds 1-8's atoms."""
    from stmsim.physics.drag_threshold import example_thresholds

    def truths(seed):
        w = SC.build_world(seed, session_dir=tmp_path / f"s{seed}")
        return [r / 1e3 for r in example_thresholds(w, 2)]

    frozen = truths(0)
    assert all(math.isfinite(v) and v > 0 for v in frozen)
    passed = 0
    for seed in range(1, 9):
        t = truths(seed)
        passed += all(C.within_scalar(f, x, SC.claims[k]["tol"]) for k, (f, x) in enumerate(zip(frozen, t)))
    assert passed <= 2, f"a frozen answer sheet passed {passed}/8 seeds"


# ── the rule card ─────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("lang", ["zh", "en"])
def test_the_card_states_every_number_the_series_rule_reads(lang):
    card = R.rule_card(SC, lang)
    ev, proto = SC.claims[0]["evidence"], SC.claims[0]["protocol"]
    for x in (ev["min_n"], ev["min_resistances"], ev["start_within_nm"], proto["length_nm"], proto["speed_nm_s"],
              100 * ev["protocol_rel"], 100 * (C.DISTINCT_RESISTANCE_FACTOR - 1), 0.6, 0.255, round(0.6 * 0.255, 3)):
        assert f"{x:g}" in card, (lang, x)
    for token in ("move:N", "move_tip", C1, C2):
        assert token in card, (lang, token)
    assert card.count("move:N") >= 2                              # the citing paragraph and the series rule


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_only_a_card_that_cites_moves_names_move_traces(lang):
    """P7's rules come with P7's card and change no other paper's (RULES_VERSION stays 2): the
    sentence on citing a move trace is in the cards of scenarios with a drag series only."""
    for path in sorted(SCEN.glob("P*.yaml")):
        sc = Scenario.load(path)
        card = R.rule_card(sc, lang)
        cites = any((c.get("evidence") or {}).get("kind") == "drag_series" for c in sc.claims)
        assert ("move:N" in card) == cites, (path.name, lang)
    assert R.RULES_VERSION == 2


def test_the_card_does_not_state_a_threshold_or_a_hidden_range():
    for lang in ("zh", "en"):
        card = R.task_text(SC, lang)
        for token in ("r_threshold", "r_pick", "v_max", "184", "210"):
            assert token not in card, (lang, token)


# ── the audit ─────────────────────────────────────────────────────────────────────────────────────
def test_the_chance_audit_reads_the_thresholds_of_the_atoms(tmp_path):
    from stmbench.trackB import audit as A

    samples = A.sample_truths(P7, [0, 1], work_dir=tmp_path)
    assert len(samples) == 2
    for s in samples:
        assert set(s) == {C1, C2} and all(isinstance(v, float) and v > 0 for v in s.values())
        assert s[C1] != s[C2]                                  # two different atoms
    assert samples[0] != samples[1]


# ── an episode on the instrument ──────────────────────────────────────────────────────────────────
def test_an_episode_on_the_instrument_ends_in_a_verdict_that_shows_the_drag_checks(tmp_path):
    """Navigate onto an atom with the truth, drag it along 1.0 nm at 0.1 nm/s at falling resistances, do
    the same for a second atom, report both citing the traces, finish: the verdict names, for each claim,
    the series' atom, its drags and a truth (it need not pass: the reference solution is another task)."""
    from stmbench.reference._common import Session
    from stmbench.reference._manip import Mover

    s = Session("P7_pull_threshold_cu111", 0, tmp_path / "run")
    inst, w = s.inst, s.w
    reg = w.surface.site.adatoms
    on, xy = reg._arrays()
    d0 = np.hypot(xy[:, 0], xy[:, 1])
    isolated = [on[int(k)] for k in np.argsort(d0)
                if int(np.count_nonzero(np.hypot(xy[:, 0] - xy[k, 0], xy[:, 1] - xy[k, 1]) <= 3e-9)) == 1]
    assert len(isolated) >= 2
    mv = Mover(s, imaging_v=0.05, imaging_pa=200.0)
    reports = {}
    for cid, atom in zip((C1, C2), isolated[:2]):
        names = []
        for r_kohm in (500, 360, 260, 190, 140, 105, 80, 60):
            ax, ay = s.image_xy(*reg.site_xy(atom.i, atom.j, atom.sub))      # the lead apex over the atom
            inst.act("move_tip", x_nm=ax, y_nm=ay)                         # onto the atom at the imaging setpoint
            mv.manipulate(r_kohm * 1e3)
            out = inst.act("move_tip", x_nm=ax + 1.0, y_nm=ay, speed_nm_s=0.1)
            mv.image()
            names.append(out["trace"])
            if atom.status != "on_surface":
                break
        reports[cid] = names
    assert len(w.drag_records) >= 8
    for cid, names in reports.items():
        inst.report(cid, 150.0, unit="kΩ", evidence=names)
    out = inst.finish()
    assert out and inst.verdict is not None
    details = inst.verdict.details["claims"]
    for cid in (C1, C2):
        d = details[cid]
        assert d["reported"] and set(d["evidence"]) >= {"kind", "atom_id", "n_qualifying", "rejected"}
        assert d["evidence"]["kind"] == "drag_series" and d["evidence"]["atom_id"] is not None
        assert d["evidence"]["n_qualifying"] >= 1
        assert "truth" in d and "truth_drags" in d and d["truth_drags"]["atom_id"] == d["evidence"]["atom_id"]
    assert details[C1]["evidence"]["atom_id"] != details[C2]["evidence"]["atom_id"]
    # the trace names the instrument returned resolve, by name and by move:N alike, to records the judge reads
    have = {r["index"] for r in w.drag_records}
    for name in reports[C1] + reports[C2]:
        e = C.resolve_citation(w.events, name)
        assert e is not None and e["move_idx"] in have
        assert C.resolve_citation(w.events, f"move:{e['move_idx']}") is e
