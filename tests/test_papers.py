"""The paper scenarios themselves: well-formed, per-seed, and not answerable from memory."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from stmbench.papers import PAPER_BASELINES
from stmbench.trackB.truth_criteria import CRITERIA, judge
from stmsim.scenario import Scenario

SCEN = Path(__file__).resolve().parent.parent / "stmbench" / "trackB" / "scenarios"
PAPERS = sorted(p for p in SCEN.glob("P*.yaml"))
PAPER_NAMES = [p.name for p in PAPERS]


def test_there_is_a_scenario_for_every_paper_in_the_ladder():
    families = {Scenario.load(p).family for p in PAPERS}
    assert families == set(PAPER_BASELINES), (families, set(PAPER_BASELINES))


@pytest.mark.parametrize("name", PAPER_NAMES)
def test_paper_scenario_is_well_formed(name):
    sc = Scenario.load(SCEN / name)                 # load() runs validate()
    assert sc.paper_id and sc.family.startswith("P")
    assert sc.success["kind"] == "claims_verified" and sc.success["kind"] in CRITERIA
    assert sc.claims and sc.hidden
    assert sc.budget["sim_hours"] > 0 and sc.budget["wire_cmds"] > 0
    assert sc.family in PAPER_BASELINES, "a paper family needs a mode-C baseline"
    for c in sc.claims:
        assert c["id"] and c["kind"] and c["truth"] and c["evidence"]
        if c["kind"] == "position":
            assert c.get("position") == "required"


@pytest.mark.parametrize("name", PAPER_NAMES)
def test_the_task_names_the_result_channel(name):
    """A model that never calls ReportResult scores zero, so the task has to say so."""
    assert "ReportResult" in Scenario.load(SCEN / name).task


def _norm(v):
    """A drawn value as something hashable: numbers rounded, choices (the tip's condition) as-is."""
    return round(float(v), 9) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


@pytest.mark.parametrize("name", PAPER_NAMES)
def test_hidden_values_differ_between_seeds(name):
    sc = Scenario.load(SCEN / name)
    draws = [sc.draw_hidden(s) for s in range(6)]
    flat = [tuple(sorted((f"{b}.{k}", _norm(v)) for b, body in d.items()
                         for k, v in body.items())) for d in draws]
    assert len(set(flat)) >= 5, "seeds must not draw the same world"
    assert flat[0] == tuple(sorted((f"{b}.{k}", _norm(v))
                                   for b, body in sc.draw_hidden(0).items()
                                   for k, v in body.items()))


@pytest.mark.parametrize("name", PAPER_NAMES)
def test_every_scalar_claim_has_a_truth_that_actually_varies(name):
    """The point of drawing per seed: a claim whose truth is the same every time could be
    answered from the literature, and would be measuring nothing."""
    sc = Scenario.load(SCEN / name)
    for c in sc.claims:
        if c["kind"] not in ("scalar", "angle") or not str(c["truth"]).startswith(("herringbone.",
                                                                                  "surface_state.",
                                                                                  "force.")):
            continue
        block, key = str(c["truth"]).split(".", 1)
        drawn = {round(float(sc.draw_hidden(s).get(block, {}).get(key, float("nan"))), 9)
                 for s in range(8) if key in sc.draw_hidden(s).get(block, {})}
        if not drawn:
            continue                    # a derived truth (force.f_min_pn) — covered below
        assert len(drawn) >= 6, (c["id"], drawn)


@pytest.mark.parametrize("name", PAPER_NAMES)
def test_a_frozen_answer_fails_most_seeds(name, tmp_path):
    """Answer every claim with the seed-0 truth and try it on eight other seeds.

    This is the rule that makes the benchmark about measuring: the hidden values are drawn
    per seed, so an answer carried in from the literature — or from a previous episode — has
    to fail most of the time. It fails here loudly if a tolerance is ever widened past what
    the range can carry."""
    sc = Scenario.load(SCEN / name)
    frozen, orient0 = _frozen_answers(sc, 0, tmp_path / "ref")
    if not frozen:
        # P4 is judged on whether the atom actually sits on the designated lattice site, so
        # there is no number to carry in from the literature and nothing to freeze.
        assert all(c["kind"] in ("position", "constraint") for c in sc.claims), sc.claims
        pytest.skip("every claim is judged on the instrument's state, not on a reported number")
    passed = 0
    for seed in range(1, 9):
        w = sc.build_world(seed, session_dir=tmp_path / f"s{seed}")
        v = judge("claims_verified", w.truth(), claims=sc.claims,
                  report={"claims": frozen}, events=_all_evidence(),
                  orientation_at=_orientation_of(w))
        if all(v.details["claims"][c]["value_ok"] for c in frozen):
            passed += 1
    assert passed <= 2, f"{name}: a frozen answer passed {passed}/8 seeds"


def _orientation_of(world):
    """The callback the harness gives the judge: the whole orientation at a place."""
    surf = world.surface
    return lambda x_nm, y_nm: surf.orientation_at(x_nm * 1e-9, y_nm * 1e-9)


def _frozen_answers(sc: Scenario, seed: int, session) -> tuple[dict, object]:
    """Seed 0's truth, in the shape a report arrives in."""
    from stmbench.trackB.claims import truth_at_path

    w = sc.build_world(seed, session_dir=session)
    truth = w.truth()
    orient = _orientation_of(w)
    out: dict[str, dict] = {}
    # two positions far enough apart to be different domains on the reference seed
    spots = [(-40.0, 25.0), (40.0, -25.0)]
    n_pos = 0
    for c in sc.claims:
        if c["kind"] == "constraint":
            continue
        if c["truth"] == "orientation_at":
            x, y = spots[min(n_pos, len(spots) - 1)]
            n_pos += 1
            val = (orient(x, y) or {}).get("stripe_deg")
            if val is None:
                continue
            out[c["id"]] = {"value": float(val), "x_nm": x, "y_nm": y}
            continue
        val = truth_at_path(truth, c["truth"])
        if isinstance(val, (list, tuple)) and val:
            val = val[0]
        if isinstance(val, (int, float)):
            out[c["id"]] = {"value": float(val)}
    return out, orient


def _scalar_claims(sc: Scenario) -> list[dict]:
    return [c for c in sc.claims if c["kind"] == "scalar"]


def _truth_of(sc: Scenario, seed: int, session) -> dict:
    return sc.build_world(seed, session_dir=session).truth()


def _truth_values(sc: Scenario, seed: int, session) -> dict:
    from stmbench.trackB.claims import truth_at_path

    truth = _truth_of(sc, seed, session)
    out = {}
    for c in _scalar_claims(sc):
        val = truth_at_path(truth, c["truth"])
        if isinstance(val, (int, float)):
            out[c["id"]] = float(val)
    return out


def _all_evidence() -> list[dict]:
    """Evidence generous enough that only the VALUE can fail the claim."""
    frame = {"kind": "scan_saved", "sim_s": 1.0, "idx": 1, "complete": True, "cx_m": 0.0,
             "cy_m": 0.0, "w_m": 200e-9, "h_m": 200e-9, "angle_deg": 0.0, "nx": 512, "ny": 512,
             "drift_m": [0.0, 0.0]}
    sts = [{"kind": "sts", "sim_s": 2.0 + i, "n": 200, "v0": -0.6, "v1": 0.4, "jump": False,
            "sx_m": i * 1e-9, "sy_m": 0.0, "near_kind": "step", "near_nm": 2.0 + i,
            "corral_centre_dist_nm": 0.2} for i in range(10)]
    zspec = [{"kind": "zspec", "sim_s": 30.0, "near_kind": "adatom", "near_nm": 0.05,
              "jump": False, "contact": False, "df_span_beyond_min_pm": 120.0},
             {"kind": "zspec", "sim_s": 31.0, "near_kind": "none", "near_nm": 9.0,
              "jump": False, "contact": False, "df_span_beyond_min_pm": 120.0}]
    hop = {"kind": "adatom_hop", "cause": "folme", "sim_s": 5.0}
    return [frame, hop, *sts, *zspec]


# ── the baselines are solvability proofs, so they must not cheat ──
BASELINE_FILES = sorted((Path(__file__).resolve().parent.parent / "stmbench" / "papers")
                        .glob("p[0-9]*.py"))


@pytest.mark.parametrize("path", BASELINE_FILES, ids=lambda p: p.name)
def test_a_baseline_never_reads_the_hidden_truth(path):
    """A scripted baseline exists to show the task is doable with what the instrument shows.
    One peek at ``world.truth()`` and it proves nothing."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    banned = {"truth", "hidden", "adatoms", "surface", "site", "force", "pll"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in banned:
            src = ast.unparse(node)
            assert not src.startswith(("world.", "host.world", "b.host.world", "w.")), \
                f"{path.name} reaches into the world: {src}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "truth", f"{path.name} calls truth()"


@pytest.mark.parametrize("name", PAPER_NAMES)
def test_the_world_a_paper_builds_has_the_physics_its_claims_read(name, tmp_path):
    """Every claim's truth path must actually resolve in the world the scenario builds."""
    from stmbench.trackB.claims import truth_at_path

    sc = Scenario.load(SCEN / name)
    truth = sc.build_world(0, session_dir=tmp_path / "s").truth()
    for c in sc.claims:
        if c["truth"] == "orientation_at":
            assert truth.get("herringbone"), f"{name}: {c['id']} needs a herringbone field"
            continue
        val = truth_at_path(truth, c["truth"])
        assert val is not None, f"{name}: claim {c['id']} truth path {c['truth']} is empty"
        if c["kind"] == "peak_in_list":
            assert isinstance(val, (list, tuple)) and len(val) >= 2, (c["id"], val)
        elif c["kind"] == "position" and isinstance(val, dict) and val.get("rule") == "any":
            # any atom will do: the rule, and the moved atoms to match a report against
            assert val["d_nm"] and isinstance(val["candidates"], list) and val["example"], (c["id"], val)
        elif c["kind"] == "position":
            assert isinstance(val, dict) and "site" in val, (c["id"], val)
        elif c.get("of_claim"):
            # a property of the atom another claim is matched to: read off its candidate list
            assert isinstance(val, list) and c["of_claim"] in {x["id"] for x in sc.claims}, (c["id"], val)
        else:
            assert isinstance(val, (int, float)), (c["id"], val)


@pytest.mark.parametrize("name", PAPER_NAMES)
def test_judging_with_no_report_does_not_raise(name, tmp_path):
    """``test_scenario_faults`` calls judge() with no extras on every scenario file."""
    sc = Scenario.load(SCEN / name)
    w = sc.build_world(0, session_dir=tmp_path / "s")
    v = judge(sc.success["kind"], {**w.truth(), "events_tail": w.events[-50:]},
              sc.success.get("thresholds"))
    assert not v.success and v.partial == 0.0


# ── the repair variant has to arrive broken ─────────────────────────────────
def test_the_corral_repair_variant_starts_below_its_own_threshold(tmp_path):
    """If the ring were already closed, the constraint would pass without an atom moving."""
    sc = Scenario.load(SCEN / "P3_corral_repair_cu111.yaml")
    claim = next(c for c in sc.claims if c["id"] == "ring_occupancy")
    for seed in range(6):
        truth = sc.build_world(seed, session_dir=tmp_path / f"s{seed}").truth()
        corral = truth["corral"]
        assert corral["ring_occupancy"] < float(claim["min"]), (seed, corral["ring_occupancy"])
        assert len(corral["gap_sites"]) >= 3
        assert corral["spares_remaining"] == len(corral["gap_sites"]),             "one spare per hole, or the task is not finishable"
    # and the measuring variant arrives whole, so the two differ only in the repair
    whole = Scenario.load(SCEN / "P3_corral_cu111.yaml")
    assert whole.build_world(0, session_dir=tmp_path / "w").truth()["corral"]["ring_occupancy"] == 1.0


def test_a_broken_ring_alone_does_not_satisfy_the_repair_claims(tmp_path):
    """The evidence rules, not the truth, are what make this a repair task."""
    sc = Scenario.load(SCEN / "P3_corral_repair_cu111.yaml")
    w = sc.build_world(0, session_dir=tmp_path / "s")
    peaks = w.truth()["corral"]["peaks_mev"]
    report = {"peak_lo_mev": {"value": peaks[0]}, "peak_hi_mev": {"value": peaks[1]}}
    # every measurement taken, but no atom ever moved: the spectra are of the broken ring
    no_hop = [e for e in _all_evidence() if e["kind"] != "adatom_hop"]
    v = judge("claims_verified", w.truth(), claims=sc.claims, report=report, events=no_hop)
    assert not v.success
    assert all(not v.details["claims"][c]["evidence_ok"] for c in ("peak_lo_mev", "ring_occupancy"))
