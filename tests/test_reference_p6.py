"""The P6 privileged reference solution: plumbing through the instrument and the judge.

These tests need the P6 scenario and ``stmbench.analysis.kondo``, which arrive with the physics,
judge and analysis branches; until then they skip. They check that the reference runs end to end
inside the budget, that its reports cite sweeps the judge's ``sts_pair`` evidence accepts, and
that the result has its documented keys. They do not check the tolerances: that is what the
calibration run on seeds 0-29 is for (docs/results/p6_reference_seeds_0-29.json).
"""
from __future__ import annotations

import pytest

from pathlib import Path

from stmbench.reference import PAPERS, p6

SCENARIO_FILE = Path(p6.__file__).parents[1] / "trackB" / "scenarios" / f"{p6.SCENARIO}.yaml"

needs_p6 = pytest.mark.skipif(not SCENARIO_FILE.exists(), reason="the P6 scenario is not on this branch yet")

KEYS = {"paper", "seed", "success", "values", "truth", "rel_err", "abs_err", "claims_ok",
        "evidence_ok", "instrument_s", "tip_condition", "attenuation", "protocol", "sweeps", "sweep_near"}


def test_p6_is_registered_with_the_reference_runner():
    assert PAPERS["P6"] is p6.run


@needs_p6
def test_a_reduced_reference_run_has_its_plumbing_in_order():
    pytest.importorskip("stmbench.analysis.kondo")
    r = p6.run(0, points=61, integration_ms=20.0, atom_sweeps=1, bare_sweeps=1, drift_comp=False)
    assert KEYS <= set(r), sorted(KEYS - set(r))
    assert set(r["values"]) == set(r["truth"]) == {"gamma_mev", "q_fano", "e_k_mev"}
    assert r["values"] != r["truth"]                       # measured, not copied
    assert r["sweeps"] == 2 and len(r["sweep_near"]) == 2
    on, off = r["sweep_near"]
    assert on["near_kind_apex"] == "adatom" and on["near_nm_apex"] <= 0.15     # the sweep on the atom
    assert off["near_kind_apex"] != "adatom" or off["near_nm_apex"] >= 2.0     # the one on bare surface
    assert r["instrument_s"] < 1.5 * 3600.0
    assert "attenuation" in r and 0.8 <= r["attenuation"] <= 1.0
    # every claim's evidence was found: the reports cite the two sweeps (sts_pair)
    assert set(r["evidence_ok"]) == set(r["values"]) and all(r["evidence_ok"].values()), r["evidence_ok"]


@needs_p6
def test_the_lead_apex_is_the_one_the_sweep_uses_not_the_one_with_the_largest_dz():
    """A two-apex tip whose larger-dz apex carries the smaller current weight: the world makes the
    spectrum through the heavier apex, so that is the one the reference must park over the atom."""
    from stmsim.physics.tip import Apex

    from stmbench.reference._common import Session

    s = Session(p6.SCENARIO, 0)
    w = s.w
    w.tip.apexes = [Apex(0.5e-9, 0.0, 0.02e-9, 0.1), Apex(0.0, 0.0, 0.0, 1.0)]
    assert p6._lead_offset(w) == (0.0, 0.0)                 # the weight wins; the largest dz would give 0.5 nm
    ax, ay = p6._isolated_atom(w)
    p6._put_apex_over(s, ax, ay, 1.0)
    s.inst.act("lockin", on=True, amplitude_mv=3.0)
    s.inst.act("sts", v_start=-0.05, v_end=0.05, points=41, settle_ms=5.0, integration_ms=5.0, backward=False)
    rec = w.sts_records[-1]
    assert rec["near_kind_apex"] == "adatom" and rec["near_nm_apex"] <= 0.15
    assert abs(rec["apex_offset_nm"][0]) < 1e-6              # the world's lead apex is the one the reference chose
