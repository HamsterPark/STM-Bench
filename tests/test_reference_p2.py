"""The P2 privileged reference solution: k(E) from dI/dV maps of a step edge, through the judge."""
from __future__ import annotations

import pytest

from stmbench.reference import p2, run_many, summarize


@pytest.mark.parametrize("seed", [0, 15])
def test_the_reference_reproduces_p2_from_measured_maps(seed):
    """Seed 0 starts with a blunt tip that turns double during the session (a map taken with
    it is retaken); seed 15 carries an adsorbate (and, with the present scipy, one of its maps
    comes back blank and is retaken). Both numbers come from the maps, not from the truth."""
    r = p2.run(seed)
    assert r["success"], r
    assert all(r["claims_ok"].values())
    assert abs(r["abs_err"]["e0_mev"]) < 5.0 and abs(r["rel_err"]["m_eff"]) < 0.015
    assert r["values"]["m_eff"] != r["truth"]["m_eff"]               # measured, not copied
    assert len(r["maps"]) == 8 and all(m["ok"] for m in r["maps"])
    assert r["terrace_nm"] >= 60.0 and abs(r["tilt_deg"]) < 2.0
    assert r["instrument_s"] < 3000.0


def test_the_runner_summarizes_p2_errors_in_the_claims_own_units():
    s = summarize(run_many("P2", [1]))
    assert s["success_rate"] == 1.0
    assert set(s["claims"]) == {"e0_mev", "m_eff"}
    assert s["claims"]["m_eff"]["relative"] and not s["claims"]["e0_mev"]["relative"]
