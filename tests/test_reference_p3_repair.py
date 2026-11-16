"""The P3 repair privileged reference solution: the gap closed by lateral manipulation, the corral
then measured as P3's reference measures it."""
from __future__ import annotations

from stmbench.reference import PAPERS, p3_repair


def test_the_reference_is_registered_as_paper_p3r():
    assert PAPERS["P3R"] is p3_repair.run


def test_the_reference_closes_the_gap_and_measures_the_corral():
    """Seed 9 arrives with a good tip and four atoms missing. No threshold is read: the ladder starts
    at 400 kOhm and goes down until the spare follows; the ring has to be back to 97 % of its sites
    and the numbers come from the spectra and the frame taken after the last hop."""
    r = p3_repair.run(9)
    assert r["success"], r
    assert all(r["claims_ok"].values())
    assert r["ring_occupancy"] >= 0.97 and r["placed"] == r["gap_sites"] and r["pickups"] == 0
    assert abs(r["abs_err"]["radius_nm"]) < 0.15
    assert all(abs(r["abs_err"][c]) < 6.0 for c in ("peak_lo_mev", "peak_hi_mev"))
    assert r["attempts"][0]["r_kohm"] == 400.0 and r["attempts"][0]["spikes"] == 0
