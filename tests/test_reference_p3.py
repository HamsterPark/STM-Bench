"""The P3 privileged reference solution: radius from an image, resonances from spectra."""
from __future__ import annotations

import pytest

from stmbench.reference import p3


@pytest.mark.parametrize("seed", [0, 13])
def test_the_reference_reproduces_p3_from_measured_data(seed):
    """Seed 0's tip is blunt (conditioned far from the ring, then the creep waited out), seed
    13's carries an adsorbate (its own states would put peaks into every spectrum)."""
    r = p3.run(seed)
    assert r["success"], r
    assert all(r["claims_ok"].values())
    assert abs(r["abs_err"]["radius_nm"]) < 0.15            # seed 0: 0.12, the worst of seeds 0-29
    assert all(abs(r["abs_err"][c]) < 6.0 for c in ("peak_lo_mev", "peak_hi_mev"))
    assert r["values"]["peak_hi_mev"] > r["values"]["peak_lo_mev"]
    assert r["tip_rounds"] >= 1
