"""The P7 privileged reference solution: two pull thresholds from the outcomes of standard drags."""
from __future__ import annotations

import math

import numpy as np

from stmbench.reference import p7


def test_the_logistic_fit_finds_the_location_and_a_separated_series_its_gap():
    rng = np.random.default_rng(0)
    r = np.exp(rng.uniform(math.log(80), math.log(160), 200))
    y = rng.random(200) < 1 / (1 + np.exp(40 * (np.log(r) - math.log(110))))
    assert abs(p7.fit_threshold(r, y)["r_hat_kohm"] / 110 - 1) < 0.05
    sep = p7.fit_threshold([100, 110, 120, 130], [1, 1, 0, 0])["r_hat_kohm"]
    assert 110 < sep < 120
    assert p7.fit_threshold([100, 110], [1, 1])["r_hat_kohm"] is None


def test_the_reference_measures_two_atoms_and_the_judge_accepts():
    """Seed 3 (an unstable tip: it is conditioned first) by the calibrated protocol, 32 drags per atom.
    Its calibration runs pass it on the desktop and on Linux (docs/results/p7_reference_seeds_0-29.json,
    khwu job 66799). A shorter series is no test: the two platforms' runs part mid-series (the reference
    computes its moves from the truth, and the platforms' floating point differs in the last bits) while a
    recorded episode replays alike on both, and at 14 drags the first atom's error was -17 % on Linux."""
    r = p7.run(3)
    assert r["success"], r
    assert len(r["atoms"]) == 2 and r["atoms"][0]["atom_id"] != r["atoms"][1]["atom_id"]
    for a in r["atoms"]:
        assert a["evidence"]["all"], a["evidence"]
        assert a["n_drags"] >= 8 and 0 < a["n_followed"] < a["n_drags"]
        assert a["start_dist_max_nm"] <= p7.START_WITHIN_NM
        assert abs(a["rel_err"]) < 0.2, a
    assert r["instrument_s"] < 9000
