"""The tip's state read off a topograph of adatoms (stmbench.analysis.tip): wide low atoms are a
blunt tip, a weaker copy of every atom at one displacement a double one."""
from __future__ import annotations

import numpy as np
import pytest

from stmbench.analysis.tip import atom_shapes, diagnose, pair_vector

ATOMS = [(-10, -9), (-3, -10), (6, -8), (-9, 0), (1, 1), (9, 2), (-6, 9), (4, 9), (11, 10)]


def _frame(atoms=ATOMS, *, size=30.0, px=300, sigma=0.38, h=60e-12, ghost=None, seed=0):
    """A topograph of Gaussian atoms (row 0 at the top), each optionally with a weaker copy at
    ``ghost`` = (dx, dy, ratio)."""
    rng = np.random.default_rng(seed)
    npx = size / px
    c = (np.arange(px) + 0.5) * npx - size / 2
    X, Y = np.meshgrid(c, -c)
    z = rng.normal(0.0, 2e-12, (px, px))
    for x, y in atoms:
        z += h * np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma ** 2))
        if ghost:
            gx, gy, r = ghost
            z += r * h * np.exp(-((X - x - gx) ** 2 + (Y - y - gy) ** 2) / (2 * sigma ** 2))
    return {"array": z, "nm_per_px": npx, "center_nm": [0.0, 0.0], "size_nm": size, "angle_deg": 0.0,
            "pixels": px}


@pytest.mark.parametrize("kw,verdict", [({}, "ok"), ({"sigma": 0.52, "h": 32e-12}, "blunt"),
                                        ({"ghost": (1.5, 1.0, 0.6)}, "double"), ({"atoms": []}, "unknown")])
def test_the_tip_is_judged_by_how_its_atoms_look(kw, verdict):
    """Sharp round atoms; wide low ones (a blunt tip); every atom with a weaker copy at one
    displacement (a double tip); none at all where the task promises some."""
    assert diagnose(_frame(**kw))["verdict"] == verdict


def test_the_atoms_are_measured_where_they_are():
    shapes = atom_shapes(_frame(sigma=0.40))
    assert len(shapes) == len(ATOMS)
    assert np.median([s["sigma_nm"] for s in shapes]) == pytest.approx(0.40, abs=0.02)
    # frame coordinates: from the lower left corner of the 30 nm frame
    found = sorted((round(s["x_nm"] - 15.0), round(s["y_nm"] - 15.0)) for s in shapes)
    assert found == sorted(ATOMS)


def test_a_double_tip_shows_one_displacement_for_every_atom():
    vec, frac, n = pair_vector(_frame(ghost=(-2.0, 1.5, 0.5)))
    assert frac >= 0.5
    assert vec[0] == pytest.approx(-2.0, abs=0.15) and vec[1] == pytest.approx(1.5, abs=0.15)
