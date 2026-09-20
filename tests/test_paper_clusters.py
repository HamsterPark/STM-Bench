"""The baselines' feature finder: compact bright bumps above a LOCAL background, so a bowed
or step-crossing frame does not hide the adatoms behind its own corners."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tests.conftest import requires_mast


def _synthetic(nmpp=0.18, n=384, bumps=((100, 120), (210, 260), (300, 90)), height=70e-12, seed=1):
    """A tilted, bowed frame with a step across one corner, noise, and three adatom bumps."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    z = 2e-12 * xx + 1.5e-12 * yy                                   # tilt
    z += 120e-12 * ((yy - n / 2) / n) ** 2                           # creep bow
    z += np.where(xx + yy > 1.6 * n, 208e-12, 0.0)                   # a step in a corner
    z += 8e-12 * np.cos(2 * np.pi * xx * nmpp / 0.255)              # lattice corrugation
    z += rng.normal(0, 4e-12, (n, n))                                # noise
    for (cx, cy) in bumps:
        z += height * np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / (0.3 / nmpp) ** 2)
    return z


def test_bumps_are_found_and_the_step_and_corners_are_not():
    from stmbench.papers._clusters import blobs_in_array

    z = _synthetic()
    got = blobs_in_array(z, 0.18)
    assert len(got) == 3, got
    for (cx, cy) in ((100, 120), (210, 260), (300, 90)):
        assert any(math.hypot(c - cx, r - cy) < 1.5 for c, r, _h in got), (cx, cy, got)
    heights = [h for _c, _r, h in got]
    assert all(50e-12 < h < 90e-12 for h in heights), heights


def test_the_finder_needs_a_bump_not_just_a_high_corner():
    from stmbench.papers._clusters import blobs_in_array

    z = _synthetic(bumps=())
    assert blobs_in_array(z, 0.18) == []


def test_nan_rows_of_an_unfinished_scan_are_ignored():
    from stmbench.papers._clusters import blobs_in_array

    z = _synthetic()
    z[250:] = np.nan                      # the scan stopped before the bump at row 260
    got = blobs_in_array(z, 0.18)
    assert len(got) == 2 and all(r < 250 for _c, r, _h in got), got


@requires_mast
def test_the_adatoms_of_a_scenario_world_are_found_where_they_are(tmp_path):
    """On the P4 world: the finder's positions, in the scan frame, land on the registry's
    atoms (put into the scan frame through the drift the frame was taken under)."""
    import time

    from stmsim.scenario import Scenario

    from stmbench.papers._clusters import bright_blobs

    sc = Scenario.load("stmbench/trackB/scenarios/P4_atom_positioning_cu111.yaml")
    # this test is about the finder, not about the tip the session arrives with (seed 0's is
# blunt, and a blunt tip's atoms are too soft for it)
    sc.hidden = {k: v for k, v in sc.hidden.items() if k != "tip"}
    w = sc.build_world(0, session_dir=tmp_path / "s")
    w.drift_v_m_per_s = np.zeros(3)      # this test is about the finder, not about the 2.5 nm a frame shears under drift
    w.scan.cx = w.scan.cy = 0.0
    w.scan.w = w.scan.h = 40e-9
    w.scan.nx = w.scan.ny = 256
# an imaging speed of 0.02 s/line: the feedback cannot follow and an adatom comes out
    # at a fifth of its height, which is a different problem from the one this test pins
    w.scan.line_time_fwd_s = w.scan.line_time_bwd_s = 0.3
    w.scan_start()
    time.sleep(0.3 * 256 * 2 / w.clock.time_scale + 1.0)
    w.scan_stop()
    path = w.save_frame()
    ev = [e for e in w.events if e["kind"] == "scan_saved"][-1]
    assert ev["complete"], ev
    reg = w.surface.site.adatoms
    dx, dy = ev["drift_m"]
    xy = reg.positions()
    every = [(x - dx, y - dy) for x, y in xy]
    # the deposit covers far more than one frame, and this finder's 3 nm local background is
    # blind by construction within 3 nm of the frame edge, a step or another atom: count the
    # atoms that sit where it is meant to work
    atoms_scan = [(ax, ay) for (ax, ay), (x, y) in zip(every, xy)
                  if abs(ax) <= 17e-9 and abs(ay) <= 17e-9
                  and w.surface.step_free_window(x, y, 6e-9)
                  and sorted(np.hypot(xy[:, 0] - x, xy[:, 1] - y))[1] >= 2e-9]
    assert len(atoms_scan) >= 3
    found = bright_blobs(path)
    hits = sum(1 for ax, ay in atoms_scan
               if any(math.hypot(ax - fx, ay - fy) < 0.5e-9 for fx, fy in found))
    assert hits >= len(atoms_scan) - 1, (hits, atoms_scan, found[:6])
    ghosts = [f for f in found if not any(math.hypot(f[0] - ax, f[1] - ay) < 0.8e-9 for ax, ay in every)]
    assert len(ghosts) <= 1, ghosts                     # no corner or step "atoms"
