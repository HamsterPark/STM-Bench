"""Reading whether a 1.0 nm drag took the atom to the end point (``stmbench.analysis.carry``) and the
logistic fit of P7's baseline. The reader's rates against the judge's record are measured on the
world's own drags (``test_a_small_set_of_the_worlds_drags_reads_right``); the rest is the logic."""
from __future__ import annotations

import ast
import inspect
import math

import numpy as np
import pytest

from stmbench.analysis import carry as C
from stmbench.baselines import p7

ATOM = [(1.0, 1.0)]


def _frame(atoms, *, size=5.0, px=50, sigma=0.35, h=60e-12, center=(0.0, 0.0), seed=0):
    rng = np.random.default_rng(seed)
    npx = size / px
    c = (np.arange(px) + 0.5) * npx - size / 2
    X, Y = np.meshgrid(c + center[0], -c + center[1])
    z = rng.normal(0.0, 1.5e-12, (px, px))
    for x, y in atoms:
        z += h * np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma ** 2))
    return {"array": z, "nm_per_px": npx, "center_nm": list(center), "size_nm": size, "angle_deg": 0.0, "pixels": px}


def test_the_atom_of_an_after_frame_is_the_one_on_the_path():
    f = _frame([(0.2, 0.0), (2.2, 2.0)])
    got = C.find_atom(f, (-0.5, 0.0), (0.5, 0.0))
    assert got is not None and abs(got[0] - 0.2) < 0.08 and abs(got[1]) < 0.08
    assert C.find_atom(_frame([(2.2, 2.0)]), (-0.5, 0.0), (0.5, 0.0)) is None      # no atom on the path: on the tip


def test_an_atom_at_the_end_is_carried_and_one_a_site_short_is_not():
    end = (0.5, 0.0)
    near = C.read_carry(None, end=end, atom=(0.45, 0.07))
    short = C.read_carry(None, end=end, atom=(0.5 - 0.255, 0.0))
    assert near["carried"] is True and near["how"] == "frame" and near["end_offset_nm"] < C.CARRY_RADIUS_NM
    assert short["carried"] is False and abs(short["end_offset_nm"] - 0.255) < 1e-9
    # the radius is the judge's: 0.6 of Cu(111)'s 0.255 nm spacing
    assert math.isclose(C.CARRY_RADIUS_NM, 0.6 * 0.255, abs_tol=1e-3)


def _trace(z_pm, feedback=True):
    z = np.asarray(z_pm, float) * 1e-12
    return {"z_abs": z, "dist": np.linspace(0.0, 1e-9, z.size), "feedback": feedback}


def test_the_trace_is_the_fallback_when_no_atom_was_found_or_no_frame_taken():
    rise, fall = np.linspace(0, 10, 21), np.linspace(0, -40, 21)
    assert C.read_carry(_trace(rise), end=(0, 0)) == {"carried": True, "how": "trace", "end_offset_nm": None,
                                                      "fall_pm": pytest.approx(C.trace_fall_pm(_trace(rise)))}
    assert C.read_carry(_trace(fall), end=(0, 0))["carried"] is False
    assert C.read_carry(_trace(fall, feedback=False), end=(0, 0))["carried"] is None     # a held Z shows nothing
    assert C.read_carry(None, end=(0, 0))["carried"] is None
    # with a frame, the frame decides whatever the trace says
    assert C.read_carry(_trace(fall), end=(0, 0), atom=(0.0, 0.05))["carried"] is True


def test_the_logistic_fit_finds_the_half_point_and_needs_both_outcomes():
    rng = np.random.default_rng(3)
    r = 200e3 * np.exp(rng.uniform(-0.4, 0.4, 400))
    p = 1.0 / (1.0 + np.exp(25.0 * (np.log(r) - np.log(200e3))))
    got = p7.fit_logistic(r, rng.random(400) < p)
    assert got is not None and abs(got["r_ohm"] / 200e3 - 1.0) < 0.04
    assert p7.fit_logistic([100e3, 120e3], [True, True]) is None and p7.fit_logistic([100e3], [False]) is None
    # separated data: the fit sits between the last carry and the first miss
    sep = p7.fit_logistic([100e3, 120e3, 140e3, 180e3, 200e3, 240e3], [True, True, True, False, False, False])
    assert 140e3 <= sep["r_ohm"] <= 180e3


def test_the_series_looks_for_both_outcomes_then_follows_its_fit_in_small_dithered_steps():
    s = p7.Series()
    r0 = s.r
    read = lambda c: {"carried": c, "how": "frame", "end_offset_nm": 0.0, "fall_pm": 0.0}   # noqa: E731
    s.add({"r_ohm": s.r, "read": read(True), "trace": "a"})
    assert math.isclose(s.r, r0 * p7.PROTO.up)                            # only carries so far: up
    s2 = p7.Series()
    s2.add({"r_ohm": s2.r, "read": read(False), "trace": "a"})
    assert math.isclose(s2.r, r0 * p7.PROTO.down)
    # both outcomes seen: the next resistance is the fit's (within x1.3 of the last), one step up or down
    rng = np.random.default_rng(1)
    s3 = p7.Series()
    for _ in range(24):
        p = 1.0 / (1.0 + math.exp(20.0 * (math.log(s3.r) - math.log(150e3))))
        s3.add({"r_ohm": s3.r, "read": read(bool(rng.random() < p)), "trace": "x"})
    assert abs(s3.estimate()["r_ohm"] / 150e3 - 1.0) < 0.15
    assert abs(s3.r / 150e3 - 1.0) < 0.2                                  # and it has gone to the half point


def test_the_reader_and_the_baseline_reach_for_no_truth():
    """Inspection, as for the P4 baseline: the code the agent's equivalent would run names no world,
    truth, hidden draw or verdict outside ``run``, which scores the episode once it is over."""
    forbidden = {"world", "truth", "hidden", "draw_hidden", "verdict", "scenario", "inst", "drag_records"}
    for mod in (C, p7):
        for node in ast.parse(inspect.getsource(mod)).body:
            if mod is p7 and isinstance(node, ast.FunctionDef) and node.name == "run":
                continue
            names = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
            names |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            assert not names & forbidden, (mod.__name__, getattr(node, "name", None), names & forbidden)


def test_an_atom_that_walked_during_the_first_series_is_not_offered_as_the_second():
    """Held-out seed 451 (2026-11-17): the first atom walked 6 nm along x during its 32 alternating
    drags and the second atom was chosen at its new place, which the survey offered because only the
    place where the series began was excluded. Every place the atom was found counts."""
    places = [(3.0, 4.1), (4.0, 4.2), (6.0, 4.4), (8.8, 4.6), (9.9, 4.6)]      # A (3, 4.1) ... B (9.9, 4.6)
    assert not p7.unclaimed(9.9, 4.6, [], places)                              # where it is now
    assert not p7.unclaimed(3.0, 4.1, [], places)                              # where it began
    assert not p7.unclaimed(6.3, 7.0, [], places)                              # beside its path
    assert p7.unclaimed(3.0, 12.0, [], places) and p7.unclaimed(17.0, 4.6, [], places)   # 5+ nm from the whole path
    assert not p7.unclaimed(20.0, 20.0, [(21.0, 20.0)], places)                # still not near an atom lost
    s = p7.Series(places=[(0.0, 0.0)])
    assert s.places == [(0.0, 0.0)] and p7.Series().places == []
