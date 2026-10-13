"""The scripted P4 baseline: nothing but what the instrument shows — the tip judged from
frames (``test_analysis_tip``), the atom chosen from a survey, the move placed against
fiducials."""
from __future__ import annotations

import numpy as np
import pytest

from stmbench.analysis.adatoms import locate
from stmbench.baselines import p4
from stmbench.baselines._common import Blind
from stmbench.instrument import Instrument

P4 = "P4_atom_positioning_cu111"
ATOMS = [(-10, -9), (-3, -10), (6, -8), (-9, 0), (1, 1), (9, 2), (-6, 9), (4, 9), (11, 10)]


def _frame(atoms=ATOMS, *, size=30.0, px=300, sigma=0.38, h=60e-12, ghost=None, step_x=None, seed=0):
    """A topograph of Gaussian atoms (row 0 at the top), optionally each with a weaker copy
    at ``ghost`` = (dx, dy, ratio), and a 208 pm step up at ``step_x`` (as in
    ``test_analysis_tip``, plus the step)."""
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
    if step_x is not None:
        z += np.where(X > step_x, 208e-12, 0.0)
    return {"array": z, "nm_per_px": npx, "center_nm": [0.0, 0.0], "size_nm": size, "angle_deg": 0.0,
            "pixels": px}


def test_an_atom_is_moved_only_where_nothing_is_in_its_way():
    """Isolated, a free path and goal inside the frame, and no step near the path: a step
    edge crossing x = 3 nm rules out the two atoms whose paths come near it."""
    d = (4.0, 0.0)
    flat = _frame()
    got = {(round(a[0]), round(a[1])) for a in p4.candidates(flat, locate(flat), d)}
    assert {(1, 1), (4, 9), (-9, 0)} <= got
    assert not got & {(9, 2), (11, 10)}                       # their goals leave the frame
    stepped = _frame(step_x=3.0)
    got = {(round(a[0]), round(a[1])) for a in p4.candidates(stepped, locate(stepped), d)}
    assert (1, 1) not in got and (4, 9) not in got and (-9, 0) in got


def test_the_displacement_is_read_off_the_task():
    assert p4.task_vector(Instrument(P4, 0).task()) == (4.0, 0.0)


class _NoWorld:
    """An instrument whose simulator cannot be reached: what an agent over MCP has."""

    def __init__(self, inst):
        self._inst = inst

    def __getattr__(self, name):
        if name in ("act", "frame", "spectrum", "status", "info", "task", "report"):
            return getattr(self._inst, name)
        raise AttributeError(f"the baseline reached for {name!r}")


def test_the_baseline_plays_through_what_an_agent_gets():
    """Seed 24 arrives with a good tip: three pokes, the drift, a survey, the ladder, and a
    verification frame the judge accepts — all through the agent's interface."""
    inst = Instrument(P4, 24)
    log: dict = {}
    got = p4.play(Blind(_NoWorld(inst)), log)
    assert got is not None and log["tip_first"] == "ok"
    x, y = got["xy"]
    inst.report("target_site", x, unit="nm", x_nm=x, y_nm=y, frame=got["frame"]["frame"],
                evidence=[got["frame"]["frame"]])
    assert inst.finish()["success"]


def test_a_blunt_tip_is_seen_and_conditioned_first():
    """Seed 3 arrives with a 4 nm tip: its atoms show wide and low, so it is pulsed far away and
    judged again on the survey before anything is moved."""
    r = p4.run(3)
    assert r["success"], r
    assert r["tip_first"] == "blunt" and r["tip_rounds"] >= 1


def test_the_baseline_names_neither_the_simulator_nor_its_truth():
    """Inspection, as for the mode-C baselines: outside ``run``, which scores the episode once
    it is over, and ``Blind``, the one place that holds the instrument, the baseline's code
    reaches for no instrument, world, truth, hidden draw or verdict."""
    import ast
    import inspect

    from stmbench.analysis import tip
    from stmbench.baselines import _common

    forbidden = {"world", "truth", "hidden", "draw_hidden", "verdict", "scenario", "inst"}
    for mod in (p4, _common, tip):
        for node in ast.parse(inspect.getsource(mod)).body:
            if (mod is p4 and isinstance(node, ast.FunctionDef) and node.name == "run") or                     (mod is _common and isinstance(node, ast.ClassDef) and node.name == "Blind"):
                continue
            names = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
            names |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            assert not names & forbidden, (mod.__name__, getattr(node, "name", None), names & forbidden)


def test_a_pair_too_close_to_tell_apart_still_counts_as_a_neighbour():
    """Two atoms 0.9 nm apart image as one elongated bump, which the compactness test that
    makes a bump an atom turns away (seed 185 moved an atom 1.8 nm from such a pair, which was
    not isolated). Checked against every bump, the atom beside the pair is not taken."""
    import math

    d = (4.0, 0.0)
    f = _frame(ATOMS + [(-1.0, 2.5), (-1.9, 2.5)])
    compact = locate(f)
    assert not any(math.hypot(a[0] + 1.45, a[1] - 2.5) < 0.8 for a in compact)
    naive = {(round(a[0]), round(a[1])) for a in p4.candidates(f, compact, d)}
    careful = {(round(a[0]), round(a[1])) for a in p4.candidates(f, compact, d, obstacles=locate(f, compact_frac=1.0))}
    assert (1, 1) in naive and (1, 1) not in careful


def test_a_row_jump_that_recurs_is_the_surface():
    """A step running along the rows steps the whole row, as a tip change does; only a retake
    tells them apart: the step comes back at the same row (seed 159)."""
    from stmbench.baselines._common import _same_rows

    assert _same_rows([120], [122]) and _same_rows([40, 200], [41, 198])
    assert not _same_rows([120], [160]) and not _same_rows([120], [120, 200]) and not _same_rows([], [])
