"""The scripted P3-repair baseline: nothing but what the instrument shows. The ring, its gap and the
spares are read from a survey, the spares are placed against the ring in the same frame, the
measurement afterwards is the P3 baseline's."""
from __future__ import annotations

import math

import numpy as np
import pytest

from stmbench.analysis.corral import ring_gap, ring_profile
from stmbench.baselines import PAPERS, p3_repair
from stmbench.baselines._common import Blind
from stmbench.instrument import Instrument

SCEN = "P3_corral_repair_cu111"
RADIUS, N_RING = 7.0, 50


def _frame(*, sigma=0.35, height=60e-12, gap=(0.0, 4), spares=((10.0, 3.0), (11.0, -2.0), (9.0, -5.5), (10.5, 0.5)),
           size=36.0, px=360, seed=0):
    """A topograph of a ring of Gaussian atoms centred on the frame's middle with ``gap[1]`` sites
    missing from site ``gap[0]`` on, and some spares outside it (row 0 at the top)."""
    rng = np.random.default_rng(seed)
    npx = size / px
    c = (np.arange(px) + 0.5) * npx - size / 2
    X, Y = np.meshgrid(c, -c)
    z = rng.normal(0.0, 2e-12, (px, px))
    atoms = [(RADIUS * math.cos(2 * math.pi * k / N_RING), RADIUS * math.sin(2 * math.pi * k / N_RING))
             for k in range(N_RING) if not gap[0] <= k < gap[0] + gap[1]] + list(spares)
    for x, y in atoms:
        z += height * np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma ** 2))
    return {"array": z, "nm_per_px": npx, "center_nm": [0.0, 0.0], "size_nm": size, "angle_deg": 0.0, "pixels": px}


def test_it_is_registered_as_paper_p3r():
    assert PAPERS["P3R"] is p3_repair.run


def test_the_ring_the_spares_and_the_gap_are_read_from_one_frame():
    f = _frame()
    r = p3_repair.ring_of(f)
    assert math.hypot(*r["centre"]) < 0.2 and abs(r["radius_nm"] - RADIUS) < 0.15
    sp = p3_repair.spares_of(f, r["centre"], r["radius_nm"])
    assert len(sp) == 4
    th, prof = ring_profile(f["array"], f["nm_per_px"], r["centre_frame_nm"], r["radius_nm"])
    g = ring_gap(th, prof, r["radius_nm"], missing=len(sp), smooth_nm=p3_repair.GAP_SMOOTH_NM)
    want = [2 * math.pi * (k) / N_RING for k in range(4)]
    got = sorted(g["sites_rad"])
    assert max(abs(a - b) for a, b in zip(got, want)) < math.radians(2.5)


def test_a_wide_tip_is_called_blunt_and_a_good_one_is_not():
    sharp = _frame()
    blunt = _frame(sigma=0.75, height=30e-12)
    for f, verdict in ((sharp, "ok"), (blunt, "blunt")):
        r = p3_repair.ring_of(f)
        sp = p3_repair.spares_of(f, r["centre"], r["radius_nm"])
        assert p3_repair.judge_tip(f, r, sp)["verdict"] == verdict, (verdict, p3_repair.judge_tip(f, r, sp))


def test_no_spare_in_sight_is_a_blunt_tip_only_before_the_repair_has_begun():
    f = _frame(spares=())
    r = p3_repair.ring_of(f)
    assert p3_repair.judge_tip(f, r, [])["verdict"] == "blunt"
    assert p3_repair.judge_tip(f, r, [], 0)["verdict"] == "ok"             # every spare is placed


def test_spares_are_paired_with_sites_in_angular_order():
    spares = [(10.0, 3.0, 1.0), (10.0, -3.0, 1.0)]                           # the upper one first when sorted from -x
    sites = [math.radians(-10), math.radians(10)]
    pairs = p3_repair._pairs(spares, sites, (0.0, 0.0))
    assert [round(p[0][1]) for p in pairs] == [-3, 3] and [round(math.degrees(p[1])) for p in pairs] == [-10, 10]


class _NoWorld:
    """An instrument whose simulator cannot be reached: what an agent over MCP has."""

    def __init__(self, inst):
        self._inst = inst

    def __getattr__(self, name):
        if name in ("act", "frame", "spectrum", "status", "info", "task", "report"):
            return getattr(self._inst, name)
        raise AttributeError(f"the baseline reached for {name!r}")


def test_the_baseline_plays_through_what_an_agent_gets():
    """Seed 9 arrives with a good tip and four atoms missing: the spares are found in a survey,
    moved one by one by the resistance ladder, and the corral then measured, all through the agent's
    interface (a few minutes of CPU)."""
    inst = Instrument(SCEN, 9)
    log: dict = {}
    m = p3_repair.play(Blind(_NoWorld(inst)), log)
    assert m is not None and log["spares_seen"] == 4 and log["placed"] == 4
    assert abs(log["planned_deg"][0] - 2.1) < 3.0          # the sites come from the ring's profile, not from the truth


def test_the_baseline_names_neither_the_simulator_nor_its_truth():
    """Inspection, as for the P4 baseline: outside ``run``, which scores the episode once it is
    over, and ``Blind``, the one place that holds the instrument, the baseline's code reaches for no
    instrument, world, truth, hidden draw or verdict."""
    import ast
    import inspect

    from stmbench.baselines import _common

    forbidden = {"world", "truth", "hidden", "draw_hidden", "verdict_obj", "scenario", "inst"}
    for mod in (p3_repair, _common):
        for node in ast.parse(inspect.getsource(mod)).body:
            if (mod is p3_repair and isinstance(node, ast.FunctionDef) and node.name in ("run", "_site_offsets")) or \
                    (mod is _common and isinstance(node, ast.ClassDef) and node.name == "Blind"):
                continue
            names = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
            names |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            assert not names & forbidden, (mod.__name__, getattr(node, "name", None), names & forbidden)
