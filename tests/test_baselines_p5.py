"""The scripted P5 baseline: force spectroscopy over one Fe atom with nothing but what the
instrument shows — the site chosen from a survey, the drift and the bond's centre found from
frames and curves, the numbers measured and judged."""
from __future__ import annotations

import math

import numpy as np

from stmbench.baselines import p5
from stmbench.baselines._common import Blind
from stmbench.instrument import Instrument

P5 = "P5_sader_jarvis_qplus"
# one atom at (-3, 0) with nothing within 3 nm and room around it, six others far from it
ATOMS = [(-3.0, 0.0), (9.0, 9.0), (-9.0, 10.0), (10.0, -9.0), (-10.0, -9.0), (2.0, 11.0), (12.0, 3.0)]


def _frame(atoms=ATOMS, *, size=36.0, px=360, sigma=0.38, h=60e-12, step_x=None, seed=0):
    """A topograph of Gaussian atoms (row 0 at the top), optionally with a 208 pm step up at ``step_x``."""
    rng = np.random.default_rng(seed)
    npx = size / px
    c = (np.arange(px) + 0.5) * npx - size / 2
    X, Y = np.meshgrid(c, -c)
    z = rng.normal(0.0, 2e-12, (px, px))
    for x, y in atoms:
        z += h * np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma ** 2))
    if step_x is not None:
        z += np.where(X > step_x, 208e-12, 0.0)
    return {"array": z, "nm_per_px": npx, "center_nm": [0.0, 0.0], "size_nm": size, "angle_deg": 0.0, "pixels": px}


class _NoWorld:
    """An instrument whose simulator cannot be reached: what an agent over MCP has."""

    def __init__(self, inst):
        self._inst = inst

    def __getattr__(self, name):
        if name in ("act", "frame", "spectrum", "status", "info", "task", "report"):
            return getattr(self._inst, name)
        raise AttributeError(f"the baseline reached for {name!r}")


def test_the_baseline_measures_one_atom_through_what_an_agent_gets():
    """Seed 1 arrives with a good tip: three pokes, the drift, a survey, the bond's centre from a
    grid of coarse curves, seven curves, and numbers the judge accepts together with the evidence
    (a development seed: the tolerances are 15 / 8 / 20 %, the errors on this seed a few per cent)."""
    inst = Instrument(P5, 1)
    log: dict = {}
    got = p5.play(Blind(_NoWorld(inst)), log)
    assert log["tip_first"] == "ok" and len(got["names"]) == 7 and len(got["atom"]) == 3
    q = inst.info()["qplus"]
    from stmbench.analysis.force import p5_numbers
    values = p5_numbers(got["atom"], got["bare"], f0=q["f0_hz"], k=q["k_n_per_m"], amplitude=got["amp_m"])
    for cid, unit in p5.UNITS.items():
        inst.report(cid, float(values[cid]), unit=unit, evidence=got["names"])
    out = inst.finish()
    det = inst.verdict.details["claims"]
    assert all(det[c]["evidence_ok"] for c in p5.UNITS), det
    assert out["success"], det
    assert inst.used_s < inst.max_sim_s / 2


def test_the_summary_row_compares_with_the_truth_only_after_the_episode():
    r = p5.run(1)
    assert r["success"] and r["paper"] == "P5" and set(r["rel_err"]) == set(p5.UNITS)
    assert r["values"] != r["truth"]
    assert r["tip_condition"] == "good" and r["actions"] > 100 and r["budget_s"] == 5400.0
    assert all(abs(x) < 0.5 for x in r["near_nm"][1:2])           # the first atom curve is on the atom


def test_a_site_needs_an_isolated_atom_a_flat_terrace_and_bare_spots_on_both_sides():
    """The atom at (-3, 0) is the one that qualifies (the others are inside the frame's margin or
    have a neighbour); its spots lie opposite each other, 2.5 nm or more from every atom. A step
    through the middle of the frame takes the terrace under the atom's spots away."""
    sites = p5.pick_site(_frame())
    assert sites and math.hypot(sites[0]["atom"][0] + 3.0, sites[0]["atom"][1]) < 0.1
    a, (s1, s2) = sites[0]["atom"], sites[0]["spots"]
    assert math.hypot(s1[0] + s2[0] - 2 * a[0], s1[1] + s2[1] - 2 * a[1]) < 1e-6      # opposite sides
    for s in (s1, s2):
        assert all(math.hypot(s[0] - x, s[1] - y) >= p5.BARE_CLEAR_NM - 0.1 for x, y in ATOMS)
    assert not any(abs(s["atom"][0] + 3.0) < 0.5 for s in p5.pick_site(_frame(step_x=-1.0)))
    crowded = ATOMS + [(-3.0, 2.0)]                                # a neighbour 2 nm away
    assert not any(abs(s["atom"][0] + 3.0) < 0.5 and abs(s["atom"][1]) < 0.5 for s in p5.pick_site(_frame(crowded)))


def test_the_baseline_names_neither_the_simulator_nor_its_truth():
    """Inspection, as for the other baselines: outside ``run``, which scores the episode once it is
    over, and ``Blind``, the one place that holds the instrument, the baseline's code reaches for no
    instrument, world, truth, hidden draw or verdict."""
    import ast
    import inspect

    from stmbench.analysis import force, tip
    from stmbench.baselines import _common

    forbidden = {"world", "truth", "hidden", "draw_hidden", "verdict", "scenario", "inst", "reference"}
    for mod in (p5, _common, tip, force):
        for node in ast.parse(inspect.getsource(mod)).body:
            if (mod is p5 and isinstance(node, ast.FunctionDef) and node.name == "run") or \
                    (mod is _common and isinstance(node, ast.ClassDef) and node.name == "Blind"):
                continue
            names = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
            names |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            names |= {a.module.split(".")[-1] for a in ast.walk(node) if isinstance(a, ast.ImportFrom) and a.module}
            assert not names & forbidden, (mod.__name__, getattr(node, "name", None), names & forbidden)
    imported = {a.module for a in ast.walk(ast.parse(inspect.getsource(p5))) if isinstance(a, ast.ImportFrom) and a.module}
    assert not any("reference" in m for m in imported), imported          # the privileged package stays out


def test_a_shallow_bond_is_centred_by_the_fit_against_the_bare_curve():
    """Seed 446 has the weakest bond of the ranges tried (a dip of 0.8 Hz). The minimum of the raw coarse
    curves is dominated by the van der Waals part and the fit failed there (se 0.15 nm); against the bare
    curve at equal loop Z the fit is accepted by its standard error and the tip ends on the atom."""
    r = p5.run(446)
    assert r["success"], r
    assert max(r["centre"][0]["se_nm"]) < p5.CENTRE_MAX_SE_NM
    assert r["centre_delta_nm"] != [0.0, 0.0]
    assert all(x < 0.05 for x in r["near_nm"][1:6:2])               # the three atom curves
