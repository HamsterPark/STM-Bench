"""The scripted P1 baseline: two rotational domains of the Au(111) herringbone from what the
instrument shows — a survey, a drift measured on an adatom, the shared stripe analysis."""
from __future__ import annotations

import ast
import inspect

from stmbench.baselines import p1
from stmbench.baselines._common import Blind
from stmbench.instrument import Instrument

P1 = "P1_barth1990_au111"


class _NoWorld:
    """An instrument whose simulator cannot be reached: what an agent over MCP has."""

    def __init__(self, inst):
        self._inst = inst

    def __getattr__(self, name):
        if name in ("act", "frame", "spectrum", "status", "info", "task", "report"):
            return getattr(self._inst, name)
        raise AttributeError(f"the baseline reached for {name!r}")


def test_the_baseline_names_neither_the_simulator_nor_its_truth():
    """Inspection: outside ``run``, which scores the episode once it is over, and ``Blind``, the one
    place that holds the instrument, the baseline's code and the analysis it uses reach for no
    instrument, world, truth, hidden draw, verdict or seed."""
    from stmbench.analysis import herringbone
    from stmbench.baselines import _common

    # ``verdict`` is not listed: the stripe analysis calls its own reading of a window a verdict, and the
    # judge's one is only reachable through ``inst``
    forbidden = {"world", "truth", "hidden", "draw_hidden", "scenario", "inst", "seed"}
    for mod in (p1, _common, herringbone):
        for node in ast.parse(inspect.getsource(mod)).body:
            if (mod is p1 and isinstance(node, ast.FunctionDef) and node.name == "run") or \
                    (mod is _common and isinstance(node, ast.ClassDef) and node.name == "Blind"):
                continue
            names = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
            names |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            assert not names & forbidden, (mod.__name__, getattr(node, "name", None), names & forbidden)


def test_two_arms_are_found_and_measured_through_what_an_agent_gets():
    """Seed 13: three pokes, a scouting frame, a drift measured on an adatom, a 120 nm frame read by the
    stripe analysis, two places of directions at least 40 degrees apart reported with the arm period.
    Development seed: the baseline's period error is a few hundredths of a nm against 0.15 and its direction
    errors a degree or two."""
    inst = Instrument(P1, 13)
    log: dict = {}
    got = p1.play(Blind(_NoWorld(inst)), log)
    a, b = got["a"], got["b"]
    assert abs(p1.circ_diff(a.direction_deg, b.direction_deg)) >= p1.MIN_SEP_DEG
    inst.report("stripe_period_nm", got["period_nm"], unit="nm", evidence=[f["frame"] for f in got["frames"]])
    for cid, r in (("stripe_orientation_deg", a), ("domain_a_orientation_deg", a), ("domain_b_orientation_deg", b)):
        inst.report(cid, r.direction_deg, unit="deg", x_nm=r.x, y_nm=r.y, frame=r.frame["frame"], evidence=[r.frame["frame"]])
    out = inst.finish()
    det = inst.verdict.details["claims"]
    assert out["success"], det
    assert all(c["evidence_ok"] for c in det.values()), det
    assert inst.used_s < inst.max_sim_s
    assert log.get("drift_nm_min"), "the drift was never measured"


def test_an_adatom_is_taken_out_of_a_frame_without_touching_the_lines():
    """An atom of 180 pm moved a window's arm period by 0.5 nm; replaced by the median of a ring around it
    it leaves the lines as they were."""
    import numpy as np

    n, d = 300, 0.4
    c = (np.arange(n) + 0.5) * d
    X, Y = np.meshgrid(c, -c)
    lines = 10e-12 * np.cos(2 * np.pi * (X * 0.6 + Y * 0.8) / 6.3)
    atom = 180e-12 * np.exp(-((X - 60) ** 2 + (Y + 60) ** 2) / (2 * 1.5 ** 2))
    f = {"array": lines + atom, "nm_per_px": d, "center_nm": [60.0, -60.0], "size_nm": 120.0, "pixels": n, "angle_deg": 0.0}
    g = p1.without_atoms(f)
    r0, c0 = 150, 150
    assert g["array"][r0, c0] < 30e-12 < f["array"][r0, c0]
    far = np.s_[:60, :60]
    assert np.allclose(g["array"][far], f["array"][far])


def test_run_returns_the_row_of_the_benchmark():
    """The CLI row: the claims, the errors against the accepted truth (after ``finish`` only), the
    instrument time and the tip the episode arrived with."""
    r = p1.run(13)
    assert r["paper"] == "P1" and r["seed"] == 13 and r["success"], r
    for k in ("claims_ok", "abs_err", "instrument_s", "budget_s", "actions", "tip_condition", "reported"):
        assert k in r
    assert abs(r["abs_err"]["period_nm"]) < 0.15
