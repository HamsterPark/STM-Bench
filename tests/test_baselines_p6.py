"""The scripted P6 baseline: nothing but what the instrument shows — the tip judged from images and
from a bare-surface sweep, the atom chosen from a survey, the sweeps centred on it by frames."""
from __future__ import annotations

import math

import numpy as np
import pytest

from stmbench.analysis.adatoms import locate
from stmbench.baselines import p6
from stmbench.baselines._common import Blind
from stmbench.instrument.core import SCENARIO_DIR

P6 = "P6_kondo_co_cu111"
needs_p6 = pytest.mark.skipif(not (SCENARIO_DIR / f"{P6}.yaml").exists(), reason="the P6 scenario is not in this checkout")


def _frame(atoms, *, size=40.0, px=400, sigma=0.38, h=60e-12, step_x=None, seed=0, centre=(0.0, 0.0)):
    """A topograph of Gaussian atoms (row 0 at the top), optionally with a 208 pm step."""
    rng = np.random.default_rng(seed)
    npx = size / px
    c = (np.arange(px) + 0.5) * npx - size / 2
    X, Y = np.meshgrid(c + centre[0], -c + centre[1])
    z = rng.normal(0.0, 2e-12, (px, px))
    for x, y in atoms:
        z += h * np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma ** 2))
    if step_x is not None:
        z += np.where(X > step_x, 208e-12, 0.0)
    return {"array": z, "nm_per_px": npx, "center_nm": [float(centre[0]), float(centre[1])], "size_nm": size, "angle_deg": 0.0,
            "pixels": px}


def _cu_didv(v, *, peak=None, noise=0.015, seed=0):
    """A bare Cu(111) sweep as the instrument reports it: the surface-state step at −0.44 V on a
    rising background, optionally with a tip's Gaussian peak (energy, height, width)."""
    from scipy.special import erf

    rng = np.random.default_rng(seed)
    y = 1.0 + 0.6 * 0.5 * (1.0 + erf((v + 0.44) / 0.012)) + 0.35 * (v + 0.1) + 0.5 * (v + 0.1) ** 2
    if peak is not None:
        e0, a, w = peak
        y = y + a * np.exp(-0.5 * ((v - e0) / w) ** 2)
    return y + rng.normal(0.0, noise * 1.6, v.size)


def test_a_bare_cu_sweep_is_clean_and_a_tip_peak_is_not():
    """The step alone passes for every noise draw; the tip's peak (50 mV wide, 0.5-0.8 of the
    density of states) is found wherever it sits clear of the band edge and of the window's end."""
    v = np.linspace(-0.6, 0.3, 150)
    for seed in range(5):
        assert p6.spectral_flaw(v, _cu_didv(v, seed=seed))["clean"]
    for e0 in (-0.3, -0.15, 0.0, 0.1):                         # the edge's own ±70 mV is not read
        for a in (0.5, 0.8):
            r = p6.spectral_flaw(v, _cu_didv(v, peak=(e0, a, 0.05), seed=1))
            assert not r["clean"], (e0, a, r)
    assert not p6.spectral_flaw(v, _cu_didv(v, peak=(0.25, 0.8, 0.05), seed=1))["clean"]


def test_a_weak_peak_at_the_window_end_is_a_known_miss():
    """A limit of the check, kept visible: a 0.5 peak at +0.25 V, at the window's end, is partly
    absorbed by the quadratic background, and its residual (0.075) stays under the threshold
    (0.08). A tip like that goes on to the atom (the poke before the check mostly buries it)."""
    v = np.linspace(-0.6, 0.3, 150)
    r = p6.spectral_flaw(v, _cu_didv(v, peak=(0.25, 0.5, 0.05), seed=1))
    assert r["clean"] and 0.06 < r["flaw"] < p6.FLAW_FRACTION, r


def test_a_sweep_that_cannot_be_read_is_not_clean():
    v = np.linspace(-0.6, 0.3, 150)
    assert not p6.spectral_flaw(v[:10], np.ones(10))["clean"]
    assert not p6.spectral_flaw(v, np.full(150, np.nan))["clean"]


def test_a_step_in_the_current_marks_its_sweep():
    """A tip change inside a sweep scales the rest of its current; the sweep is found against
    the median of its fellows."""
    v = np.linspace(-0.05, 0.05, 101)
    rng = np.random.default_rng(3)
    base = 1e-9 * v * (1.0 + 2.0 * v)
    sweeps = [base * (1.0 + rng.normal(0.0, 0.01, v.size)) for _ in range(5)]
    assert p6.outlier_sweeps(sweeps, v) == []
    jumped = np.array(sweeps[2])
    jumped[50:] *= 1.3
    sweeps[2] = jumped
    assert p6.outlier_sweeps(sweeps, v) == [2]
    assert p6.outlier_sweeps(sweeps[:2], v) == []              # two sweeps cannot tell which one


ATOMS = [(-10, -9), (-3, -10), (6, -8), (-9, 0), (1, 1), (9, 2), (-6, 9), (4, 9), (11, 10)]


def test_an_atom_needs_isolation_a_terrace_and_a_bare_patch():
    f = _frame(ATOMS + [(1.0, 3.5)])                           # (1, 1) now has a neighbour 2.5 nm away
    got = {(round(a[0][0]), round(a[0][1])): a[1] for a in p6.candidates(f, locate(f), locate(f, compact_frac=1.0))}
    assert (1, 1) not in got and (1, 4) not in got and (-9, 0) in got
    for (x, y), (bx, by) in got.items():
        assert 3.4 <= math.hypot(bx - x, by - y) <= 5.6        # the patch is 3.5-5.5 nm away
        assert all(math.hypot(bx - ax, by - ay) >= p6.BARE_CLEAR_NM - 0.01 for ax, ay in ATOMS + [(1.0, 3.5)])
    stepped = _frame(ATOMS, step_x=3.0)                        # a step 2 nm beside (1, 1)
    got = {(round(a[0][0]), round(a[0][1])) for a in p6.candidates(stepped, locate(stepped), locate(stepped, compact_frac=1.0))}
    assert (1, 1) not in got and (-9, 0) in got


class _Frames:
    """An ``Ops`` that images a fixed set of atoms about whatever centre it is asked for."""

    def __init__(self, atoms):
        self.atoms, self.v, self.n = atoms, None, 0

    def frame(self, cx, cy, size, px, line_s):
        self.n += 1
        return _frame(self.atoms, size=size, px=px, centre=(cx, cy), seed=self.n), 0.0


def test_the_bare_patch_follows_the_atom_as_the_sample_drifts():
    """Seed 6 of the dev runs: the patch was worked out in the survey's coordinates and the sample
    drifted 1.7 nm before the reference sweeps, which then lay 1.84 nm from the atom. It keeps
    its offset from the atom as the atom is now."""
    atom, bare, here = (-8.5, -2.2), (-5.0, -2.3), (-6.8, -0.5)
    got = p6.bare_from(atom, bare, here)
    assert got == pytest.approx((-3.3, -0.6))
    assert math.hypot(got[0] - here[0], got[1] - here[1]) == pytest.approx(math.hypot(bare[0] - atom[0], bare[1] - atom[1]))


def test_the_bare_patch_keeps_2_8_nm_from_every_bump_the_atom_measured_included():
    """The atom measured counts as an obstacle even when the frame about the patch does not show
    it (the judge asks for 2 nm from every atom), and so does a bump the survey missed."""
    from stmbench.baselines.p6 import Bench

    here = (0.0, 0.0)
    spot = (1.84, 0.0)                                         # as in the failure: 1.84 nm from the atom
    for atoms in ([], [(3.0, 2.0)]):
        bench = Bench(_Frames(atoms), {}, 4.36)
        got = bench.clear_bare(spot, here)
        for ax, ay in [here] + atoms:
            assert math.hypot(got[0] - ax, got[1] - ay) >= 2.8, (atoms, got)
    assert Bench(_Frames([]), {}, 4.36).clear_bare((5.0, 0.0), here) == (5.0, 0.0)     # far enough: left alone


def test_an_atom_lost_is_not_taken_again():
    f = _frame(ATOMS)
    first = p6.candidates(f, locate(f), locate(f, compact_frac=1.0))
    assert first
    x, y = first[0][0][:2]
    again = p6.candidates(f, locate(f), locate(f, compact_frac=1.0), avoid=[(x, y)])
    assert all(math.hypot(a[0][0] - x, a[0][1] - y) > p6.AVOID_NM for a in again)


def test_the_baseline_names_neither_the_simulator_nor_its_truth():
    """Inspection, as for the P4 baseline: outside ``run``, which scores the episode once it is
    over, and ``Blind``, the one place that holds the instrument, the baseline's code reaches for
    no instrument, world, truth, hidden draw or verdict."""
    import ast
    import inspect

    from stmbench.baselines import _common

    forbidden = {"world", "truth", "hidden", "draw_hidden", "verdict", "scenario", "inst"}
    for mod in (p6, _common):
        for node in ast.parse(inspect.getsource(mod)).body:
            if (mod is p6 and isinstance(node, ast.FunctionDef) and node.name == "run") or \
                    (mod is _common and isinstance(node, ast.ClassDef) and node.name == "Blind"):
                continue
            names = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
            names |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            assert not names & forbidden, (mod.__name__, getattr(node, "name", None), names & forbidden)


class _NoWorld:
    """An instrument whose simulator cannot be reached: what an agent over MCP has."""

    def __init__(self, inst):
        self._inst = inst

    def __getattr__(self, name):
        if name in ("act", "frame", "spectrum", "status", "info", "task", "report"):
            return getattr(self._inst, name)
        raise AttributeError(f"the baseline reached for {name!r}")


@needs_p6
def test_a_reduced_session_reaches_a_report_that_cites_sweeps(monkeypatch):
    """One seed through the agent's interface only (the guard raises at any other attribute):
    survey, pokes, the check of the tip, the drift, the atom, sweeps on it and on its bare patch,
    the fit; the report cites the sweeps, and the episode finishes with the evidence judged."""
    pytest.importorskip("stmbench.analysis.kondo")
    from stmbench.instrument import Instrument

    monkeypatch.setattr(p6, "N_ATOM", 3)
    monkeypatch.setattr(p6, "N_BARE", 2)
    inst = Instrument(P6, 24)
    log: dict = {}
    got = p6.play(Blind(_NoWorld(inst)), log)
    assert got is not None and log["n_sweeps"] >= 6
    assert set(got["evidence"]) <= set(inst.spectra) and len(got["atom_sweeps"]) >= 2 and got["bare_sweeps"]
    for cid in ("gamma_mev", "q_fano", "e_k_mev"):
        inst.report(cid, got["fit"][cid], evidence=got["evidence"])
    out = inst.finish()
    ev = inst.verdict.details["claims"]
    assert all(c["evidence_ok"] for c in ev.values()), (out, ev)
    assert inst.used_s < 5400.0
