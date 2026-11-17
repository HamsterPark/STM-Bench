"""The scripted P3 baseline: nothing but what the instrument shows — the ring found in a survey,
the tip judged from the ring and the spare atoms, a bare spot picked from the survey."""
from __future__ import annotations

import numpy as np
import pytest

from stmbench.analysis.corral import corral_radius, ghost_strength, locate_corral, ring_width_nm
from stmbench.baselines import p3
from stmbench.baselines._common import Blind
from stmbench.instrument import Instrument

P3 = "P3_corral_cu111"


def _ring_frame(*, centre=(-5.0, 3.0), radius=7.0, n=48, size=60.0, px=300, sigma=0.3, h=60e-12, ghost=None,
                step_x=None, spares=(), seed=0):
    """A topograph (row 0 at the top) of a ring of Gaussian atoms, optionally with a weaker copy
    of everything at ``ghost`` = (dx, dy, ratio), a 208 pm step at ``step_x`` and spare atoms."""
    rng = np.random.default_rng(seed)
    npx = size / px
    c = (np.arange(px) + 0.5) * npx - size / 2
    X, Y = np.meshgrid(c, -c)
    z = rng.normal(0.0, 2e-12, (px, px))
    th = np.arange(n) * 2 * np.pi / n
    atoms = [(centre[0] + radius * np.cos(t), centre[1] + radius * np.sin(t)) for t in th] + list(spares)
    for x, y in atoms:
        z += h * np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma ** 2))
        if ghost:
            gx, gy, r = ghost
            z += r * h * np.exp(-((X - x - gx) ** 2 + (Y - y - gy) ** 2) / (2 * sigma ** 2))
    if step_x is not None:
        z += np.where(X > step_x, 208e-12, 0.0)
    return {"array": z, "nm_per_px": npx, "center_nm": [0.0, 0.0], "size_nm": size, "angle_deg": 0.0, "pixels": px}


def test_the_ring_is_found_anywhere_in_a_survey():
    f = _ring_frame()
    L = p3.find_corral(f)
    assert np.hypot(L["centre"][0] + 5.0, L["centre"][1] - 3.0) < 0.3
    assert abs(L["radius_nm"] - 7.0) < 0.2 and L["coverage"] > 0.9


def test_a_frame_that_cuts_the_ring_or_has_none_is_no_ring():
    with pytest.raises(p3.NoRing):                      # the ring runs off the frame's left edge
        p3.find_corral(_ring_frame(centre=(-27.0, 0.0)))
    with pytest.raises(p3.NoRing):
        p3.find_corral(_ring_frame(n=0, spares=[(3.0, 4.0), (-10.0, 9.0), (12.0, -15.0)]))


def test_a_blunt_tip_widens_the_ring_and_a_second_apex_copies_it():
    sharp, blunt = _ring_frame(sigma=0.3), _ring_frame(sigma=0.6, h=40e-12)
    ws = ring_width_nm(sharp["array"], sharp["nm_per_px"], (25.0, 33.0), 7.0)
    wb = ring_width_nm(blunt["array"], blunt["nm_per_px"], (25.0, 33.0), 7.0)
    assert ws < p3.BLUNT_FWHM_NM < wb
    double = _ring_frame(ghost=(1.0, 2.6, 0.7))
    g, lag = ghost_strength(double["array"], double["nm_per_px"], (25.0, 33.0), 7.0)
    g0, _ = ghost_strength(sharp["array"], sharp["nm_per_px"], (25.0, 33.0), 7.0)
    assert g >= p3.GHOST_MIN > g0
    assert abs(abs(lag[0]) - 1.0) < 0.4 and abs(abs(lag[1]) - 2.6) < 0.4


def test_the_tip_is_judged_on_the_frame():
    spares = [(15.0, 20.0), (-22.0, -18.0), (20.0, -12.0), (-18.0, 22.0), (24.0, 8.0)]
    ok = _ring_frame(spares=spares)
    assert p3.judge(ok, p3.find_corral(ok))["verdict"] == "ok"
    for kw, verdict in ((dict(sigma=0.6, h=40e-12), "blunt"), (dict(ghost=(1.0, 2.6, 0.7)), "double")):
        f = _ring_frame(spares=spares, **kw)
        assert p3.judge(f, p3.find_corral(f))["verdict"] == verdict


def test_a_bare_spot_keeps_away_from_the_ring_a_step_and_the_edge():
    f = _ring_frame(centre=(-8.0, 0.0), radius=7.0, step_x=12.0, spares=[(8.0, 20.0)])
    L = p3.find_corral(f)
    (x, y), clear = p3.bare_site(f, L["centre"], L["radius_nm"])
    assert np.hypot(x + 8.0, y) >= 7.0 + p3.BARE_BEYOND_NM - 0.5
    assert abs(x - 12.0) >= 4.0 and np.hypot(x - 8.0, y - 20.0) >= 4.0
    assert clear >= 4.0 and abs(x) <= 26.0 and abs(y) <= 26.0


def test_a_tip_feature_is_a_peak_above_the_band_edge_that_noise_is_not():
    v = np.linspace(-0.55, 0.15, 351)
    rng = np.random.default_rng(1)
    flat = 1.0 + 0.0 * v + np.where(v < -0.45, -0.4, 0.0) + rng.normal(0, 0.06, v.size)
    dirty = flat + 0.7 * np.exp(-((v + 0.12) / 0.04) ** 2)
    assert p3.tip_feature(v, flat, -0.45) < p3.FEATURE_MAX < p3.tip_feature(v, dirty, -0.45)


def test_the_results_are_read_off_the_rule_card():
    zh = Instrument(P3, 0).task()
    assert p3.claim_ids(zh) == ["peak_lo_mev", "peak_hi_mev", "radius_nm"]
    from stmbench.trackB.rules import task_text
    en = task_text(Instrument(P3, 0).scenario, "en")
    assert p3.claim_ids(en) == ["peak_lo_mev", "peak_hi_mev", "radius_nm"]


class _NoWorld:
    """An instrument whose simulator cannot be reached: what an agent over MCP has."""

    def __init__(self, inst):
        self._inst = inst

    def __getattr__(self, name):
        if name in ("act", "frame", "spectrum", "status", "info", "task", "report"):
            return getattr(self._inst, name)
        raise AttributeError(f"the baseline reached for {name!r}")


def test_the_baseline_plays_through_what_an_agent_gets():
    """Seed 15 arrives with a good tip and a ring 15 nm from the origin: three pokes, the survey,
    a bare spectrum, the drift, the radius frame and three spectra at the centre — all through
    the agent's interface — and the judge accepts the three reports."""
    inst = Instrument(P3, 15)
    got = p3.play(Blind(_NoWorld(inst)), {})
    assert set(got) == {"peak_lo_mev", "peak_hi_mev", "radius_nm"}
    for cid, (val, unit, ev) in got.items():
        inst.report(cid, val, unit=unit, evidence=ev)
    assert inst.finish()["success"]


def test_the_baseline_names_neither_the_simulator_nor_its_truth():
    """Inspection, as for the P4 baseline: outside ``run``, which scores the episode once it is
    over, the baseline's code reaches for no instrument, world, truth, hidden draw or verdict."""
    import ast
    import inspect

    from stmbench.analysis import corral

    forbidden = {"world", "truth", "hidden", "draw_hidden", "verdict", "scenario", "inst"}
    for mod in (p3, corral):
        for node in ast.parse(inspect.getsource(mod)).body:
            if mod is p3 and isinstance(node, ast.FunctionDef) and node.name == "run":
                continue
            names = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
            names |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            assert not names & forbidden, (mod.__name__, getattr(node, "name", None), names & forbidden)


def test_a_ring_the_survey_misses_is_looked_for_in_a_wider_one():
    """The task promises the ring within 60 nm of the origin, not within the first survey: with
    the survey centred 45 nm off, seed 15's ring (15 nm out, 9 nm across) is cut by its edge, so
    the wide survey finds it."""
    inst = Instrument(P3, 15)
    log: dict = {}
    c = p3.Corral(Blind(_NoWorld(inst)), log)
    c.centre = (45.0, 0.0)
    L = c.survey()
    assert "wide_survey" in log
    assert np.hypot(L["centre"][0] - 15.0, L["centre"][1] - 8.0) < 3.0 and abs(L["radius_nm"] - 8.9) < 0.3


def test_a_spectrum_in_which_the_tip_changed_is_told_from_its_repeats():
    v = np.linspace(-0.55, 0.15, 351)
    rng = np.random.default_rng(2)
    base = 1.0 + 0.5 * np.exp(-((v + 0.2) / 0.03) ** 2)
    good = [base + rng.normal(0, 0.05, v.size) for _ in range(3)]
    jumped = base * np.where(v > -0.1, 0.6, 1.0) + rng.normal(0, 0.05, v.size)
    assert p3.spectrum_outliers(good) == set()
    assert p3.spectrum_outliers(good + [jumped]) == {3}
