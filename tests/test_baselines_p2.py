"""The scripted P2 baseline: the standing waves of a Cu(111) step edge from nothing but what the
instrument shows — the staircase found in a survey, a place chosen on it, maps taken there."""
from __future__ import annotations

import math

import numpy as np

from stmbench.baselines import p2
from stmbench.baselines._common import Blind
from stmbench.instrument import Instrument

P2 = "P2_standing_waves_cu111"
STEP_M = 208e-12                                    # Cu(111) single step


def _survey(*, alpha_deg=33.0, widths=(25, 40, 90, 30, 22, 65, 35, 100, 28), amp=4.0, period=250.0, size=400.0, px=200,
            adatoms=(), noise_pm=3.0, seed=0):
    """A topograph (row 0 the high-y edge) of a staircase along the normal at ``alpha_deg`` whose
    steps wander together (the surface's own form: one function of the coordinate along the
    steps shifts them all), with soft edges, adatoms (x, y, height_m, sigma_nm) and noise; the
    step positions along the normal, from the frame centre, are returned with it."""
    rng = np.random.default_rng(seed)
    npx = size / px
    c = (np.arange(px) + 0.5) * npx - size / 2
    X, Y = np.meshgrid(c, -c)
    a = math.radians(alpha_deg)
    u = X * math.cos(a) + Y * math.sin(a)
    v = -X * math.sin(a) + Y * math.cos(a)
    uw = u + amp * np.sin(2 * math.pi * v / period) + 0.4 * amp * np.sin(2 * math.pi * v / (0.37 * period) + 1.3)
    b = np.cumsum(widths) - np.sum(widths) / 2.0     # the edges, centred
    z = rng.normal(0.0, noise_pm * 1e-12, (px, px))
    for e in b:
        z += STEP_M / (1.0 + np.exp(-(uw - e) / 0.5))
    for x, y, h, s in adatoms:
        z += h * np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * s ** 2))
    return {"array": z, "nm_per_px": npx, "center_nm": [0.0, 0.0], "size_nm": size, "angle_deg": 0.0}, b


def test_the_mean_direction_of_a_wandering_staircase_is_found_to_a_degree():
    for alpha in (12.0, 33.0, 95.0, 160.0):
        f, _ = _survey(alpha_deg=alpha)
        th = p2.measure_axis(f["array"], f["nm_per_px"])
        assert abs((th - alpha + 90.0) % 180.0 - 90.0) < 1.5


def test_the_steps_and_their_wander_are_read_off_a_survey():
    f, b = _survey(alpha_deg=33.0)
    th = p2.measure_axis(f["array"], f["nm_per_px"])
    st = p2.staircase(f["array"], f["nm_per_px"], th)
    got = np.asarray(st["edges"])
    # every step in the frame is found within 2 nm, none is made up
    inside = [e for e in b if abs(e) < 170.0]
    assert all(np.min(np.abs(got - e)) < 2.0 for e in inside)
    assert all(np.min(np.abs(b - g)) < 2.5 for g in got)
    sh = st["shift"][np.isfinite(st["shift"])]
    assert 4.0 < np.ptp(sh) < 14.0                   # the wander (amplitude 4 nm + 1.6 nm) is seen, to a nanometre or so


def test_a_place_needs_a_wide_terrace_a_straight_edge_and_no_adatom_near():
    # the terraces are 25-100 nm: only the 90 and the 100 nm ones are wide enough
    f, b = _survey(alpha_deg=33.0, amp=2.0)
    th = p2.measure_axis(f["array"], f["nm_per_px"])
    st = p2.staircase(f["array"], f["nm_per_px"], th)
    cands = p2.pick_site(st, [])
    assert cands and cands[0]["ok"] and cands[0]["width_nm"] >= p2.MIN_TERRACE_NM
    for c in cands:
        assert c["width_nm"] >= p2.MIN_TERRACE_NM
    # an adatom 10 nm from the best centre takes that place away; the next one is elsewhere
    cu, cv = cands[0]["centre"]
    adatom = (cu + 10.0, cv, 0.0)
    again = p2.pick_site(st, [adatom])
    assert again[0]["ok"] and math.hypot(again[0]["centre"][0] - cu, again[0]["centre"][1] - cv) > 5.0
    assert min(math.hypot(a - again[0]["centre"][0], b_ - again[0]["centre"][1]) for a, b_, _ in [adatom]) >= p2.ADATOM_CLEAR_NM


def test_a_survey_without_a_wide_terrace_has_no_place():
    f, _ = _survey(widths=(25, 30, 22, 35, 28, 40, 30, 25, 33, 38, 27))
    th = p2.measure_axis(f["array"], f["nm_per_px"])
    st = p2.staircase(f["array"], f["nm_per_px"], th)
    assert not [c for c in p2.pick_site(st, []) if c["ok"]]


def test_adatoms_are_found_in_a_survey_and_steps_are_not_taken_for_them():
    # three adatoms in the middle of terraces (u along the normal, v along the steps, from the centre)
    alpha = 33.0
    a = math.radians(alpha)
    on_terrace = [(28.0, -66.0, 140e-12, 1.2), (-47.0, 107.0, 200e-12, 1.5), (72.0, 68.0, 180e-12, 1.2)]
    spares = [(u * math.cos(a) - v * math.sin(a), u * math.sin(a) + v * math.cos(a), h, s_) for u, v, h, s_ in on_terrace]
    f, _ = _survey(alpha_deg=alpha, adatoms=spares)
    th = p2.measure_axis(f["array"], f["nm_per_px"])
    st = p2.staircase(f["array"], f["nm_per_px"], th)
    found = p2.survey_adatoms(f["array"], f["nm_per_px"], st)
    for u, v, _, _ in on_terrace:
        assert min(math.hypot(u - fu, v - fv) for fu, fv, _ in found) < 4.0
    assert len(found) <= len(on_terrace) + 2          # a few false ones at most


def test_the_edge_looked_for_is_the_one_expected_not_its_neighbour():
    """Two steps 15 nm apart in a 40 nm frame, the neighbour the steeper one: the edge near the
    expected column is the one found, and a slope the sample's drift gives the rows is followed."""
    npx, n_col, n_row = 0.2, 200, 120
    x = (np.arange(n_col) + 0.5) * npx
    drift = 0.04                                     # nm per row
    rows = np.arange(n_row)
    z = np.zeros((n_row, n_col))
    for r in rows:
        z[r] = STEP_M / (1 + np.exp(-(x - (12.0 + drift * r)) / 0.4)) + 0.9 * STEP_M / (1 + np.exp(-(x - (27.0 + drift * r)) / 0.3))
    ed = p2.crop_edges(z, npx, 12.0 + drift * n_row / 2, 4.0, drift)
    assert abs(ed["slope_nm_per_row"] - drift) < 0.005 and ed["n_good"] > 100
    assert abs(ed["offset_nm"] - 12.0) < 0.3


def test_the_planned_staircase_is_found_again_in_a_frame_taken_after_a_creep():
    npx, n_col, n_row = 0.3, 200, 200
    x = (np.arange(n_col) + 0.5) * npx - n_col * npx / 2
    planned = [-22.0, -6.0, 9.0]                     # steps from the frame centre as planned
    z = np.zeros((n_row, n_col))
    for e in planned:
        z += STEP_M / (1 + np.exp(-(x - (e + 11.0)) / 0.5))[None, :]     # the sample crept 11 nm
    off, r = p2.match_steps(z + np.random.default_rng(1).normal(0, 3e-12, z.shape), npx, planned)[0]
    assert abs(off - 11.0) < 0.6 and r > 0.8


def test_two_steps_one_terrace_apart_fit_again_one_terrace_over_and_the_other_side_decides():
    """The pattern of the survey (a step, and its neighbour 23 nm along) is matched by the right
    shift and, less well, by one terrace over: both are offered, best first. At the wrong one the
    planned edge sits on the neighbour and a step is in view on the profile's side."""
    npx, n_col, n_row = 0.3, 200, 120
    x = (np.arange(n_col) + 0.5) * npx
    z = np.zeros((n_row, n_col))
    for e in (20.0, 43.0):                           # the frame's true steps, from its left side (the frame is 60 nm)
        z += STEP_M / (1 + np.exp(-(x - e) / 0.5))[None, :]
    z += np.random.default_rng(2).normal(0, 3e-12, z.shape)
    planned = [-33.0, -10.0]                         # the survey's pair, one terrace short of where the frame has it
    alts = p2.match_steps(z, npx, planned, search_nm=28.0)
    assert abs(alts[0][0] - 23.0) < 0.6              # both steps match at 23 nm ...
    assert any(abs(a[0]) < 0.6 for a in alts[1:])    # ... and the neighbour alone at no shift: offered second
    # the profile side (towards lower columns) of the step at 43 nm holds the step at 20 nm: not clear;
    # that of the step at 20 nm, looked at from the neighbour's side away from it, is
    assert not p2.side_is_clear(z, npx, 43.0, -1)
    assert p2.side_is_clear(z, npx, 20.0, -1)


def test_an_edge_is_told_from_the_noise_a_line_would_follow_where_there_is_none():
    npx, n_col, n_row = 0.4, 250, 125
    x = (np.arange(n_col) + 0.5) * npx
    rng = np.random.default_rng(3)
    step = np.vstack([STEP_M / (1 + np.exp(-(x - (30.0 + 0.01 * r)) / 0.5)) for r in range(n_row)]) + rng.normal(0, 3e-12, (n_row, n_col))
    flat = 0.002e-9 * (np.arange(n_row)[:, None] / n_row) + rng.normal(0, 3e-12, (n_row, n_col))
    for z, there in ((step, True), (flat, False)):
        ed = p2.crop_edges(z, npx, 30.0, 9.0)
        assert p2.edge_present(z, npx, ed) == there


def test_a_second_apex_is_seen_in_the_steps_drawn_twice():
    npx, n_col, n_row = 0.4, 250, 125
    x = (np.arange(n_col) + 0.5) * npx
    rng = np.random.default_rng(4)

    def frame(copy_at, weight=0.6, steps=(18.0, 47.0, 71.0)):
        z = np.zeros((n_row, n_col))
        for e in steps:
            z += STEP_M / (1 + np.exp(-(x - e) / 0.5))[None, :]
            if copy_at:
                z += weight * STEP_M / (1 + np.exp(-(x - e - copy_at) / 0.5))[None, :]
        return z + rng.normal(0, 3e-12, z.shape)

    g = p2.ghost_steps(frame(3.2), npx)
    assert g is not None and abs(g["offset_nm"] - 3.2) < 0.6 and g["pairs"] >= 2
    assert abs(p2.ghost_steps(frame(-2.5), npx)["offset_nm"] + 2.5) < 0.6      # to the other side
    assert p2.ghost_steps(frame(0.0), npx) is None                              # one apex
    # two steps a terrace of 4 nm apart is a pair, not two pairs at one distance: not a second apex
    assert p2.ghost_steps(frame(0.0, steps=(18.0, 22.0, 71.0)), npx) is None


class _NoWorld:
    """An instrument whose simulator cannot be reached: what an agent over MCP has."""

    def __init__(self, inst):
        self._inst = inst

    def __getattr__(self, name):
        if name in ("act", "frame", "spectrum", "status", "info", "task", "report"):
            return getattr(self._inst, name)
        raise AttributeError(f"the baseline reached for {name!r}")


def test_the_baseline_names_neither_the_simulator_nor_its_truth():
    """Inspection, as for the P3 baseline: outside ``run``, which scores the episode once it is
    over, the baseline's code reaches for no instrument, world, truth, hidden draw or verdict."""
    import ast
    import inspect

    forbidden = {"world", "truth", "hidden", "draw_hidden", "verdict", "scenario", "inst", "seed"}
    for node in ast.parse(inspect.getsource(p2)).body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            continue
        names = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
        names |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        assert not names & forbidden, (getattr(node, "name", None), names & forbidden)


def test_the_baseline_plays_through_what_an_agent_gets_and_the_judge_accepts_it():
    """Seed 1 arrives with a good tip: a survey, a place, a check frame, a trial map and eight
    maps — all through the agent's interface — and the judge accepts both reports. About a
    minute of wall time (the maps are 120 x 120 pixels each, eight of them, drifted and noisy)."""
    import time

    inst = Instrument(P2, 1)
    t0 = time.time()
    got = p2.play(Blind(_NoWorld(inst)), {})
    wall = time.time() - t0
    assert set(got) == {"e0_mev", "m_eff"}
    for cid, (val, unit, ev) in got.items():
        inst.report(cid, val, unit=unit, evidence=ev)
    out = inst.finish()
    assert out["success"] and wall < 90.0
