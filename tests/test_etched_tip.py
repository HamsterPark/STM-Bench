"""The STM rig's tip is an etched W wire, not a qPlus fork (decision of 2026-11-17).

The tip form acts only through a poke above 50 mV (a fork rings up) and a reshaping pulse's odds
of leaving a better tip (0.50 with a fork, 0.55 without). These tests pin that down: the rigs say
which tip they carry, a world starts bit for bit the same whichever form its tip has (layout,
hidden draws, truth, every random stream), and an episode without a pulse or a poke above 50 mV
measures the same frames and spectra with either form, so a record of one replays on the other.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from stmbench.instrument import Instrument
from stmsim.physics.tip import Tip
from stmsim.physics.rig import RigProfile
from stmsim.scenario import Scenario

SCEN = Path(__file__).resolve().parent.parent / "stmbench" / "trackB" / "scenarios"
STM_PAPERS = sorted(p for p in SCEN.glob("P*.yaml")
                    if yaml.safe_load(p.read_text(encoding="utf-8")).get("rig") == "reference-stm")


def _with_form(sc: Scenario, form: str | None) -> Scenario:
    """The scenario with its tip's form set (``None``: as the file has it, i.e. the rig's)."""
    if form is None:
        return sc
    ini = json.loads(json.dumps(sc.initial or {}))
    ini["tip"] = {**ini.get("tip", {}), "form": form}
    return dataclasses.replace(sc, initial=ini)


def _generators(obj, prefix: str, seen: set, depth: int = 0) -> dict:
    """Every numpy Generator reachable from ``obj`` through stmsim objects, by attribute path."""
    out: dict = {}
    if id(obj) in seen or depth > 4 or not hasattr(obj, "__dict__"):
        return out
    seen.add(id(obj))
    for k, v in vars(obj).items():
        if isinstance(v, np.random.Generator):
            out[f"{prefix}.{k}"] = v.bit_generator.state
        elif type(v).__module__.startswith("stmsim"):
            out.update(_generators(v, f"{prefix}.{k}", seen, depth + 1))
    return out


def _without_form(d):
    if isinstance(d, dict):
        return {k: _without_form(v) for k, v in d.items() if k != "form"}
    if isinstance(d, list):
        return [_without_form(v) for v in d]
    return d


def test_the_stm_rig_carries_an_etched_tip_and_only_the_qplus_rig_a_fork(tmp_path):
    assert RigProfile.load("reference-stm").get("tip.form") == "etched"
    assert RigProfile.load("reference-stm-qplus").get("tip.form") == "qplus"
    assert Tip().form == "etched"                          # a tip built without a form is an STM tip
    assert STM_PAPERS and all(Scenario.load(p).build_world(0, tmp_path / p.stem, virtual_clock=True).tip.form
                              == "etched" for p in STM_PAPERS)
    p5 = Scenario.load(SCEN / "P5_sader_jarvis_qplus.yaml")
    assert p5.build_world(0, tmp_path / "p5", virtual_clock=True).tip.form == "qplus"


@pytest.mark.parametrize("path", STM_PAPERS, ids=lambda p: p.stem)
@pytest.mark.parametrize("seed", [0, 7])
def test_a_world_starts_bit_for_bit_the_same_whichever_form_its_tip_has(path, seed, tmp_path):
    sc = Scenario.load(path)
    worlds = {}
    for form in ("etched", "qplus"):
        s = _with_form(sc, form)
        assert s.draw_hidden(seed) == sc.draw_hidden(seed)
        worlds[form] = s.build_world(seed, tmp_path / form, virtual_clock=True)
    a, b = worlds["etched"], worlds["qplus"]
    assert (a.tip.form, b.tip.form) == ("etched", "qplus")
    assert json.dumps(_without_form(a.truth()), sort_keys=True, default=str) == \
        json.dumps(_without_form(b.truth()), sort_keys=True, default=str)
    ga, gb = _generators(a, "world", set()), _generators(b, "world", set())
    assert ga and ga.keys() == gb.keys()
    assert {k for k in ga if ga[k] != gb[k]} == set()
    assert _without_form(a.tip.snapshot()) == _without_form(b.tip.snapshot())


def _episode(sc: Scenario, out: Path, poke_bias_v: float) -> tuple[list[np.ndarray], dict]:
    inst = Instrument(sc, 0, out)
    inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=20.0, pixels=64, line_time_s=0.05)
    inst.act("move_tip", x_nm=3.0, y_nm=-2.0)
    sp = inst.act("sts", v_start=-0.3, v_end=0.3, points=40, backward=False)["spectrum"]
    inst.act("move_tip", x_nm=40.0, y_nm=40.0)
    inst.act("poke", depth_nm=0.4, dwell_s=0.5, bias_v=poke_bias_v)
    inst.act("wait", seconds=120.0)
    fr = inst.act("scan", center_x_nm=0.0, center_y_nm=0.0, size_nm=20.0, pixels=64, line_time_s=0.05)["frame"]
    arrays = [np.asarray(inst.frame(fr)["array"], float)] + \
             [np.asarray(v, float) for k, v in sorted(inst.spectrum(sp).items()) if isinstance(v, (list, np.ndarray))]
    return arrays, _generators(inst.world, "world", set())


def test_an_episode_without_a_pulse_or_a_high_bias_poke_measures_the_same_with_either_form(tmp_path):
    """So a record of the qPlus rig with neither replays on the etched one, bit for bit."""
    sc = Scenario.load(SCEN / "P4_atom_positioning_cu111.yaml")
    e, ge = _episode(_with_form(sc, "etched"), tmp_path / "e", 0.02)
    q, gq = _episode(_with_form(sc, "qplus"), tmp_path / "q", 0.02)
    assert len(e) == len(q) and all(np.array_equal(x, y, equal_nan=True) for x, y in zip(e, q))
    assert ge == gq


def test_a_poke_above_50_mv_rings_only_a_fork_up(tmp_path):
    """The positive control of the test above: at 100 mV the fork draws for its ring-up and the
    etched tip does not, so the two episodes part from there."""
    sc = Scenario.load(SCEN / "P4_atom_positioning_cu111.yaml")
    _, ge = _episode(_with_form(sc, "etched"), tmp_path / "e", 0.1)
    _, gq = _episode(_with_form(sc, "qplus"), tmp_path / "q", 0.1)
    assert ge["world.tip.rng"] != gq["world.tip.rng"]
