"""Reading the trace of a drag (``stmbench.analysis.trace``): did the atom hop, and how often?

The traces are the simulator's own (P4 worlds, a tip put on the atom as a slow frame would put
it, a drag of 4 nm at 0.25 nm/s at a rung of the resistance ladder), with the truth read from the
world's events afterwards: that is what the reader is checked against, never what it reads."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from stmbench.analysis import trace as T
from stmsim.scenario import Scenario

P4 = "P4_atom_positioning_cu111"
SC = Scenario.load(Path("stmbench/trackB/scenarios") / f"{P4}.yaml")


def _drag(tmp_path, seed: int, rung: int, *, speed: float = 0.25, dx: float = 4.0):
    """The record of a drag at rung ``rung`` of the ladder (400 kohm, then by 0.8 each) as
    ``Instrument.spectrum`` gives it, the number of hops the world made, and whether it picked the atom up."""
    r_kohm = 400.0 * 0.8 ** rung
    w = SC.build_world(seed, tmp_path / f"s{seed}_{rung}", virtual_clock=True)
    reg = w.surface.site.adatoms
    atom = reg.atoms[reg.target["atom_id"]]
    ax, ay = reg.site_xy(atom.i, atom.j)
    off = np.random.default_rng(seed * 1000 + int(r_kohm)).normal(0.0, 0.05e-9, 2)   # where a slow frame places an atom
    w.move_xy(ax + off[0], ay + off[1])
    w.preamp.full_scale_a = 100e-9
    v = min(0.005, r_kohm * 1e3 * 80e-9)
    w.set_bias(v)
    w.set_setpoint(v / (r_kohm * 1e3))
    w.transients.clear()
    events = w.move_xy_path(ax + off[0] + dx * 1e-9, ay + off[1], speed_m_s=speed * 1e-9, trace=True)
    tr = w.last_move_trace
    rec = {"z_abs": tr["z_abs"], "dist": tr["dist"], "i": tr["i"], "t": tr["t"], "feedback": tr["feedback"]}
    hops = sum(int(e.get("n_hops", 0)) for e in events if e["kind"] == "adatom_hop")
    return rec, hops, any(e["kind"] == "adatom_picked" for e in events)


def test_an_atom_left_where_it_was_reads_no_hop(tmp_path):
    """Seed 24 (a good tip) at 400 kohm, above the atom's threshold: it stays, the bump falls away
    from the start of the trace, and the reading says so."""
    rec, hops, picked = _drag(tmp_path, 28, 0)                          # 320 k is its first hop: 400 k is none
    assert hops == 0 and not picked
    r = T.read_drag(rec)
    assert r["hops"] == 0 and r["left_behind"] and not r["carried"] and r["steps"] == 0
    assert T.MIN_NOISE_PM <= r["noise_pm"] < 2.0                      # read off the trace, never under the floor
    assert T.count_hops(rec) == 0


@pytest.mark.parametrize("seed,rung", [(84, 1), (31, 0), (34, 1), (23, 2)])
def test_a_hop_after_a_long_lag_is_read_from_its_jump(tmp_path, monkeypatch, seed, rung):
    """One hop, late in the drag (0.35 nm or more along it): the atom lagged until the pull made
    it hop, so the hop is a jump of 11-23 pm out of a fall, which the schedule the fit assumes has no
    place for. The fit alone reads these as no hop (the last one's jump is even taken out as a
    step); the jump is what says the atom moved."""
    rec, hops, picked = _drag(tmp_path, seed, rung)
    assert hops == 1 and not picked
    assert len(T.hop_steps(np.asarray(rec["z_abs"]) * 1e12)) == 1
    r = T.read_drag(rec)
    assert r["hops"] == 1 and r["left_behind"] and not r["carried"]
    monkeypatch.setattr(T, "hop_steps", lambda z: [])
    assert T.read_drag(rec)["hops"] == 0


def test_hop_steps_finds_a_jump_up_out_of_a_fall_and_nothing_else():
    rng = np.random.default_rng(4)
    k = np.arange(60)
    fall = -1.5 * k + rng.normal(0.0, 0.5, 60)                       # pm: the tip pulling away from the atom
    jump = fall.copy()
    jump[20:] += 12.0
    assert T.hop_steps(jump) == [20]
    two = jump.copy()
    two[40:] += 9.0
    assert T.hop_steps(two) == [20, 40]
    # not a hop: a bump coming under the tip (a ramp), a step with no fall before it (a tip that
    # changed), a step down, a step too small, one before the fall can be seen, a trace too short
    ramp = fall.copy()
    ramp[20:25] += np.linspace(0.0, 14.0, 5)
    ramp[25:] += 14.0
    assert T.hop_steps(ramp) == []
    flat = rng.normal(0.0, 0.5, 60)
    flat[30:] += 14.0
    assert T.hop_steps(flat) == []
    down = fall.copy()
    down[20:] -= 12.0
    assert T.hop_steps(down) == []
    small = fall.copy()
    small[20:] += 3.0
    assert T.hop_steps(small) == []
    early = fall.copy()
    early[2:] += 12.0
    assert T.hop_steps(early) == []
    assert T.hop_steps(jump[:5]) == []


@pytest.mark.parametrize("seed,rung", [(24, 3), (28, 5), (7, 4), (22, 5)])
def test_an_atom_carried_the_whole_way_is_one_the_bump_never_leaves(tmp_path, seed, rung):
    rec, hops, picked = _drag(tmp_path, seed, rung)
    assert hops >= 14 and not picked
    r = T.read_drag(rec)
    assert r["carried"] and not r["left_behind"]
    assert abs(r["hops"] - hops) <= 4                                 # as many as kept it under the tip


@pytest.mark.parametrize("seed,rung,want", [(24, 0, 3), (23, 3, 3), (70, 4, 4), (28, 1, 2), (76, 3, 7)])
def test_an_atom_dropped_after_a_few_hops_reads_about_as_many(tmp_path, seed, rung, want):
    """Where the bump falls says how far the atom came: these hopped a few times and were left
    behind, a nanometre or so on."""
    rec, hops, picked = _drag(tmp_path, seed, rung)
    assert hops == want and not picked
    r = T.read_drag(rec)
    assert r["left_behind"] and abs(r["hops"] - hops) <= 1, r


def test_the_reading_is_right_across_tips_and_the_resistance_ladder(tmp_path):
    """The record kept in docs/FIDELITY.md, in small: over 24 seeds of all five tips and four rungs,
    an atom that hopped is read as hopping and one that did not, as not, in all but a few drags."""
    wrong, n = [], 0
    for seed in range(24):
        for rung in (0, 2, 4, 6):
            rec, hops, picked = _drag(tmp_path, seed, rung)
            if picked:
                continue
            n += 1
            if (T.count_hops(rec) > 0) != (hops > 0):
                wrong.append((seed, rung, hops, T.count_hops(rec)))
    assert n >= 80 and len(wrong) <= 0.06 * n, wrong


def test_a_trace_that_cannot_be_read_reads_nothing():
    z = np.zeros(40)
    d = np.linspace(0.0, 2e-9, 40)
    for rec in ({"z_abs": z, "dist": d, "feedback": False},                 # no loop: Z shows nothing
                {"z_abs": z[:2], "dist": d[:2], "feedback": True},          # too short
                {"z_abs": z, "dist": d * 50.0, "feedback": True},           # 100 nm: no drag of one atom
                {"z_abs": z * np.nan, "dist": d, "feedback": True},
                {"z_abs": z, "dist": d[:30], "feedback": True}):
        r = T.read_drag(rec)
        assert (r["hops"], r["left_behind"], r["carried"], r["steps"]) == (0, False, False, 0) and np.isnan(r["noise_pm"])
        assert T.count_hops(rec) == 0


def test_a_flat_trace_is_one_the_bump_never_leaves():
    """Noise on a level, no fall: an atom that stayed under the tip, or none at the start. The
    reading assumes the tip started over the atom and says so with ``carried``."""
    rng = np.random.default_rng(0)
    rec = {"z_abs": rng.normal(0.0, 2e-12, 81), "dist": np.linspace(0.0, 4e-9, 81), "feedback": True}
    r = T.read_drag(rec)
    assert r["carried"] and not r["left_behind"] and r["hops"] == round(4.0 / (np.mean(T.ADVANCES) * 0.255))


def test_a_bump_that_falls_at_the_start_is_no_hop_whatever_its_size_or_width():
    rng = np.random.default_rng(1)
    s = np.linspace(0.0, 4e-9, 81)
    for amp, width in ((15e-12, 0.5e-9), (50e-12, 0.3e-9), (90e-12, 0.4e-9)):
        rec = {"z_abs": amp * np.exp(-0.5 * ((s + 0.05e-9) / width) ** 2) + rng.normal(0.0, 2e-12, 81) + 3e-12 * s * 1e9,
               "dist": s, "feedback": True}
        r = T.read_drag(rec)
        assert r["left_behind"] and r["hops"] == 0, (amp, width, r)


def test_the_reader_sees_nothing_but_the_record():
    """No world, truth, hidden draw, verdict or instrument: the module is pure arrays."""
    forbidden = {"world", "truth", "hidden", "draw_hidden", "verdict", "scenario", "inst", "events", "Instrument"}
    tree = ast.parse(inspect.getsource(T))
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert not names & forbidden
    imports = {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    assert not {i for i in imports if "stmsim" in i}
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert modules <= {"__future__", "functools"}


def test_a_step_in_z_is_taken_out_before_it_can_be_read_as_a_fall():
    """A tip that changes mid-drag steps Z by 20-150 pm: that is neither a tooth nor a bump. A
    carried atom's flat trace with a step up late in it is still carried, not an atom left behind."""
    rng = np.random.default_rng(2)
    s = np.linspace(0.0, 4e-9, 81)
    z = rng.normal(0.0, 2e-12, 81)
    z[74:] += 87e-12
    r = T.read_drag({"z_abs": z, "dist": s, "feedback": True})
    assert r["steps"] == 1 and r["carried"] and not r["left_behind"]
    z = 55e-12 * np.exp(-0.5 * ((s + 0.05e-9) / 0.35e-9) ** 2) + rng.normal(0.0, 2e-12, 81)
    z[40:] -= 60e-12                                                # the same fall as an atom left behind, plus a step down
    r = T.read_drag({"z_abs": z, "dist": s, "feedback": True})
    assert r["steps"] == 1 and r["left_behind"] and r["hops"] == 0


def test_the_last_hop_of_an_atom_nearly_home_reads_as_a_hop_and_a_hop_that_cannot_happen_does_not():
    """A drag of 0.2 nm (five samples) is the last hop of an atom one site from its goal. It has to
    read as moved: a baseline that took it for no hop would treat the atom's own step as drift. A
    drag shorter than the lag a hop needs (0.15 nm) cannot hop."""
    rng = np.random.default_rng(3)
    for length, want in ((0.2e-9, 1), (0.1e-9, 0)):
        n = int(round(length / 0.05e-9)) + 1
        rec = {"z_abs": rng.normal(0.0, 2e-12, n), "dist": np.linspace(0.0, length, n), "feedback": True}
        assert T.count_hops(rec) == want, length
