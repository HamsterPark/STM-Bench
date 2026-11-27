"""Polling the status must not change what a later action records (screen trial 26).

``zctrl.z_n`` is a cache of the loop's Z. An action that reads it (a Z spectrum's ``z0_n_m``, a file
header, the start of a setpoint transient) must bring it to the present first; otherwise the
record depends on whether a client polled ``status`` in between, and ``verify`` (which replays
actions only) disagrees with a ledger written next to a console that polls all the time.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from stmbench.instrument import Instrument, verify

P5 = "P5_sader_jarvis_qplus"
P4 = "P4_atom_positioning_cu111"
ZSPEC = dict(sweep_nm=0.3, points=20)
SEQ = [("move_tip", dict(x_nm=3.0, y_nm=2.0, speed_nm_s=20.0)),
       ("zspec", ZSPEC),
       ("move_tip", dict(x_nm=-3.0, y_nm=1.0, speed_nm_s=20.0)),
       ("sts", dict(v_start=-0.5, v_end=0.5, points=20)),
       ("set_setpoint", dict(current_pa=200.0)),
       ("move_tip", dict(x_nm=1.0, y_nm=-3.0, speed_nm_s=20.0)),
       ("zspec", ZSPEC),
       ("scan", dict(center_x_nm=0.0, center_y_nm=0.0, size_nm=6.0, pixels=16, line_time_s=0.05))]


def _run(seed: int, polls: int, scen: str = P5):
    inst = Instrument(scen, seed)
    for name, args in SEQ:
        for _ in range(polls):
            inst.status()
        inst.act(name, **args)
    for _ in range(polls):
        inst.status()
    w = inst.world
    events = [{k: v for k, v in e.items() if k not in ("wall_s", "path")} for e in w.events]
    arrays = {n: {a: np.asarray(b) for a, b in s.items() if isinstance(b, np.ndarray)} for n, s in inst.spectra.items()}
    dats = [open(p, "rb").read() for p in w.dats_saved]
    return inst, events, arrays, dats


@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("polls", [1, 5])
def test_polling_status_changes_no_record(seed, polls):
    _, ev0, sp0, d0 = _run(seed, 0)
    _, ev1, sp1, d1 = _run(seed, polls)
    assert ev0 == ev1                                   # z0_n_m of every Z spectrum included
    assert sp0.keys() == sp1.keys()
    for n in sp0:
        for a in sp0[n]:
            assert np.array_equal(sp0[n][a], sp1[n][a], equal_nan=True), (n, a)
    assert d0 == d1                                     # .dat headers carry the loop's Z


def test_a_zspec_records_the_loop_z_at_the_tip_now():
    inst = Instrument(P5, 0)
    inst.act("move_tip", x_nm=3.0, y_nm=2.0, speed_nm_s=20.0)
    w = inst.world
    # the loop is on: the record is the equilibrium Z of this place at the moment the sweep starts
    # (read before the action, because the action then spends time and the drift moves it on)
    want = float(w.z_ctrl_from_tip(w._equilibrium_z_tip()))
    inst.act("zspec", **ZSPEC)
    rec = w.zspec_records[-1]
    assert rec["z0_n_m"] == pytest.approx(want, abs=1e-15)


def test_a_setpoint_transient_starts_from_the_loop_z_before_the_change():
    starts = []
    for polls in (0, 3):
        inst = Instrument(P5, 0)
        inst.act("move_tip", x_nm=3.0, y_nm=2.0, speed_nm_s=20.0)
        for _ in range(polls):
            inst.status()
        inst.act("set_setpoint", current_pa=200.0)
        tr = inst.world.transients[-1] if inst.world.transients else None
        starts.append(None if tr is None else float(np.asarray(tr.z_tip)[0]))
    assert starts[0] == starts[1]


def test_verify_matches_a_ledger_written_while_polling(tmp_path):
    inst = Instrument(P4, 5, out_dir=tmp_path / "run")
    for name, args in [("move_tip", dict(x_nm=3.0, y_nm=2.0, speed_nm_s=100.0)),
                       ("zspec", ZSPEC),
                       ("move_tip", dict(x_nm=-2.0, y_nm=4.0, speed_nm_s=100.0)),
                       ("zspec", ZSPEC)]:
        for _ in range(7):
            inst.status()
        try:
            inst.act(name, **args)
        except ValueError:
            pass                                        # a scenario without the qPlus refuses zspec: still a replay
    inst.finish()
    res = verify(tmp_path / "run")
    assert res["ok"], res
    # the raw recorded events (what a reader of events.json sees) equal the replay's, not only the portable ones
    from stmbench.instrument import replay

    rec = json.loads((tmp_path / "run" / "events.json").read_text(encoding="utf-8"))
    again = replay(tmp_path / "run").world.events
    z_rec = [e["z0_n_m"] for e in rec if e.get("kind") == "zspec"]
    z_again = [e["z0_n_m"] for e in again if e.get("kind") == "zspec"]
    assert z_rec == pytest.approx(z_again, abs=1e-15)
