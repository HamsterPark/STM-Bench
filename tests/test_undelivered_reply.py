"""A command whose reply never reaches the client still acts on the world; the world's
events say the reply was not delivered, and a normal call leaves no such stamp."""
from __future__ import annotations

from stmsim.physics.rig import RigProfile
from stmsim.physics.world import World
from stmsim.modules import build_dispatcher


def _tunnelling_world(tmp_path, seed=21, time_scale=20.0):
    w = World(rig=RigProfile.load("reference-stm"), seed=seed, session_dir=tmp_path / "s", time_scale=time_scale)
    w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
    w.withdrawn = False
    w.zctrl_set(True)
    w.transients.clear()
    w.achievable_z_tip()
    return w


def test_undelivered_reply_marks_world_events(tmp_path):
    w = _tunnelling_world(tmp_path)
    d = build_dispatcher(w)
    d.log_calls = True
    import struct

    n0 = len(w.events)
    # a ZCtrl.Withdraw (wait=0, timeout=-1) whose reply the client will never receive
    d.call("ZCtrl.Withdraw", struct.pack(">Ii", 0, -1), undelivered=True)
    new = w.events[n0:]
    assert any(e["kind"] == "withdraw" for e in new), new
    assert all(e.get("delivered") is False for e in new)
    assert d.call_log[-1][3].startswith("ok(undelivered)")
    # a normal call leaves no delivered stamp at all
    n1 = len(w.events)
    d.call("Bias.Get", b"")
    assert all("delivered" not in e for e in w.events[n1:])
    assert d.call_log[-1][3] == "ok"
