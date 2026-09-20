"""Round-3 validity pins: undelivered replies are not progress, paused-clock commands do
not spend the budget, [DONE] tolerates trailing punctuation, and BindAllToolNames is part
of every LLM episode."""
from __future__ import annotations

import socket
import time

import pytest

from stmsim.physics.rig import RigProfile
from stmsim.physics.world import World
from stmsim.modules import build_dispatcher
from stmsim.wire import codec
from stmsim.wire.server import WireServer


def _tunnelling_world(tmp_path, seed=21, time_scale=20.0):
    w = World(rig=RigProfile.load("reference-stm"), seed=seed, session_dir=tmp_path / "s", time_scale=time_scale)
    w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
    w.withdrawn = False
    w.zctrl_set(True)
    w.transients.clear()
    w.achievable_z_tip()
    return w


def test_call_log_stamps_paused_and_budget_probe_ignores_it(tmp_path):
    from stmbench.harness.ic_driver import _budget_probe

    w = _tunnelling_world(tmp_path)
    d = build_dispatcher(w)
    d.log_calls = True
    body = b""
    d.call("Bias.Get", body)
    with w.clock.paused():
        d.call("Bias.Get", body)
        d.call("Current.Get", body)
    d.call("Bias.Get", body)
    assert [c[4] for c in d.call_log] == [False, True, True, False]

    class _Host:
        world = w
        dispatcher = d
    sim, cmds = _budget_probe(_Host())
    assert cmds == 2


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


def test_wire_latency_past_timeout_marks_undelivered(tmp_path):
    """The connection handler sleeps past MAST's 5 s recv timeout → the command still runs
    but its events carry delivered=False and the progress check ignores them."""
    from stmbench.harness.episode import _progress_check

    w = _tunnelling_world(tmp_path)
    d = build_dispatcher(w)
    d.log_calls = True
    with WireServer(d, ports=[0]) as srv:
        srv.latency["Bias.Get"] = 5.2
        port = srv.bound_ports[0]
        s = socket.create_connection(("127.0.0.1", port), timeout=1.0)
        s.sendall(codec.build_request_frame("Bias.Get", b"", send_response_back=True))
        with pytest.raises((TimeoutError, socket.timeout, OSError)):
            s.recv(40)
        s.close()
        t0 = time.time()
        while time.time() - t0 < 8 and not any(c[1] == "Bias.Get" for c in d.call_log):
            time.sleep(0.05)
    assert any(c[1] == "Bias.Get" and "undelivered" in c[3] for c in d.call_log), d.call_log[-3:]
    # progress check: an sts event stamped undelivered is not progress
    w.events.append({"kind": "sts", "delivered": False, "sim_s": 0.0, "wall_s": 0.0})
    assert not _progress_check("sts_acquired_under_3s_sweeps", w, {})
    w.events.append({"kind": "sts", "sim_s": 0.0, "wall_s": 0.0})
    assert _progress_check("sts_acquired_under_3s_sweeps", w, {})


def test_done_marker_tolerates_trailing_punctuation_and_markdown():
    from stmbench.harness.ic_driver import marker_outcome

    assert marker_outcome("完成。[DONE]")[0] == "done"
    assert marker_outcome("完成 [DONE]。")[0] == "done"
    assert marker_outcome("**[DONE]**")[0] == "done"
    assert marker_outcome("[ABORT] 原因：x")[0] is None            # marker not at the end
    assert marker_outcome("无法完成。[ABORT]")[0] == "abort"
    assert marker_outcome("先做 [DONE] 再说别的")[1] == "marker_in_body"
