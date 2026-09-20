"""MAST's real ConnectionPool + InstrumentState against the simulator (docs/DESIGN.md §4.6.6)."""
from __future__ import annotations

import os
import time

import pytest

from stmsim.modules import build_dispatcher
from stmsim.physics.rig import RigProfile
from stmsim.physics.world import World
from stmsim.wire.server import WireServer

from tests.conftest import requires_mast


@pytest.fixture
def sim(tmp_path):
    w = World(rig=RigProfile.load("reference-stm"), seed=2, session_dir=tmp_path / "s", time_scale=50.0)
    w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
    w.withdrawn = False
    w.zctrl_set(True)
    d = build_dispatcher(w)
    with WireServer(d, ports=[0, 0, 0, 0]) as srv:
        yield w, srv


@requires_mast
def test_connection_pool_connects_and_state_refreshes(sim, monkeypatch):
    w, srv = sim
    ports = srv.bound_ports
    for role, port in zip(("MAIN", "MONITOR", "DATA", "EMERGENCY"), ports):
        monkeypatch.setenv(f"MAST_NANONIS_PORT_{role}", str(port))
    from mast.config import NanonisConfig
    from mast.core.connection import ConnectionPool
    from mast.core.state import InstrumentState

    cfg = NanonisConfig()
    pool = ConnectionPool(cfg)
    ok = pool.connect_all()
    assert all(ok.values()), ok
    try:
        rec = pool.safe_call("Util_VersionGet")
        assert not rec.error, rec.error
        rec = pool.safe_call("Bias_Set", 0.02)
        assert not rec.error
        state = InstrumentState(pool)
        hs = state.refresh()
        assert hs.bias_v == pytest.approx(0.02)
        assert hs.z_controller_on is True
        assert hs.z_controller_name in ("Current", "log Current", "df")
        assert hs.setpoint_a == pytest.approx(w.zctrl.setpoint_a)
        assert hs.current_a is not None and abs(hs.current_a) > 1e-12
        assert hs.scan_running is False
        assert hs.scan_width_m == pytest.approx(w.scan.w)
        assert hs.withdrawn is False
        assert hs.lockin_mod_on is False
        # withdraw through the emergency role like MAST's e-stop does
        rec = pool.safe_call("ZCtrl_Withdraw", 1, -1, role="emergency")
        assert not rec.error
        hs = state.refresh()
        assert hs.withdrawn is True and hs.z_controller_on is False
        assert pool.comms_healthy()
    finally:
        pool.close_all()


@requires_mast
def test_monitoring_osci2t_probe_and_timebase(sim, monkeypatch):
    w, srv = sim
    for role, port in zip(("MAIN", "MONITOR", "DATA", "EMERGENCY"), srv.bound_ports):
        monkeypatch.setenv(f"MAST_NANONIS_PORT_{role}", str(port))
    from mast.config import NanonisConfig
    from mast.core.connection import ConnectionPool
    from mast.monitoring import pump as pump_mod

    pool = ConnectionPool(NanonisConfig())
    assert all(pool.connect_all().values())
    try:
        ok, detail = pump_mod.Osci2TProbe.available(pool)
        assert ok, detail
        assert pump_mod.OsciHRProbe.available(pool) is False
        p = pump_mod.make_pump(lambda: pool, strategy="osci2t", window_target_s=6.4)
        assert p.STRATEGY == "osci2t", p.STRATEGY
        cfg = p.configure()
        assert p._timebase_check == "ok", (p._timebase_check, cfg)
        chunk = None
        for _ in range(5):
            chunk = p.poll_once()
            if chunk is not None:
                break
            time.sleep(0.05)
        assert chunk is not None
        assert chunk.y.size >= 256 and chunk.n == chunk.y.size
        assert abs(chunk.dt - w.rig.osci_dt_s) < 1e-6
    finally:
        pool.close_all()
