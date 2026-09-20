"""World + dispatcher + wire server, driven by MAST's real patched client over TCP."""
from __future__ import annotations

import socket
import time
from pathlib import Path

import numpy as np
import pytest

from stmsim.modules import build_dispatcher
from stmsim.physics.rig import RigProfile
from stmsim.physics.world import World
from stmsim.wire.server import WireServer
from stmsim.wire.spec import commands

from tests.conftest import requires_mast


def _approached_world(tmp_path: Path, seed: int = 1, time_scale: float = 50.0) -> World:
    w = World(rig=RigProfile.load("reference-stm"), seed=seed, session_dir=tmp_path / "session",
              time_scale=time_scale)
    w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
    w.withdrawn = False
    w.zctrl_set(True)
    return w


def test_dispatcher_covers_the_local_registry():
    w = World(rig=RigProfile.load("reference-stm"), seed=0)
    d = build_dispatcher(w)
    # Every locally registered command for a loaded module has a simulator handler.
    loaded = w.rig.modules_loaded
    missing = [command for command, spec in commands().items()
               if spec.module in loaded and d.handler_for(command) is None]
    assert not missing, f"registered commands without handlers: {missing}"


def test_world_current_matches_setpoint_when_feedback_on(tmp_path):
    w = _approached_world(tmp_path)
    w.set_bias(0.1)
    w.set_setpoint(100e-12)
    time.sleep(0.7)  # let the setpoint transient finish
    i = np.array([w.current_now() for _ in range(200)])
    assert abs(np.median(np.abs(i)) - 100e-12) / 100e-12 < 0.15
    assert not w.withdrawn
    z = w.z_now()
    assert w.zctrl.limit_low_m < z < w.zctrl.limit_high_m


@requires_mast
def test_scan_save_roundtrip_through_real_client_and_reader(tmp_path):
    import mast.core.nanonis_patch  # noqa: F401
    from nanonis_spm import Nanonis as Client   # MAST's client class
    from mast.io.nanonis_files import read_sxm, sxm_oriented_frames
    from mast.vision.frame_validity import acquired_row_mask

    w = _approached_world(tmp_path, time_scale=20.0)   # 64 rows × 0.1 s sim = 320 ms wall
    d = build_dispatcher(w)
    with WireServer(d, ports=[0]) as srv:
        sock = socket.create_connection(("127.0.0.1", srv.bound_ports[0]), timeout=5)
        nn = Client(sock)
        try:
            assert nn.Bias_Set(0.05)[0] == ""
            assert nn.ZCtrl_SetpntSet(50e-12)[0] == ""
            assert nn.Scan_BufferSet([0, 14], 64, 64)[0] == ""
            assert nn.Scan_FrameSet(0.0, 0.0, 50e-9, 50e-9, 0.0)[0] == ""
            assert nn.Scan_SpeedSet(0.0, 0.0, 0.05, 0.05, 1, 1.0)[0] == ""
            err, _, v = nn.Scan_FrameGet()
            assert err == "" and v[2] == pytest.approx(50e-9)
            assert nn.Scan_Action(0, 0)[0] == ""
            assert nn.Scan_StatusGet()[2][0] == 1
            # live buffer: unacquired rows are zeros
            err, _, v = nn.Scan_FrameDataGrab(14, 1)
            assert err == "" and v[1] == "Z (m)" and v[2] == 64 and v[3] == 64
            grab = np.asarray(v[4], dtype=np.float64).reshape(64, 64)
            assert np.isfinite(grab).all()
            # wait for the frame (64 rows × 0.1 s sim / 20× = 320 ms wall)
            deadline = time.monotonic() + 10
            while nn.Scan_StatusGet()[2][0] == 1 and time.monotonic() < deadline:
                time.sleep(0.05)
            assert nn.Scan_StatusGet()[2][0] == 0
            err, _, v = nn.Scan_FrameDataGrab(14, 1)
            full = np.asarray(v[4], dtype=np.float64).reshape(64, 64)
            assert (np.abs(full) > 0).all()
            assert nn.Scan_Save(1, 5000)[0] == ""
            err, _, v = nn.Util_SessionPathGet()
            sess = Path(v[1])
        finally:
            sock.close()
    files = sorted(sess.glob("*.sxm"))
    assert len(files) == 1
    scan = read_sxm(str(files[0]))
    assert scan["header"]["scan_pixels"] == [64, 64]
    assert "Z" in scan["channels"] and "Current" in scan["channels"]
    fr = sxm_oriented_frames(scan, "Z")
    fwd, bwd = fr["forward"], fr["backward"]
    assert fwd.shape == (64, 64) and bwd.shape == (64, 64)
    mask = acquired_row_mask(fwd, bwd)
    assert mask.all()
    # forward and mirrored-back backward agree well on a flat-ish terrace image
    c = np.corrcoef(fwd.ravel(), bwd.ravel())[0, 1]
    assert c > 0.6, c
    assert fr["nm_per_px"] == pytest.approx(50 / 64, rel=1e-3)


@requires_mast
def test_aborted_scan_saves_nan_rows(tmp_path):
    from mast.io.nanonis_files import read_sxm
    from mast.vision.frame_validity import acquired_row_mask

    w = _approached_world(tmp_path, time_scale=1.0)
    w.scan.nx = w.scan.ny = 32
    w.scan.line_time_fwd_s = w.scan.line_time_bwd_s = 0.02
    w.scan_start()
    time.sleep(0.4)          # ~10 of 32 rows
    w.scan_stop()
    path = w.save_frame()
    scan = read_sxm(path)
    z = scan["channels"]["Z"]["forward"]
    mask = acquired_row_mask(z)
    assert 3 <= mask.sum() < 32
    assert np.isnan(z[-1]).all()
    # live buffer for the same frame: zeros, not NaN
    live = w.frame.live_buffer(14, "fwd")
    assert np.isfinite(live).all() and (live[-1] == 0).all()


def test_truth_and_events(tmp_path):
    w = _approached_world(tmp_path)
    t = w.truth()
    assert t["tip"]["n_apex"] == 1 and not t["withdrawn"]
    w.withdraw()
    assert w.truth()["withdrawn"] and w.events[-1]["kind"] == "withdraw"


def test_piezo_limits_are_volts_so_the_scanner_has_travel(tmp_path):
    """``Piezo.XYZLimitsGet`` returns DAC volts, not metres.

    MAST works out a reachable half range as ``calibration × min(10 V, |limit|)``. Reporting
    the limits in metres makes that product ~1e-13 m, so the scanner looks like it has no
    travel at all and every ``ConfigureScan`` is refused as out of range — which is how this
    was found. With volts the half range comes out at exactly ``Piezo.RangeGet / 2``.
    """
    from stmsim.modules import build_dispatcher

    w = _approached_world(tmp_path)
    d = build_dispatcher(w)
    enabled, xlo, xhi, ylo, yhi, zlo, zhi = d.handler_for("Piezo_XYZLimitsGet")()
    rx, ry, rz = d.handler_for("Piezo_RangeGet")()
    cx, cy, cz = d.handler_for("Piezo_CalibrGet")()
    assert enabled == 1
    assert (xlo, xhi, ylo, yhi) == (-10.0, 10.0, -10.0, 10.0)
    assert cx * min(10.0, abs(xlo), abs(xhi)) == pytest.approx(rx / 2, rel=1e-9)
    assert cy * min(10.0, abs(ylo), abs(yhi)) == pytest.approx(ry / 2, rel=1e-9)
    # Z carries the controller's own limits, converted through the same calibration
    assert cz * zhi == pytest.approx(w.zctrl.limit_high_m, rel=1e-9)
    assert cz * zlo == pytest.approx(w.zctrl.limit_low_m, rel=1e-9)
    assert -10.0 <= zlo < zhi <= 10.0


def test_preamp_gain_table_round_trips_and_the_widest_range_is_10_uA():
    """Gain index i is a full scale of 10 V / 10^(6+i); index 0 (1E6 V/A) is the widest
    range the preamp switches to, which is what the harness registers as the preamp's
    full scale (MAST clamps the setpoint ceiling to that fact)."""
    from stmsim.physics.junction import Preamp

    p = Preamp()
    assert p.max_full_scale_a == pytest.approx(10e-6)
    assert p.full_scale_for(3) == pytest.approx(10e-9) and p.full_scale_for(5) == pytest.approx(100e-12)
    for i in range(6):
        assert Preamp.index_for(p.full_scale_for(i)) == i
