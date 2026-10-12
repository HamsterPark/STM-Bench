"""RuntimeHost: MAST's CoreRuntime on the simulator, with the world's facts synced in."""
from __future__ import annotations

import pytest

from stmsim.physics.rig import RigProfile
from stmsim.physics.world import World

from tests.conftest import requires_mast

pytestmark = requires_mast


def test_host_registers_tip_opens_experiment_and_runs_skills(tmp_path):
    from stmbench.harness.runtime_host import RuntimeHost

    w = World(rig=RigProfile.load("reference-stm"), seed=7, session_dir=tmp_path / "s", time_scale=20.0)
    w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
    w.withdrawn = False
    w.zctrl_set(True)
    w.transients.clear()
    w.achievable_z_tip()
    host = RuntimeHost(w, tmp_path / "out")
    host.start()
    try:
        assert host.t_setup_s < 60
        f = host.facts
        assert f["tip"]["ok"] and f["is_qplus"] is True
        # the preamp fact is the WIDEST switchable range, so MAST's clamp leaves the
        # setpoint ceiling at its 100 nA default and a 57 nA manipulation setpoint is legal
        # (declaring the 10 nA imaging range clamped every MoveAtomTo, 2026-09-11)
        assert f["instrument_profile"]["preamp_full_scale_a"] == pytest.approx(w.preamp.max_full_scale_a)
        from mast.config import SafetyLimits
        from mast.core.safety import SafetyGuard
        assert SafetyGuard(SafetyLimits())._limits.setpoint_max_a == pytest.approx(100e-9)
        assert f["experiment"]["ok"], f["experiment"]
        assert f["vacuum_attest"]["ok"], f["vacuum_attest"]
        from mast.core.vacuum_interlock import check as vac_check
        assert vac_check().allow, vac_check().reason
        from mast.core import tip_state
        assert tip_state.current_tip_facts()["form"] == "qplus"
        r = host.run_skill("GetBias")
        assert r.success
        r = host.run_skill("MeasureBarrierHeight")
        assert r.success and r.data.get("n_usable", 0) >= 3, r.data
        # a composite that the sample_gate guards must now run
        r = host.run_skill("FindCleanSpot", {})
        assert r.success, r.error
        errs = [c for c in host.dispatcher.call_log if not c[3].startswith("ok") and "NeedModule" not in c[3]]
        assert not errs, errs[:5]

        # ── the context must carry the runtime's marker sink ──────────────────
        # Without it a RelocateCoarseXY nested in ForgeAuTip never advances
        # coord_epoch, and every fresh coarse site inherits the previous site's
        # pulse marks (host trial 9: "surface spent" ×5, verdict "spinning").
        from mast.core.coord_epoch import read_current_epoch
        from mast.core.map_scope import load_markers, record_damage_marker

        ctx = host.context()
        assert callable(getattr(ctx, "marker_sink", None))
        assert record_damage_marker(0.0, 0.0, kind="pulse", skill_name="test") is not None
        e0 = read_current_epoch()
        assert e0 is not None
        before, _, known = load_markers()
        assert known and any(getattr(m, "kind", None) == "pulse" for m in before)
        ctx.marker_sink({"skill": "RelocateCoarseXY", "success": True,
                         "params": {"direction": "x+", "steps": 5},
                         "data": {"direction": "x+", "steps": 5, "lateral_steps_taken": 5}})
        assert read_current_epoch() == e0 + 1
        after, _, _ = load_markers()
        assert not any(getattr(m, "kind", None) == "pulse" for m in after), \
            "old-epoch pulse mark still blocks the fresh site"
    finally:
        host.stop()


@pytest.mark.parametrize("timeout_s", [None, 60.0])
def test_controller_timeout_reaches_the_runtime_config(tmp_path, monkeypatch, timeout_s):
    """The simulator renders a whole frame inside Scan_Action; a large Au(111) frame took
    2-15 s on a loaded machine, past MAST's 5 s reply timeout, and MAST dropped the
    connection. The host can raise that timeout; None keeps MAST's own default."""
    import mast.core.runtime as mast_runtime
    from mast.config import NanonisConfig

    from stmbench.harness import runtime_host

    seen = {}

    class CapturingRuntime:
        def __init__(self, cfg):
            seen["cfg"] = cfg

        def setup(self):
            pass

    monkeypatch.setattr(mast_runtime, "CoreRuntime", CapturingRuntime)
    monkeypatch.setattr(runtime_host, "_export_api_keys_from_repo", lambda: {})
    monkeypatch.setattr(runtime_host.RuntimeHost, "sync_facts", lambda self: {})
    # start() points these at its own root and ports; registering them restores them afterwards
    for var in ("MAST2_PROJECT_ROOT", "MAST2_USER_ROOT", "MAST_NANONIS_PORT_MAIN",
                "MAST_NANONIS_PORT_MONITOR", "MAST_NANONIS_PORT_DATA", "MAST_NANONIS_PORT_EMERGENCY"):
        monkeypatch.setenv(var, "")

    w = World(rig=RigProfile.load("reference-stm"), seed=7, session_dir=tmp_path / "s")
    host = runtime_host.RuntimeHost(w, tmp_path / "out", controller_timeout_s=timeout_s)
    try:
        host.start()
    finally:
        host.stop()
    controllers = [v for v in vars(seen["cfg"]).values() if isinstance(v, NanonisConfig)]
    expected = NanonisConfig().timeout_s if timeout_s is None else timeout_s
    assert [c.timeout_s for c in controllers] == [expected]
