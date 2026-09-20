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
