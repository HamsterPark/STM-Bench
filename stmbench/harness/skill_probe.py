"""Run MAST's real skills against the simulator, one after another, and print what
each returned. Exploratory harness for the P2 gate (docs/DESIGN.md §4.6.6).

Usage (MAST venv, PYTHONPATH=MASTv2;STM-Bench)::

    python -m stmbench.harness.skill_probe --root "$STM_BENCH_DATA/probe_root" [--approached]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from stmbench.paths import data_path


def _setup_env(root: Path, ports: list[int]) -> None:
    os.environ["MAST2_PROJECT_ROOT"] = str(root)
    os.environ["MAST2_USER_ROOT"] = str(root)
    for role, port in zip(("MAIN", "MONITOR", "DATA", "EMERGENCY"), ports):
        os.environ[f"MAST_NANONIS_PORT_{role}"] = str(port)


def _short(v, n=300):
    s = json.dumps(v, default=str, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "…"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(data_path("probe_root")), help="default $STM_BENCH_DATA/probe_root")
    ap.add_argument("--approached", action="store_true")
    ap.add_argument("--time-scale", type=float, default=10.0)
    ap.add_argument("--skills", default="", help="comma list to restrict")
    a = ap.parse_args(argv)

    from stmsim.modules import build_dispatcher
    from stmsim.physics.rig import RigProfile
    from stmsim.physics.world import World
    from stmsim.wire.server import WireServer

    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    world = World(rig=RigProfile.load("reference-stm"), seed=3, session_dir=root / "session",
                  time_scale=a.time_scale)
    if a.approached:
        world.coarse.coarse_gap_m = world.surface_height_here() + 0.6e-9
        world.withdrawn = False
        world.zctrl_set(True)
    else:
        world.coarse.coarse_gap_m = 3e-6      # a few coarse cycles away
        plan_first = [("AutoApproach", {"wait_timeout_s": 120})]
    disp = build_dispatcher(world)
    disp.log_calls = True
    srv = WireServer(disp, ports=[0, 0, 0, 0]).start()
    _setup_env(root, srv.bound_ports)

    from mast.config import NanonisConfig
    from mast.core.connection import ConnectionPool
    from mast.core.execution_context import ExecutionContext
    from mast.core.registry import SkillRegistry
    from mast.core.state import InstrumentState

    pool = ConnectionPool(NanonisConfig())
    print("connect:", pool.connect_all(), flush=True)
    state = InstrumentState(pool)
    state.refresh()
    registry = SkillRegistry()
    n = registry.discover()
    print(f"registry: {n if n is not None else len(registry.list_skills())} skills", flush=True)

    def ctx():
        return ExecutionContext(pool=pool, state=state, registry=registry, approval_source="llm")

    plan: list[tuple[str, dict]] = [
        ("GetBias", {}),
        ("SetBias", {"bias_v": 0.1}),
        ("SetSetpoint", {"setpoint_a": 50e-12}),
        ("GetZPosition", {}),
        ("MeasureBarrierHeight", {}),
        ("SetScanBuffer", {"pixels": 64, "lines": 64}),
        ("ConfigureScan", {"center_x_m": 0.0, "center_y_m": 0.0, "width_m": 50e-9, "height_m": 50e-9,
                           "angle_deg": 0.0, "line_time_s": 0.1}),
        ("StartScan", {}),
        ("WaitScanComplete", {"timeout_ms": 120000}),
        ("SaveScan", {}),
        ("GetLatestScanFile", {}),
        ("GrabScanFrameData", {"channel_index": 14, "direction": 1}),
        ("CheckScanForCrash", {}),
        ("AcquireSTS", {}),
        ("BiasPulseWithReadback", {"bias_v": 3.0, "width_s": 0.1}),
        ("TipShapeWithReadback", {"tip_lift_m": -0.3e-9, "lift_height_m": 0.3e-9}),
        ("PreScanCheck", {"center_x_m": 0.0, "center_y_m": 0.0, "width_m": 50e-9, "pixels": 64, "line_time_s": 0.1}),
        ("FindCleanSpot", {}),
        ("WithdrawTip", {}),
        ("AutoApproach", {"wait_timeout_s": 120}),
        ("MeasureBarrierHeight", {}),
    ]
    if not a.approached:
        plan = [("AutoApproach", {"wait_timeout_s": 120})] + plan
    only = {s.strip() for s in a.skills.split(",") if s.strip()}
    for name, params in plan:
        if only and name not in only:
            continue
        sk = registry.get(name) if hasattr(registry, "get") else None
        if sk is None:
            print(f"--- {name}: NOT IN REGISTRY", flush=True)
            continue
        meta = (sk() if isinstance(sk, type) else sk).metadata()
        pnames = [p.name for p in meta.parameters]
        t0 = time.perf_counter()
        try:
            res = ctx().run(name, params)
            dt = time.perf_counter() - t0
            print(f"--- {name} ({dt:.1f}s) params={pnames}\n    success={res.success} error={res.error!r}\n"
                  f"    data={_short(res.data)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"--- {name}: EXC {type(exc).__name__}: {exc}", flush=True)
        state.refresh()
    print("truth:", _short(world.truth(), 600))
    print("events:", _short([e["kind"] for e in world.events], 400))
    errs = [c for c in disp.call_log if not c[3].startswith("ok")]
    print(f"calls={len(disp.call_log)} non-ok={len(errs)}")
    from collections import Counter
    print(Counter((c[1], c[3][:50]) for c in errs).most_common(20))
    pool.close_all()
    srv.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
