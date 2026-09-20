"""Big trial: MAST's ``ForgeAuTip`` (the mode-C scripted baseline) on a deliberately damaged
tip in the simulator. Prints the narration/progress it emits, the outcome, and the hidden
truth before/after.

    python -m stmbench.harness.forge_trial --root "$STM_BENCH_DATA/forge_root" --time-scale 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from stmbench.paths import data_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(data_path("forge_root")), help="default $STM_BENCH_DATA/forge_root")
    ap.add_argument("--time-scale", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--budget-h", type=float, default=0.3)
    ap.add_argument("--skill", default="ForgeAuTip")
    ap.add_argument("--params", default="{}")
    a = ap.parse_args(argv)

    from stmsim.modules import build_dispatcher
    from stmsim.physics.rig import RigProfile
    from stmsim.physics.tip import Apex, Tip
    from stmsim.physics.world import World
    from stmsim.wire.server import WireServer
    import numpy as np

    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    os.environ["MAST2_PROJECT_ROOT"] = str(root)
    os.environ["MAST2_USER_ROOT"] = str(root)
    tip = Tip(material="W", form="qplus", radius_m=8e-9, rng=np.random.default_rng(a.seed + 1),
              apexes=[Apex(0, 0, 0, 1.0), Apex(2.5e-9, -1.5e-9, -0.2354e-9, 0.8)], lambda_per_s=2e-3)
    world = World(rig=RigProfile.load("reference-stm"), seed=a.seed, session_dir=root / "session",
                  time_scale=a.time_scale, tip=tip)
    world.coarse.coarse_gap_m = world.surface_height_here() + 0.6e-9
    world.withdrawn = False
    world.zctrl_set(True)
    world.false_landing_p = 0.05
    disp = build_dispatcher(world)
    disp.log_calls = True
    srv = WireServer(disp, ports=[0, 0, 0, 0]).start()
    for role, port in zip(("MAIN", "MONITOR", "DATA", "EMERGENCY"), srv.bound_ports):
        os.environ[f"MAST_NANONIS_PORT_{role}"] = str(port)

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
    registry.discover()
    print("truth before:", json.dumps(world.truth()["tip"], default=str), flush=True)
    params = {"time_budget_h": a.budget_h, "forge_pixels": 128}
    params.update(json.loads(a.params))
    ctx = ExecutionContext(pool=pool, state=state, registry=registry, approval_source="llm")
    t0 = time.perf_counter()
    try:
        res = ctx.run(a.skill, params)
        print(f"=== {a.skill} done in {time.perf_counter()-t0:.0f}s wall, {world.clock.sim():.0f}s sim")
        print("success:", res.success, "error:", res.error)
        (root / "result.json").write_text(json.dumps({"success": res.success, "error": res.error, "data": res.data},
                                                     default=str, ensure_ascii=False, indent=1), encoding="utf-8")
        print("data:", json.dumps(res.data, default=str, ensure_ascii=False)[:3000])
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print("EXC", exc)
    print("truth after:", json.dumps(world.truth(), default=str, ensure_ascii=False)[:1500], flush=True)
    kinds = [e["kind"] for e in world.events]
    from collections import Counter
    print("events:", Counter(kinds))
    print("tip events:", [(round(e.sim_s), e.kind, e.detail.get("outcome")) for e in world.tip.events][:60])
    errs = [c for c in disp.call_log if not c[3].startswith("ok")]
    print(f"calls={len(disp.call_log)} non-ok={len(errs)}", Counter((c[1], c[3][:60]) for c in errs).most_common(15))
    pool.close_all()
    srv.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
