"""Trial: MAST CoreRuntime hosted on the simulator; registers the tip; runs a composite.

    python -m stmbench.harness.host_trial --root "$STM_BENCH_DATA/host_root" --skill ForgeAuTip
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from stmbench.paths import data_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(data_path("host_root")), help="default $STM_BENCH_DATA/host_root")
    ap.add_argument("--time-scale", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--skill", default="ForgeAuTip")
    ap.add_argument("--params", default='{"time_budget_h": 0.3, "forge_pixels": 128}')
    ap.add_argument("--vision", default="", help="MAST_VISION_BACKEND override (mock|legacy|vigil)")
    a = ap.parse_args(argv)

    from stmsim.physics.rig import RigProfile
    from stmsim.physics.tip import Apex, Tip
    from stmsim.physics.world import World
    from .runtime_host import RuntimeHost

    tip = Tip(material="W", form="qplus", radius_m=8e-9, rng=np.random.default_rng(a.seed + 1),
              apexes=[Apex(0, 0, 0, 1.0), Apex(2.5e-9, -1.5e-9, -0.2354e-9, 0.8)], lambda_per_s=2e-3)
    world = World(rig=RigProfile.load("reference-stm"), seed=a.seed, session_dir=Path(a.root) / "session",
                  time_scale=a.time_scale, tip=tip)
    world.coarse.coarse_gap_m = world.surface_height_here() + 0.6e-9
    world.withdrawn = False
    world.zctrl_set(True)
    world.transients.clear()
    world.achievable_z_tip()
    host = RuntimeHost(world, a.root, vision_backend=(a.vision or None))
    t0 = time.perf_counter()
    host.start()
    print(f"host up in {host.t_setup_s:.1f}s; facts={json.dumps(host.facts, default=str, ensure_ascii=False)}", flush=True)
    print("truth before:", json.dumps(world.truth()["tip"], default=str), flush=True)
    try:
        r = host.run_skill("FindCleanSpot", {})
        print("FindCleanSpot:", r.success, json.dumps(r.data, default=str)[:300], flush=True)
        params = json.loads(a.params)
        t1 = time.perf_counter()
        res = host.run_skill(a.skill, params)
        print(f"=== {a.skill} done in {time.perf_counter()-t1:.0f}s wall, {world.clock.sim():.0f}s sim; success={res.success}")
        print("error:", res.error)
        (Path(a.root) / "result.json").write_text(json.dumps({"success": res.success, "error": res.error, "data": res.data},
                                                             default=str, ensure_ascii=False, indent=1), encoding="utf-8")
        print("data:", json.dumps(res.data, default=str, ensure_ascii=False)[:2500])
    finally:
        truth_after = world.truth()
        from stmbench.trackB.truth_criteria import judge
        verdict = judge("tip_repaired", truth_after).as_dict()
        (Path(a.root) / "truth_after.json").write_text(json.dumps(
            {"truth_after": truth_after, "verdict": verdict,
             "tip_events": [(e.sim_s, e.kind, e.detail) for e in world.tip.events],
             "events": world.events}, default=str, ensure_ascii=False, indent=1), encoding="utf-8")
        print("truth after:", json.dumps(truth_after, default=str, ensure_ascii=False)[:1200], flush=True)
        print("TRUTH VERDICT:", verdict, flush=True)
        print("events:", Counter(e["kind"] for e in world.events))
        print("tip events:", [(round(e.sim_s), e.kind, e.detail.get("outcome")) for e in world.tip.events][:40])
        errs = [c for c in host.dispatcher.call_log if not c[3].startswith("ok")]
        print(f"calls={len(host.dispatcher.call_log)} non-ok={len(errs)}", Counter((c[1], c[3][:60]) for c in errs).most_common(10))
        host.stop()
        print(f"total wall {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
