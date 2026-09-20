"""Probe: what does the provider return when the model calls a tool that is NOT in the
bound list? Logs, per model call, how many tools were bound, whether the names the model
returned were among them, and the raw tool_calls — to a file (MAST captures stdout).

    python -m stmbench.harness.probe_tool_names --model kimi-k3 --out "$STM_BENCH_DATA/tmp_dbg/probe.json"
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from stmbench.paths import data_path


class _LoggingPort:
    """ChatModelPort proxy: records bound tool names and returned tool calls."""

    def __init__(self, inner):
        self._inner = inner
        self.log: list[dict] = []

    def invoke(self, request):
        bound = [getattr(t, "name", None) or (t.get("function", {}).get("name") if isinstance(t, dict) else None)
                 for t in request.tools]
        resp = self._inner.invoke(request)
        calls = [{"name": c.get("name"), "args": c.get("args")} for c in (resp.tool_calls or [])]
        self.log.append({"n_bound": len(bound), "bound_sample": sorted(b for b in bound if b)[:400],
                         "calls": calls,
                         "calls_not_bound": [c["name"] for c in calls if c["name"] not in bound],
                         "text": (resp.text or "")[:300]})
        return resp

    def stream(self, request):
        return self._inner.stream(request)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="kimi-k3")
    ap.add_argument("--out", default=str(data_path("tmp_dbg", "probe.json")))
    ap.add_argument("--mode", default="B0")
    a = ap.parse_args(argv)
    logging.disable(logging.WARNING)

    from stmsim.physics.rig import RigProfile
    from stmsim.physics.world import World
    from stmbench.harness.runtime_host import RuntimeHost

    w = World(rig=RigProfile.load("reference-stm"), seed=3, session_dir=data_path("tmp_dbg", "probe_s"), time_scale=20.0)
    w.coarse.coarse_gap_m = w.surface_height_here() + 0.6e-9
    w.withdrawn = False
    w.zctrl_set(True)
    w.transients.clear()
    w.achievable_z_tip()
    host = RuntimeHost(w, str(data_path("tmp_dbg", "probe_host")))
    host.start()
    out = {"model": a.model, "mode": a.mode}
    try:
        from mast.agentruntime.model import LangChainModelPort
        from mast.agents._shared.models import make_chat_model
        from stmbench.harness.ic_driver import run_llm_episode

        port = _LoggingPort(LangChainModelPort(make_chat_model(model_id=a.model, usage_source="probe")))
        t0 = time.time()
        r = run_llm_episode(
            host,
            "只做一件事：把扫描分辨率设为 128×128 像素（SetScanBuffer），然后读回扫描缓冲区设置（GetScanBuffer）确认，"
            "报告结果后写 [DONE]。不要做别的。",
            model_id=a.model, mode=a.mode, episode_id="probe", max_model_calls=8, max_tool_calls=10,
            max_turns=2, model=port)
        out["driver"] = {"outcome": r.outcome, "turns": r.turns, "model_calls": r.model_calls,
                         "tool_calls": r.tool_calls, "error": r.error,
                         "executed": [e.get("name") for e in r.events if e.get("kind") == "tool_start"],
                         "final": r.final_text[:800], "wall_s": time.time() - t0}
        out["model_log"] = port.log
    finally:
        host.stop()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
