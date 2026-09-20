"""Shared plumbing for the scripted paper baselines."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..harness.results import ResultSink


class Baseline:
    """A recorded run of MAST skills, with the same result channel an LLM episode uses."""

    def __init__(self, host, scenario, seed: int = 0, run_id: str = "baseline"):
        self.host = host
        self.scenario = scenario
        self.seed = seed
        self.ctx = host.context(run_id=f"{scenario.id}-{run_id}")
        self.sink = ResultSink()
        self.steps: list[dict] = []
        self.error: str | None = None

    # ── instrument ──
    def run(self, skill: str, params: dict | None = None, *, optional: bool = False):
        params = dict(params or {})
        try:
            res = self.ctx.run(skill, params)
        except Exception as exc:  # noqa: BLE001 — a baseline records failures, it does not raise
            self.steps.append({"skill": skill, "success": False, "error": f"{type(exc).__name__}: {exc}"})
            if not optional and self.error is None:
                self.error = f"{skill}: {exc}"
            return None
        self.steps.append({"skill": skill, "success": bool(res.success), "error": res.error or ""})
        if not res.success and not optional and self.error is None:
            self.error = f"{skill}: {res.error}"
        return res

    @staticmethod
    def data(res, key: str, default=None):
        if res is None or not getattr(res, "data", None):
            return default
        return res.data.get(key, default) if isinstance(res.data, dict) else default

    # ── files the instrument left behind ──
    def session_dir(self) -> Path:
        return Path(self.host.world.session_dir)

    def frames(self) -> list[Path]:
        return sorted(self.session_dir().glob("*.sxm"))

    def latest_frame(self) -> Path | None:
        got = self.frames()
        return got[-1] if got else None

    def dats(self, pattern: str = "*.dat") -> list[Path]:
        return sorted(self.session_dir().glob(pattern))

    # ── result ──
    def report(self, *args, **kw) -> None:
        self.sink.report(*args, **kw)

    def finish(self, **data: Any) -> dict:
        folded = self.sink.folded()
        return {"skill": "paper_baseline", "paper": self.scenario.paper_id,
                "steps": self.steps, "success": self.error is None,
                "error": self.error, "results": folded,
                "data": {"n_claims_reported": len(folded["claims"]), **data}}
