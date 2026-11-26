"""The window's one door to the instrument (no Qt).

Why it exists. The desktop console runs the :class:`~stmbench.instrument.Instrument` in the same
process as its window, and what the window may know must stay what an MCP agent is told. This class is
the only code that holds the instrument: it exposes the same actions, observations, reports and
refusals, plus the task text, the manual and the reportable claims, and nothing from the episode's
truth, the seed or the verdict. :meth:`finish` returns nothing.

Every method takes one lock (the window's worker thread and its first reads share the instrument) and
syncs the export directory (:class:`~stmbench.instrument.export.ExportWriter`) afterwards, also when
the instrument refuses: a refusal is a record in the ledger, and an episode that is stopped from
outside is graded from the export as it stands.

:func:`read_spec` reads the sealed mode's one line of input. Its errors never repeat the line: it
holds the seed, which must appear nowhere else.
"""
from __future__ import annotations

import json
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..instrument import ActionError, Instrument
from ..instrument.core import SCENARIO_DIR
from ..instrument.export import ExportWriter
from ..trackB.rules import asks_citation

#: report fields beyond claim_id and value, in the order the result channel names them
REPORT_FIELDS = ("x_nm", "y_nm", "frame", "evidence", "unit", "sigma", "note")
_NUMERIC_REPORT_FIELDS = ("x_nm", "y_nm", "sigma")
_SCENARIO_ID = re.compile(r"[A-Za-z0-9_]{1,80}")


class SpecError(ValueError):
    """The sealed mode's input line is not a usable ``{"scenario": ..., "seed": ...}``."""


def read_spec(stream) -> tuple[str, int]:
    """The scenario id and the seed from one JSON line of ``stream`` (sealed mode's stdin).

    Reads one line only, so a launcher may keep the pipe open. Errors say what is wrong with the
    line, never what it holds: it carries the seed."""
    line = stream.readline()
    if isinstance(line, bytes):
        line = line.decode("utf-8", "replace")
    if not line.strip():
        raise SpecError("no spec on stdin: expected one JSON line {\"scenario\": \"<id>\", \"seed\": <int>}")
    try:
        obj = json.loads(line)
    except ValueError:
        raise SpecError("the spec line is not JSON") from None
    if not isinstance(obj, dict) or set(obj) != {"scenario", "seed"}:
        raise SpecError("the spec must be one JSON object with exactly the keys scenario and seed")
    scenario, seed = obj["scenario"], obj["seed"]
    if not isinstance(scenario, str) or not _SCENARIO_ID.fullmatch(scenario) \
            or not (SCENARIO_DIR / f"{scenario}.yaml").is_file():
        raise SpecError("the spec names no known scenario")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 63:
        raise SpecError("the seed must be a non-negative integer")
    return scenario, seed


def make_private_dir(path: str | Path) -> Path:
    """An empty directory only its owner can enter (POSIX mode 0700; elsewhere the mode is a no-op).
    The sealed mode's run directory holds the seed and, after the end, the truth."""
    p = Path(path)
    if p.exists() and any(p.iterdir()):
        raise FileExistsError(f"{p} is not empty: a private directory holds one episode")
    p.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == "posix":
        os.chmod(p, 0o700)
    return p


class GuiSession:
    """One episode behind the window: the instrument, its export, one lock."""

    def __init__(self, scenario: str | Path, seed: int, *, run_dir: str | Path, export_dir: str | Path,
                 lang: str = "zh"):
        run_dir = Path(run_dir)
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"{run_dir} is not empty: use a new directory for each episode")
        self.lang = "en" if lang == "en" else "zh"
        self._lock = threading.RLock()
        self._inst = Instrument(scenario, int(seed), out_dir=run_dir)
        self._export = ExportWriter(self._inst, export_dir)
        self.scenario_id: str = self._inst.scenario.id

    @contextmanager
    def _call(self) -> Iterator[Instrument]:
        with self._lock:
            try:
                yield self._inst
            finally:
                self._export.sync()

    # ── what the agent is told ──
    def task_text(self) -> str:
        """The task with its rule card, in the window's language."""
        with self._call() as inst:
            return inst.task(lang=self.lang)

    def manual(self) -> str:
        with self._call() as inst:
            return inst.manual()

    def claims(self) -> list[dict]:
        """The reportable claims, as the rule card states them (constraints are not reported); ``cite``: whether the card's
        evidence line asks to cite data in the evidence field."""
        with self._call() as inst:
            return [{"id": str(c["id"]), "kind": c.get("kind"), "unit": c.get("unit") or "",
                     "position": c.get("position") == "required", "say": str(c.get("say") or "").strip(),
                     "say_en": str(c.get("say_en") or "").strip(), "cite": asks_citation(c)}
                    for c in inst.scenario.claims if c.get("kind") != "constraint"]

    # ── the instrument ──
    def act(self, name: str, args: dict | None = None) -> dict:
        """The instrument's reply to one action; :class:`ActionError` with its words if it refuses."""
        with self._call() as inst:
            return inst.act(str(name), **dict(args or {}))

    def status(self) -> dict:
        with self._call() as inst:
            return inst.status()

    def info(self) -> dict:
        """The instrument's fixed facts: ranges, budget, temperature, material, a qPlus sensor's constants."""
        with self._call() as inst:
            return inst.info()

    def data(self) -> dict:
        """The saved frames (geometry, start time, channels) and spectra, oldest first."""
        with self._call() as inst:
            out = inst.list_data()
            for f in out["frames"]:                          # the channels the scan reply listed
                rec = inst.frames[f["frame"]]
                f["channels"] = ["z", "current"] + (["didv"] if rec.didv_fwd is not None else [])
            return out

    def frame(self, name: str, channel: str = "z", direction: str = "forward") -> dict:
        with self._call() as inst:
            return inst.frame(name, channel, direction)

    def spectrum(self, name: str) -> dict:
        with self._call() as inst:
            return inst.spectrum(name)

    # ── results ──
    def report(self, fields: dict) -> str:
        """Report one result from the form's fields, cleaned as the web console cleans them:
        evidence may be a comma-separated string; numbers arrive as typed."""
        b = dict(fields or {})
        if b.get("value") in (None, ""):
            raise ActionError("report: value is required")
        ev = b.get("evidence")
        if isinstance(ev, str):
            b["evidence"] = [e.strip() for e in ev.replace("，", ",").split(",") if e.strip()]
        args: dict[str, Any] = {}
        for k in REPORT_FIELDS:
            v = b.get(k)
            if v in (None, "", []):
                continue
            args[k] = float(v) if k in _NUMERIC_REPORT_FIELDS else v
        with self._call() as inst:
            return inst.report(str(b.get("claim_id") or ""), float(b["value"]), **args)

    def finish(self) -> None:
        """End the episode and write its record. Nothing comes back: not the verdict, not the summary."""
        with self._call() as inst:
            inst.finish()

    @property
    def finished(self) -> bool:
        return bool(self._inst.finished)
