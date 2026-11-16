"""The agent-visible record of an episode, written while it happens.

Why it exists. A screen agent works on a machine where the instrument's process and its private
ledger (``episode.json``: the seed, the truth, the verdict) are out of its reach; what leaves that
machine to be graded is an export directory. It holds the calls in order - the records a run
directory's ``actions.jsonl`` holds - appended as they happen, so an episode that never reaches
``finish`` (the machine was stopped, the agent ran out of steps) is graded as it stands; and the
hashes and fingerprints of what was measured, so a replay elsewhere can show that it measured the same. It holds
nothing the agent may not see: no seed (the grader holds it), no truth, no verdict.

Version 2 adds ``"fp"`` to every digests line: per array, a coarse fingerprint
(:func:`~stmbench.instrument.core.array_fingerprint`: shape, non-finite mask, min, max, mean, std
and block means). Equal hashes remain the first, exact test; when they differ, the grader compares
the fingerprints, so a replay on another machine whose arrays differ in the last bits (another libm
or BLAS kernel: measured on trial 6, see docs/DESIGN.md) is not mistaken for a different
measurement. A fingerprint summarises arrays the agent itself measured, so it shows it nothing new.
Version 1 exports (hashes only) are still read, and graded exactly.

    <export>/
      session.json   format, versions, scenario id, finished, record count      (replaced)
      actions.jsonl  the instrument's log records, in order                      (appended)
      digests.jsonl  one line per saved frame, spectrum or move trace: hashes, fingerprints (appended)
      report.json    the results reported so far, folded as the judge reads them (replaced)

:class:`ExportWriter` writes it beside an :class:`~stmbench.instrument.core.Instrument`;
:func:`read_export` reads it back for a grader.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..trackB.rules import RULES_VERSION
from .core import API_VERSION, Instrument, _json_default

EXPORT_FORMAT = "stmbench-export"
EXPORT_VERSION = 2
#: versions :func:`read_export` accepts: 1 has hashes only, 2 adds fingerprints beside them
READABLE_VERSIONS = (1, 2)


class ExportDamaged(ValueError):
    """The submission itself is damaged: a file missing or unreadable, not JSON, not the shape of an
    export, a record missing or out of order, a log shorter than ``session.json`` counts. An export
    that cannot be verified is not a reproduction, so the grade command scores it 0 (exit 0), unlike
    a mismatch of setup (another scenario, rules, API or export version), which is a plain
    :class:`ValueError` (exit 2: the grader's or the deployment's problem, not the agent's).
    ``reason`` is the text without the directory, safe to repeat in a result."""

    def __init__(self, reason: str, where=None):
        self.reason = reason
        super().__init__(f"{where}: {reason}" if where is not None else reason)


def _chmod(path: Path, mode: int) -> None:
    """Readable by every account, writable only by the instrument's own (POSIX; elsewhere a no-op)."""
    if os.name == "posix":
        os.chmod(path, mode)


def _replace(path: Path, obj) -> None:
    """Write a whole JSON file so that a reader never finds it half written."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, default=_json_default, ensure_ascii=False, indent=1), encoding="utf-8")
    _chmod(tmp, 0o644)
    os.replace(tmp, path)


def _append(path: Path, objs: list[dict]) -> None:
    """Append JSON lines and push them to the disk: a record written is a record kept."""
    if not objs:
        return
    with path.open("a", encoding="utf-8") as fh:
        for o in objs:
            fh.write(json.dumps(o, default=_json_default, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


class ExportWriter:
    """Keeps an export directory up to date with one :class:`Instrument` episode: call
    :meth:`sync` after every call on the instrument (actions, refusals, observations, reports,
    ``finish``). One episode per directory."""

    def __init__(self, inst: Instrument, export_dir: str | Path):
        self.inst = inst
        self.dir = Path(export_dir)
        if self.dir.exists() and any(self.dir.iterdir()):
            raise FileExistsError(f"{self.dir} is not empty: an export directory holds one episode")
        self.dir.mkdir(parents=True, exist_ok=True)
        _chmod(self.dir, 0o755)
        for name in ("actions.jsonl", "digests.jsonl"):
            p = self.dir / name
            p.touch()
            _chmod(p, 0o644)
        self._n = 0
        _replace(self.dir / "report.json", inst.sink.folded())
        self._write_session()

    def sync(self) -> int:
        """Write what the instrument logged since the last call; returns how many records."""
        inst = self.inst
        new = inst.log[self._n:]
        digests = []
        for rec in new:
            if rec.get("kind") != "act":
                continue
            res = rec.get("result") or {}
            if res.get("frame") in inst.frames:
                digests.append({"i": rec["i"], "frame": res["frame"], "digest": inst.frame_digest(res["frame"]),
                                "fp": inst.frame_fingerprint(res["frame"])})
            if res.get("spectrum") in inst.spectra:
                digests.append({"i": rec["i"], "spectrum": res["spectrum"],
                                "digest": inst.spectrum_digest(res["spectrum"]),
                                "fp": inst.spectrum_fingerprint(res["spectrum"])})
            # a move's trace is kept like a spectrum, and the grader re-checks it like one
            if res.get("trace") in inst.spectra:
                digests.append({"i": rec["i"], "spectrum": res["trace"],
                                "digest": inst.spectrum_digest(res["trace"]),
                                "fp": inst.spectrum_fingerprint(res["trace"])})
        _append(self.dir / "actions.jsonl", new)
        _append(self.dir / "digests.jsonl", digests)
        if any(r.get("kind") == "report" for r in new):
            _replace(self.dir / "report.json", inst.sink.folded())
        self._n += len(new)
        if new:
            self._write_session()
        return len(new)

    def _write_session(self) -> None:
        inst = self.inst
        _replace(self.dir / "session.json", {
            "format": EXPORT_FORMAT, "version": EXPORT_VERSION, "api_version": API_VERSION,
            "rules_version": RULES_VERSION, "scenario_id": inst.scenario.id,
            "finished": bool(inst.finished), "records": self._n})


def _read_jsonl(path: Path) -> tuple[list[dict], bool]:
    """The lines of a JSON-lines file, and whether its last line was cut off mid-write (then left
    out). A damaged line anywhere else is not a cut-off write: :class:`ExportDamaged`."""
    lines = path.read_text(encoding="utf-8").split("\n")
    out: list[dict] = []
    for k, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if k == len(lines) - 1:               # no newline after it: the write was interrupted
                return out, True
            raise ExportDamaged(f"{path.name}: line {k + 1} is not a JSON record") from None
    return out, False


def read_export(export_dir: str | Path) -> dict:
    """An export directory as a grader reads it: ``session``, ``records`` (the instrument's log,
    in order), ``digests``, and ``torn`` - True when the log's last line was cut off mid-write
    (it is left out). :class:`ExportDamaged` for anything else that is not a complete export (a
    missing or unreadable file raises the underlying :class:`OSError` or :class:`json.JSONDecodeError`,
    which the grader turns into :class:`ExportDamaged`); a plain :class:`ValueError` for an export
    of a version this code does not read, which is a mismatch of setup."""
    d = Path(export_dir)
    session = json.loads((d / "session.json").read_text(encoding="utf-8"))
    version = session.get("version")
    if session.get("format") != EXPORT_FORMAT or not isinstance(version, int) or isinstance(version, bool):
        raise ExportDamaged(f"session.json is not that of a {EXPORT_FORMAT} directory")
    if version not in READABLE_VERSIONS:
        raise ValueError(f"{d} is not a version-{' or '.join(map(str, READABLE_VERSIONS))} {EXPORT_FORMAT} directory")
    records, torn = _read_jsonl(d / "actions.jsonl")
    for k, rec in enumerate(records):
        if not isinstance(rec, dict) or rec.get("i") != k:
            raise ExportDamaged(f"actions.jsonl: record {k} carries i = {rec.get('i') if isinstance(rec, dict) else rec!r}: a record is missing or out of order")
    digests, _ = _read_jsonl(d / "digests.jsonl") if (d / "digests.jsonl").exists() else ([], False)
    return {"session": session, "records": records, "digests": digests, "torn": torn}


__all__ = ["EXPORT_FORMAT", "EXPORT_VERSION", "ExportDamaged", "ExportWriter", "read_export"]
