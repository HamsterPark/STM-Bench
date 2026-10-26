"""Grading an exported episode on the host that holds the seed (the export: :mod:`.export`).

Why it exists. An export leaves a machine the agent worked on, so nothing in it is trusted
for what was measured or how the episode was judged: the records only say what was *done*.
The grader holds the scenario and the seed that machine never had, runs the recorded calls
again on its own instrument (:func:`~stmbench.instrument.core.replay_records`) and judges that
run, as it stands when the export ends.

The replay also has to *be* the episode. Every action's reply and instrument times, and every
saved frame's and spectrum's hashes, must equal what the export recorded, and every file the
replay saved must have its hash line. When they do not (another numerical environment, an
edited log, a file lost on the way out) the graded run is not the one the agent saw: by default
(``strict``) it then scores 0 and is not a success, and the result says what differed;
``strict=False`` keeps the replay's own score and still reports the divergence.

What a result may hold. It repeats nothing of the world beyond counts: no seed, no truth, no
per-claim detail of the verdict (``Verdict.details`` carries each claim's true value, the
atoms' positions and the tolerances). Text that came out of the export is clipped before it
is repeated.
"""
from __future__ import annotations

import json
from pathlib import Path

from stmsim.scenario import Scenario

from ..trackB.rules import RULES_VERSION
from .core import API_VERSION, SCENARIO_DIR, Instrument, replay_records
from .export import read_export

#: longest piece of export-supplied text a result repeats
_CLIP = 120


def _clip(x) -> str:
    s = "".join(c if c.isprintable() else "?" for c in str(x))
    return s if len(s) <= _CLIP else s[:_CLIP - 3] + "..."


def _scrub(m: dict) -> dict:
    """A mismatch with its export-supplied text clipped (record numbers stay numbers)."""
    def one(v):
        if v is None or (isinstance(v, int) and not isinstance(v, bool)):
            return v
        if isinstance(v, list):
            return [_clip(x) for x in v]
        return _clip(v)
    return {k: one(v) for k, v in m.items()}


def _load_scenario(scenario: Scenario | str | Path) -> Scenario:
    if isinstance(scenario, Scenario):
        return scenario
    p = Path(scenario)
    path = p if p.is_file() else SCENARIO_DIR / f"{scenario}.yaml"
    if not path.is_file():
        raise ValueError(f"no scenario {_clip(scenario)!r}")
    return Scenario.load(path)


def _read(export_dir: str | Path) -> dict:
    """:func:`read_export`, with everything that makes a directory not an export a :class:`ValueError`."""
    d = Path(export_dir)
    try:
        return read_export(d)
    except OSError as exc:
        raise ValueError(f"{d}: cannot read the export ({exc})") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{d}: a file is not JSON in UTF-8 ({exc})") from None
    except (AttributeError, TypeError) as exc:                  # JSON, but not the shape an export has
        raise ValueError(f"{d}: not an export directory ({type(exc).__name__}: {exc})") from None


def _digest_mismatches(inst: Instrument, digests: list, records: list[dict]) -> list[dict]:
    """Every digest line against what the replay saved, and every file the replay saved against its line."""
    tables = {"frame": (inst.frames, inst.frame_digest), "spectrum": (inst.spectra, inst.spectrum_digest)}
    made_at: dict[tuple[str, str], int] = {}              # which record saved a file, as the export says
    for rec in records:
        res = rec.get("result") if rec.get("kind") == "act" else None
        for kind in tables:
            if isinstance(res, dict) and isinstance(res.get(kind), str):
                made_at[(kind, res[kind])] = rec["i"]
    out: list[dict] = []
    seen: dict[str, set] = {kind: set() for kind in tables}
    for line in digests:
        kinds = [k for k in tables if isinstance(line, dict) and k in line]
        if len(kinds) != 1 or not isinstance(line[kinds[0]], str):
            out.append({"i": line.get("i") if isinstance(line, dict) else None,
                        "reason": "a digest line names no frame or spectrum"})
            continue
        kind = kinds[0]
        name = line[kind]
        table, digest_of = tables[kind]
        seen[kind].add(name)
        if name not in table:
            out.append({"i": line.get("i"), kind: name, "reason": "the replay saved no such file"})
            continue
        want, got = line.get("digest"), digest_of(name)
        if want != got:
            keys = set(got) | (set(want) if isinstance(want, dict) else set())
            differ = sorted(k for k in keys if not isinstance(want, dict) or want.get(k) != got.get(k))
            out.append({"i": line.get("i"), kind: name, "reason": "the hashes differ", "arrays": differ})
    for kind, (table, _) in tables.items():
        out.extend({"i": made_at.get((kind, name)), kind: name, "reason": "the export has no digest line for it"}
                   for name in table if name not in seen[kind])
    return out


def grade_export(export_dir: str | Path, scenario: Scenario | str | Path, seed: int, *,
                 strict: bool = True) -> dict:
    """Replay an export with the scenario and seed the grader holds, and grade the replay.

    ``score`` (0-1) and ``success`` are the grade. ``partial`` (the fraction of claims verified,
    capped when over budget), ``claims_verified``, ``instrument_time_s`` and ``actions`` describe
    the replayed episode; ``claims_total`` is the scenario's. ``finished_in_episode`` is whether
    the episode reached ``finish`` before the export was taken; one that did not is judged as it
    stands. ``replay`` says whether the replay was the recorded episode (``consistent``), where it
    was not (``result_mismatches``: replies and instrument times, ``digest_mismatches``: hashes of
    frames and spectra), whether the log's last line was cut off and left out (``torn``) and how
    many records were replayed.

    ``strict``: an inconsistent replay scores 0 and is no success, and the numbers that describe
    it are ``None``: it was not the recorded episode, so they are those of a world the agent never
    saw, and they stay out of the result as unknown, not zero. Without it the replay's own grade
    stands and the divergence is reported beside it.

    :class:`ValueError` when the directory is not a gradable export of this scenario: not an
    export at all, another scenario, another rules, API or export version, or a ``session.json``
    that counts more records than the log holds (a log cut short)."""
    export = _read(export_dir)
    session, records = export["session"], export["records"]
    sc = _load_scenario(scenario)
    if session.get("scenario_id") != sc.id:
        raise ValueError(f"the export is of scenario {_clip(session.get('scenario_id'))!r}, not {sc.id!r}")
    for key, ours in (("rules_version", RULES_VERSION), ("api_version", API_VERSION)):
        if session.get(key) != ours:
            raise ValueError(f"the export is of {key} {_clip(session.get(key))}; this grader is {key} {ours}")
    counted = session.get("records")
    if isinstance(counted, int) and not isinstance(counted, bool) and counted > len(records):
        raise ValueError(f"session.json counts {counted} records but the log holds {len(records)}")

    inst, result_mismatches = replay_records(sc, seed, records)
    digest_mismatches = _digest_mismatches(inst, export["digests"], records)
    consistent = not result_mismatches and not digest_mismatches
    judged = consistent or not strict                         # whether the replay's own judgement is the grade
    summary = inst.finish()                                   # finished by the replay: its judged summary
    keys = ("partial", "claims_verified", "instrument_time_s", "actions")
    replayed = {k: summary[k] if judged else None for k in keys}
    finished = bool(session.get("finished"))
    parts = [f"{sc.id}: {summary['claims_verified']}/{summary['claims_total']} claims verified, "
             f"{summary['instrument_time_s']:.0f} s of instrument time, {summary['actions']} actions"
             if judged else sc.id]
    if judged and summary.get("budget_overrun"):
        parts.append("over the instrument-time budget")
    if not finished:
        parts.append("the episode never reached finish: judged as it stands")
    if export["torn"]:
        parts.append("the log's last line was cut off and left out")
    if consistent:
        parts.append("replay consistent")
    else:
        parts.append(f"replay differs from the recorded episode ({len(result_mismatches)} result and "
                     f"{len(digest_mismatches)} digest mismatches); "
                     + ("scored 0 (strict)" if strict else "scored as replayed (not strict)"))
    return {"score": float(summary["partial"]) if judged else 0.0, "success": bool(summary["success"]) and judged,
            "partial": replayed["partial"], "claims_total": summary["claims_total"],
            "claims_verified": replayed["claims_verified"], "instrument_time_s": replayed["instrument_time_s"],
            "actions": replayed["actions"], "finished_in_episode": finished,
            "replay": {"consistent": consistent,
                       "result_mismatches": [_scrub(m) for m in result_mismatches],
                       "digest_mismatches": [_scrub(m) for m in digest_mismatches],
                       "torn": bool(export["torn"]), "records": len(records)},
            "scenario_id": sc.id, "rules_version": session["rules_version"], "export_version": session["version"],
            "note": "; ".join(parts)}


__all__ = ["grade_export"]
