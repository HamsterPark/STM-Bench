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
from .core import RTOL, SCENARIO_DIR, Instrument, compare_fingerprints, replay_records, require_api_version
from .export import ExportDamaged, read_export

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
    """:func:`read_export`, with everything that makes a directory not an export an :class:`ExportDamaged`
    (its ``reason`` names the damage without the path or the OS's text)."""
    d = Path(export_dir)
    try:
        return read_export(d)
    except OSError as exc:
        raise ExportDamaged("an export file is missing or cannot be read", d) from None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ExportDamaged("an export file is not JSON in UTF-8", d) from None
    except (AttributeError, TypeError) as exc:                  # JSON, but not the shape an export has
        raise ExportDamaged(f"not an export directory ({type(exc).__name__})", d) from None


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def damaged_result(reason: str, scenario: Scenario | str | Path | None = None) -> dict:
    """The scored result of a damaged submission: score 0, no success, the numbers that describe the
    replay ``None`` (there was none), ``replay.consistent`` false and ``replay.damaged`` naming the damage.
    What the grade command prints for an :class:`ExportDamaged`."""
    try:
        scenario_id = _load_scenario(scenario).id if scenario is not None else None
    except Exception:  # noqa: BLE001 - the id is only information
        scenario_id = None
    why = _clip(reason)
    return {"score": 0.0, "success": False, "partial": None, "claims_total": None, "claims_verified": None,
            "instrument_time_s": None, "actions": None, "finished_in_episode": None,
            "replay": {"consistent": False, "damaged": why, "result_mismatches": [], "digest_mismatches": [],
                       "hashes_equal": False, "rounding_differences": [], "torn": False, "records": None},
            "scenario_id": scenario_id, "rules_version": RULES_VERSION, "export_version": None,
            "note": f"the export is damaged: {why}; scored 0"}


def _digest_check(inst: Instrument, digests: list, records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Every digest line against what the replay saved, and every file the replay saved against its line.

    Returns ``(mismatches, rounded)``. An array with equal hashes is consistent. With unequal hashes
    and a fingerprint on the line (export version 2) the fingerprints decide: within
    :data:`~stmbench.instrument.core.RTOL` it is a rounding difference (listed in ``rounded``, not a
    mismatch); beyond it, or in its structure, a mismatch. Without a fingerprint (version 1) unequal
    hashes are a mismatch."""
    fp_of = {"frame": inst.frame_fingerprint, "spectrum": inst.spectrum_fingerprint}
    tables = {"frame": (inst.frames, inst.frame_digest), "spectrum": (inst.spectra, inst.spectrum_digest)}
    made_at: dict[tuple[str, str], int] = {}              # which record saved a file, as the export says
    named_by = (("frame", "frame"), ("spectrum", "spectrum"), ("spectrum", "trace"))   # table, key of the reply
    for rec in records:
        res = rec.get("result") if rec.get("kind") == "act" else None
        for kind, key in named_by:                        # a move's reply names its trace, kept as a spectrum
            if isinstance(res, dict) and isinstance(res.get(key), str):
                made_at[(kind, res[key])] = rec["i"]
    out: list[dict] = []
    rounded: list[dict] = []
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
            fp_want = line.get("fp")
            if not isinstance(want, dict) or not isinstance(fp_want, dict):
                out.append({"i": line.get("i"), kind: name, "reason": "the hashes differ", "arrays": differ})
                continue
            fp_got = fp_of[kind](name)
            beyond: list[str] = []
            structure: list[str] = []
            worst_beyond = 0.0
            for arr in differ:
                if arr not in fp_got or arr not in want or not isinstance(fp_want.get(arr), dict):
                    structure.append(arr)
                    continue
                verdict, rel = compare_fingerprints(fp_want[arr], fp_got[arr])
                if verdict == "rounding":
                    rounded.append({"i": line.get("i"), kind: name, "array": arr, "max_rel_diff": rel})
                elif verdict == "beyond":
                    beyond.append(arr)
                    worst_beyond = max(worst_beyond, rel)
                else:
                    structure.append(arr)
            if beyond or structure:
                m = {"i": line.get("i"), kind: name, "arrays": sorted(beyond + structure)}
                if structure:
                    m["reason"] = "the hashes differ and the fingerprints do not match in shape or non-finite values"
                else:
                    m["reason"] = (f"the values differ beyond rounding (largest relative difference "
                                   f"{worst_beyond:.3g}, tolerance {RTOL:g})")
                out.append(m)
    for kind, (table, _) in tables.items():
        out.extend({"i": made_at.get((kind, name)), kind: name, "reason": "the export has no digest line for it"}
                   for name in table if name not in seen[kind])
    return out, rounded


def _digest_mismatches(inst: Instrument, digests: list, records: list[dict]) -> list[dict]:
    """The mismatches of :func:`_digest_check`."""
    return _digest_check(inst, digests, records)[0]


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
    many records were replayed. Records beyond the count in ``session.json`` (the console moved on
    between the copies of its files) are left out with their digest lines and said so in ``note``;
    ``replay.records`` is what was judged.

    ``strict``: an inconsistent replay scores 0 and is no success, and the numbers that describe
    it are ``None``: it was not the recorded episode, so they are those of a world the agent never
    saw, and they stay out of the result as unknown, not zero. Without it the replay's own grade
    stands and the divergence is reported beside it.

    Two kinds of input that cannot be graded. A **damaged submission** (a file missing, not JSON,
    not the shape of an export, a record missing or out of order, a log shorter than ``session.json``
    counts) raises :class:`~stmbench.instrument.export.ExportDamaged`, a :class:`ValueError` subclass:
    an export that cannot be verified is not a reproduction, so the grade command scores it 0
    (:func:`damaged_result`, exit 0). A **mismatch of setup** raises a plain :class:`ValueError`
    (exit 2, the grader's or the deployment's problem, which the OSWorld adapter reports as pending). The
    latter: the directory is not a gradable export of this scenario: not an
    export at all, another scenario, another rules, API or export version, or a ``session.json``
    that counts more records than the log holds (a log cut short). An export of another API
    version is a :class:`~stmbench.instrument.core.VersionError`, whose message says that it needs
    the code of its own version: its calls and records are not this version's, so a replay here
    would only differ."""
    export = _read(export_dir)
    session, records = export["session"], export["records"]
    sc = _load_scenario(scenario)
    if not isinstance(session.get("scenario_id"), str) or not _is_int(session.get("api_version")) \
            or not _is_int(session.get("rules_version")):
        raise ExportDamaged("session.json lacks a scenario, API version or rules version")
    if session.get("scenario_id") != sc.id:
        raise ValueError(f"the export is of scenario {_clip(session.get('scenario_id'))!r}, not {sc.id!r}")
    require_api_version(session.get("api_version"), "the export")      # first: it says which code can replay it
    if session.get("rules_version") != RULES_VERSION:
        raise ValueError(f"the export is of rules_version {_clip(session.get('rules_version'))}; "
                         f"this grader is rules_version {RULES_VERSION}")
    counted = session.get("records")
    if _is_int(counted) and counted > len(records):
        raise ExportDamaged(f"session.json counts {counted} records but the log holds {len(records)}")
    # session.json is written last by every sync and copied first, so a log (or digest file) copied
    # after the console had moved on may hold records the count does not: the recorded episode is
    # the first `counted` records. Digest lines of later records leave with them (report.json is
    # not read: the replay re-sends the report records itself).
    digests = export["digests"]
    left_out = 0
    if _is_int(counted) and counted < len(records):
        left_out = len(records) - counted
        records = records[:counted]
        digests = [d for d in digests if not (isinstance(d, dict) and _is_int(d.get("i")) and d["i"] >= counted)]

    inst, result_mismatches = replay_records(sc, seed, records)
    digest_mismatches, rounded = _digest_check(inst, digests, records)
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
    if left_out:
        parts.append(f"{left_out} records beyond the count in session.json were left out")
    if consistent:
        parts.append("replay consistent" + (f" (rounding differences in {len(rounded)} arrays, within {RTOL:g})"
                                            if rounded else ""))
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
                       "hashes_equal": not rounded and not digest_mismatches,
                       "rounding_differences": [{**_scrub({k: v for k, v in m.items() if k != "max_rel_diff"}),
                                                 "max_rel_diff": m["max_rel_diff"]} for m in rounded],
                       "torn": bool(export["torn"]), "records": len(records)},
            "scenario_id": sc.id, "rules_version": session["rules_version"], "export_version": session["version"],
            "note": "; ".join(parts)}


__all__ = ["damaged_result", "grade_export"]
