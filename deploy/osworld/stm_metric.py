"""OSWorld-Science ``module_metric`` adapter for STM-Bench.

The harness imports this file by path and calls :func:`grade_episode` as
``fn(result_path, {"type": "rule", "rules": {"oracle_dir": <dir>}})``. ``result_path`` is the
collected ``actions.jsonl``; the rest of the export (``session.json``, ``digests.jsonl``,
``report.json``) was collected beside it. ``oracle_dir`` holds ``gt.json``::

    {"scenario": "P4_atom_positioning_cu111", "seed": 0, "grader_image": "stmbench-grader:<tag>"}

The runner requires only Python 3.11 and brings its own dependencies, while STM-Bench needs
Python 3.13 and numpy 2. So this module only runs the real grader (``python -m stmbench.cli
grade``) as a separate process and reads its JSON: in the Docker image ``grader_image`` names (no
network, the export mounted read-only) or, when ``STMBENCH_GRADER_CMD`` is set, through that
command prefix (``<python> -m stmbench.cli``, split like a shell would). Standard library only.

Two kinds of outcome. A grade that was made returns ``(score, note)``; an inconsistent replay or
a failed claim is a real ``0.0``. A fault on the grading side (no Docker, no image, a timeout, a
crash, a grader exit status other than 0, output that is not a result, no readable ``gt.json``)
says nothing about the submission, so it returns ``{"score": 0.0, "pending": <reason>, "note":
...}``: the runner's scoring turns a ``pending`` detail into "no score, no verdict" rather than a
zero. The adapter never raises.

The seed is on the host's command line while the grader runs (``--seed N``, also inside ``docker
run``): anyone who can read the host's process list can see it, and only there. It is replaced by
``<seed>`` in everything this module returns (notes, reasons, the grader's stderr).
"""
from __future__ import annotations

import json
import math
import os
import re
import shlex
import subprocess
import uuid
from pathlib import Path

CMD_ENV = "STMBENCH_GRADER_CMD"
#: a replay of a long episode takes minutes; a hung one must not hold the harness for ever
TIMEOUT_S = 30 * 60
_CLIP = 300


def _scrub(text, seed: str | None) -> str:
    """``text`` with the seed's digits replaced by ``<seed>``. A seed of fewer than four digits is replaced
    only where it stands alone (it would otherwise mangle every number of the text)."""
    s = str(text)
    if seed:
        s = s.replace(seed, "<seed>") if len(seed) >= 4 else re.sub(rf"(?<!\d){re.escape(seed)}(?!\d)", "<seed>", s)
    return s


def _clip(x, seed: str | None = None) -> str:
    s = " ".join(_scrub(x, seed).split())
    return s if len(s) <= _CLIP else s[:_CLIP - 3] + "..."


class _Fault(Exception):
    """A fault on the grading side; the text is the reason (already free of the seed)."""


def grade_episode(result_path, expected):
    seed = None
    try:
        gt = _read_gt(expected)
        seed = str(int(gt["seed"]))
    except Exception as exc:  # noqa: BLE001 - no usable gt.json: the grading side is broken
        return _pending(f"no usable gt.json: {_clip(exc) or type(exc).__name__}", None)
    try:
        return _grade(Path(result_path), gt, seed)
    except _Fault as fault:
        return _pending(str(fault), seed)
    except Exception as exc:  # noqa: BLE001 - never raises: an unexpected failure is a grader fault
        return _pending(_clip(exc, seed) or type(exc).__name__, seed)


def _pending(reason: str, seed: str | None) -> dict:
    reason = _clip(reason, seed)
    return {"score": 0.0, "pending": f"grader fault: {reason}", "note": f"grader fault: {reason}"}


def _read_gt(expected: dict) -> dict:
    oracle_dir = Path(expected["rules"]["oracle_dir"])
    gt = json.loads((oracle_dir / "gt.json").read_text(encoding="utf-8"))
    str(gt["scenario"]), int(gt["seed"])
    return gt


def _grade(result_path: Path, gt: dict, seed: str) -> tuple[float, str]:
    export_dir = result_path.resolve().parent
    scenario = str(gt["scenario"])
    prefix = os.environ.get(CMD_ENV, "").strip()
    container = None
    if prefix:
        cmd = [*shlex.split(prefix), "grade", str(export_dir), "--scenario", scenario, "--seed", seed]
    else:
        container = f"stmbench-grade-{uuid.uuid4().hex[:12]}"
        cmd = ["docker", "run", "--rm", "--name", container, "--network", "none",
               "-v", f"{export_dir}:/export:ro", str(gt["grader_image"]),
               "grade", "/export", "--scenario", scenario, "--seed", seed]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        if container:                       # the timeout kills the docker client, not the container
            try:
                subprocess.run(["docker", "kill", container], capture_output=True, timeout=60)
            except Exception:  # noqa: BLE001 - best effort
                pass
        raise _Fault(f"no result within {TIMEOUT_S // 60} min") from None
    except OSError as exc:
        raise _Fault(f"the grader could not be started: {_clip(exc, seed)}") from None
    if proc.returncode != 0:
        lines = (proc.stderr or proc.stdout).strip().splitlines()
        raise _Fault(f"exit {proc.returncode}: {_clip(lines[-1], seed) if lines else 'no output'}")
    try:
        res = json.loads(proc.stdout)
        score = float(res["score"])
        note = str(res["note"])
    except (ValueError, KeyError, TypeError):
        raise _Fault("the grader's output is not a result") from None
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise _Fault("the grader returned a score outside 0 to 1")
    return score, _scrub(note, seed)
