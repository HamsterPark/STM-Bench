"""OSWorld-Science ``module_metric`` adapter for STM-Bench.

The harness imports this file by path and calls :func:`grade_episode` as
``fn(result_path, {"type": "rule", "rules": {"oracle_dir": <dir>}})``. ``result_path`` is the
collected ``actions.jsonl``; the rest of the export (``session.json``, ``digests.jsonl``,
``report.json``) was collected beside it. ``oracle_dir`` holds ``gt.json``::

    {"scenario": "P4_atom_positioning_cu111", "seed": 0, "grader_image": "stmbench-grader:<tag>"}

The harness runs Python 3.11 with its own dependencies; STM-Bench needs Python 3.13 and numpy 2,
so this module only runs the real grader (``python -m stmbench.cli grade``) as a separate
process and reads its JSON: inside the Docker image ``grader_image`` names (no network, the
export mounted read-only), or, when ``STMBENCH_GRADER_CMD`` is set, through that command prefix
(``<python> -m stmbench.cli``, split like a shell would). Standard library only. It never
raises: a grade that could not be made is ``(0.0, "grader failed: <why>")``.
"""
from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import uuid
from pathlib import Path

CMD_ENV = "STMBENCH_GRADER_CMD"
#: a replay of a long episode takes minutes; a hung one must not hold the harness for ever
TIMEOUT_S = 30 * 60
_CLIP = 300


def _clip(x) -> str:
    s = " ".join(str(x).split())
    return s if len(s) <= _CLIP else s[:_CLIP - 3] + "..."


def grade_episode(result_path, expected) -> tuple[float, str]:
    try:
        return _grade(Path(result_path), expected)
    except Exception as exc:  # noqa: BLE001 - a failed grade is a score of 0 with its reason
        return 0.0, f"grader failed: {_clip(exc) or type(exc).__name__}"


def _grade(result_path: Path, expected: dict) -> tuple[float, str]:
    export_dir = result_path.resolve().parent
    oracle_dir = Path(expected["rules"]["oracle_dir"])
    gt = json.loads((oracle_dir / "gt.json").read_text(encoding="utf-8"))
    scenario, seed = str(gt["scenario"]), str(int(gt["seed"]))
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
        return 0.0, f"grader failed: no result within {TIMEOUT_S // 60} min"
    if proc.returncode != 0:
        lines = (proc.stderr or proc.stdout).strip().splitlines()
        return 0.0, f"grader failed: exit {proc.returncode}: {_clip(lines[-1]) if lines else 'no output'}"
    res = json.loads(proc.stdout)
    score = float(res["score"])
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"the grader returned the score {score}")
    return score, str(res["note"])
