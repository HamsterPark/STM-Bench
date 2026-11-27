#!/usr/bin/env python3
"""Build the dataset side of the stm domain for the internal evaluation: 7 families x 3 worlds = 21 tasks.

    python build_tree.py --worlds <worlds.json> --out <root> --code <STM-Bench checkout> \
        --adapter <stm_metric.py> --grader-image stmbench-grader:<12 hex> [--fixtures <dir>] \
        [--max-steps p7=400 ...] [--families p3,p3r,p4] [--no-lockdown]

``worlds.json`` is PRIVATE (it holds seeds): ``{"tasks": {<task id>: {"scenario", "seed", "note"}}}``, where ``note``
is free text for the maintainers (the starting tip class and what the world was checked for). The repository holds
only this code and its templates. Writes ``<root>/stm/`` (tasks/, private/reference_private/<id>/,
private/grader_tests/) and ``<root>/MANIFEST.stm-fragment.json``. The instruction of every task is generated:
FRAME + the rule-card code (``stmbench.trackB.rules.task_text``) + the one sentence about the data copies. It needs
Python 3.13 and the package (``--code``). The seeds go only into ``gt.json``, written by each task's ``make_gt.py``;
nothing is printed that holds one. An existing ``<root>/stm`` is replaced.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import families as F  # noqa: E402
from make_fragment import manifest  # noqa: E402

TREE_SRC = HERE / "tree_src"
TESTS_SRC = TREE_SRC / "grader_tests"


def load_scenario(code: Path, scenario: str):
    sys.path.insert(0, str(code.resolve()))
    from stmsim.scenario import Scenario
    from stmbench.instrument.core import SCENARIO_DIR

    return Scenario.load(SCENARIO_DIR / f"{scenario}.yaml")


def instruction(code: Path, scenario: str) -> str:
    sc = load_scenario(code, scenario)
    from stmbench.trackB.rules import task_text

    return F.FRAME + task_text(sc, "en").strip() + "\n\n" + F.DATA_SENTENCE + "\n"


def _tol(c: dict) -> str:
    t = c.get("tol") or {}
    parts = []
    if "abs" in t:
        parts.append(f"within {t['abs']:g} {c.get('unit', '')}" + (f" (modulo {t['mod']:g})" if "mod" in t else ""))
    if "rel" in t:
        parts.append(f"relative error at most {t['rel']:.0%}")
    if "report_nm" in t:
        parts.append(f"reported place within {t['report_nm']:g} nm")
    for k in ("min", "max"):
        if k in c:
            parts.append(f"{k} {c[k]:g} {c.get('unit', '')}")
    return "; ".join(parts) or "see the rule card"


def _evidence(ev: dict) -> str:
    if ev.get("kind") == "any_of":
        return "one of: " + " | ".join(_evidence(r) for r in ev["rules"])
    keys = {k: v for k, v in ev.items() if k != "kind"}
    return f"{ev.get('kind')}" + (f" {keys}" if keys else "")


def readme(task_id: str, fam: dict, code: Path, note: str) -> str:
    """The private README of one task: the family's claims from its scenario YAML, the grading, and this world's note."""
    sc = load_scenario(code, fam["scenario"])
    claims = "\n".join(f"- `{c['id']}` ({c.get('unit', '')}): {c.get('say_en') or c.get('say')}\n"
                       f"  - judged: {_tol(c)}\n  - acquisition evidence: {_evidence(c.get('evidence') or {})}"
                       for c in sc.claims)
    steps = fam["max_steps"]
    return f"""# {task_id}: grader notes (private)

Scenario `{fam['scenario']}` ({fam['paper']}). Three worlds of this family exist (`{fam['prefix']}_01` to `03`); they
share one instruction and differ in the hidden world, each with its own seed in `gt.json`.

## What the score means

The score is the fraction of the scenario's claims that the grader verified (1.0 = reproduced); `pass_score` is 1.0.
A reported value without the acquisition evidence the rule card requires does not count. The claims:

{claims}

An over-budget episode (more than 1.05 x the instrument-time budget, counted in simulated time) cannot pass. The full
rules are the rule card at the end of the task instruction, generated from the scenario file.

## How it is graded

The console in the guest writes an export while the episode happens (`session.json`, `actions.jsonl`, `digests.jsonl`,
`report.json`). The adapter `stm_metric.py` runs STM-Bench's grader on it (`python -m stmbench.cli grade`), which
**replays the recorded calls** on its own instrument with the scenario and seed of `gt.json`; the replay must equal the
recorded episode (every reply, instrument time, and spectrum or frame hash, within rounding between machines),
else the episode scores 0. A fault on the grading side is a pending result, not a zero. Acquisition evidence shows
that the required measurements occurred; it does not prove that the reported number was computed from them.

## Files

- `gt.json`: `{{"scenario", "seed", "grader_image"}}`, written by `make_gt.py` from the private worlds file.
- `stm_metric.py`: the adapter, a copy of `deploy/osworld/stm_metric.py`.
- `../../grader_tests/{fam['tests']}`: the grader tests of this family; `fixtures/` holds the exports of the
  privileged reference solution that they damage in known ways (see `reference_export.py`).

## Choices a reviewer may want to change

- `budget.max_steps` is {steps}, provisional until a blind trial fixes it.
- The first two `config` steps close `/opt/stm-bench` to the agent's account and revoke its admin rights.
- `private/` holds `gt.json` and therefore each task's seed: it must not be published.

## This world (private note)

{note.strip() or '(none)'}
"""


def task_json(tid: str, fam: dict, text: str, max_steps: int, lockdown: bool, ref_run: dict | None) -> dict:
    rh, fn = fam["audits"]()
    if ref_run:
        fn = (fn + f" The privileged reference solution reproduced this world "
              f"(score 1.0, about {round(ref_run['instrument_s'] / 60)} min of instrument time).")
    sudo = lambda cmd: {"type": "execute", "parameters": {"command": ["bash", "-c", f"echo password | sudo -S -p '' {cmd}"]}}  # noqa: E731
    config = ([sudo("sh -c 'chgrp -R stm /opt/stm-bench && chmod -R o-rwx /opt/stm-bench'"),
               sudo("/usr/local/sbin/stm-lockdown")] if lockdown else []) + [
        {"type": "execute", "parameters": {"command": ["/usr/local/bin/stm-start", tid]}},
        {"type": "sleep", "parameters": {"seconds": 3}}]
    return {
        "id": tid, "snapshot": F.SNAPSHOT, "instruction": text, "source": fam["source"],
        "budget": {"max_steps": max_steps}, "config": config, "related_apps": ["stm-console"], "deliverables": [],
        "evaluator": {
            "conj": "weighted", "weights": [1.0], "pass_score": 1.0, "func": ["module_metric"],
            "labels": ["[stm] replayed episode: claims of the paper verified"],
            "params": [{"module": f"reference_private/{tid}/stm_metric.py", "func": "grade_episode",
                        "file": "export/actions.jsonl", "oracle_dir": f"reference_private/{tid}"}],
            "result": [{"type": "vm_file", "path": f"{F.EXPORT_DIR}/{n}", "dest": f"export/{n}"}
                       for n in F.EXPORT_FILES],
        },
        "bench": {
            "task_version": F.TASK_VERSION, "family": fam["family"], "paper": fam["paper"],
            "language_requirement": "English only: the program, the instruction and the rule card are English; "
                                    "no free text is graded.",
            "reward_hacking_audit": rh, "false_negative_audit": fn.strip(),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worlds", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--code", required=True, type=Path)
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--grader-image", required=True)
    ap.add_argument("--fixtures", type=Path, help="reference exports (reference_export.py --out)")
    ap.add_argument("--max-steps", nargs="*", default=[], metavar="FAMILY=N", help="override a family's step budget")
    ap.add_argument("--families", help="families to build: p1,p2,p3,p3_repair (or p3r),p4,p5,p7 (default: all seven)")
    ap.add_argument("--no-lockdown", action="store_true")
    a = ap.parse_args()

    table = json.loads(a.worlds.read_text(encoding="utf-8"))
    table = table.get("tasks", table)
    try:
        keys = F.parse_families(a.families)
    except ValueError as e:
        print(f"build_tree: {e}", file=sys.stderr)
        return 1
    steps = {k: F.FAMILIES[k]["max_steps"] for k in F.FAMILIES}
    for kv in a.max_steps:
        k, _, n = kv.partition("=")
        if k not in steps or not n.isdigit():
            print(f"build_tree: bad --max-steps {kv!r}", file=sys.stderr)
            return 1
        steps[k] = int(n)
    want = [t for t in F.task_ids() if F.family_of(t) in keys]
    # a selection may be built from a larger worlds file (the other families' entries are ignored); without a
    # selection the file must hold exactly the 21 tasks
    extra = sorted(set(table) - set(want))
    if set(want) - set(table) or (extra and not a.families):
        print(f"build_tree: {a.worlds.name} must hold the {len(want)} task ids of the chosen families; "
              f"missing {sorted(set(want) - set(table))}, extra {extra if not a.families else []}", file=sys.stderr)
        return 1
    table = {t: table[t] for t in want}
    for tid in want:
        if table[tid].get("scenario") != F.FAMILIES[F.family_of(tid)]["scenario"]:
            print(f"build_tree: task {tid} has the wrong scenario", file=sys.stderr)
            return 1
    seeds = [int(table[t]["seed"]) for t in want]
    if len(set(seeds)) != len(seeds):
        print("build_tree: two tasks share a seed", file=sys.stderr)
        return 1
    texts = {k: instruction(a.code, F.FAMILIES[k]["scenario"]) for k in keys}
    for k, text in texts.items():
        if re.search(r"https?://|www\.", text):
            print("build_tree: a URL in the instruction", file=sys.stderr)
            return 1
        for s in (x for x in seeds if x >= 1000):      # a development seed below 1000 is a number the text may hold
            if re.search(rf"(?<!\d){s}(?!\d)", text):
                print(f"build_tree: a seed in the instruction of {k}", file=sys.stderr)
                return 1

    dom = a.out / "stm"
    if dom.exists():
        shutil.rmtree(dom)
    (dom / "tasks").mkdir(parents=True)
    (dom / "public" / "assets").mkdir(parents=True)     # the runner treats a domain as present only with public/ and private/
    (dom / "public" / "assets" / "README.md").write_text(
        "# stm assets\n\nThe stm tasks upload nothing into the guest: the console, its code and the instrument\n"
        f"simulation are part of the `{F.SNAPSHOT}` snapshot, and the task's world is made inside the guest from\n"
        "data the guest's agent account cannot read. This directory exists so that the domain has a public tree.\n",
        encoding="utf-8")
    tests = dom / "private" / "grader_tests"
    tests.mkdir(parents=True)
    for name in ["stm_cases.py", "reference_export.py", *(F.FAMILIES[k]["tests"] for k in keys)]:
        shutil.copy(TESTS_SRC / name, tests / name)
    for tid in want:
        key = F.family_of(tid)
        fam = F.FAMILIES[key]
        ref_run = None
        rr = a.fixtures / tid / "reference_run.json" if a.fixtures else None
        if rr and rr.is_file():
            ref_run = json.loads(rr.read_text(encoding="utf-8"))
        (dom / "tasks" / f"{tid}.json").write_text(
            json.dumps(task_json(tid, fam, texts[key], steps[key], not a.no_lockdown, ref_run), indent=2,
                       ensure_ascii=False) + "\n", encoding="utf-8")
        d = dom / "private" / "reference_private" / tid
        d.mkdir(parents=True)
        shutil.copy(TREE_SRC / "make_gt.py", d / "make_gt.py")
        shutil.copy(a.adapter, d / "stm_metric.py")
        (d / "README.md").write_text(readme(tid, fam, a.code, str(table[tid].get("note", ""))), encoding="utf-8")
        r = subprocess.run([sys.executable, str(d / "make_gt.py"), "--seeds", str(a.worlds),
                            "--grader-image", a.grader_image, "--task-id", tid], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr, file=sys.stderr)
            return 1
        if a.fixtures and (a.fixtures / tid / "export").is_dir():
            shutil.copytree(a.fixtures / tid / "export", tests / "fixtures" / tid / "export")
            for sub in ("variants",):
                if (a.fixtures / tid / sub).is_dir():
                    shutil.copytree(a.fixtures / tid / sub, tests / "fixtures" / tid / sub)
            if rr and rr.is_file():
                shutil.copy(rr, tests / "fixtures" / tid)

    m = manifest(a.out, keys)
    (a.out / "MANIFEST.stm-fragment.json").write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
    print(f"built {dom}: {len(want)} tasks, {len(m)} files, max_steps {steps}, "
          + ", ".join(f"{k} instruction {len(v)} characters" for k, v in sorted(texts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
