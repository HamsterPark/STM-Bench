#!/usr/bin/env python3
"""Shared parts of the grader tests of the stm tasks (every family's ``run_tests_<family>.py`` imports this).

A family's runner holds only its table of cases; this module holds the export model, the mutations that damage a
correct export the ways a solver without the measurement could, the in-tolerance shifts, and the loop that grades
every case the way the OSWorld-Science runner does: ``<suite root>/score.py <task> --submission <dir> --json``
(the task's ``module_metric`` and the adapter ``stm_metric.py``). The adapter needs the STM-Bench grader: set
``STMBENCH_GRADER_CMD`` to ``<python 3.13 with the pins> -m stmbench.cli`` or have Docker and the image named in
``gt.json``. Without either the suite says SKIPPED; so it does when the family has no reference solution yet
(no fixtures) -- a skipped suite is not a pass. ``STM_TEST_WORKERS`` sets how many cases are graded at once.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # suite root: tasks/, reference_private/, score.py
FIXTURES = HERE / "fixtures"
SCORE = ROOT / "score.py"
PASS, FAIL = "PASS", "FAIL"


class Export:
    """An export directory as parsed data; every mutation works on a copy."""

    def __init__(self, session: dict, records: list[dict], digests: list[dict], report: dict):
        self.session, self.records, self.digests, self.report = session, records, digests, report

    @classmethod
    def load(cls, d: Path) -> Export:
        def lines(name):
            return [json.loads(x) for x in (d / name).read_text(encoding="utf-8").splitlines() if x.strip()]
        return cls(json.loads((d / "session.json").read_text(encoding="utf-8")), lines("actions.jsonl"),
                   lines("digests.jsonl"), json.loads((d / "report.json").read_text(encoding="utf-8")))

    def copy(self) -> Export:
        return copy.deepcopy(self)

    def renumber(self) -> Export:
        """Close up the record numbers and make session.json count what the log holds."""
        for k, rec in enumerate(self.records):
            rec["i"] = k
        self.session["records"] = len(self.records)
        return self

    def write(self, dest: Path) -> Path:
        """Write ``dest/export/`` readable by every account (the grader image runs as another user)."""
        out = dest / "export"
        out.mkdir(parents=True, exist_ok=True)
        out.chmod(0o755)
        dump = lambda objs: "".join(json.dumps(o, ensure_ascii=False) + "\n" for o in objs)  # noqa: E731
        files = {"session.json": json.dumps(self.session, indent=1), "actions.jsonl": dump(self.records),
                 "digests.jsonl": dump(self.digests), "report.json": json.dumps(self.report, indent=1)}
        for name, text in files.items():
            (out / name).write_text(text, encoding="utf-8")
            (out / name).chmod(0o644)
        dest.chmod(0o755)
        return dest


# ── reports ─────────────────────────────────────────────────────────────────────────────────
def reports(e: Export, claim: str) -> list[dict]:
    return [r for r in e.records if r.get("kind") == "report" and r["args"].get("claim_id") == claim]


def reported(e: Export, claim: str) -> float:
    rs = reports(e, claim)
    assert rs, f"the fixture holds no report of {claim}"
    return float(rs[-1]["args"]["value"])


def edit_reports(e: Export, claim: str, **changes) -> None:
    """Apply ``changes`` to every report of ``claim`` in the log, and to report.json alongside."""
    rs = reports(e, claim)
    assert rs, f"the fixture holds no report of {claim}"
    for r in rs:
        r["args"].update(changes)
    for r in e.report.get("reports", []):
        if r.get("args", {}).get("claim_id") == claim:
            r["args"].update(changes)
    if claim in e.report.get("claims", {}):
        e.report["claims"][claim].update(changes)


def shifted(e: Export, claim: str, delta: float) -> Export:
    """The value reported for ``claim`` moved by ``delta`` (in its own unit)."""
    m = e.copy()
    edit_reports(m, claim, value=reported(m, claim) + delta)
    return m


def scaled(e: Export, claim: str, factor: float) -> Export:
    """The value reported for ``claim`` multiplied by ``factor``."""
    m = e.copy()
    edit_reports(m, claim, value=reported(m, claim) * factor)
    return m


def swapped(e: Export, a: str, b: str) -> Export:
    m = e.copy()
    va, vb = reported(m, a), reported(m, b)
    edit_reports(m, a, value=vb)
    edit_reports(m, b, value=va)
    return m


def set_evidence(e: Export, claims, evidence: list[str], **extra) -> Export:
    m = e.copy()
    for c in claims:
        edit_reports(m, c, evidence=list(evidence), **extra)
    return m


def evidence_of(e: Export, claim: str) -> list[str]:
    return list(reports(e, claim)[-1]["args"].get("evidence") or [])


def act_results(e: Export, name: str, key: str) -> list[str]:
    """The ``key`` of the reply of every act ``name``, in order (a sweep's reply names its file)."""
    return [r["result"][key] for r in e.records
            if r.get("kind") == "act" and r.get("name") == name and isinstance(r.get("result"), dict)
            and r["result"].get(key)]


def spectrum_names(e: Export) -> list[str]:
    return act_results(e, "sts", "spectrum")


def frame_names(e: Export) -> list[str]:
    return act_results(e, "scan", "frame")


# ── damaging a log ──────────────────────────────────────────────────────────────────────────
def cut_in_half(e: Export) -> Export:
    """The log's first half; session.json still counts the whole episode."""
    m = e.copy()
    m.records = m.records[:len(m.records) // 2]
    return m


def cut_before_report(e: Export) -> Export:
    """Everything up to the first report kept, the count corrected: the solver stops before it reports."""
    m = e.copy()
    k = next(i for i, r in enumerate(m.records) if r.get("kind") == "report")
    m.records = m.records[:k]
    m.session["finished"] = False
    m.report = {"claims": {}}
    return m.renumber()


def cut_before_first(e: Export, act_name: str) -> Export:
    """Everything from the first act ``act_name`` on gone, the count corrected, nothing reported."""
    m = e.copy()
    k = next(i for i, r in enumerate(m.records) if r.get("kind") == "act" and r.get("name") == act_name)
    m.digests = [d for d in m.digests if d.get("i", 0) < k]
    m.records = m.records[:k]
    m.session["finished"] = False
    m.report = {"claims": {}}
    return m.renumber()


def unfinished(e: Export) -> Export:
    """The episode ends without ``finish`` (the machine was stopped): judged as it stands."""
    m = e.copy()
    assert m.records[-1]["kind"] == "finish"
    m.records = m.records[:-1]
    m.session["finished"] = False
    return m.renumber()


def middle_act(e: Export) -> int:
    """Index of a move of the tip near the middle of the log (any act if the log holds no tip move)."""
    acts = [i for i, r in enumerate(e.records) if r.get("kind") == "act" and r.get("name") == "move_tip"]
    acts = acts or [i for i, r in enumerate(e.records) if r.get("kind") == "act"]
    assert acts, "the fixture holds no act"
    return acts[len(acts) // 2]


def drop_middle(e: Export, close_up: bool) -> Export:
    m = e.copy()
    del m.records[middle_act(m)]
    return m.renumber() if close_up else m


def alter_call(e: Export) -> Export:
    """One act changed (a tip move aimed 0.5 nm elsewhere), the rest of the log and the digests untouched."""
    m = e.copy()
    rec = m.records[middle_act(m)]
    p = rec["params"]
    if "x_nm" in p:
        p["x_nm"] = float(p["x_nm"]) + 0.5
    else:
        key = next(k for k, v in p.items() if isinstance(v, (int, float)) and not isinstance(v, bool))
        p[key] = float(p[key]) * 1.1 + 0.1
    return m


def flip_hash(e: Export) -> Export:
    """One character of one digest hash changed, its fingerprint left alone (rounding between machines)."""
    m = e.copy()
    line = next(d for d in m.digests if isinstance(d.get("digest"), dict))
    key = sorted(line["digest"])[0]
    h = line["digest"][key]
    line["digest"][key] = h[:-1] + ("0" if h[-1] != "0" else "1")
    return m


def tamper_digest(e: Export) -> Export:
    """A digest line that describes another measurement: hash changed, fingerprint mean moved by a tenth of range."""
    m = flip_hash(e)
    line = next(d for d in m.digests if isinstance(d.get("digest"), dict))
    key = sorted(line["digest"])[0]
    fp = line["fp"][key]
    fp["mean"] = float(fp["mean"]) + 0.1 * (abs(float(fp["max"])) or 1.0)
    return m


def drop_digests(e: Export) -> Export:
    m = e.copy()
    m.digests = []
    return m


def empty_episode(e: Export) -> Export:
    m = e.copy()
    m.records, m.digests, m.report = [], [], {"claims": {}}
    m.session["finished"] = False
    return m.renumber()


def report_only(e: Export) -> Export:
    """The correct reports and no instrument calls before them: the numbers without the measurement."""
    m = e.copy()
    m.records = [r for r in m.records if r.get("kind") in ("report", "finish")]
    m.digests = []
    return m.renumber()


# ── tolerances and the reference's own error ────────────────────────────────────────────────
def rule_card(task: str) -> str:
    return json.loads((ROOT / "tasks" / f"{task}.json").read_text(encoding="utf-8")).get("instruction", "")


def card_abs_tol(task: str, claim: str, unit: str, default: float) -> float:
    """The absolute tolerance the rule card states for ``claim`` ("differs by at most N <unit>")."""
    m = re.search(rf"{claim}:.*?differs by at most ([0-9.]+) {unit}", rule_card(task), re.S)
    return float(m.group(1)) if m else default


def card_rel_tol(task: str, claim: str, default: float) -> float:
    """The relative tolerance the rule card states for ``claim`` ("the relative error is at most N%"), as a fraction."""
    m = re.search(rf"{claim}:.*?relative error is at most ([0-9.]+)\s*%", rule_card(task), re.S)
    return float(m.group(1)) / 100.0 if m else default


def reference_error(task: str, claim: str) -> tuple[str, float]:
    """The reference solution's own error for ``claim`` when the fixture was made (``reference_run.json``):
    ("abs", signed absolute error) or ("rel", signed relative error); ("abs", 0.0) if unknown. An in-tolerance variant
    must stay inside, and a world whose reference value already sits near the edge allows a shift one way only."""
    p = FIXTURES / task / "reference_run.json"
    if p.is_file():
        run = json.loads(p.read_text(encoding="utf-8"))
        for key, kind in (("abs_err", "abs"), ("rel_err", "rel")):
            v = (run.get(key) or {}).get(claim)
            if v is not None:
                return kind, float(v)
    return "abs", 0.0


def inside(task: str, claim: str, kind: str, shift: float, tol: float) -> bool:
    """Whether moving ``claim``'s value by ``shift`` (absolute, or a relative factor minus one) leaves the error
    inside ``tol`` with a margin of 5 %."""
    k, err = reference_error(task, claim)
    if k != kind:
        return k == "abs" and err == 0.0
    total = err + shift if kind == "abs" else (1 + err) * (1 + shift) - 1
    return abs(total) <= 0.95 * tol


def abs_shift_cases(e: Export, task: str, claims, tol: float, label: str, together: bool = True) -> list:
    """PASS variants: ``claims`` moved by half of ``tol`` (absolute), in either direction where the error allows.
    ``together`` moves all the claims at once (two peaks), else one case per claim."""
    out = []
    groups = [tuple(claims)] if together else [(c,) for c in claims]
    for g in groups:
        for sign in (+1, -1):
            d = sign * tol / 2
            if all(inside(task, c, "abs", d, tol) for c in g):
                m = e
                for c in g:
                    m = shifted(m, c, d)
                out.append((f"{'/'.join(g)} moved by {d:+g} {label} (half of the {tol:g} allowed)", PASS, m))
    return out


def rel_shift_cases(e: Export, task: str, tols: dict[str, float]) -> list:
    """PASS variants: each claim's value scaled by 1 +/- tol/2, one case per claim and direction."""
    out = []
    for c, tol in tols.items():
        for sign in (+1, -1):
            d = sign * tol / 2
            if inside(task, c, "rel", d, tol):
                out.append((f"{c} moved by {d:+.1%} (half of the {tol:.0%} allowed)", PASS, scaled(e, c, 1 + d)))
    return out


# ── running the grader ──────────────────────────────────────────────────────────────────────
def grade(task: str, sub: Path) -> dict:
    r = subprocess.run([sys.executable, str(SCORE), task, "--submission", str(sub), "--json"],
                       capture_output=True, text=True, timeout=7200)
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"verdict": None, "score": None, "error": (r.stdout + r.stderr)[-400:]}


def note_of(res: dict) -> str:
    for p in res.get("pending") or []:
        return "PENDING " + " ".join(str(p.get("reason")).split())[:200]
    for c in res.get("checks") or []:
        d = c.get("detail") or {}
        if d.get("note") or d.get("error"):
            return " ".join(str(d.get("note") or d.get("error")).split())[:160]
    return res.get("error", "")


def adapter(task: str):
    path = ROOT / "reference_private" / task / "stm_metric.py"
    spec = importlib.util.spec_from_file_location(f"stm_metric_{task}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gt_of(task: str) -> dict:
    return json.loads((ROOT / "reference_private" / task / "gt.json").read_text(encoding="utf-8"))


def grade_with_gt(task: str, sub: Path, gt: dict) -> dict:
    """The adapter on a ground truth of our choosing (the task's own is never changed)."""
    with tempfile.TemporaryDirectory() as oracle:
        (Path(oracle) / "gt.json").write_text(json.dumps(gt), encoding="utf-8")
        score, note = adapter(task).grade_episode(
            str(sub / "export" / "actions.jsonl"), {"type": "rule", "rules": {"oracle_dir": oracle}})
    return {"verdict": PASS if score >= 1.0 - 1e-9 else FAIL, "score": score,
            "checks": [{"detail": {"note": note}}]}


def common_cases(task: str, tasks: list[str], fx: dict[str, Export]) -> list:
    """The cases every family has: the correct episode and the damaged logs. ``(name, want, Export | callable)``;
    a callable gets the submission directory (of the correct episode) and returns a result dict."""
    e = fx[task]
    other = tasks[(tasks.index(task) + 1) % len(tasks)]
    gt, gt_other = gt_of(task), gt_of(other)
    return [
        ("correct episode", PASS, e),
        ("episode never reached finish", PASS, unfinished(e)),
        ("digest hash differs in its last bits, fingerprint equal (rounding between machines)", PASS, flip_hash(e)),
        (f"export of another task ({other})", FAIL, fx[other]),
        ("another task's seed in the ground truth", FAIL,
         lambda sub: grade_with_gt(task, sub, {**gt, "seed": gt_other["seed"]})),
        ("log cut to its first half", FAIL, cut_in_half(e)),
        ("stopped before the report, count corrected", FAIL, cut_before_report(e)),
        ("record removed from the middle, numbers closed up", FAIL, drop_middle(e, True)),
        ("record removed from the middle, numbers left", FAIL, drop_middle(e, False)),
        ("one call altered", FAIL, alter_call(e)),
        ("digest line describing another measurement", FAIL, tamper_digest(e)),
        ("digest lines removed", FAIL, drop_digests(e)),
        ("empty episode", FAIL, empty_episode(e)),
        ("report with no instrument calls", FAIL, report_only(e)),
    ]


def run_suite(title: str, task_glob: str, cases, reference: bool = True) -> int:
    """Grade every case of every task matching ``task_glob``. ``cases(task, tasks, fx)`` is the family's table.
    ``reference=False`` marks a family whose reference solution does not exist yet: the suite is SKIPPED."""
    if not reference:
        print(f"SKIPPED: {title}: no reference solution yet, so there are no fixtures and these tests have "
              "not been run")
        return 0
    if not os.environ.get("STMBENCH_GRADER_CMD") and not shutil.which("docker"):
        print("SKIPPED: neither STMBENCH_GRADER_CMD nor docker is available, so the STM-Bench grader cannot be run")
        return 0
    tasks = sorted(p.stem for p in (ROOT / "tasks").glob(f"{task_glob}.json"))
    if not tasks:
        print(f"SKIPPED: no {title} tasks under tasks/")
        return 0
    absent = [t for t in tasks if not (FIXTURES / t / "export" / "actions.jsonl").is_file()]
    if absent:
        print(f"SKIPPED: no fixtures for {len(absent)} task(s) (reference_export.py makes them)")
        return 0
    fx = {t: Export.load(FIXTURES / t / "export") for t in tasks}
    work = Path(tempfile.mkdtemp(prefix="stm_grader_tests_"))
    jobs = []
    for t in tasks:
        for k, (name, want, what) in enumerate(cases(t, tasks, fx)):
            sub = work / t / f"{k:02d}"
            (what if isinstance(what, Export) else fx[t]).write(sub)   # a callable gets the correct episode
            jobs.append((t, name, want, what, sub))

    def run(job):
        t, name, want, what, sub = job
        return job, (what(sub) if callable(what) else grade(t, sub))

    bad = 0
    with ThreadPoolExecutor(max_workers=int(os.environ.get("STM_TEST_WORKERS", "4"))) as pool:
        for (t, name, want, _w, _s), res in pool.map(run, jobs):
            verdict, score = res.get("verdict"), res.get("score")
            ok = verdict == want and (score is not None and (score >= 1.0 - 1e-9 if want == PASS else True))
            bad += not ok
            shown = "n/a" if score is None else f"{score:.3f}"
            print(f"{'ok  ' if ok else 'BAD '} {t} {name}: want {want}, got {verdict} (score {shown})"
                  + ("" if ok else f"  [{note_of(res)}]"), flush=True)
    shutil.rmtree(work, ignore_errors=True)
    print(f"\n{len(jobs)} cases over {len(tasks)} tasks, {bad} unexpected")
    if bad:
        return 1
    print("ALL BEHAVED AS INTENDED")
    return 0
