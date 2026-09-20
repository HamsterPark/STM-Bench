"""``python -m stmbench.trackA.cli {build|render|score|paired}``

  build   → delegates to ``stmbench.trackA.manifests`` (T1–T4 manifests under ``$STM_BENCH_DATA``;
            frame references are written relative to ``$STM_BENCH_RAW`` / ``--raw-root``)
  render  → PNGs + a ``prompts.jsonl`` (system/user text, image files, schema) for one task's manifest;
            nothing is sent anywhere — a model runner consumes the JSONL and writes an answers parquet.
            T1: the earlier frames of the run are rendered in full, the judged frame (the last
            input) only down to its ``head_rows`` — its outcome is the label.
  score   → metrics for ``--baseline <name>`` or ``--answers <parquet>`` (columns sample_id, answer JSON).
            T3 is three-class (stay / relocate / long_stop) by default; ``--t3-binary`` folds
            relocate + long_stop into relocate on both the label and the answer.
  paired  → the guide's 69–78 % figure from a step table (e.g. the read-only ``glance_drift.csv``)

No LLM is called from here. Data root = ``$STM_BENCH_DATA``; relative frame references resolve
under ``$STM_BENCH_RAW`` (see ``stmbench.paths``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .render import data_root


# ── tables ──────────────────────────────────────────────────────────────────

def _pd():
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise SystemExit("pandas + pyarrow are required for Track A tables: pip install 'stm-bench[bench]'") from e
    return pd


def load_table(path: str | Path):
    pd = _pd()
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"table not found: {p}")
    suf = p.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(p)
    if suf == ".csv":
        return pd.read_csv(p)
    if suf in (".jsonl", ".ndjson"):
        return pd.read_json(p, lines=True)
    if suf == ".json":
        return pd.read_json(p)
    raise SystemExit(f"unknown table format {suf!r}: {p}")


def save_table(rows: Any, path: str | Path) -> Path:
    pd = _pd()
    df = rows if hasattr(rows, "to_parquet") else pd.DataFrame(list(rows))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".csv":
        df.to_csv(p, index=False)
    else:
        df.to_parquet(p, index=False)
    return p


def default_manifest(task: str) -> Path:
    root = data_root() / "trackA"
    for cand in (root / "manifests" / f"{task}.parquet", root / f"{task}.parquet",
                 root / "manifests" / f"{task.lower()}.parquet", root / f"{task.lower()}.parquet"):
        if cand.exists():
            return cand
    return root / "manifests" / f"{task}.parquet"


def _dump(obj: Any, out: str | None) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    print(text)


def _task_kw(a) -> dict:
    """Scorer / baseline keywords that depend on the task (only T3 has the binary fold)."""
    return {"t3_binary": bool(getattr(a, "t3_binary", False))} if a.task.upper() == "T3" else {}


# ── commands ────────────────────────────────────────────────────────────────

def cmd_build(a) -> int:
    try:
        from . import manifests  # written by the manifest agent
    except ImportError as e:
        raise SystemExit(f"stmbench.trackA.manifests is not importable ({e}); nothing to build") from e
    rest = list(a.rest or [])
    if hasattr(manifests, "main"):
        rc = manifests.main(rest)
        return int(rc or 0)
    for name in ("build_all", "build_manifests", "build"):
        fn = getattr(manifests, name, None)
        if callable(fn):
            res = fn(*rest) if rest else fn()
            print(res if res is not None else f"{name}() done")
            return 0
    raise SystemExit("stmbench.trackA.manifests exposes neither main() nor build_all()/build()")


def cmd_render(a) -> int:
    from . import schema as SCHEMA
    from .baselines import as_paths
    from .prompts import build_prompt
    from .render import render_frame, render_settings_for_task
    from .score import normalize_manifest

    task = a.task.upper()
    df = load_table(a.manifest or default_manifest(task))
    rows = [r for r in normalize_manifest(df, task) if str(r.get("task", task)).upper() == task]
    if a.split:
        rows = [r for r in rows if str(r.get("split", "")) == a.split]
    if a.limit:
        rows = rows[:a.limit]
    out = Path(a.out or (data_root() / "trackA" / "render" / task))
    (out / "png").mkdir(parents=True, exist_ok=True)
    settings = render_settings_for_task(task)
    n_ok = n_fail = 0
    with open(out / "prompts.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            sid = row["sample_id"]
            paths = as_paths(row.get("input_paths"))       # relative entries resolve under the raw mirror root
            # T1: the judged frame (the last input) is shown only down to its head rows — the rest IS
            # the label (schema.T1: ``stopped`` = this frame). Every earlier frame has already ended
            # and is rendered in full, acquired rows and all.
            hr = None
            if task == "T1":
                v = row.get("head_rows")
                hr = int(v) if v is not None and v == v else SCHEMA.T1_HEAD_ROWS
            try:
                renders = [render_frame(p, **settings) for p in paths[:-1]]
                if paths:
                    renders.append(render_frame(paths[-1], head_rows=hr, **settings))
            except Exception as e:  # noqa: BLE001 — one unreadable file must not kill the batch
                n_fail += 1
                fh.write(json.dumps({"sample_id": sid, "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False) + "\n")
                continue
            prompt = build_prompt(task, row, renders)
            files = []
            for i, r in enumerate(renders):
                f = out / "png" / f"{sid}_{i:02d}.png"
                f.write_bytes(r.png)
                files.append(str(f.relative_to(out)))
            rec = {"sample_id": sid, "task": task, "system": prompt["system"], "user": prompt["user"],
                   "images": files, "frames": [r.as_dict() for r in renders], "schema": prompt["schema"],
                   "input_paths": paths, "head_rows": hr}
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            n_ok += 1
    print(f"rendered {n_ok} samples ({n_fail} failed) → {out}")
    return 0 if n_ok else 1


def cmd_score(a) -> int:
    from . import score as S
    from .baselines import NEGATIVE_CONTROLS, run_baseline, step_table_from_manifest, paired_final_more_stable

    task = a.task.upper()
    kw = _task_kw(a)
    df = load_table(a.manifest or default_manifest(task))
    if a.baseline:
        answers = run_baseline(a.baseline, df, task=task, split=a.split, seed=a.seed, **kw)
        source = {"baseline": a.baseline, "negative_control": a.baseline in NEGATIVE_CONTROLS}
        if a.save_answers:
            save_table(answers, a.save_answers)
    elif a.answers:
        answers = load_table(a.answers)
        source = {"answers": str(a.answers)}
    else:
        raise SystemExit("score needs --baseline <name> or --answers <table>")
    result: dict[str, Any] = {"source": source, "manifest": str(a.manifest or default_manifest(task)), "split": a.split}
    if task == "T3":
        result["label_mode"] = "binary" if kw.get("t3_binary") else "three_class"
    result["overall"] = S.score(task, df, answers, split=a.split, **kw)
    if a.by:
        result["by"] = {col: S.score_by(task, df, answers, col, split=a.split, **kw) for col in a.by}
    if task == "T1" and a.paired:
        rows = [r for r in df.to_dict("records") if not a.split or str(r.get("split", "")) == a.split]
        result["paired_final_more_stable"] = paired_final_more_stable(step_table_from_manifest(rows))
    _dump(result, a.out)
    return 0


def cmd_paired(a) -> int:
    from .baselines import paired_final_more_stable
    df = load_table(a.csv)
    res = paired_final_more_stable(df)
    _dump({"source": str(a.csv), "n_steps": int(len(df)), "guide_reference": "69–78 % (data guide §1, README §4.5)",
           "paired_final_more_stable": res}, a.out)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="stmbench.trackA", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build T1–T4 manifests (delegates to stmbench.trackA.manifests)")
    b.add_argument("rest", nargs=argparse.REMAINDER, help="arguments passed through to manifests.main")
    b.set_defaults(fn=cmd_build)

    r = sub.add_parser("render", help="render PNGs + prompts.jsonl for a task")
    r.add_argument("--task", required=True, choices=["T1", "T2", "T3", "T4", "t1", "t2", "t3", "t4"])
    r.add_argument("--manifest", default=None)
    r.add_argument("--split", default="test")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--out", default=None)
    r.set_defaults(fn=cmd_render)

    s = sub.add_parser("score", help="score a baseline or an answers table")
    s.add_argument("--task", required=True, choices=["T1", "T2", "T3", "T4", "t1", "t2", "t3", "t4"])
    s.add_argument("--manifest", default=None)
    s.add_argument("--baseline", default=None,
                   help="e.g. drift_rule, step_order / pixel_head (negative controls), majority, center, random, "
                        "flattest, material_median")
    s.add_argument("--answers", default=None, help="table with columns sample_id, answer (JSON)")
    s.add_argument("--split", default="test")
    s.add_argument("--by", nargs="*", default=None, help="strat columns to report separately (era, genre, instrument …)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--paired", action="store_true", help="T1: also report the paired final-step-more-stable fraction")
    s.add_argument("--t3-binary", action="store_true",
                   help="T3: fold relocate + long_stop into relocate (label and answer) instead of three classes")
    s.add_argument("--save-answers", default=None)
    s.add_argument("--out", default=None, help="write the JSON report here too")
    s.set_defaults(fn=cmd_score)

    p = sub.add_parser("paired", help="69–78 %% check on a step table (ep, is_final, dz_pm, dx_nm, corr)")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_paired)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
