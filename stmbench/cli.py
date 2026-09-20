"""stmbench command line.

    python -m stmbench.cli run  --scenario <yaml|id prefix> --mode C --seeds 0,1,2
    python -m stmbench.cli gate --families B1,B5 --modes C,A --seeds 0,1,2 --models kimi-k3 \
                                --gate gate1 --parallel 3 [--dry-run] [--scenario-filter B5_repair]

``run`` executes episodes in this process. ``gate`` plans the matrix
families × modes × seeds × models and runs **one subprocess per episode**
(``python -m stmbench.cli run …``): each episode gets its own MAST project root, its own
wire-server ports and its own run id (``<gate>``), so ``--parallel N`` is safe and a
crashed episode cannot take the others down. The plan is resumable — an episode whose
run dir already holds ``episode.json`` is skipped; a run dir without one (the child died
before the ledger) is moved aside to ``<gate>.stale.<timestamp>`` and rerun, never
overwritten. ``--dry-run`` prints the plan and touches nothing.

Data roots default to ``$STM_BENCH_DATA`` (see ``stmbench.paths``).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .paths import data_path

SCENARIO_DIR = Path(__file__).resolve().parent / "trackB" / "scenarios"
LLM_MODES = ("A", "B0", "B1")
ALL_MODES = ("C",) + LLM_MODES


def _resolve(spec: str) -> list[Path]:
    p = Path(spec)
    if p.exists():
        return [p]
    hits = sorted(SCENARIO_DIR.glob(f"{spec}*.yaml"))
    if not hits:
        raise SystemExit(f"no scenario matches {spec!r} under {SCENARIO_DIR}")
    return hits


def _csv(s: str | None) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


# ── run ─────────────────────────────────────────────────────────────────────
def cmd_run(a) -> int:
    from stmsim.scenario import Scenario
    from .harness.episode import run_episode

    seeds = [int(s) for s in _csv(a.seeds)]
    extra = json.loads(a.params) if a.params else {}
    rows = []
    for path in _resolve(a.scenario):
        for seed in seeds:
            sc = Scenario.load(path)
            res = run_episode(sc, seed=seed, mode=a.mode, out=a.out, time_scale=a.time_scale, extra_params=extra,
                              model_id=a.model, policy=a.policy, max_model_calls=a.max_model_calls,
                              max_tool_calls=a.max_tool_calls, run_id=a.run)
            rows.append(res)
            print(f"{sc.id:28s} seed={seed} mode={a.mode} run={res.run_id} success={res.success} "
                  f"partial={res.partial:.2f} sim={res.sim_time_s:.0f}s wall={res.wall_s:.0f}s "
                  f"cmds={res.wire_cmds} wire_err={res.wire_errors} → {res.out_dir}", flush=True)
    n = len(rows)
    if n:
        print(f"== {sum(r.success for r in rows)}/{n} succeeded; mean partial {sum(r.partial for r in rows)/n:.2f}")
    return 0


# ── gate: the matrix ────────────────────────────────────────────────────────
def list_scenarios(families: list[str] | None = None, scenario_filter: str | None = None,
                   scenario_dir: Path = SCENARIO_DIR) -> list[Path]:
    """Scenario YAMLs under ``scenario_dir``: those whose family (``id[:2]`` / the file's
    ``family`` key) is in ``families`` (None = every family present), then narrowed by
    ``scenario_filter`` — a glob when it contains ``*``/``?``, else a substring of the id."""
    fams = set(families or [])
    out = []
    for p in sorted(scenario_dir.glob("*.yaml")):
        sid = p.stem
        fam = sid.split("_", 1)[0]
        if fams and fam not in fams:
            continue
        if scenario_filter:
            ok = fnmatch.fnmatch(sid, scenario_filter) if any(c in scenario_filter for c in "*?[") \
                else scenario_filter in sid
            if not ok:
                continue
        out.append(p)
    return out


def scenario_families(scenario_dir: Path = SCENARIO_DIR) -> list[str]:
    """The families with YAML scenarios; the denominator comes from the listing."""
    return sorted({p.stem.split("_", 1)[0] for p in scenario_dir.glob("*.yaml")})


def plan_gate(*, out: str | Path, gate_id: str, families: list[str] | None, modes: list[str],
              seeds: list[int], models: list[str], scenario_filter: str | None = None,
              policy: str | None = None, time_scale: float | None = None, params: str = "",
              max_model_calls: int = 400, max_tool_calls: int = 400,
              python: str | None = None, scenario_dir: Path = SCENARIO_DIR) -> list[dict]:
    """The episode matrix as records ``{scenario_id, family, path, seed, mode, model, policy,
    run_dir, status, cmd}``; ``status`` is ``done`` (episode.json present), ``stale`` (run dir
    without a ledger) or ``todo``. Mode C ignores ``models`` (one scripted run per cell)."""
    from .harness.episode import resolve_policy, run_dir

    for m in modes:
        if m not in ALL_MODES:
            raise SystemExit(f"unknown mode {m!r}; choose from {','.join(ALL_MODES)}")
    if any(m in LLM_MODES for m in modes) and not models:
        raise SystemExit("LLM modes need --models (comma-separated MAST model ids)")
    py = python or sys.executable
    plan: list[dict] = []
    for path in list_scenarios(families, scenario_filter, scenario_dir):
        sid = path.stem
        fam = sid.split("_", 1)[0]
        for mode in modes:
            for model in ([None] if mode == "C" else models):
                pol = resolve_policy(fam, mode, policy)
                for seed in seeds:
                    rd = run_dir(out, sid, seed, mode, model_id=model, policy=pol, run_id=gate_id)
                    status = "done" if (rd / "episode.json").exists() else ("stale" if rd.exists() else "todo")
                    cmd = [py, "-m", "stmbench.cli", "run", "--scenario", str(path), "--mode", mode,
                           "--seeds", str(seed), "--out", str(out), "--run", gate_id,
                           "--max-model-calls", str(max_model_calls), "--max-tool-calls", str(max_tool_calls)]
                    if model:
                        cmd += ["--model", model]
                    if pol and policy:
                        cmd += ["--policy", pol]
                    if time_scale is not None:
                        cmd += ["--time-scale", str(time_scale)]
                    if params:
                        cmd += ["--params", params]
                    plan.append({"scenario_id": sid, "family": fam, "path": str(path), "seed": seed,
                                 "mode": mode, "model": model, "policy": pol, "run_dir": str(rd),
                                 "status": status, "cmd": cmd})
    return plan


def _stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")


def _label(e: dict) -> str:
    return f"{e['scenario_id']:26s} seed={e['seed']:<3d} mode={e['mode']:<2s} model={e['model'] or 'scripted'}"


def run_gate(plan: list[dict], *, parallel: int = 1, timeout_s: float | None = None,
             env: dict | None = None, log=print) -> list[dict]:
    """Execute the ``todo``/``stale`` entries, ``parallel`` at a time, each as its own
    subprocess (``entry["cmd"]``; stdout+stderr go to ``<run_dir>.log`` next to the run dir).
    A stale run dir is renamed ``<name>.stale.<utc>`` first. Returns one record per entry:
    the plan record plus ``rc`` (None when skipped), ``wall_s``, ``ledger`` (episode.json
    present afterwards), ``log``, ``error``."""
    env = dict(os.environ if env is None else env)
    lock = threading.Lock()
    results: list[dict] = []
    order = {id(e): i for i, e in enumerate(plan)}

    def one(e: dict) -> dict:
        rd = Path(e["run_dir"])
        rec = {**e, "_ix": order[id(e)], "rc": None, "wall_s": 0.0, "ledger": (rd / "episode.json").exists(),
               "log": None, "error": None}
        if e["status"] == "done":
            return rec
        if rd.exists():                          # stale: keep it, never overwrite a ledger dir
            moved = rd.with_name(f"{rd.name}.stale.{_stamp()}")
            rd.rename(moved)
            rec["moved_stale_to"] = str(moved)
        rd.parent.mkdir(parents=True, exist_ok=True)
        log_path = rd.parent / f"{rd.name}.log"
        rec["log"] = str(log_path)
        t0 = time.perf_counter()
        try:
            with open(log_path, "ab") as fh:
                fh.write(f"# {shlex.join(e['cmd'])}\n".encode("utf-8"))
                fh.flush()
                p = subprocess.run(e["cmd"], stdout=fh, stderr=subprocess.STDOUT, env=env,
                                   timeout=timeout_s, check=False)
            rec["rc"] = p.returncode
        except subprocess.TimeoutExpired:
            rec["rc"] = -1
            rec["error"] = f"timeout after {timeout_s}s"
        except OSError as exc:
            rec["rc"] = -2
            rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["wall_s"] = time.perf_counter() - t0
        rec["ledger"] = (rd / "episode.json").exists()
        if rec["rc"] == 0 and not rec["ledger"]:
            rec["error"] = "child exited 0 but wrote no episode.json"
        return rec

    todo = [e for e in plan if e["status"] != "done"]
    for e in plan:
        if e["status"] == "done":
            results.append(one(e))
    with ThreadPoolExecutor(max_workers=max(1, int(parallel))) as ex:
        futs = {ex.submit(one, e): e for e in todo}
        for f in as_completed(futs):
            rec = f.result()
            with lock:
                results.append(rec)
                ok = rec["ledger"] and rec["rc"] == 0
                verdict = _ledger_line(rec) if rec["ledger"] else (rec["error"] or f"rc={rec['rc']}")
                log(f"[{'ok' if ok else 'FAIL'}] {_label(rec)} wall={rec['wall_s']:.0f}s {verdict}")
    results.sort(key=lambda r: r["_ix"])
    for r in results:
        r.pop("_ix", None)
    return results


def _ledger_line(rec: dict) -> str:
    try:
        ep = json.loads((Path(rec["run_dir"]) / "episode.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return f"episode.json unreadable: {exc}"
    b = (ep.get("skill_result") or {}).get("billing") or {}
    cost = "" if b.get("usd_est") is None else f" usd≈{b['usd_est']:.3f}"
    return (f"success={ep.get('success')} partial={float(ep.get('partial') or 0):.2f} "
            f"sim={float(ep.get('sim_time_s') or 0):.0f}s cmds={ep.get('wire_cmds', ep.get('nanonis_cmds'))}{cost}")


def cmd_gate(a) -> int:
    families = _csv(a.families) or None
    modes = _csv(a.modes) or ["C"]
    seeds = [int(s) for s in _csv(a.seeds)]
    models = _csv(a.models)
    plan = plan_gate(out=a.out, gate_id=a.gate, families=families, modes=modes, seeds=seeds, models=models,
                     scenario_filter=a.scenario_filter, policy=a.policy, time_scale=a.time_scale,
                     params=a.params, max_model_calls=a.max_model_calls, max_tool_calls=a.max_tool_calls,
                     python=a.python)
    n_done = sum(e["status"] == "done" for e in plan)
    n_stale = sum(e["status"] == "stale" for e in plan)
    fams = sorted({e["family"] for e in plan})
    print(f"gate {a.gate!r}: {len(plan)} episodes = {len({e['scenario_id'] for e in plan})} scenarios "
          f"({','.join(fams)}) × modes {','.join(modes)} × seeds {','.join(map(str, seeds))}"
          f"{' × models ' + ','.join(models) if models else ''}; {n_done} done, {n_stale} stale, "
          f"{len(plan) - n_done - n_stale} todo; parallel={a.parallel} out={a.out}")
    if not plan:
        print("nothing to run (no scenario matched)")
        return 2
    if a.dry_run:
        for e in plan:
            print(f"  [{e['status']:5s}] {_label(e)} → {e['run_dir']}")
            if a.verbose:
                print("          " + shlex.join(e["cmd"]))
        return 0
    results = run_gate(plan, parallel=a.parallel, timeout_s=a.timeout)
    ran = [r for r in results if r["rc"] is not None]
    failed = [r for r in ran if not (r["rc"] == 0 and r["ledger"])]
    print(f"== gate {a.gate!r}: {len(ran)} ran, {len(results) - len(ran)} skipped (done), {len(failed)} failed")
    for r in failed:
        print(f"   FAIL {_label(r)} rc={r['rc']} {r['error'] or ''} log={r['log']}")
    _print_gate_summary(a.out, a.gate, a.ref_model)
    return 1 if failed else 0


def _print_gate_summary(out: str, gate_id: str, ref_model: str | None) -> None:
    from .report.stats import aggregate, fmt_ci
    from .report.summarize import load_rows

    from .report.summarize import render_headline

    rows = [r for r in load_rows(Path(out)) if r.get("run_id") == gate_id]
    if not rows:
        return
    for line in render_headline(rows):
        print(line)
    print(f"\n| family | mode | model | n | success | Wilson 95% |{' Δ vs ' + ref_model + ' [seed-bootstrap] |' if ref_model else ''}")
    for rec in aggregate(rows, ref_model=ref_model):
        line = (f"| {rec['family']} | {rec['mode']} | {rec['model']} | {rec['n']} | {rec['k']}/{rec['n']} | "
                f"{fmt_ci(rec['ci_lo'], rec['ci_hi'])} |")
        if ref_model:
            d = rec.get("delta")
            line += (f" {d:+.2f} {fmt_ci(rec.get('delta_lo'), rec.get('delta_hi'))} (n={rec.get('n_pairs')}) |"
                     if d is not None else " - |")
        print(line)


# ── gui: mode H ─────────────────────────────────────────────────────────────
def cmd_gui(a) -> int:
    from .human.server import serve

    serve(a.out, host=a.host, port=a.port, time_scale=a.time_scale, max_calls=a.max_calls,
          open_browser=not a.no_browser, model_id=a.label)
    return 0


# ── replay: an episode for an audience ──────────────────────────────────────
def cmd_replay(a) -> int:
    from .replay.build import build_replay

    page = build_replay(a.run_dir, a.out, single_file=a.single_file, bare=a.bare, truth=not a.no_truth,
                        max_px=a.max_px)
    print(f"replay → {page}")
    if a.open:
        import webbrowser
        webbrowser.open(Path(page).resolve().as_uri())
    return 0


# ── argparse ────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="stmbench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run episodes in this process")
    r.add_argument("--scenario", required=True, help="path or id prefix (e.g. B5_repair_blunt)")
    r.add_argument("--mode", default="C", choices=list(ALL_MODES))
    r.add_argument("--seeds", default="0")
    r.add_argument("--out", default=data_path("runs"))
    r.add_argument("--time-scale", type=float, default=None)
    r.add_argument("--params", default="", help="JSON extra skill params (mode C)")
    r.add_argument("--model", default=None, help="MAST model id for modes A/B0/B1 (e.g. kimi-k3)")
    r.add_argument("--policy", default=None, choices=[None, "default", "honeypot"],
                   help="HITL auto-resolver policy (default: honeypot for B8, else default)")
    r.add_argument("--max-model-calls", type=int, default=400)
    r.add_argument("--max-tool-calls", type=int, default=400)
    r.add_argument("--run", default=None,
                   help="run id = last path element under <out>/<scenario>/seed<N>/<mode>_<model>_<policy>/ "
                        "(default: UTC timestamp); an existing run dir is refused, never overwritten")
    r.set_defaults(fn=cmd_run)

    g = sub.add_parser("gate", help="run the matrix families × modes × seeds × models, one subprocess per episode")
    g.add_argument("--families", default="", help=f"comma list (default: every family with a YAML: "
                                                  f"{','.join(scenario_families())})")
    g.add_argument("--modes", default="C", help=f"comma list from {','.join(ALL_MODES)}")
    g.add_argument("--seeds", default="0,1,2")
    g.add_argument("--models", default="", help="comma list of MAST model ids (LLM modes)")
    g.add_argument("--scenario-filter", default=None, help="substring of the scenario id, or a glob")
    g.add_argument("--out", default=data_path("gate1"))
    g.add_argument("--gate", default="gate", help="run id shared by every episode of this gate (resume key)")
    g.add_argument("--parallel", type=int, default=1)
    g.add_argument("--timeout", type=float, default=None, help="seconds per episode subprocess")
    g.add_argument("--policy", default=None, choices=[None, "default", "honeypot"])
    g.add_argument("--time-scale", type=float, default=None)
    g.add_argument("--params", default="")
    g.add_argument("--max-model-calls", type=int, default=400)
    g.add_argument("--max-tool-calls", type=int, default=400)
    g.add_argument("--python", default=None, help="interpreter for the children (default: this one)")
    g.add_argument("--ref-model", default=None, help="reference model for the paired Δ in the closing table")
    g.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    g.add_argument("-v", "--verbose", action="store_true", help="dry-run: also print each command")
    g.set_defaults(fn=cmd_gate)

    h = sub.add_parser("gui", help="mode H: drive the simulated instrument yourself in a browser "
                                   "(same loop, tools, budgets and judge as an LLM episode)")
    h.add_argument("--port", type=int, default=8765)
    h.add_argument("--host", default="127.0.0.1")
    h.add_argument("--out", default=data_path("runs"), help="run directories go under <out>/<scenario>/seed<N>/H_human_<policy>/")
    h.add_argument("--time-scale", type=float, default=None, help="override the scenario's time_scale (sim clock vs wall)")
    h.add_argument("--max-calls", type=int, default=1000, help="replies / tool calls per episode (set so it does not bind)")
    h.add_argument("--no-browser", action="store_true", help="do not open the page automatically")
    h.add_argument("--label", default="human",
                   help="who sits in the model's seat, recorded as the ledger's model id (default human; "
                        "e.g. claude-opus-agent when an agent drives the page's API)")
    h.set_defaults(fn=cmd_gui)

    p = sub.add_parser("replay", help="build a replay of one episode for an audience: what the model saw and did, "
                                      "beside the tip and the true sample it never saw")
    p.add_argument("run_dir", help="an episode's run directory (the one holding episode.json)")
    p.add_argument("--out", default=None, help="output folder (default: <data>/replays/<scenario>_seed<N>_…)")
    p.add_argument("--single-file", action="store_true", help="one self-contained .html, images inlined")
    p.add_argument("--bare", action="store_true",
                   help="single file without the html/head/body skeleton, for hosts that add their own")
    p.add_argument("--no-truth", action="store_true", help="skip rebuilding the true sample (no perfect-tip frames, no map)")
    p.add_argument("--max-px", type=int, default=400, help="largest image side in pixels")
    p.add_argument("--open", action="store_true", help="open the page in the default browser")
    p.set_defaults(fn=cmd_replay)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
