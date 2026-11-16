"""stmbench command line.

    python -m stmbench.cli episode --scenario P5_sader_jarvis_qplus --seed 0 --out <run dir>   # MCP stdio
    python -m stmbench.cli verify <run dir>                                                     # replay + compare
    python -m stmbench.cli grade <export dir> --scenario P4_atom_positioning_cu111 --seed 0 [--no-strict] [--out result.json]
    python -m stmbench.cli task --scenario P4_atom_positioning_cu111 [--lang zh|en]            # the task text
    python -m stmbench.cli reference --paper P4 [--seeds 30] [--jobs 4] [--json out.json]      # privileged reference runs
    python -m stmbench.cli baseline --paper P4 [--seeds 30] [--jobs 4] [--json out.json]       # scripted, no truth
    python -m stmbench.cli audit chance [--scenario P3] [--seeds 200] [--jobs 4] [--json out.json]
    python -m stmbench.cli agent --scenario P4_atom_positioning_cu111 --seed 0 --out <run dir> \
                                 [--provider anthropic|openai] [--model ID]                    # a model alone
    python -m stmbench.cli console --scenario P4_atom_positioning_cu111 --seed 0 --out <run dir> [--port 8765]
    python -m stmbench.cli gui --scenario P4_atom_positioning_cu111 --seed 0 --out <run dir> [--lang zh|en]
    python -m stmbench.cli gui --sealed --private-dir <dir> --export-dir <dir>      # the seed on stdin, as one JSON line

Data roots default to ``$STM_BENCH_DATA`` (see ``stmbench.paths``)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCENARIO_DIR = Path(__file__).resolve().parent / "trackB" / "scenarios"


def _resolve(spec: str) -> list[Path]:
    p = Path(spec)
    if p.exists():
        return [p]
    hits = sorted(SCENARIO_DIR.glob(f"{spec}*.yaml"))
    if not hits:
        raise SystemExit(f"no scenario matches {spec!r} under {SCENARIO_DIR}")
    return hits


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


# ── audit ───────────────────────────────────────────────────────────────────
def cmd_audit(a) -> int:
    """How much of each paper a fixed answer sheet reproduces (stmbench.trackB.audit)."""
    from .trackB.audit import audit_scenario, format_report

    paths = _resolve(a.scenario) if a.scenario else sorted(SCENARIO_DIR.glob("P*.yaml"))
    results = []
    for path in paths:
        res = audit_scenario(path, a.seeds, jobs=a.jobs, first_seed=a.first_seed)
        print(format_report(res), flush=True)
        print()
        results.append(res)
    if a.json:
        Path(a.json).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"audit → {a.json}")
    return 0


# ── the benchmark instrument ───────────────────────────────────────────────
def cmd_episode(a) -> int:
    """Serve one episode over MCP stdio (stmbench.instrument.mcp_server)."""
    from .instrument import Instrument
    from .instrument.mcp_server import EpisodeServer

    EpisodeServer(Instrument(a.scenario, a.seed, out_dir=a.out)).serve_stdio()
    return 0


def cmd_verify(a) -> int:
    """Replay an instrument episode from its ledger and compare measurements and verdict: 0 when
    it re-checks, 1 when it does not, 2 when the ledger is of another API version (it needs the
    code of its own)."""
    from .instrument import VersionError, verify

    try:
        res = verify(a.run_dir)
    except VersionError as exc:
        print(f"verify: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0 if res["ok"] else 1


def cmd_grade(a) -> int:
    """Grade an exported episode by replaying it with the scenario and seed this host holds
    (stmbench.instrument.grade): 0 when graded, 2 when the setup does not match (another scenario,
    rules, API or export version). A damaged export is graded too: it scores 0, exit 0."""
    from .instrument.export import ExportDamaged
    from .instrument.grade import damaged_result, grade_export

    try:
        try:
            res = grade_export(a.export_dir, a.scenario, a.seed, strict=not a.no_strict)
        except ExportDamaged as exc:
            print(f"grade: {exc}", file=sys.stderr)
            res = damaged_result(exc.reason, a.scenario)
    except ValueError as exc:
        print(f"grade: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(res, indent=1)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    return 0


def cmd_task(a) -> int:
    """Print what an agent is told for a scenario: its task and, for a paper, the rule card
    (stmbench.trackB.rules.task_text), for building the task files of other harnesses."""
    from stmsim.scenario import Scenario
    from .trackB.rules import task_text

    exact = SCENARIO_DIR / f"{a.scenario}.yaml"
    hits = [exact] if exact.is_file() else _resolve(a.scenario)
    if len(hits) != 1:
        raise SystemExit(f"{a.scenario!r} matches {len(hits)} scenarios ({', '.join(h.stem for h in hits)}); "
                         f"name one")
    try:
        text = task_text(Scenario.load(hits[0]), a.lang)
    except ValueError as e:                  # e.g. a scenario without its English text
        raise SystemExit(str(e))
    # UTF-8 bytes with plain line breaks, whatever the console's code page and newline translation:
    # the output is meant to be redirected into a task file
    sys.stdout.flush()
    out = getattr(sys.stdout, "buffer", None)
    if out is None:
        print(text)
    else:
        out.write(text.encode("utf-8") + b"\n")
        out.flush()
    return 0


def cmd_reference(a) -> int:
    """Privileged reference solutions over many seeds (stmbench.reference)."""
    from .reference import format_summary, run_many, summarize

    rows = run_many(a.paper, list(range(a.first_seed, a.first_seed + a.seeds)), jobs=a.jobs)
    s = summarize(rows)
    print(format_summary(a.paper, s))
    if a.json:
        Path(a.json).write_text(json.dumps({"summary": s, "runs": rows}, ensure_ascii=False, indent=1),
                                encoding="utf-8")
        print(f"reference → {a.json}")
    return 0


def cmd_baseline(a) -> int:
    """Scripted baselines over many seeds: no truth (stmbench.baselines)."""
    from .baselines import format_summary, run_many, summarize

    rows = run_many(a.paper, list(range(a.first_seed, a.first_seed + a.seeds)), jobs=a.jobs)
    s = summarize(rows)
    print(format_summary(a.paper, s))
    if a.json:
        Path(a.json).write_text(json.dumps({"summary": s, "runs": rows}, ensure_ascii=False, indent=1, default=str),
                                encoding="utf-8")
        print(f"baseline → {a.json}")
    return 0


def cmd_agent(a) -> int:
    """One episode with a language model at the instrument (stmbench.agent)."""
    from .agent import AnthropicModel, OpenAICompatibleModel, run_agent_episode

    if a.provider == "anthropic":
        model = AnthropicModel(a.model or "claude-opus-5-5", effort=a.effort,
                               max_tokens=a.max_tokens or 64000, fallbacks=a.fallbacks)
    else:
        if not a.model:
            raise SystemExit("--model is required with --provider openai")
        model = OpenAICompatibleModel(a.model, base_url=a.base_url, api_key_env=a.api_key_env,
                                      max_tokens=a.max_tokens or 16000, vision=not a.no_images)
    res = run_agent_episode(a.scenario, a.seed, a.out, model, max_turns=a.max_turns,
                            max_tool_calls=a.max_tool_calls,
                            max_wall_s=a.max_wall_min * 60.0 if a.max_wall_min else None,
                            log=print if a.verbose else None)
    print(json.dumps(res.as_dict(), ensure_ascii=False, indent=1, default=str))
    return 1 if res.stop.startswith("error") else 0


def cmd_console(a) -> int:
    """The instrument in a web page (stmbench.console)."""
    from .console import serve

    if a.seed is not None and a.scenario is None:
        raise SystemExit("--seed needs --scenario (without one, the page offers the scenarios)")
    serve(a.out, scenario=a.scenario, seed=a.seed, host=a.host, port=a.port, open_browser=not a.no_browser)
    return 0


def cmd_gui(a) -> int:
    """The instrument in a native window (stmbench.gui)."""
    if a.sealed:
        if a.scenario is not None or a.seed is not None or a.out is not None:
            raise SystemExit("--sealed reads the scenario and the seed from one JSON line on stdin: "
                             "do not give --scenario, --seed or --out")
        if not (a.private_dir and a.export_dir):
            raise SystemExit("--sealed needs --private-dir and --export-dir")
    else:
        if a.scenario is None or a.out is None:
            raise SystemExit("gui needs --scenario and --out (or --sealed with --private-dir and --export-dir)")
        if a.private_dir or a.export_dir:
            raise SystemExit("--private-dir and --export-dir belong to --sealed")
    try:
        from .gui.app import run
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] != "PySide6":
            raise
        raise SystemExit('the desktop console needs Qt: pip install "stm-bench[gui]"') from exc
    return run(scenario=a.scenario, seed=0 if a.seed is None else a.seed, out=a.out, lang=a.lang,
               ui_scale=a.ui_scale, sealed=a.sealed, private_dir=a.private_dir, export_dir=a.export_dir,
               ready_file=a.ready_file)


# ── argparse ────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="stmbench")
    sub = ap.add_subparsers(dest="cmd", required=True)
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

    u = sub.add_parser("audit", help="benchmark self-checks that need no model: 'chance' = how often a fixed "
                                     "answer sheet passes each paper claim")
    u.add_argument("what", choices=["chance"])
    u.add_argument("--scenario", default=None, help="path or id prefix (default: every paper scenario P*)")
    u.add_argument("--seeds", type=int, default=200, help="number of seeds drawn")
    u.add_argument("--first-seed", type=int, default=0)
    u.add_argument("--jobs", type=int, default=1, help="processes building worlds in parallel")
    u.add_argument("--json", default=None, help="also write the results here")
    u.set_defaults(fn=cmd_audit)

    e = sub.add_parser("episode", help="serve one episode of the instrument over MCP stdio "
                                       "(any MCP agent can take the instrument's seat)")
    e.add_argument("--scenario", required=True, help="scenario id or YAML path")
    e.add_argument("--seed", type=int, required=True)
    e.add_argument("--out", required=True, help="run directory: ledger and saved files")
    e.set_defaults(fn=cmd_episode)

    v = sub.add_parser("verify", help="re-check an instrument episode: replay its actions and compare "
                                      "every measurement and the verdict")
    v.add_argument("run_dir")
    v.set_defaults(fn=cmd_verify)

    gr = sub.add_parser("grade", help="grade an exported episode: replay its recorded calls with the scenario "
                                      "and seed only this host holds, and judge the replay")
    gr.add_argument("export_dir", help="the export directory (session.json, actions.jsonl, digests.jsonl)")
    gr.add_argument("--scenario", required=True, help="scenario id or YAML path")
    gr.add_argument("--seed", type=int, required=True)
    gr.add_argument("--no-strict", action="store_true",
                    help="score a replay that differs from the recorded episode as replayed (default: 0)")
    gr.add_argument("--out", default=None, help="also write the result here")
    gr.set_defaults(fn=cmd_grade)

    k = sub.add_parser("task", help="print the task text an agent is given: the scenario's task and, for a "
                                    "paper, its rule card")
    k.add_argument("--scenario", required=True, help="scenario id or YAML path")
    k.add_argument("--lang", choices=["zh", "en"], default="zh",
                   help="zh (default) or en; en needs the scenario's task_en and its claims' say_en")
    k.set_defaults(fn=cmd_task)

    f = sub.add_parser("reference", help="privileged reference solutions: the truth navigates, every reported "
                                         "number is measured; reports success, error spread and instrument time")
    f.add_argument("--paper", required=True, help="P3, P3R (P3's repair variant), P4 or P5")
    f.add_argument("--seeds", type=int, default=30, help="number of seeds")
    f.add_argument("--first-seed", type=int, default=0)
    f.add_argument("--jobs", type=int, default=1, help="processes in parallel")
    f.add_argument("--json", default=None, help="also write every run and the summary here")
    f.set_defaults(fn=cmd_reference)
    b = sub.add_parser("baseline", help="scripted baselines: the instrument an agent gets and nothing else "
                                        "(no truth); reports success by the tip's state and instrument time")
    b.add_argument("--paper", required=True, help="P4")
    b.add_argument("--seeds", type=int, default=30, help="number of seeds")
    b.add_argument("--first-seed", type=int, default=0)
    b.add_argument("--jobs", type=int, default=1, help="processes in parallel")
    b.add_argument("--json", default=None, help="also write every run and the summary here")
    b.set_defaults(fn=cmd_baseline)

    g = sub.add_parser("agent", help="one episode with a language model at the instrument, with the instrument's "
                                     "own tools and nothing else (the model-alone condition)")
    g.add_argument("--scenario", required=True, help="scenario id or YAML path")
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--out", required=True, help="a new run directory")
    g.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic",
                   help="anthropic: Claude through the anthropic SDK (pip install \"stm-bench[agent]\"); "
                        "openai: any OpenAI-compatible Chat Completions endpoint")
    g.add_argument("--model", default=None, help="model id (anthropic default: claude-opus-5-5)")
    g.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"],
                   help="anthropic: the effort level (recorded in driver.json)")
    g.add_argument("--max-tokens", type=int, default=None, help="per turn (anthropic 64000, openai 16000)")
    g.add_argument("--fallbacks", action="store_true",
                   help="anthropic: allow server-side refusal fallbacks to another model (off by default: a "
                        "turn answered by another model does not measure the model under test)")
    g.add_argument("--base-url", default="https://api.openai.com/v1", help="openai: the endpoint's base URL")
    g.add_argument("--api-key-env", default="OPENAI_API_KEY", help="openai: the variable holding the key")
    g.add_argument("--no-images", action="store_true", help="openai: the model gets no images (text-only models)")
    g.add_argument("--max-turns", type=int, default=400)
    g.add_argument("--max-tool-calls", type=int, default=1000)
    g.add_argument("--max-wall-min", type=float, default=None, help="wall-clock limit in minutes")
    g.add_argument("-v", "--verbose", action="store_true", help="print one line per turn")
    g.set_defaults(fn=cmd_agent)

    w = sub.add_parser("console", help="the instrument in a web page, for a person or an agent working through a "
                                       "screen; the same ledger as an MCP episode")
    w.add_argument("--out", required=True, help="the run directory (with --scenario) or a directory for the "
                                                "episode the page starts")
    w.add_argument("--scenario", default=None, help="scenario id; without it the page offers the paper scenarios")
    w.add_argument("--seed", type=int, default=None, help="with --scenario (default 0)")
    w.add_argument("--host", default="127.0.0.1",
                   help="interface to bind (default 127.0.0.1; whoever reaches the page can operate the instrument)")
    w.add_argument("--port", type=int, default=8765)
    w.add_argument("--no-browser", action="store_true", help="do not open a browser")
    w.set_defaults(fn=cmd_console)

    gu = sub.add_parser("gui", help="the instrument in a native window (needs the gui extra), for a person or an "
                                    "agent working through a screen; the same ledger as an MCP episode, plus an "
                                    "export directory")
    gu.add_argument("--scenario", default=None, help="scenario id or YAML path (not with --sealed)")
    gu.add_argument("--seed", type=int, default=None, help="with --scenario (default 0); not with --sealed")
    gu.add_argument("--out", default=None, help="a new run directory; the export goes to <out>/export (not with --sealed)")
    gu.add_argument("--lang", choices=["zh", "en"], default="zh",
                    help="zh: each label in Chinese followed by its English (default); en: English only")
    gu.add_argument("--ui-scale", type=float, default=1.0, help="multiplies every font and size (1.0 suits 1920x1080)")
    gu.add_argument("--sealed", action="store_true",
                    help="read {\"scenario\": ..., \"seed\": ...} as one JSON line on stdin and keep the seed out of "
                         "argv, the window, the export and every message; the window opens full screen and its close "
                         "button does nothing")
    gu.add_argument("--private-dir", default=None, help="with --sealed: the run directory, made readable by its owner "
                                                        "only (it holds the seed and, after the end, the truth)")
    gu.add_argument("--export-dir", default=None, help="with --sealed: where the agent-visible export is written")
    gu.add_argument("--ready-file", default=None,
                    help="write this file (one line, 0644) once the window is on screen: a launcher waits for it")
    gu.set_defaults(fn=cmd_gui)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
