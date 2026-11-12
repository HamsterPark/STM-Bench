"""stmbench command line.

    python -m stmbench.cli episode --scenario P5_sader_jarvis_qplus --seed 0 --out <run dir>   # MCP stdio
    python -m stmbench.cli verify <run dir>                                                     # replay + compare
    python -m stmbench.cli reference --paper P4 [--seeds 30] [--jobs 4] [--json out.json]      # privileged reference runs
    python -m stmbench.cli baseline --paper P4 [--seeds 30] [--jobs 4] [--json out.json]       # scripted, no truth
    python -m stmbench.cli audit chance [--scenario P3] [--seeds 200] [--jobs 4] [--json out.json]
    python -m stmbench.cli agent --scenario P4_atom_positioning_cu111 --seed 0 --out <run dir> \
                                 [--provider anthropic|openai] [--model ID]                    # a model alone
    python -m stmbench.cli console --scenario P4_atom_positioning_cu111 --seed 0 --out <run dir> [--port 8765]

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
    """Replay an instrument episode from its ledger and compare measurements and verdict."""
    from .instrument import verify

    res = verify(a.run_dir)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0 if res["ok"] else 1


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

    f = sub.add_parser("reference", help="privileged reference solutions: the truth navigates, every reported "
                                         "number is measured; reports success, error spread and instrument time")
    f.add_argument("--paper", required=True, help="P3, P4 or P5")
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
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
