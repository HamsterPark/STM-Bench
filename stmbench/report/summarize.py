"""Walk episode directories and print / write the family × mode × model table.

    python -m stmbench.report.summarize --runs <data>/gate1 [--ref-model kimi-k3] [--csv out.csv] [--md out.md]

One row per ``episode.json``; columns: scenario, family, seed, mode, model, provider,
policy, run_id, success, partial, checks (failed ones), driver outcome, sim_s / budget,
wire_cmds, model_calls, tool_calls, hitl, stub_calls, diag_correct (the ledger's
``1`` / ``0`` / ``abstain``: ReportTipState vs truth), envelope_violations (tool results
MAST's safety stack refused), tokens in/out and usd_est from the billing ledger (``-``
when unknown), wall_s.

The §5.2 ledger keys (``provider``, ``diag_correct``, ``envelope_violations``, ``billing``
…) are read from the top level of ``episode.json``; a ledger written before that schema
gets ``provider`` and ``diag_correct`` recomputed from what it does carry and ``-`` for
the rest — never a 0 it did not record.

The aggregate (``stats.aggregate``) gives per (family, mode, model) and per (ALL, mode,
model): n, success k/n with its Wilson 95 % CI, mean partial, median sim_s / cmds, the
diagnosis accuracy over the episodes that made a statement (``stats.diag_accuracy``) and —
with ``--ref-model`` — the paired Δ against that model with a seed-bootstrap CI. A last
block reports the LLM-sampling vs sim variance split when some seeds were run more than
once (``stats.variance_split``).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ..paths import data_path
from .stats import aggregate, fmt_ci, variance_split

_MISSING = object()


def _ledger_diag_correct(ep: dict):
    """``diag_correct`` as the ledger wrote it; recomputed for a ledger that predates the key."""
    v = ep.get("diag_correct", _MISSING)
    if v is not _MISSING:
        return v
    from ..harness.episode import diag_correct

    sr = ep.get("skill_result") or {}
    thr = ((ep.get("verdict") or {}).get("details") or {}).get("thresholds")
    return diag_correct(sr.get("diagnosis"), ep.get("truth_after"), thr if isinstance(thr, dict) else None)


def _ledger_provider(ep: dict, model: str | None):
    v = ep.get("provider", _MISSING)
    if v is not _MISSING:
        return v
    from ..harness.episode import provider_of

    return provider_of(model if model not in (None, "scripted", "?") else None)


def _opt(v, nd: int | None = None):
    """A ledger number for a table cell: ``-`` when unknown (None), never 0."""
    if v is None:
        return "-"
    return round(float(v), nd) if nd is not None else v


def load_rows(runs: Path) -> list[dict]:
    rows = []
    for ep_path in sorted(runs.rglob("episode.json")):
        try:
            ep = json.loads(ep_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(ep, dict):
            continue
        sr = ep.get("skill_result") or {}
        checks = (ep.get("verdict") or {}).get("details", {}).get("checks", {}) or {}
        failed = [k for k, v in checks.items() if not v]
        budget = sr.get("budget") or {}
        billing = ep.get("billing") if isinstance(ep.get("billing"), dict) else (sr.get("billing") or {})
        model = sr.get("model_id") or ep.get("model_id") or ("scripted" if ep.get("mode") == "C" else "?")
        diag = _ledger_diag_correct(ep)
        rows.append({
            "scenario": ep.get("scenario_id"),
            # the family is the id up to the first underscore, the same rule the gate planner
            # uses (cli.py) — a two-character slice broke the moment a family was not "B<n>"
            "family": str(ep.get("scenario_id", "")).split("_", 1)[0],
            "paper": ep.get("paper_id") or "-",
            "reproduced": ep.get("reproduced") if ep.get("reproduced") is not None else "-",
            "claims": (f"{ep.get('claims_verified')}/{ep.get('claims_total')}"
                       if ep.get("claims_total") is not None else "-"),
            "seed": ep.get("seed"), "mode": ep.get("mode"),
            "model": model, "provider": _ledger_provider(ep, model) or "-",
            "policy": sr.get("policy") or ep.get("policy") or "-", "run_id": ep.get("run_id") or ep_path.parent.name,
            "success": ep.get("success"), "partial": round(float(ep.get("partial") or 0), 2),
            "failed_checks": ",".join(failed) or "-",
            "outcome": sr.get("outcome") or ("skill:" + str(sr.get("success"))),
            "sim_s": int(sr.get("sim_s") or ep.get("sim_time_s") or 0),
            "budget_sim_s": int(budget.get("max_sim_s") or 0) or "-",
            # ledgers written before 2026-09-13 carry the field under its former name
            "cmds": (sr.get("wire_cmds") or ep.get("wire_cmds")
                     or sr.get("nanonis_cmds") or ep.get("nanonis_cmds")),
            "model_calls": sr.get("model_calls", "-"), "tool_calls": sr.get("tool_calls", "-"),
            "hitl": sr.get("hitl_questions", "-"), "stubs": len(sr.get("stub_calls") or []) if sr.get("stub_calls") is not None else "-",
            "diag_correct": diag if diag is not None else "-",
            "envelope_violations": _opt(ep.get("envelope_violations")),
            "tokens_in": _opt(billing.get("tokens_in")), "tokens_out": _opt(billing.get("tokens_out")),
            "usd_est": _opt(billing.get("usd_est"), 4),
            "cost_known": billing.get("cost_known") if billing else "-",
            "wall_s": int(ep.get("wall_s") or 0),
            "path": str(ep_path.parent),
        })
    return rows


ROW_COLS = ["scenario", "paper", "seed", "mode", "model", "provider", "policy", "success", "claims", "partial", "failed_checks",
            "outcome", "sim_s", "budget_sim_s", "cmds", "model_calls", "tool_calls", "hitl", "stubs",
            "diag_correct", "envelope_violations", "tokens_in", "tokens_out", "usd_est", "wall_s"]


def _f(v, nd=2) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def render_headline(rows: list[dict], n_boot: int = 2000) -> list[str]:
    """paper × mode × model: how often each paper was reproduced, and what it cost.

    This is the table the benchmark exists to produce, so it goes first. Empty when no paper
    episode is in the run — the B families are regressions, not a headline."""
    papers = [r for r in rows if str(r.get("family", "")).startswith("P")]
    if not papers:
        return []
    cols = ["paper", "mode", "model", "n", "reproduced", "Wilson 95%", "mean claims",
            "median sim h", "median usd"]
    out = ["", "### 论文复现率", "",
           "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for rec in aggregate(papers, n_boot=n_boot):
        if rec["family"] == "ALL" and len({r["family"] for r in papers}) == 1:
            continue                       # one paper: the ALL row would repeat it
        cell = [r for r in papers if rec["family"] in ("ALL", r["family"])
                and r.get("mode") == rec["mode"] and r.get("model") == rec["model"]]
        claims = [_claim_fraction(r) for r in cell]
        claims = [c for c in claims if c is not None]
        usd = [r.get("usd_est") for r in cell if isinstance(r.get("usd_est"), (int, float))]
        sim_h = [float(r["sim_s"]) / 3600.0 for r in cell
                 if isinstance(r.get("sim_s"), (int, float))]
        out.append("| " + " | ".join(str(c) for c in [
            rec["family"], rec["mode"], rec["model"], rec["n"], f"{rec['k']}/{rec['n']}",
            fmt_ci(rec["ci_lo"], rec["ci_hi"]),
            _f(sum(claims) / len(claims)) if claims else "-",
            _f(_median(sim_h), 2), _f(_median(usd), 3)]) + " |")
    return out


def _claim_fraction(row: dict) -> float | None:
    got = str(row.get("claims") or "-")
    if "/" not in got:
        return None
    a, b = got.split("/", 1)
    try:
        return float(a) / float(b) if float(b) else None
    except ValueError:
        return None


def _median(values) -> float | None:
    vals = sorted(v for v in values if isinstance(v, (int, float)))
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def render_md(rows: list[dict], ref_model: str | None = None, n_boot: int = 2000) -> str:
    out = render_headline(rows, n_boot=n_boot)
    if out:
        out.append("")
    out += ["| " + " | ".join(ROW_COLS) + " |", "|" + "|".join("---" for _ in ROW_COLS) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in ROW_COLS) + " |")
    # aggregate: family × mode × model (+ ALL) with Wilson CI and the paired Δ vs the reference
    cols = ["family", "mode", "model", "n", "success", "Wilson 95%", "mean partial", "median sim_s", "median cmds",
            "diag acc", "diag abstain"]
    if ref_model:
        cols += [f"Δ vs {ref_model}", "seed-bootstrap 95%", "pairs/seeds"]
    out += ["", "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for rec in aggregate(rows, ref_model=ref_model, n_boot=n_boot):
        acc = rec.get("diag_acc")
        cells = [rec["family"], rec["mode"], rec["model"], rec["n"], f"{rec['k']}/{rec['n']}",
                 fmt_ci(rec["ci_lo"], rec["ci_hi"]), _f(rec["mean_partial"]),
                 _f(rec["median_sim_s"], 0), _f(rec["median_cmds"], 0),
                 "-" if acc is None else f"{rec.get('diag_correct_n')}/{rec.get('diag_n')}",
                 f"{rec.get('diag_abstain', 0)}/{rec['n']}"]
        if ref_model:
            d = rec.get("delta")
            cells += ["-" if d is None else f"{d:+.2f}", fmt_ci(rec.get("delta_lo"), rec.get("delta_hi")),
                      "-" if d is None else f"{rec.get('n_pairs')}/{rec.get('n_seeds')}"]
        out.append("| " + " | ".join(str(c) for c in cells) + " |")
    vs = variance_split(rows)
    out.append("")
    if vs["var_llm"] is None:
        out.append(f"variance split: no repeated seed among {vs['n_episodes']} episodes — LLM-sampling vs sim "
                   f"variance unknown (rerun ~20 % of the seeds twice)")
    else:
        out.append(f"variance split over {vs['n_repeated_groups']} repeated seeds / {vs['n_groups']} groups / "
                   f"{vs['n_episodes']} episodes: var_llm={vs['var_llm']:.3f} var_sim={_f(vs['var_sim'], 3)} "
                   f"frac_llm={_f(vs['frac_llm'])} frac_sim={_f(vs['frac_sim'])}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=data_path("gate1"))
    ap.add_argument("--ref-model", default=None, help="paired Δ (same scenario/seed/mode) against this model")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--md", default=None)
    a = ap.parse_args(argv)
    rows = load_rows(Path(a.runs))
    md = render_md(rows, ref_model=a.ref_model, n_boot=a.n_boot)
    print(md)
    if a.md:
        Path(a.md).write_text(md, encoding="utf-8")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["scenario"])
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
