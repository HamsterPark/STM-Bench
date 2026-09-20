"""The §5.2 episode ledger schema (docs/DESIGN.md): ``episode.json`` carries provenance
(provider, sampling params, both git SHAs, sim version, settings snapshot sha) and outcome
facts (envelope violations, tip_dead, damage area, wrong-action count, diag_correct,
billing) with the right types, and every unknown is ``None`` — never a default.

* a mode-C run on the smallest scenario (needs MAST) writes every key;
* ``diag_correct`` / ``count_envelope_violations`` / ``sampling_params_for`` are pinned on
  hand-made dicts;
* ``summarize.load_rows`` reads the new keys and still copes with a ledger written before them.
"""
from __future__ import annotations

import dataclasses
import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from stmbench.harness.episode import (
    BILLING_NO_MODEL, BILLING_UNKNOWN, ENVELOPE_MARKERS, EpisodeResult, count_envelope_violations, diag_correct,
    file_sha256, git_sha, provenance, provider_of, sim_version, stmbench_root,
)
from stmbench.harness.ic_driver import MODEL_CALL_DEFAULTS, observed_sampling_params, sampling_params_for
from stmbench.report.stats import aggregate, diag_accuracy
from stmbench.report.summarize import ROW_COLS, load_rows, render_md
from tests.conftest import requires_mast

NONE = type(None)
#: the §5.2 keys and the types each may take in ``episode.json``
LEDGER_TYPES: dict[str, tuple] = {
    "provider": (str, NONE), "sampling_params_json": (dict, NONE),
    "mast_sha": (str, NONE), "stmbench_sha": (str, NONE), "sim_version": (str, NONE),
    "settings_snapshot_sha": (str, NONE), "envelope_violations": (int, NONE), "tip_dead": (bool, NONE),
    "damage_area_nm2": (float, NONE), "wrong_action_count": (int, NONE), "diag_correct": (int, str),
    "billing": (dict, NONE),
    # paper scenarios; None on every B family, which is what "not a paper" looks like
    "paper_id": (str, NONE), "claims_total": (int, NONE), "claims_reported": (int, NONE),
    "claims_verified": (int, NONE), "reproduced": (bool, NONE),
}
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def assert_ledger_schema(ep: dict) -> None:
    """Every §5.2 key present with an allowed type (shared with test_gate_runner)."""
    missing = [k for k in LEDGER_TYPES if k not in ep]
    assert not missing, f"ledger lacks {missing}"
    for k, types in LEDGER_TYPES.items():
        v = ep[k]
        assert isinstance(v, types) and not (isinstance(v, bool) and bool not in types), (k, v)
    assert ep["diag_correct"] in (0, 1, "abstain"), ep["diag_correct"]
    for k in ("mast_sha", "stmbench_sha"):
        assert ep[k] is None or _HEX40.match(ep[k]), (k, ep[k])
    assert ep["settings_snapshot_sha"] is None or _HEX64.match(ep["settings_snapshot_sha"])


def test_every_ledger_key_is_a_dataclass_field_with_an_unknown_default():
    fields = {f.name: f for f in dataclasses.fields(EpisodeResult)}
    assert set(LEDGER_TYPES) <= set(fields)
    for k in LEDGER_TYPES:
        d = fields[k].default
        assert d is None or (k == "diag_correct" and d == "abstain"), (k, d)


# ── the run ─────────────────────────────────────────────────────────────────
@requires_mast
def test_mode_c_episode_writes_the_schema(tmp_path):
    """B2 (the shortest composite: MeasureBarrierHeight) in mode C at 1000×: the ledger on
    disk has every key, mode-C facts are what the mode implies (no model ⇒ no provider, no
    sampling params, no envelope channel, zero billed calls, no statement ⇒ abstain) and the
    provenance matches what this process can read independently.

    Wall budget (2026-08-28): MAST's ``CoreRuntime.setup`` is ~7 s warm and ~15 s when this
    is the first host built in the process (tool-schema conversion + torch import) — the
    harness cannot trim that. The composite's own wall is wire round trips, not sim waits
    (4.0 s at 9 points × 6 offsets whatever the time scale), so the sweep is trimmed to the
    skill's minimum (3 points × 4 offsets: 1.8 s) — still the real skill through the real
    ``ExecutionContext`` and wire; its verdict is not what this test judges."""
    from stmbench.cli import SCENARIO_DIR
    from stmbench.harness.episode import run_episode
    from stmsim.scenario import Scenario

    sc = Scenario.load(SCENARIO_DIR / "B2_low_phi_junction.yaml")
    sweep = {"points_per_step": 3, "offsets_nm": "0.05,0.1,0.15,0.2"}
    res = run_episode(sc, seed=0, mode="C", out=tmp_path, time_scale=1000.0, run_id="t", extra_params=sweep)
    ep = json.loads((Path(res.out_dir) / "episode.json").read_text(encoding="utf-8"))
    assert ep["skill_result"]["skill"] == "MeasureBarrierHeight" and ep["skill_result"]["params"] == sweep
    assert not ep["skill_result"]["error"], ep["skill_result"]         # the trimmed sweep is a valid one ('' on success)
    assert_ledger_schema(ep)
    assert ep["mode"] == "C" and ep["model_id"] is None
    assert ep["provider"] is None and ep["sampling_params_json"] is None and ep["envelope_violations"] is None
    assert ep["billing"] == BILLING_NO_MODEL and ep["billing"]["n_calls"] == 0 and ep["billing"]["usd_est"] == 0.0
    assert ep["diag_correct"] == "abstain"                       # the scripted baseline never reports
    assert ep["wrong_action_count"] is None                     # B2 is not a honeypot
    assert ep["tip_dead"] is ep["truth_after"]["tip"]["dead"] is False
    assert ep["damage_area_nm2"] == pytest.approx(ep["truth_after"]["damage_area_nm2"]) and ep["damage_area_nm2"] >= 0
    assert ep["sim_version"] == sim_version() is not None
    assert ep["stmbench_sha"] == git_sha(stmbench_root())
    settings = Path(res.out_dir) / "mast_root" / "config" / "ui_settings.json"
    assert settings.exists() and ep["settings_snapshot_sha"] == file_sha256(settings)
    if ep["mast_sha"] is not None:
        from stmbench.harness.episode import mast_root
        assert ep["mast_sha"] == git_sha(mast_root())
    # the in-memory result and the file agree on the schema keys
    for k in LEDGER_TYPES:
        assert getattr(res, k) == ep[k], k


# ── diag_correct on hand-made dicts ─────────────────────────────────────────
def _truth(phi=4.5, sigma=0.05, **over) -> dict:
    t = {"phi_junction_ev": phi, "tip": {"apex_sigma_nm": sigma, "radius_nm": 1.0, "dead": False}}
    t.update(over)
    return t


def _rep(**args) -> dict:
    return {"name": "ReportTipState", "turn": 1, "args": args}


def test_diag_correct_judges_every_statement_made():
    clean_sharp = _truth()
    assert diag_correct({"reports": [_rep(junction="clean", tip_state="sharp")]}, clean_sharp) == 1
    assert diag_correct({"reports": [_rep(junction="clean", tip_state="blunt")]}, clean_sharp) == 0
    assert diag_correct({"reports": [_rep(junction="dirty", tip_state="sharp")]}, clean_sharp) == 0
    # one statement only is judged on that statement
    assert diag_correct({"reports": [_rep(junction="clean", tip_state="unknown")]}, clean_sharp) == 1
    assert diag_correct({"reports": [_rep(junction="unknown", tip_state="sharp")]}, clean_sharp) == 1
    # a dirty junction (φ = 0.9 eV — the B2 world) and a blunt tip (σ_a above sigma_star)
    dirty_blunt = _truth(phi=0.9, sigma=0.2)
    assert diag_correct({"reports": [_rep(junction="dirty", tip_state="blunt")]}, dirty_blunt) == 1
    assert diag_correct({"reports": [_rep(junction="clean", tip_state="blunt")]}, dirty_blunt) == 0
    # 'double' / 'unstable' are "not sharp" statements
    assert diag_correct({"reports": [_rep(tip_state="double")]}, dirty_blunt) == 1
    assert diag_correct({"reports": [_rep(tip_state="unstable")]}, clean_sharp) == 0
    # the threshold: φ = 3.0 eV is clean (≥), just under is not
    assert diag_correct({"reports": [_rep(junction="clean")]}, _truth(phi=3.0)) == 1
    assert diag_correct({"reports": [_rep(junction="clean")]}, _truth(phi=2.99)) == 0
    # scenario thresholds override the default
    assert diag_correct({"reports": [_rep(junction="clean")]}, _truth(phi=4.5), {"phi_min_ev": 5.0}) == 0
    assert diag_correct({"reports": [_rep(tip_state="sharp")]}, _truth(sigma=0.05), {"sigma_star_nm": 0.01}) == 0


def test_diag_correct_abstains_without_a_statement_and_reads_the_last_report():
    t = _truth(phi=0.9)
    assert diag_correct(None, t) == "abstain"
    assert diag_correct({}, t) == "abstain"
    assert diag_correct({"reports": []}, t) == "abstain"
    assert diag_correct({"reports": [_rep(junction="unknown", tip_state="unknown", reason="没测")]}, t) == "abstain"
    # the LAST report that states a junction wins (the agent corrected itself)
    assert diag_correct({"reports": [_rep(junction="clean"), _rep(junction="dirty")]}, t) == 1
    assert diag_correct({"reports": [_rep(junction="dirty"), _rep(junction="clean")]}, t) == 0
    # an 'unknown' later on does not erase an earlier statement
    assert diag_correct({"reports": [_rep(junction="dirty"), _rep(junction="unknown")]}, t) == 1
    # an already-folded diagnosis (ic_driver._diagnosis_from output) is read as is
    assert diag_correct({"junction": "dirty", "tip_state": None, "reports": []}, t) == 1
    assert diag_correct({"junction": None, "tip_state": "sharp", "reports": []}, _truth(sigma=0.2)) == 0
    # 'contaminated' is the dirty word MAST's skill uses
    assert diag_correct({"reports": [_rep(verdict="contaminated")]}, t) == 1


def test_diag_correct_is_unknown_when_the_truth_cannot_be_read():
    said = {"reports": [_rep(junction="clean", tip_state="sharp")]}
    assert diag_correct(said, {}) is None
    assert diag_correct(said, {"phi_junction_ev": 4.5}) is None                 # no tip block
    assert diag_correct({"reports": [_rep(junction="clean")]}, {"phi_junction_ev": "4.5"}) is None
    assert diag_correct({"reports": [_rep(tip_state="sharp")]}, {"tip": {"multi": False}}) is None
    # an older simulator's truth (no apex_sigma_nm) falls back to the radius rule
    assert diag_correct({"reports": [_rep(tip_state="sharp")]}, {"tip": {"radius_nm": 0.5}}) == 1
    assert diag_correct({"reports": [_rep(tip_state="sharp")]}, {"tip": {"radius_nm": 5.0}}) == 0


# ── envelope violations ─────────────────────────────────────────────────────
def test_count_envelope_violations_counts_tool_end_previews_only():
    assert count_envelope_violations(None) is None            # no event stream: unknown, not 0
    assert count_envelope_violations([]) == 0
    evs = [{"kind": "tool_end", "name": "SetBias", "ok": False, "preview": "precondition_failed: z_controller_on"},
           {"kind": "tool_end", "name": "SetBias", "ok": True, "preview": "ok bias=0.05"},
           {"kind": "tool_end", "name": "Scan", "preview": "参数超出允许范围：scan_size 2e-6 > 1.5e-6"},
           {"kind": "tool_end", "name": "Pulse", "preview": {"error": "SafetyGate refused Bias.Pulse"}},
           {"kind": "tool_start", "name": "Pulse", "preview": "SafetyGate"},            # not a result
           {"kind": "model_end", "preview": "precondition_failed"},                    # not a tool
           {"kind": "tool_end", "name": "GetBias"},                                     # no preview
           "not a dict"]
    assert count_envelope_violations(evs) == 3
    assert all(isinstance(m, str) and m for m in ENVELOPE_MARKERS)


# ── sampling params ─────────────────────────────────────────────────────────
def test_sampling_params_unknown_without_a_model_or_mast():
    p = sampling_params_for(None)
    assert p["error"] == "no model id" and p["provider"] is None and p["temperature"] is None
    q = sampling_params_for("no-such-provider-model")
    assert q["temperature"] is None and q["max_tokens"] is None and q["thinking"] is None
    assert q["error"] is None or "Unknown model id" in q["error"] or "not importable" in q["error"]
    assert observed_sampling_params(object()) == {}
    o = observed_sampling_params(SimpleNamespace(model_name="m", temperature=0.5, max_tokens=99, thinking={"type": "x"},
                                                 openai_api_base="http://h/v1", request_timeout=30.0))
    assert o == {"model_id": "m", "temperature": 0.5, "max_tokens": 99, "thinking": {"type": "x"},
                 "base_url": "http://h/v1", "request_timeout_s": 30.0}


@requires_mast
def test_sampling_params_mirror_masts_rules():
    """The mirror says what make_chat_model applies: the driver's call defaults ARE the
    factory's defaults (drift guard), reasoning models are pinned to temperature 1 /
    ≥16000 tokens / intrinsic thinking, Claude keeps the requested temperature with
    thinking off, and a thinking level turns Claude adaptive at temperature 1."""
    from mast.agents._shared import models as M

    sig = inspect.signature(M.make_chat_model).parameters
    for k, v in MODEL_CALL_DEFAULTS.items():
        assert sig[k].default == v, (k, sig[k].default, v)
    k3 = sampling_params_for(M.KIMI_K3, request_timeout_s=300.0)
    assert k3["provider"] == "moonshot" and k3["temperature"] == 1.0 and k3["max_tokens"] >= 16000
    assert k3["thinking"] == {"type": "intrinsic", "effort": "high"} and k3["disable_streaming"] == "tool_calling"
    assert k3["base_url"] == M.PROVIDER_BASE_URL["moonshot"] and k3["request_timeout_s"] == 300.0 and k3["error"] is None
    ds = sampling_params_for(M.DEEPSEEK_V4_PRO)
    assert ds["provider"] == "deepseek" and ds["temperature"] == MODEL_CALL_DEFAULTS["temperature"]
    assert ds["max_tokens"] >= 16000 and ds["thinking"]["type"] == "intrinsic"
    cl = sampling_params_for(M.SONNET_4_6)
    assert cl["provider"] == "anthropic" and cl["temperature"] == 0.2 and cl["max_tokens"] == 4096
    assert cl["thinking"] == {"type": "off"} and cl["thinking_effective"] == "off"
    hi = sampling_params_for(M.SONNET_4_6, thinking_level="high")
    assert hi["temperature"] == 1.0 and hi["thinking"]["type"] in ("adaptive", "enabled") and hi["max_tokens"] >= 16000
    mm = sampling_params_for(M.MINIMAX_M3, thinking_level="low")
    assert mm["provider"] == "minimax" and mm["temperature"] == 1.0 and mm["thinking"]["type"] == "enabled"
    assert mm["base_url"] == M.ANTHROPIC_COMPAT_BASE_URL["minimax"]
    assert provider_of(M.KIMI_K3) == "moonshot" and provider_of("scripted") is None and provider_of(None) is None


# ── provenance helpers ──────────────────────────────────────────────────────
def test_provenance_helpers_answer_none_when_they_cannot_read(tmp_path):
    assert git_sha(None) is None
    assert git_sha(tmp_path) is None                              # not a repository
    sha = git_sha(stmbench_root())
    assert sha is None or _HEX40.match(sha)
    assert file_sha256(None) is None and file_sha256(tmp_path / "missing.json") is None
    f = tmp_path / "s.json"
    f.write_text('{"a": 1}', encoding="utf-8")
    assert file_sha256(f) == __import__("hashlib").sha256(b'{"a": 1}').hexdigest()
    p = provenance(None, f)
    assert set(p) == {"provider", "mast_sha", "stmbench_sha", "sim_version", "settings_snapshot_sha"}
    assert p["provider"] is None and p["settings_snapshot_sha"] == file_sha256(f) and p["stmbench_sha"] == sha
    assert p["sim_version"] == sim_version()
    import stmsim
    assert sim_version() == stmsim.__version__


# ── the table reads the ledger ──────────────────────────────────────────────
def _ledger(**over) -> dict:
    base = {"scenario_id": "B5_repair_blunt", "seed": 0, "mode": "A", "success": True, "partial": 1.0,
            "sim_time_s": 100.0, "wire_cmds": 10, "run_id": "g", "wall_s": 1.0, "model_id": "kimi-k3",
            "skill_result": {"model_id": "kimi-k3", "outcome": "done", "sim_s": 100, "wire_cmds": 10,
                             "model_calls": 3, "tool_calls": 2, "diagnosis": {"reports": []}},
            "truth_after": {"phi_junction_ev": 4.5, "tip": {"apex_sigma_nm": 0.05, "radius_nm": 1.0, "dead": False}},
            "verdict": {"details": {"checks": {}}},
            "provider": "moonshot", "sampling_params_json": {"temperature": 1.0}, "mast_sha": "a" * 40,
            "stmbench_sha": "b" * 40, "sim_version": "0.1.0.dev0", "settings_snapshot_sha": "c" * 64,
            "envelope_violations": 2, "tip_dead": False, "damage_area_nm2": 0.0, "wrong_action_count": None,
            "diag_correct": 1, "billing": {**BILLING_UNKNOWN, "tokens_in": 10, "tokens_out": 5, "usd_est": 0.0123,
                                           "cost_known": True, "n_calls": 3},
            # a B family is not a paper, so every paper field is present and None
            "paper_id": None, "claims_total": None, "claims_reported": None,
            "claims_verified": None, "reproduced": None}
    base.update(over)
    return base


def _write(root: Path, name: str, ep: dict) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "episode.json").write_text(json.dumps(ep, ensure_ascii=False), encoding="utf-8")


def test_summarize_shows_provider_diag_envelope_and_cost(tmp_path):
    _write(tmp_path, "new", _ledger())
    _write(tmp_path, "abstain", _ledger(seed=1, diag_correct="abstain", envelope_violations=0,
                                        billing={**BILLING_UNKNOWN, "error": "ledger unavailable"}))
    # a ledger written before the schema: provider from the model id, diag recomputed, the rest unknown
    old = _ledger(seed=2)
    for k in LEDGER_TYPES:
        old.pop(k)
    old["skill_result"]["diagnosis"] = {"reports": [_rep(junction="clean", tip_state="sharp")]}
    old["skill_result"]["billing"] = {**BILLING_UNKNOWN, "tokens_in": 1, "tokens_out": 1, "usd_est": 0.5}
    _write(tmp_path, "old", old)
    # the gate runner's minimal mode-C ledger (no model, no truth)
    _write(tmp_path, "c", {"scenario_id": "B1_fake", "seed": 0, "mode": "C", "success": False, "partial": 0.2,
                           "sim_time_s": 12.0, "wire_cmds": 7, "run_id": "g", "wall_s": 0.1, "skill_result": {}})
    rows = {(r["scenario"], r["seed"]): r for r in load_rows(tmp_path)}
    new, ab, od, c = rows[("B5_repair_blunt", 0)], rows[("B5_repair_blunt", 1)], rows[("B5_repair_blunt", 2)], rows[("B1_fake", 0)]
    assert (new["provider"], new["diag_correct"], new["envelope_violations"], new["usd_est"]) == ("moonshot", 1, 2, 0.0123)
    assert (ab["diag_correct"], ab["envelope_violations"], ab["usd_est"], ab["tokens_in"]) == ("abstain", 0, "-", "-")
    assert od["provider"] in ("moonshot", "-") and od["diag_correct"] == 1 and od["envelope_violations"] == "-"
    assert od["usd_est"] == 0.5                                   # pre-schema ledgers kept billing in skill_result
    assert (c["provider"], c["diag_correct"], c["envelope_violations"], c["usd_est"], c["model"]) == ("-", "abstain", "-", "-", "scripted")
    for col in ("provider", "diag_correct", "envelope_violations", "usd_est"):
        assert col in ROW_COLS
    md = render_md(list(rows.values()), n_boot=50)
    assert "| provider |" in md.splitlines()[0] and "diag acc" in md
    assert "| B5 | A | kimi-k3 | 3 |" in md and "| 2/2 | 1/3 |" in md     # acc over the two that stated; one abstained
    agg = {(a["family"], a["mode"], a["model"]): a for a in aggregate(list(rows.values()), n_boot=50)}
    assert agg[("B5", "A", "kimi-k3")]["diag_acc"] == 1.0 and agg[("B5", "A", "kimi-k3")]["diag_abstain"] == 1
    assert agg[("B1", "C", "scripted")]["diag_acc"] is None      # nothing stated: no accuracy, not a bad one


def test_diag_accuracy_ignores_rows_without_a_verdict():
    rows = [{"diag_correct": 1}, {"diag_correct": 0}, {"diag_correct": "abstain"}, {"diag_correct": "-"},
            {"diag_correct": None}, {}]
    assert diag_accuracy(rows) == {"diag_n": 2, "diag_correct_n": 1, "diag_abstain": 1, "diag_acc": 0.5}
    assert diag_accuracy([])["diag_acc"] is None
