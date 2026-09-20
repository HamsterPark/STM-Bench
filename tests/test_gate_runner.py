"""The gate matrix runner (cli gate) and the billing read-back from MAST's usage ledger."""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from stmbench import cli
from stmbench.cli import list_scenarios, main, plan_gate, run_gate, scenario_families
from stmbench.harness.episode import (BILLING_UNKNOWN, episode_id_for, ledger_db_path, read_billing,
                                      resolve_policy, run_dir)
from tests.conftest import MASTV2, requires_mast

REPO = Path(__file__).resolve().parents[1]


# ── the plan ────────────────────────────────────────────────────────────────
def test_families_come_from_the_scenario_listing():
    fams = scenario_families()
    # B… = the instrument-operation regression families, P… = one per reproduced paper
    assert fams and all(f[0] in "BP" for f in fams)
    assert {p.stem.split("_", 1)[0] for p in list_scenarios(fams[:1])} == {fams[0]}
    first = list_scenarios()[0].stem
    assert [p.stem for p in list_scenarios(None, first)] == [first]            # substring
    assert [p.stem for p in list_scenarios(None, first[:4] + "*")]              # glob
    assert list_scenarios(None, "no-such-scenario") == []


def test_plan_is_the_full_matrix_with_status_and_policy(tmp_path):
    fams = scenario_families()[:2]
    n_sc = len(list_scenarios(fams))
    plan = plan_gate(out=tmp_path, gate_id="g", families=fams, modes=["C", "A"], seeds=[0, 1],
                     models=["m1", "m2"], python="py")
    assert len(plan) == n_sc * (1 + 2) * 2                     # C ignores models
    c_entries = [e for e in plan if e["mode"] == "C"]
    assert all(e["model"] is None and e["policy"] is None and "--model" not in e["cmd"] for e in c_entries)
    a_entries = [e for e in plan if e["mode"] == "A"]
    assert {e["model"] for e in a_entries} == {"m1", "m2"}
    assert all(e["policy"] == resolve_policy(e["family"], "A", None) for e in a_entries)
    e = a_entries[0]
    assert e["cmd"][:4] == ["py", "-m", "stmbench.cli", "run"] and "--run" in e["cmd"] and "g" in e["cmd"]
    assert Path(e["run_dir"]) == run_dir(tmp_path, e["scenario_id"], e["seed"], "A", model_id=e["model"],
                                         policy=e["policy"], run_id="g")
    assert all(e["status"] == "todo" for e in plan) and not any(tmp_path.iterdir())   # planning touches nothing
    # a ledger already there → done; a run dir without one → stale
    done, stale = plan[0], plan[1]
    Path(done["run_dir"]).mkdir(parents=True)
    (Path(done["run_dir"]) / "episode.json").write_text("{}", encoding="utf-8")
    Path(stale["run_dir"]).mkdir(parents=True)
    again = plan_gate(out=tmp_path, gate_id="g", families=fams, modes=["C", "A"], seeds=[0, 1],
                      models=["m1", "m2"], python="py")
    st = {(e["scenario_id"], e["seed"], e["mode"], e["model"]): e["status"] for e in again}
    assert st[(done["scenario_id"], done["seed"], done["mode"], done["model"])] == "done"
    assert st[(stale["scenario_id"], stale["seed"], stale["mode"], stale["model"])] == "stale"
    assert list(st.values()).count("todo") == len(plan) - 2


def test_plan_refuses_llm_mode_without_models_and_unknown_mode(tmp_path):
    with pytest.raises(SystemExit):
        plan_gate(out=tmp_path, gate_id="g", families=None, modes=["A"], seeds=[0], models=[])
    with pytest.raises(SystemExit):
        plan_gate(out=tmp_path, gate_id="g", families=None, modes=["Z"], seeds=[0], models=["m"])


def test_honeypot_policy_lands_in_the_run_dir_only_when_b8_exists(tmp_path):
    fams = scenario_families()
    if "B8" not in fams:
        pytest.skip("no B8 scenario in the listing")
    plan = plan_gate(out=tmp_path, gate_id="g", families=["B8"], modes=["B0"], seeds=[3], models=["m"])
    assert plan and all(e["policy"] == "honeypot" and "B0_m_honeypot" in e["run_dir"] for e in plan)
    assert episode_id_for(plan[0]["scenario_id"], 3, "B0", "m") == f"{plan[0]['scenario_id']}.seed3.B0.m"


def test_dry_run_prints_the_plan_and_creates_nothing(tmp_path, capsys):
    first = list_scenarios()[0].stem
    rc = main(["gate", "--dry-run", "--out", str(tmp_path), "--modes", "C", "--seeds", "0,1",
               "--scenario-filter", first, "--gate", "g", "-v"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "2 episodes" in out and out.count("[todo ]") == 2 and first in out
    assert "stmbench.cli run" in out                               # -v shows the command
    assert not any(tmp_path.iterdir())
    assert main(["gate", "--dry-run", "--out", str(tmp_path), "--scenario-filter", "no-such"]) == 2


# ── running: subprocess per episode, resume, stale ──────────────────────────
FAKE_CHILD = '''
import json, sys, pathlib
rd = pathlib.Path(sys.argv[1]); rc = int(sys.argv[2]); ok = sys.argv[3] == "1"
rd.mkdir(parents=True, exist_ok=False)
print("child says hello")
if rc == 0:
    (rd / "episode.json").write_text(json.dumps({"scenario_id": sys.argv[4], "seed": int(sys.argv[5]),
        "mode": "C", "success": ok, "partial": 1.0 if ok else 0.2, "sim_time_s": 12.0, "wire_cmds": 7,
        "run_id": rd.name, "wall_s": 0.1, "skill_result": {}}), encoding="utf-8")
sys.exit(rc)
'''


def _fake_plan(tmp_path, n_scen=2, seeds=(0, 1)):
    child = tmp_path / "fake_child.py"
    child.write_text(FAKE_CHILD, encoding="utf-8")
    plan = []
    for i in range(n_scen):
        sid = f"B{i + 1}_fake"
        for seed in seeds:
            rd = run_dir(tmp_path / "out", sid, seed, "C", model_id=None, policy=None, run_id="g")
            plan.append({"scenario_id": sid, "family": sid[:2], "path": "-", "seed": seed, "mode": "C",
                         "model": None, "policy": None, "run_dir": str(rd), "status": "todo",
                         "cmd": [sys.executable, str(child), str(rd), "0", "1", sid, str(seed)]})
    return plan


def test_run_gate_spawns_children_in_parallel_and_skips_done(tmp_path):
    plan = _fake_plan(tmp_path)
    # entry 0: already done — must be left untouched
    Path(plan[0]["run_dir"]).mkdir(parents=True)
    ledger = Path(plan[0]["run_dir"]) / "episode.json"
    ledger.write_text('{"scenario_id": "B1_fake", "seed": 0, "success": false, "run_id": "g"}', encoding="utf-8")
    plan[0]["status"] = "done"
    # entry 1: stale run dir (a crashed child) with a leftover file
    Path(plan[1]["run_dir"]).mkdir(parents=True)
    (Path(plan[1]["run_dir"]) / "junk.txt").write_text("x", encoding="utf-8")
    plan[1]["status"] = "stale"
    # entry 3: the child fails
    plan[3]["cmd"][3] = "3"
    lines = []
    t0 = time.perf_counter()
    res = run_gate(plan, parallel=3, log=lines.append)
    assert time.perf_counter() - t0 < 15
    assert [r["scenario_id"] + str(r["seed"]) for r in res] == [e["scenario_id"] + str(e["seed"]) for e in plan]
    assert res[0]["rc"] is None and res[0]["ledger"] and ledger.read_text(encoding="utf-8").startswith('{"scenario_id": "B1_fake", "seed": 0, "success": false')
    assert res[1]["rc"] == 0 and res[1]["ledger"] and "moved_stale_to" in res[1]
    assert Path(res[1]["moved_stale_to"]).joinpath("junk.txt").exists()
    assert not (Path(plan[1]["run_dir"]) / "junk.txt").exists()
    assert res[2]["rc"] == 0 and res[2]["ledger"]
    assert res[3]["rc"] == 3 and not res[3]["ledger"]
    assert Path(res[3]["log"]).read_text(encoding="utf-8").count("child says hello") == 1
    assert Path(res[2]["log"]).read_text(encoding="utf-8").startswith("# ")     # the command is logged
    assert sum(l.startswith("[ok]") for l in lines) == 2 and sum(l.startswith("[FAIL]") for l in lines) == 1
    # resume: plan again from the same tree → three done, one todo (the failed one wrote no ledger)
    st = {Path(e["run_dir"]).exists() and (Path(e["run_dir"]) / "episode.json").exists() for e in plan}
    assert st == {True, False}


def test_run_gate_child_that_exits_0_without_a_ledger_is_a_failure(tmp_path):
    plan = _fake_plan(tmp_path, n_scen=1, seeds=(0,))
    child = tmp_path / "silent.py"
    child.write_text("import sys, pathlib; pathlib.Path(sys.argv[1]).mkdir(parents=True); sys.exit(0)\n",
                     encoding="utf-8")
    plan[0]["cmd"] = [sys.executable, str(child), plan[0]["run_dir"]]
    res = run_gate(plan, parallel=1, log=lambda s: None)
    assert res[0]["rc"] == 0 and not res[0]["ledger"] and "no episode.json" in res[0]["error"]


def test_run_gate_timeout_is_reported_not_raised(tmp_path):
    plan = _fake_plan(tmp_path, n_scen=1, seeds=(0,))
    child = tmp_path / "slow.py"
    child.write_text("import time; time.sleep(30)\n", encoding="utf-8")
    plan[0]["cmd"] = [sys.executable, str(child)]
    res = run_gate(plan, parallel=1, timeout_s=2.0, log=lambda s: None)
    assert res[0]["rc"] == -1 and "timeout" in res[0]["error"] and not res[0]["ledger"]


@requires_mast
def test_gate_mode_c_one_tiny_scenario_end_to_end(tmp_path, monkeypatch, capsys):
    """A real child: ``python -m stmbench.cli run`` on the smallest scenario, mode C, no LLM;
    the second invocation finds the ledger and skips (resumable)."""
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(MASTV2), str(REPO)]))
    out = tmp_path / "gate"
    args = ["gate", "--out", str(out), "--modes", "C", "--seeds", "0", "--scenario-filter", "B2_",
            "--time-scale", "1000", "--gate", "t", "--parallel", "1"]
    rc = main(args)
    text = capsys.readouterr().out
    assert rc == 0, text
    plan = plan_gate(out=out, gate_id="t", families=None, modes=["C"], seeds=[0], models=[], scenario_filter="B2_")
    assert len(plan) == 1 and plan[0]["status"] == "done"
    ep = json.loads((Path(plan[0]["run_dir"]) / "episode.json").read_text(encoding="utf-8"))
    assert ep["mode"] == "C" and ep["run_id"] == "t" and ep["seed"] == 0
    # the child wrote the §5.2 ledger schema (tests/test_ledger_schema.py owns the full check)
    from tests.test_ledger_schema import assert_ledger_schema
    assert_ledger_schema(ep)
    assert ep["provider"] is None and ep["diag_correct"] == "abstain" and ep["billing"]["n_calls"] == 0
    assert "| family | mode | model |" in text and "| B2 | C | scripted | 1 |" in text
    assert main(args + ["--dry-run"]) == 0
    assert "[done ]" in capsys.readouterr().out


# ── billing ─────────────────────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE usage_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, kind TEXT NOT NULL, provider TEXT NOT NULL,
    model TEXT NOT NULL, source TEXT, input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0,
    chars INTEGER DEFAULT 0, seconds REAL DEFAULT 0, cost REAL DEFAULT 0, currency TEXT,
    cost_known INTEGER DEFAULT 1, meta TEXT);
"""


def _fake_ledger(path: Path, rows):
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA)
    conn.executemany("INSERT INTO usage_events (ts, kind, provider, model, source, input_tokens, output_tokens, "
                     "cost, currency, cost_known) VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def test_read_billing_sums_only_this_episode_since_its_start(tmp_path):
    db = tmp_path / "usage_ledger.sqlite"
    eid = "B5_x.seed0.A.kimi-k3"
    _fake_ledger(db, [
        (1000.0, "llm", "moonshot", "kimi-k3", eid, 100, 10, 0.5, "CNY", 1),          # an earlier run
        (2000.0, "llm", "moonshot", "kimi-k3", eid, 1000, 200, 3.6, "CNY", 1),
        (2001.0, "llm", "moonshot", "kimi-k3", eid, 500, 100, 1.8, "CNY", 1),
        (2002.0, "llm", "moonshot", "kimi-k3", "someone-else", 9999, 9999, 99.0, "CNY", 1),
    ])
    b = read_billing(eid, since=1500.0, db_path=db, fx_usd_to_cny=7.2)
    assert (b["tokens_in"], b["tokens_out"], b["n_calls"]) == (1500, 300, 2)
    assert b["usd_est"] == pytest.approx(5.4 / 7.2) and b["cost_known"] is True
    assert b["currencies"]["CNY"]["cost"] == pytest.approx(5.4) and b["error"] is None
    assert b["ledger"] == str(db)
    all_runs = read_billing(eid, db_path=db, fx_usd_to_cny=7.2)
    assert all_runs["tokens_in"] == 1600 and all_runs["n_calls"] == 3


def test_read_billing_unknown_stays_unknown(tmp_path):
    db = tmp_path / "usage_ledger.sqlite"
    missing = read_billing("e", db_path=db)
    assert missing == {**BILLING_UNKNOWN, "error": "ledger unavailable"}
    _fake_ledger(db, [(1.0, "llm", "p", "m", "other", 5, 5, 0.1, "USD", 1)])
    none = read_billing("e", db_path=db)
    assert none["tokens_in"] is None and none["usd_est"] is None and none["cost_known"] is False
    assert none["n_calls"] == 0 and "no ledger rows" in none["error"]
    # not a ledger at all
    (tmp_path / "junk.sqlite").write_bytes(b"not a database")
    bad = read_billing("e", db_path=tmp_path / "junk.sqlite")
    assert bad["tokens_in"] is None and "unreadable" in bad["error"]
    assert ledger_db_path(tmp_path / "nowhere") is None


def test_read_billing_cost_known_needs_price_and_fx(tmp_path):
    db = tmp_path / "l.sqlite"
    _fake_ledger(db, [
        (1.0, "llm", "anthropic", "claude", "e", 10, 5, 0.02, "USD", 1),
        (2.0, "llm", "anthropic", "claude", "e", 10, 5, 0.00, "USD", 0),       # MAST could not price it
    ])
    b = read_billing("e", db_path=db)
    assert b["tokens_in"] == 20 and b["usd_est"] == pytest.approx(0.02) and b["cost_known"] is False
    _fake_ledger(tmp_path / "eur.sqlite", [(1.0, "llm", "x", "m", "e", 1, 1, 1.0, "EUR", 1)])
    eur = read_billing("e", db_path=tmp_path / "eur.sqlite")
    assert eur["tokens_in"] == 1 and eur["usd_est"] is None and eur["cost_known"] is False
    _fake_ledger(tmp_path / "cny.sqlite", [(1.0, "llm", "x", "m", "e", 1, 1, 7.2, "CNY", 1)])
    cny = read_billing("e", db_path=tmp_path / "cny.sqlite", fx_usd_to_cny=7.2)
    assert cny["usd_est"] == pytest.approx(1.0) and cny["cost_known"] is True and cny["fx_usd_to_cny"] == 7.2


def test_ledger_db_path_falls_back_to_the_isolated_project_root(tmp_path, monkeypatch):
    try:
        from mast.billing import ledger as mod
    except Exception:  # noqa: BLE001
        mod = None
    if mod is not None:
        monkeypatch.setattr(mod, "_LEDGER", None)
    assert ledger_db_path(tmp_path) is None
    p = tmp_path / "mast_root" / "experiments" / "usage_ledger.sqlite"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"")
    assert ledger_db_path(tmp_path) == p


@requires_mast
def test_read_billing_through_masts_own_ledger(tmp_path):
    """The production writer (UsageLedger.record / LlmUsageCallback's record_llm) and our
    reader agree on the schema — the fake above is not the only witness."""
    from mast.billing import ledger as mod
    from mast.billing.capture import record_llm

    real = mod.UsageLedger(tmp_path / "usage_ledger.sqlite")
    prev = getattr(mod, "_LEDGER", None)
    mod.set_ledger_for_test(real)
    try:
        eid = episode_id_for("B5_repair_blunt", 1, "A", "kimi-k3")
        record_llm(model="kimi-k3", input_tokens=1200, output_tokens=300, source=eid)
        record_llm(model="kimi-k3", input_tokens=800, output_tokens=100, source=eid)
        record_llm(model="kimi-k3", input_tokens=5, output_tokens=5, source="not-this-episode")
        assert ledger_db_path() == real._path
        b = read_billing(eid)
        assert (b["tokens_in"], b["tokens_out"], b["n_calls"]) == (2000, 400, 2)
        assert b["error"] is None and b["currencies"]
        if b["cost_known"]:
            assert b["usd_est"] is not None and b["usd_est"] >= 0.0
    finally:
        mod.set_ledger_for_test(prev)
        real.close()
