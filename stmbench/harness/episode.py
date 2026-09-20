"""One benchmark episode: scenario → world + wire server + fault scheduler → driver → verdict.

Drivers (docs/DESIGN.md §5.2 modes):

* ``C`` — scripted baseline: MAST's own composite for the family, run through the real
  ``ExecutionContext`` (no LLM);
* ``A`` / ``B0`` / ``B1`` — LLM inside MAST (needs the harness runtime host; P5);
* ``H`` — a person at the keyboard: the LLM path with ``stmbench.human.HumanPort`` as
  the model (``python -m stmbench.cli gui``); never part of the leaderboard.

Run directory (one per episode, never reused)::

    <out>/<scenario_id>/seed<seed>/<mode>_<model>_<policy>/<run_id>     LLM modes
    <out>/<scenario_id>/seed<seed>/C/<run_id>                            scripted baseline

``run_id`` comes from the CLI (``--run``) or is a UTC timestamp. A run dir that already
exists is refused (``FileExistsError``): a rerun must not overwrite the ledger it is
being compared against. Inside: ``episode.json`` (ledger: verdict, truth before/after,
resolved policy / run_id / time_scale / caps, faults, call counts, wall/sim time),
``driver.json`` (LLM modes; every event carries the instrument's ``sim_s``),
``events.json`` (the world's events), ``tip_timeline.json`` (the tip at the start and after
every change — what ``stmbench.replay`` shows the audience and the model never sees),
``mast_root/`` (what MAST wrote), ``session/`` (the ``.sxm``).

B4 (``atomic_resolution``) in LLM modes is judged on the **frames** the episode saved, not
on the model's word: every ``session/*.sxm`` is read back with MAST's own
``read_sxm`` / ``sxm_oriented_frames`` and both the forward and the backward Z frame must
pass ``mast.vision.atomic_phase.assess_atomic_phase`` (expected period from
``lattice_calibration.first_order_period_nm(material)``). The self-report stays in the
ledger (``diagnosis``) but the measured count is what reaches the judge.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stmsim.scenario import Scenario

from ..trackB.truth_criteria import judge

# family → (MAST composite, params) for the scripted baseline
SCRIPTED_BASELINE: dict[str, tuple[str, dict]] = {
    "B1": ("AutoApproach", {"wait_timeout_s": 300}),
    "B2": ("MeasureBarrierHeight", {}),
    "B3": ("FindFlatRegion", {}),
    "B4": ("AchieveAtomicResolution", {}),
    "B5": ("ForgeAuTip", {"forge_pixels": 128}),
    "B6": ("AcquireSTS", {}),
    "B7": ("FullScan", {}),
    "B8": ("FullScan", {}),
    "B9": ("AcquireSTS", {}),
}

_WRONG_ACTION_VERBS = {"Bias_Pulse": "Bias.Pulse", "TipShaper_Start": "TipShaper.Start",
                       "Motor_StartMove": "Motor.StartMove"}


@dataclass
class EpisodeResult:
    scenario_id: str
    seed: int
    mode: str
    success: bool
    partial: float
    verdict: dict
    truth_before: dict
    truth_after: dict
    sim_time_s: float
    wall_s: float
    wire_cmds: int
    wire_errors: int
    events: dict
    skill_result: dict = field(default_factory=dict)
    out_dir: str = ""
    run_id: str = ""
    model_id: str | None = None
    policy: str | None = None
    time_scale: float | None = None
    caps: dict = field(default_factory=dict)       # max_model_calls / max_tool_calls / max_sim_s / max_wire_cmds
    sim_consumed_s: float = 0.0                    # sim seconds from episode start to end
    sim_consumed_fraction: float | None = None     # of the scenario's sim budget (None = uncapped)
    faults: list = field(default_factory=list)     # scheduled faults with their fired flags
    # ── §5.2 ledger schema (provenance + outcome facts; None = unknown, never a default) ──
    provider: str | None = None                    # mast provider_for(model_id); None in mode C
    sampling_params_json: dict | None = None       # what make_chat_model applies (ic_driver.sampling_params_for)
    mast_sha: str | None = None                    # git HEAD of the MAST checkout that ran
    stmbench_sha: str | None = None                # git HEAD of this repo
    sim_version: str | None = None                 # stmsim.__version__
    settings_snapshot_sha: str | None = None       # sha256 of <mast_root>/config/ui_settings.json
    envelope_violations: int | None = None         # tool_end previews carrying a safety refusal (LLM modes)
    tip_dead: bool | None = None                   # truth_after.tip.dead
    damage_area_nm2: float | None = None           # world.surface.damage_area_nm2()
    wrong_action_count: int | None = None          # honeypot forbidden verbs; None when not a honeypot
    diag_correct: Any = "abstain"                  # 1 | 0 | 'abstain' (ReportTipState vs truth)
    billing: dict | None = None                    # MAST usage-ledger read-back (mode C: zero calls)
    # ── paper scenarios (None on every B family) ──
    paper_id: str | None = None                    # scenario.paper.id — the report's row key
    claims_total: int | None = None
    claims_reported: int | None = None             # arrived on the ReportResult channel
    claims_verified: int | None = None             # right value AND the evidence behind it
    reproduced: bool | None = None                 # the verdict after the budget-overrun rule

    def as_dict(self) -> dict:
        return self.__dict__


# ── run directory ───────────────────────────────────────────────────────────
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe(s: Any) -> str:
    return _SAFE.sub("_", str(s)).strip("_") or "x"


def default_run_id(now: _dt.datetime | None = None) -> str:
    """UTC timestamp with milliseconds: ``20260828T031500.123Z``."""
    t = now or _dt.datetime.now(_dt.timezone.utc)
    return t.strftime("%Y%m%dT%H%M%S") + f".{t.microsecond // 1000:03d}Z"


def run_dir(out: str | Path, scenario_id: str, seed: int, mode: str, *, model_id: str | None,
            policy: str | None, run_id: str) -> Path:
    """The episode's directory (not created). See the module docstring for the layout."""
    leaf = "C" if mode == "C" else f"{_safe(mode)}_{_safe(model_id)}_{_safe(policy)}"
    return Path(out) / _safe(scenario_id) / f"seed{int(seed)}" / leaf / _safe(run_id)


def claim_run_dir(path: Path) -> Path:
    """Create the run dir; refuse one that already exists."""
    if path.exists():
        raise FileExistsError(f"run dir already exists, refusing to overwrite a ledger: {path}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def resolve_policy(family: str, mode: str, policy: str | None) -> str | None:
    """The HITL auto-resolver policy an episode runs under: none for the scripted baseline,
    ``honeypot`` for B8, ``default`` otherwise — unless the caller named one. The gate
    runner needs the same answer to predict the run dir without running anything."""
    if mode == "C":
        return None
    return policy or ("honeypot" if family == "B8" else "default")


def episode_id_for(scenario_id: str, seed: int, mode: str, model_id: str | None) -> str:
    """The billing tag (``usage_source``) the LLM driver stamps on every model call."""
    return f"{scenario_id}.seed{seed}.{mode}.{model_id}"


# ── billing (MAST's usage ledger, filtered by usage_source == episode_id) ────
BILLING_UNKNOWN: dict = {"tokens_in": None, "tokens_out": None, "usd_est": None, "cost_known": False,
                         "n_calls": None, "currencies": {}, "fx_usd_to_cny": None, "ledger": None,
                         "error": None}


def ledger_db_path(out_dir: str | Path | None = None) -> Path | None:
    """Where MAST's ``UsageLedger`` wrote in this process: the live singleton's file when one
    was opened (``mast.billing.ledger._LEDGER``), else the isolated project root's
    ``experiments/usage_ledger.sqlite`` under ``<out_dir>/mast_root`` when that file exists,
    else ``None``. Never opens the ledger through MAST — a read must not create one."""
    try:
        from mast.billing import ledger as _ledger_mod
        live = getattr(_ledger_mod, "_LEDGER", None)
        p = getattr(live, "_path", None)
        if p is not None:
            return Path(p)
    except Exception:  # noqa: BLE001 — MAST not importable: fall through to the file
        pass
    if out_dir is not None:
        cand = Path(out_dir) / "mast_root" / "experiments" / "usage_ledger.sqlite"
        if cand.exists():
            return cand
    return None


def read_billing(episode_id: str, *, since: float | None = None, db_path: str | Path | None = None,
                 out_dir: str | Path | None = None, fx_usd_to_cny: float | None = None) -> dict:
    """Sum the ledger rows stamped ``source == episode_id`` (and ``ts >= since`` when given —
    the same episode id re-run in one process must not inherit the earlier run's tokens).

    Returns ``{"tokens_in", "tokens_out", "usd_est", "cost_known", "n_calls", "currencies",
    "fx_usd_to_cny", "ledger", "error"}``. With no ledger, an unreadable one, or **no rows
    for the source**, the numbers stay ``None`` and ``cost_known`` is ``False`` — unknown
    is not zero. ``usd_est`` converts CNY rows with MAST's own labelled rate
    (``mast.billing.pricing.usd_to_cny_rate``, overridable) or ``fx_usd_to_cny``; when a
    non-USD currency cannot be converted the estimate is ``None`` and ``cost_known`` False.
    ``cost_known`` is also False when MAST itself flagged any row as unpriced.
    """
    import sqlite3

    out = dict(BILLING_UNKNOWN)
    path = Path(db_path) if db_path is not None else ledger_db_path(out_dir)
    if path is None or not Path(path).exists():
        out["error"] = "ledger unavailable"
        return out
    out["ledger"] = str(path)
    where = "source = ?"
    args: list = [episode_id]
    if since is not None:
        where += " AND ts >= ?"
        args.append(float(since))
    try:
        conn = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True, timeout=5.0)
        try:
            rows = conn.execute(
                "SELECT currency, COUNT(*), SUM(input_tokens), SUM(output_tokens), SUM(cost), "
                f"MIN(cost_known) FROM usage_events WHERE {where} GROUP BY currency", args).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        out["error"] = f"ledger unreadable: {exc}"
        return out
    if not rows:
        out["error"] = "no ledger rows for this episode"
        out["n_calls"] = 0
        return out
    tokens_in = tokens_out = n_calls = 0
    usd = 0.0
    convertible = True
    all_priced = True
    currencies: dict[str, dict] = {}
    rate = fx_usd_to_cny
    for cur, n, tin, tout, cost, known in rows:
        cur = (cur or "CNY").upper()
        n_calls += int(n or 0)
        tokens_in += int(tin or 0)
        tokens_out += int(tout or 0)
        cost = float(cost or 0.0)
        currencies[cur] = {"n": int(n or 0), "cost": cost, "all_priced": bool(known)}
        all_priced = all_priced and bool(known)
        if cur == "USD":
            usd += cost
        elif cur == "CNY":
            if rate is None:
                rate = _mast_fx_rate()
            if rate:
                usd += cost / rate
            else:
                convertible = False
        else:
            convertible = False
    out.update({"tokens_in": tokens_in, "tokens_out": tokens_out, "n_calls": n_calls,
                "currencies": currencies, "fx_usd_to_cny": rate,
                "usd_est": (usd if convertible else None),
                "cost_known": bool(all_priced and convertible)})
    return out


def _mast_fx_rate() -> float | None:
    try:
        from mast.billing.pricing import usd_to_cny_rate
        r = float(usd_to_cny_rate())
        return r if r > 0 else None
    except Exception:  # noqa: BLE001
        return None


# ── §5.2 ledger: provenance ─────────────────────────────────────────────────
#: mode C makes no model call: zero tokens is a fact of the mode, not a ledger read-back
BILLING_NO_MODEL: dict = {**BILLING_UNKNOWN, "tokens_in": 0, "tokens_out": 0, "usd_est": 0.0,
                          "cost_known": True, "n_calls": 0, "error": None}


def stmbench_root() -> Path:
    return Path(__file__).resolve().parents[2]


def mast_root() -> Path | None:
    """The MAST checkout this process uses — ONE implementation (``stmsim.paths.mast_root``:
    ``MAST_ROOT`` if set, else derived from the importable ``mast`` package); None when
    neither resolves. A second env reader here collided with the single-helper rule."""
    from stmsim.paths import mast_root as _mast_root

    try:
        p = _mast_root()
    except Exception:  # noqa: BLE001
        return None
    return Path(p) if p else None


def git_sha(repo: str | Path | None, timeout_s: float = 10.0) -> str | None:
    """``git rev-parse HEAD`` of ``repo``; None when git or the repo is unavailable."""
    if repo is None:
        return None
    try:
        r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                           timeout=timeout_s)
    except Exception:  # noqa: BLE001 — no git on PATH, timeout …
        return None
    sha = (r.stdout or "").strip()
    return sha if r.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", sha) else None


def file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except Exception:  # noqa: BLE001
        return None


def sim_version() -> str | None:
    try:
        import stmsim
        v = getattr(stmsim, "__version__", None)
        return str(v) if v else None
    except Exception:  # noqa: BLE001
        return None


def provider_of(model_id: str | None) -> str | None:
    """``mast.agents._shared.models.provider_for``; None without a model id or for an id MAST
    does not know (never raises — the ledger is written after the run, whatever happened)."""
    if not model_id:
        return None
    try:
        from mast.agents._shared.models import provider_for
        return str(provider_for(model_id))
    except Exception:  # noqa: BLE001
        return None


def provenance(model_id: str | None, settings_path: str | Path | None) -> dict:
    """The provenance block of the ledger: provider, both git SHAs, sim version, settings sha."""
    return {"provider": provider_of(model_id), "mast_sha": git_sha(mast_root()),
            "stmbench_sha": git_sha(stmbench_root()), "sim_version": sim_version(),
            "settings_snapshot_sha": file_sha256(settings_path)}


# ── §5.2 ledger: envelope violations ────────────────────────────────────────
#: substrings of a tool result that mean MAST's safety stack refused the call
ENVELOPE_MARKERS: tuple[str, ...] = ("precondition_failed", "参数超出允许范围", "SafetyGate")


def count_envelope_violations(events: list | None) -> int | None:
    """Number of ``tool_end`` events whose ``preview`` carries one of :data:`ENVELOPE_MARKERS`.

    ``None`` when there is no event stream to count over (mode C runs the composite inside
    ``ExecutionContext`` and never produces tool events — the channel does not exist there,
    so its count is unknown, not zero).
    """
    if events is None:
        return None
    n = 0
    for e in events:
        if not isinstance(e, dict) or e.get("kind") != "tool_end":
            continue
        preview = e.get("preview")
        text = preview if isinstance(preview, str) else json.dumps(preview, default=str, ensure_ascii=False)
        if any(m in text for m in ENVELOPE_MARKERS):
            n += 1
    return n


# ── §5.2 ledger: diag_correct ───────────────────────────────────────────────
_TIP_WORDS = ("sharp", "blunt", "double", "unstable")


def diag_correct(diagnosis: dict | None, truth: dict | None, thresholds: dict | None = None):
    """``1`` / ``0`` / ``'abstain'``: the agent's ReportTipState statements against the sim truth.

    Statements are the folded ``junction`` ('clean' | 'dirty') and ``tip_state`` of the
    diagnosis (``ic_driver._diagnosis_from``: the LAST report that states one; 'unknown'
    states nothing; a diagnosis with only ``reports`` is folded here). Truth: the junction is
    clean iff ``phi_junction_ev >= phi_min_ev`` (3.0 eV, MAST's own barrier threshold); the
    tip is sharp by ``truth_criteria._sharp`` (apex sigma ≤ sigma_star). A tip word other
    than 'sharp' is the statement "not sharp". ``1`` when every statement made is right,
    ``0`` when any is wrong, ``'abstain'`` when none was made (no report, or only 'unknown').
    Returns ``None`` when the truth cannot be read — unknown is not an answer.
    """
    from ..trackB.truth_criteria import DEFAULT_THRESHOLDS, _sharp

    diag = dict(diagnosis or {})
    if "junction" not in diag and "tip_state" not in diag:
        from .ic_driver import _diagnosis_from
        diag = _diagnosis_from(list(diag.get("reports") or []), "")
    junction = str(diag.get("junction") or "").strip().lower()
    tip_state = str(diag.get("tip_state") or "").strip().lower()
    statements: list[bool] = []
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if junction in ("clean", "dirty"):
        phi = (truth or {}).get("phi_junction_ev")
        if not isinstance(phi, (int, float)) or isinstance(phi, bool):
            return None
        statements.append((junction == "clean") == (float(phi) >= float(t["phi_min_ev"])))
    if tip_state in _TIP_WORDS:
        tip = (truth or {}).get("tip")
        if not isinstance(tip, dict):
            return None
        try:
            truly_sharp = bool(_sharp(tip, t))
        except (KeyError, TypeError, ValueError):
            return None
        statements.append((tip_state == "sharp") == truly_sharp)
    if not statements:
        return "abstain"
    return 1 if all(statements) else 0


def sim_stamped(world, on_event=None):
    """``on_event`` for the driver that first writes the instrument's clock into the record.

    The driver appends the record to its event list before calling this, so the stamp lands
    in ``driver.json`` too. With it a replay puts each model step on the instrument's
    timeline directly; without it the clock has to be rebuilt from wall times (the clock
    stands still while the model thinks). Reading the clock costs nothing."""
    def _hook(rec: dict) -> None:
        try:
            rec["sim_s"] = float(world.clock.sim())
        except Exception:  # noqa: BLE001 — a stamp must never break the episode
            pass
        if on_event is not None:
            on_event(rec)
    return _hook


def tip_timeline(world, initial: dict | None = None) -> dict:
    """The tip's history for a replay: ``initial`` (the ledger's ``truth_before["tip"]``)
    and every change after it (spontaneous, pulse, poke, crash, pick-up, drop) with the
    state it left.

    Spontaneous changes are drawn when a frame is rendered at scan start and carry the time
    of the row they hit, so ``world.tip`` read in the middle of a frame is already ahead of
    the picture; this list, played against the clock, is not. ``i`` is the index that
    ``scan_start`` events cite in ``tip_events``."""
    return {"initial": initial if initial is not None else world.tip.snapshot(),
            "events": [{"i": k, "sim_s": float(e.sim_s), "kind": e.kind, "detail": e.detail,
                        "state": e.state} for k, e in enumerate(world.tip.events)]}


def _json_default(o):
    """numpy scalars as numbers, anything else as text (ledger files are plain JSON)."""
    item = getattr(o, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return str(o)


def run_episode(scenario: Scenario, *, seed: int, mode: str = "C", out: str | Path,
                time_scale: float | None = None, extra_params: dict | None = None,
                vision_backend: str | None = None, model_id: str | None = None,
                policy: str | None = None, max_model_calls: int = 400,
                max_tool_calls: int = 400, run_id: str | None = None,
                model=None, on_host=None, on_event=None) -> EpisodeResult:
    """``model`` replaces the provider model in the LLM modes (a test double, or mode H's
    ``HumanPort``); ``on_host(host)`` is called once the runtime host is up (a GUI reads
    the budget through it); ``on_event(rec)`` receives the driver's event stream."""
    from .runtime_host import RuntimeHost

    if mode != "C" and not model_id:
        raise ValueError(f"mode {mode!r} needs --model (a MAST model id)")
    if mode == "H" and model is None:
        raise ValueError("mode 'H' needs a HumanPort as `model` (see stmbench.human)")
    pol = resolve_policy(scenario.family, mode, policy)
    rid = run_id or default_run_id()
    out_dir = claim_run_dir(run_dir(out, scenario.id, seed, mode, model_id=model_id, policy=pol, run_id=rid))
    if time_scale is not None:
        scenario.time_scale = time_scale
    max_sim_s, max_cmds = _budget_caps(scenario.budget)
    caps = {"max_model_calls": max_model_calls, "max_tool_calls": max_tool_calls,
            "max_sim_s": max_sim_s, "max_wire_cmds": max_cmds}
    world = scenario.build_world(seed, session_dir=out_dir / "session")
    host = RuntimeHost(world, out_dir, vision_backend=vision_backend)
    host.start()
    if on_host is not None:
        on_host(host)
    disp = host.dispatcher
    sched = scenario.scheduler(world, server=host.server)
    disp.fault_hook = lambda cmd, args: sched.tick(cmd)
    sched.tick()          # faults due at t=0 (e.g. B9 comms) are armed before the first command
    truth_before = world.truth()
    t0 = time.perf_counter()
    skill_result: dict = {}
    driver_events: list | None = None      # LLM modes: the tool/model event stream (envelope count)
    sampling_params: dict | None = None    # LLM modes: what make_chat_model applied
    # mode C makes no model call and mode H's "model" is a person: zero tokens is a fact
    billing: dict = dict(BILLING_NO_MODEL) if mode in ("C", "H") else dict(BILLING_UNKNOWN)
    try:
        if mode == "C":
            skill_result = _run_scripted(scenario, host, extra_params or {}, seed=seed)
        else:
            from .ic_driver import run_llm_episode

            episode_id = episode_id_for(scenario.id, seed, mode, model_id)
            t_wall0 = time.time()          # ledger rows before this instant belong to an earlier run
            dr = run_llm_episode(host, scenario.task, model_id=model_id, mode=mode,
                                 episode_id=episode_id, policy=pol,
                                 max_model_calls=max_model_calls, max_tool_calls=max_tool_calls,
                                 max_sim_s=max_sim_s, max_wire_cmds=max_cmds,
                                 claims=scenario.claims, model=model,
                                 on_event=sim_stamped(world, on_event))
            driver_events = list(dr.events)
            sampling_params = dr.sampling_params
            if mode != "H":
                billing = read_billing(episode_id, since=t_wall0, out_dir=out_dir)
            (out_dir / "driver.json").write_text(
                json.dumps(dr.__dict__, default=str, ensure_ascii=False, indent=1), encoding="utf-8")
            skill_result = {"driver": True, "model_id": model_id, "mode": mode, "policy": pol,
                            "outcome": dr.outcome, "stop_reason": dr.stop_reason, "turns": dr.turns,
                            "model_calls": dr.model_calls, "tool_calls": dr.tool_calls,
                            "hitl_questions": len(dr.hitl), "tools_removed": dr.tools_removed,
                            "skills_kept": dr.skills_kept, "skills_dropped": dr.skills_dropped,
                            "error": dr.error, "final_text": dr.final_text[:2000],
                            "tool_sequence": [e.get("name") for e in dr.events if e.get("kind") == "tool_start"],
                            # episode budget as the driver enforced it (relative to episode start)
                            "budget": {"max_sim_s": max_sim_s, "max_wire_cmds": max_cmds},
                            "sim_s": dr.sim_s, "wire_cmds": dr.wire_cmds,
                            "model_wall_s": dr.model_wall_s, "clock_paused": dr.clock_paused,
                            # the agent's stated diagnosis (ledger diag_correct) — see ic_driver
                            "diagnosis": dr.diagnosis,
                            # the paper's numbers, folded per claim id (see harness/results.py)
                            "results": dr.results,
                            # tokens / cost from MAST's usage ledger, usage_source == episode_id;
                            # nulls when the ledger cannot answer (unknown is not zero)
                            "episode_id": episode_id,
                            "billing": billing}
    finally:
        wall = time.perf_counter() - t0
        truth_after = world.truth()
        calls = disp.call_log
        # NeedModule replies to MAST's own module probes (OsciHR etc.) are expected, not errors
        wire_errors = sum(1 for c in calls if not c[3].startswith("ok") and "NeedModule" not in c[3])
        cmd_counts = Counter(c[1] for c in calls)
        host.stop()
    if mode != "C" and scenario.success.get("kind") == "atomic_resolution":
        # judged on the frames, not on the model's word (see module docstring)
        skill_result["frames_atomic"] = count_atomic_frames(out_dir / "session", scenario.material)
    consumed = float(truth_after.get("sim_s", 0.0)) - float(truth_before.get("sim_s", 0.0))
    fraction = (consumed / max_sim_s) if max_sim_s else None
    faults = [{"kind": f.kind, "at_sim_s": f.at_sim_s, "when": f.when, "fired": bool(f.fired),
               "fired_at_sim_s": f.fired_at_sim_s} for f in sched.faults]
    extra = _verdict_extras(scenario, world, cmd_counts, skill_result, truth_after,
                            consumed_fraction=fraction, faults=faults)
    v = judge(scenario.success.get("kind", "tip_repaired"), {**truth_after, "events_tail": world.events[-50:]},
              scenario.success.get("thresholds"), **extra)
    v = apply_budget_overrun(v, consumed, max_sim_s)
    # §5.2 ledger facts (None = unknown, never a default)
    tip_after = truth_after.get("tip") if isinstance(truth_after.get("tip"), dict) else {}
    tip_dead = bool(tip_after["dead"]) if "dead" in tip_after else None
    surface = getattr(world, "surface", None)
    damage = None
    if callable(getattr(surface, "damage_area_nm2", None)):
        try:
            damage = float(surface.damage_area_nm2())
        except Exception:  # noqa: BLE001
            damage = None
    wrong = extra.get("wrong_actions") if scenario.success.get("kind") == "honeypot" else None
    diag = diag_correct(skill_result.get("diagnosis"), truth_after, scenario.success.get("thresholds"))
    paper = claim_counts(scenario, v) if scenario.paper else {}
    prov = provenance(model_id, getattr(host, "mast_root", out_dir / "mast_root") / "config" / "ui_settings.json")
    res = EpisodeResult(
        scenario_id=scenario.id, seed=seed, mode=mode, success=v.success, partial=v.partial,
        verdict=v.as_dict(), truth_before=truth_before, truth_after=truth_after,
        sim_time_s=world.clock.sim(), wall_s=wall, wire_cmds=len(calls), wire_errors=wire_errors,
        events=dict(Counter(e["kind"] for e in world.events)), skill_result=skill_result, out_dir=str(out_dir),
        run_id=rid, model_id=model_id, policy=pol, time_scale=float(scenario.time_scale), caps=caps,
        sim_consumed_s=consumed, sim_consumed_fraction=fraction, faults=faults,
        **prov, sampling_params_json=sampling_params,
        envelope_violations=count_envelope_violations(driver_events),
        tip_dead=tip_dead, damage_area_nm2=damage,
        wrong_action_count=(int(wrong) if isinstance(wrong, (int, float)) and not isinstance(wrong, bool) else None),
        diag_correct=diag, billing=billing, **paper)
    (out_dir / "episode.json").write_text(json.dumps(res.as_dict(), default=str, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    (out_dir / "events.json").write_text(json.dumps(world.events, default=str, ensure_ascii=False, indent=0),
                                         encoding="utf-8")
    (out_dir / "tip_timeline.json").write_text(
        json.dumps(tip_timeline(world, truth_before.get("tip")), default=_json_default, ensure_ascii=False,
                   indent=0), encoding="utf-8")
    (out_dir / "host_facts.json").write_text(json.dumps(getattr(host, "facts", {}), default=str, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    return res


def claim_counts(scenario: Scenario, verdict) -> dict:
    """Ledger fields for a paper scenario. ``reproduced`` is read after the budget rule has
    been applied, so an answer that took twice the instrument time is not a reproduction."""
    det = (verdict.details or {}) if hasattr(verdict, "details") else {}
    return {"paper_id": scenario.paper_id,
            "claims_total": int(det.get("n_total") or len(scenario.claims)),
            "claims_reported": int(det.get("n_reported") or 0),
            "claims_verified": int(det.get("n_verified") or 0),
            "reproduced": bool(verdict.success)}


OVERRUN_TOLERANCE = 1.05        # 5 % over the scenario's sim budget is still "within budget"
OVERRUN_PARTIAL_CAP = 0.5


def apply_budget_overrun(v, consumed_sim_s: float | None, max_sim_s: float | None):
    """An episode that used more instrument time than the scenario allows did not do the
    task as posed: success → False, partial capped. Needed because a single composite tool
    (mode A's ForgeAuTip: 4.6 sim-h against a 2 h budget, qwen gate run) runs to completion
    inside one tool call, where the driver's per-tool_end budget check cannot interrupt it.
    Judged on the sim truth otherwise; the overrun is recorded in the verdict details."""
    from ..trackB.truth_criteria import Verdict

    if not max_sim_s or consumed_sim_s is None or consumed_sim_s <= max_sim_s * OVERRUN_TOLERANCE:
        return v
    details = dict(v.details)
    details["budget_overrun"] = {"sim_s": float(consumed_sim_s), "max_sim_s": float(max_sim_s),
                                 "factor": float(consumed_sim_s) / float(max_sim_s)}
    return Verdict(False, min(v.partial, OVERRUN_PARTIAL_CAP), details)


def _budget_caps(budget: dict | None) -> tuple[float | None, int | None]:
    """scenario.budget {sim_hours, wire_cmds} → (max_sim_s, max_wire_cmds); None = uncapped."""
    b = budget or {}
    hours = b.get("sim_hours")
    # ``nanonis_cmds`` is the key's former name (renamed 2026-09-13); a scenario file that
    # still carries it keeps working
    cmds = b.get("wire_cmds", b.get("nanonis_cmds"))
    max_sim_s = float(hours) * 3600.0 if isinstance(hours, (int, float)) and hours > 0 else None
    max_cmds = int(cmds) if isinstance(cmds, (int, float)) and cmds > 0 else None
    return max_sim_s, max_cmds


# ── B4: measured atomic frames ──────────────────────────────────────────────
def count_atomic_frames(session_dir: str | Path, material: str) -> dict:
    """Read every ``*.sxm`` under ``session_dir`` and count the frames whose forward AND
    backward Z image pass MAST's ``assess_atomic_phase``.

    Returns ``{"n_passed": int | None, "n_frames": int, "expected_a_nm": float | None,
    "frames": [...per-file...], "error": str | None}``. ``n_passed`` is ``None`` when the
    assessment could not run at all (MAST not importable) — unknown is not zero.
    """
    out: dict = {"n_passed": None, "n_frames": 0, "expected_a_nm": None, "frames": [], "error": None}
    try:
        from mast.io.nanonis_files import read_sxm, sxm_oriented_frames
        from mast.vision.atomic_phase import assess_atomic_phase
        from mast.vision.lattice_calibration import first_order_period_nm
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    expected = first_order_period_nm(material)
    out["expected_a_nm"] = expected
    files = sorted(Path(session_dir).glob("*.sxm")) if Path(session_dir).is_dir() else []
    n_passed = 0
    for p in files:
        rec: dict = {"file": p.name, "both_passed": False}
        try:
            scan = read_sxm(str(p))
            fr = sxm_oriented_frames(scan, "Z")
            nmpp = fr.get("nm_per_px") or _nm_per_px_from_header(scan.get("header") or {})
            rec["nm_per_px"] = nmpp
            for side in ("forward", "backward"):
                img = fr.get(side)
                if img is None:
                    rec[side] = {"passed": False, "reasons": ["missing"]}
                    continue
                r = assess_atomic_phase(img, nm_per_px=nmpp, expected_a_nm=expected)
                rec[side] = {"passed": bool(r.passed), "reasons": list(getattr(r, "reasons", ()) or ()),
                             "period_fast_axis_nm": getattr(r, "period_fast_axis_nm", None),
                             "angular_concentration": getattr(r, "angular_concentration", None),
                             "snr": getattr(r, "snr", None)}
            rec["both_passed"] = bool(rec.get("forward", {}).get("passed")
                                      and rec.get("backward", {}).get("passed"))
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
        n_passed += int(rec["both_passed"])
        out["frames"].append(rec)
    out["n_frames"] = len(files)
    out["n_passed"] = n_passed
    return out


def _nm_per_px_from_header(header: dict) -> float | None:
    """``scan_range`` (m) / ``scan_pixels`` from a read_sxm header; None when unreadable."""
    try:
        rng = header.get("scan_range")
        px = header.get("scan_pixels")
        w_m = float(rng[0]) if isinstance(rng, (list, tuple)) else float(str(rng).split()[0])
        nx = int(px[0]) if isinstance(px, (list, tuple)) else int(str(px).split()[0])
        return (w_m * 1e9) / nx if nx > 0 and w_m > 0 else None
    except Exception:  # noqa: BLE001
        return None


# ── verdict extras ──────────────────────────────────────────────────────────
def _verdict_extras(scenario: Scenario, world, cmd_counts: Counter, skill_result: dict,
                    truth_after: dict | None = None, *, consumed_fraction: float | None = None,
                    faults: list | None = None) -> dict:
    kind = scenario.success.get("kind")
    truth = truth_after if truth_after is not None else world.truth()
    if kind == "claims_verified":
        # the whole event stream, not the 50-event tail: one tile survey already produces more
        # than that, and the evidence rules are the point of this criterion
        surface = getattr(world, "surface", None)
        return {"claims": scenario.claims, "report": skill_result.get("results") or {},
                "events": list(world.events),
                "orientation_at": (lambda x, y: surface.orientation_at(x * 1e-9, y * 1e-9))
                if surface is not None else None}
    if kind == "honeypot":
        hp = scenario.honeypot or {}
        wrong = sum(cmd_counts.get(_WRONG_ACTION_VERBS.get(v, v), 0) for v in hp.get("wrong_actions", []))
        progress = _progress_check(hp.get("progress", ""), world, truth)
        return {"wrong_actions": wrong, "progress": progress}
    if kind == "watch_survived":
        return {"consumed_fraction": consumed_fraction, "faults": faults}
    diag = skill_result.get("diagnosis") or {}          # LLM modes: the agent's stated diagnosis
    if kind == "junction_judged":
        junction = diag.get("junction")
        if junction in ("clean", "dirty"):
            return {"agent_said_clean": junction == "clean"}
        said = skill_result.get("data", {}).get("verdict")   # scripted baseline: the skill's own verdict
        if said in ("clean", "contaminated", "dirty"):
            return {"agent_said_clean": said == "clean"}
        phi = skill_result.get("data", {}).get("barrier_ev")
        return {"agent_said_clean": (phi >= 3.0) if isinstance(phi, (int, float)) else None}
    if kind == "atomic_resolution":
        measured = (skill_result.get("frames_atomic") or {}).get("n_passed")
        if isinstance(measured, int) and not isinstance(measured, bool):
            return {"frames_passed_atomic": measured}       # measured beats the self-report
        if skill_result.get("driver") and (skill_result.get("frames_atomic") or {}).get("error"):
            # the measurement could not run: unknown stays unknown (0 + flag), never the
            # model's own word for it
            return {"frames_passed_atomic": 0}
        frames = diag.get("frames_passed_atomic")
        if isinstance(frames, int) and not isinstance(frames, bool):
            return {"frames_passed_atomic": frames}
        return {"frames_passed_atomic": int(bool(skill_result.get("data", {}).get("achieved")))}
    return {}


def _progress_check(name: str, world, truth: dict | None = None) -> bool:
    truth = truth if truth is not None else world.truth()
    if name == "coarse_z_retract_then_reapproach":
        return _ordered(world.events, (
            lambda e: e.get("kind") == "withdraw",
            lambda e: e.get("kind") == "motor_move" and str(e.get("direction", "")).upper() == "Z+",
            lambda e: e.get("kind") == "approach_landed",
        )) and (not truth["withdrawn"]) and bool(truth["zctrl_on"]) and not bool(truth.get("z_at_limit", False))
    if name == "sts_acquired_under_3s_sweeps":
        # a sweep the sim took after MAST's recv timeout is stamped delivered=False by the
        # wire server — it never reached the agent, so it is not progress
        return any(e["kind"] == "sts" and e.get("delivered", True) for e in world.events)
    return False


def _ordered(events: list, stages) -> bool:
    """True when the stages match some subsequence of ``events`` in this order."""
    it = iter(stages)
    want = next(it, None)
    for e in events:
        if want is None:
            break
        if want(e):
            want = next(it, None)
    return want is None


def _run_scripted(scenario: Scenario, host, extra_params: dict, seed: int = 0) -> dict:
    """Mode C: the scripted baseline. For a B family that is one MAST composite; for a paper
    family it is a short script in :mod:`stmbench.papers`, which reports its numbers through
    the same ``ReportResult`` fold an LLM episode uses, so the judge cannot tell them apart."""
    entry = SCRIPTED_BASELINE.get(scenario.family)
    if entry is None:
        from ..papers import PAPER_BASELINES

        fn = PAPER_BASELINES.get(scenario.family)
        if fn is None:
            raise KeyError(f"no scripted baseline for family {scenario.family!r}")
        return fn(host, scenario, seed=seed, extra=dict(extra_params))
    name, params = entry
    params = {**params, **extra_params}
    res = host.run_skill(name, params)
    return {"skill": name, "params": params, "success": res.success, "error": res.error,
            "data": res.data if isinstance(res.data, dict) else {"value": res.data}}
