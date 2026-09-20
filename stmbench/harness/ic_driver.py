"""Drive MAST's instrument-control agent on the simulator (DESIGN.md §5.3).

One episode = a multi-turn conversation with the IC loop over the task text, with the
FULL hardware safety stack (``CoreRuntime.build_instrument_loop_for``), the model under
test (``make_chat_model(model_id=…, usage_source=episode_id)`` so billing is per
episode), the mode's tool surface (``modes.py``) and an auto-resolver standing in for
the operator on every HITL question.

Why multi-turn: the IC prompt is written for a chat — one operator message, one reply —
and ``AgentLoop.run`` ends the moment the model answers without a tool call. Unattended,
a model that says "next I will start the overview scan" and stops would score zero for
a plan it never executed. So the driver answers every such stop with a fixed "continue"
message until the model writes ``[DONE]`` / ``[ABORT]``, the episode budget (model
calls, tool calls, turns, wall, **sim seconds, controller commands**) is spent, or the loop
ends for another reason (error / aborted / stalled). The continue text is part of the
benchmark protocol and identical for every model.

Benchmark protocol — the clock and the model
--------------------------------------------
**Sim time is paused while the model thinks.** The simulator's only time source is
``Clock.wall()`` (monotonic − t0); scan progress and slow physics derive from it. A model
that takes 40 s per reply on a slow provider would otherwise watch 800 sim-seconds of
drift go by at 20× *per turn* — a penalty on the provider's latency, not on the model's
decisions. So the model handed to the loop is wrapped in :class:`ClockPausedPort`, which
calls ``clock.pause()`` before every ``invoke`` / ``stream`` and ``clock.resume()`` after
(``clock.paused()`` context manager when the clock offers one). Sim time therefore
advances only while *tools* run — which is exactly the time a real instrument would
spend. The wrapper is a pure port (``invoke`` + ``stream``, no ``bind_tools``) so
``ic_assembly._as_port`` passes it through untouched. If the clock has no pause (older
simulator), the wrapper degrades to a pass-through and ``DriverResult.clock_paused`` says
so — the ledger must not claim a protocol it did not run.

Diagnosis channel (ledger ``diag_correct``)
-------------------------------------------
The harness injects ONE tool of its own into the loop in every mode, after assembly:
``ReportTipState(junction, atomic_resolution, tip_state, reason)``. It records the
statement into ``DriverResult.diagnosis["reports"]`` and answers with a short ack; it
never touches the instrument. The judge reads *only* this channel (``episode.py``
``_verdict_extras``) — nothing is parsed out of free text, and no report ⇒ ``None``
(unknown is not an answer). Older name guesses (``report_tip_state`` …) issued as
non-existent tools are still captured from ``tool_start`` events for the ledger.

Sim / command budgets (scenario YAML ``budget: {sim_hours, wire_cmds}``) are polled on
every ``tool_end`` event, not only between turns: a single tool call can burn hours of sim
time (a long scan, an approach), so waiting for the model to stop talking would let a
model overrun the budget by a whole turn. On overrun the driver sets ``ctx.abort`` —
MAST's loop checks it at the top of every hop and before every tool in a batch — so the
loop returns ``aborted`` at the next hop and the driver records ``budget_sim`` /
``budget_cmds`` instead. Both budgets count from the episode start (scenarios may
``advance_sim`` at setup and the host issues its own probes before the model runs).

Call budgets: ``max_model_calls`` / ``max_tool_calls`` are episode totals. Each turn's
per-run cap in MAST's loop is what is left of them; a turn is not started when nothing is
left (outcome ``budget``), and the recorded counters are clamped to the cap — the loop
counts the call that *tripped* its limit, which was never made.

End markers: ``[DONE]`` / ``[ABORT]`` count only when the reply's stripped text ENDS with
the marker. A marker in the body ("…once that is [DONE] I will…") is logged on the turn
(``marker_in_body``) and the conversation continues.
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable

from .hitl_autoresolver import AutoResolver
from .modes import MODES, Mode, apply_tool_deny, filtered_registry
from .results import fold_results, is_result_tool, missing_claims

# ── the harness-owned diagnosis tool ────────────────────────────────────────
REPORT_TOOL = "ReportTipState"
REPORT_JUNCTIONS = ("clean", "dirty", "unknown")
REPORT_TIP_STATES = ("sharp", "blunt", "double", "unstable", "unknown")
REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "junction": {"type": "string", "enum": list(REPORT_JUNCTIONS),
                     "description": "隧穿结是否干净：clean（φ ≥ 3 eV 量级）/ dirty / unknown（没测、测不出来）"},
        "atomic_resolution": {"type": "boolean",
                              "description": "你认为当前已经拿到一张原子分辨图（true/false）"},
        "tip_state": {"type": "string", "enum": list(REPORT_TIP_STATES),
                      "description": "针尖形态：sharp / blunt / double（双针尖/鬼影）/ unstable / unknown"},
        "reason": {"type": "string", "description": "依据（测到的 φ、晶格周期、对称性、噪声…），一两句"},
    },
    "required": ["junction", "atomic_resolution", "tip_state", "reason"],
}
REPORT_DESCRIPTION = (
    "报告你对隧穿结与针尖状态的判断。这是评分唯一读取的通道：不报告 = 未判断。"
    "可以多次调用，以最后一次为准。此工具不碰仪器。"
)

# older name guesses (normalised: lower-case, no '_'/'-') a model may issue as a non-existent tool
DIAGNOSIS_TOOL_NAMES = frozenset({"reporttipstate", "reporttip", "reportdiagnosis", "tipdiagnosis"})
_JUNCTION_WORDS = {"clean": "clean", "dirty": "dirty", "contaminated": "dirty"}


def _norm_tool_name(name: str) -> str:
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


def is_diagnosis_tool(name: str) -> bool:
    return _norm_tool_name(name) in DIAGNOSIS_TOOL_NAMES


def make_report_tool(sink: list, turn_of: Callable[[], int] | None = None):
    """The ``ReportTipState`` ToolSpec. ``sink`` receives one record per call."""
    from mast.agentruntime.tools import ToolSpec

    def _fn(args: dict, ctx: Any = None, **_kw) -> str:
        a = dict(args) if isinstance(args, dict) else {}
        sink.append({"name": REPORT_TOOL, "turn": int(turn_of()) if turn_of else 0, "args": a})
        return (f"已记录：结面={a.get('junction', '?')}，原子分辨={a.get('atomic_resolution', '?')}，"
                f"针尖={a.get('tip_state', '?')}。评分只看这个通道；判断更新了可再次报告。")

    return ToolSpec(name=REPORT_TOOL, description=REPORT_DESCRIPTION, schema=dict(REPORT_SCHEMA),
                    fn=_fn, touches_instrument=False)


def make_result_tool(sink: list, turn_of: Callable[[], int] | None = None, *, claims=()):
    """The ``ReportResult`` ToolSpec for one scenario's claim list."""
    from mast.agentruntime.tools import ToolSpec

    from .results import RESULT_DESCRIPTION, RESULT_TOOL, ack_text, fold_results, result_schema

    ids = [str(c.get("id")) for c in claims]

    def _fn(args: dict, ctx: Any = None, **_kw) -> str:
        a = dict(args) if isinstance(args, dict) else {}
        sink.append({"name": RESULT_TOOL, "turn": int(turn_of()) if turn_of else 0, "args": a})
        return ack_text(a, fold_results(sink), ids)

    return ToolSpec(name=RESULT_TOOL, description=RESULT_DESCRIPTION,
                    schema=result_schema(ids), fn=_fn, touches_instrument=False)


def inject_result_tool(loop, sink: list, turn_of: Callable[[], int] | None = None, *, claims=()):
    from .results import RESULT_TOOL

    spec = make_result_tool(sink, turn_of, claims=claims)
    loop.tools[RESULT_TOOL] = spec
    _make_visible(loop, RESULT_TOOL, spec.description)
    return spec


def inject_report_tool(loop, sink: list, turn_of: Callable[[], int] | None = None):
    """Put ``ReportTipState`` into an assembled loop (every mode) and make it visible.

    Visibility matters: IC's ``ToolVisibilityMiddleware`` narrows ``request.tools`` to
    "core + loaded packs" from its catalog, and a tool the catalog has never heard of is
    dropped from what the model sees — a tool that is in ``loop.tools`` but never offered.
    """
    spec = make_report_tool(sink, turn_of)
    loop.tools[REPORT_TOOL] = spec
    _make_visible(loop, REPORT_TOOL, spec.description)
    return spec


def _make_visible(loop, name: str, summary: str = "") -> bool:
    """Add ``name`` to the core set of every tool-pack catalog on the loop's stack."""
    done = False
    for mw in getattr(loop, "stack", None) or ():
        inner = getattr(mw, "wrapped", mw)
        cat = getattr(inner, "catalog", None)
        by_pack = getattr(cat, "tools_by_pack", None)
        by_tool = getattr(cat, "packs_by_tool", None)
        if not isinstance(by_pack, dict) or not isinstance(by_tool, dict):
            continue
        try:
            from mast.agents._shared.tool_packs import CORE
        except Exception:  # noqa: BLE001
            CORE = "core"  # noqa: N806
        core = tuple(by_pack.get(CORE, ()))
        if name not in core:
            by_pack[CORE] = core + (name,)
        by_tool[name] = frozenset({CORE})
        summ = getattr(cat, "summary_by_tool", None)
        if isinstance(summ, dict):
            summ[name] = summary[:80]
        done = True
    return done


def _diagnosis_from(reports: list[dict], final_text: str) -> dict:
    """Fold the recorded statements into what the judge reads.

    ``junction``: 'clean' | 'dirty' | None — from the LAST report that states one
    (``junction`` / ``verdict`` / ``state``; 'unknown' states nothing). ``tip_state``:
    the last stated ``tip_state`` other than 'unknown', else None.
    ``frames_passed_atomic``: int | None — from ``frames_passed_atomic`` (int) or
    ``atomic_resolution`` / ``atomic`` (bool). Free text is kept for the ledger but
    never interpreted.
    """
    junction = None
    tip_state = None
    frames = None
    for rep in reports:                       # later statements override earlier ones
        args = rep.get("args") or {}
        if not isinstance(args, dict):
            continue
        for key in ("junction", "verdict", "state"):
            word = _JUNCTION_WORDS.get(str(args.get(key, "")).strip().lower())
            if word is not None:
                junction = word
                break
        ts = str(args.get("tip_state", "")).strip().lower()
        if ts in REPORT_TIP_STATES and ts != "unknown":
            tip_state = ts
        if isinstance(args.get("frames_passed_atomic"), (int, float)) \
                and not isinstance(args.get("frames_passed_atomic"), bool):
            frames = int(args["frames_passed_atomic"])
        elif isinstance(args.get("atomic_resolution"), bool):
            frames = 1 if args["atomic_resolution"] else 0
        elif isinstance(args.get("atomic"), bool):
            frames = 1 if args["atomic"] else 0
    return {"reports": list(reports), "final_text": final_text,
            "junction": junction, "tip_state": tip_state, "frames_passed_atomic": frames}


# ── end markers ─────────────────────────────────────────────────────────────
DONE_MARK, ABORT_MARK = "[DONE]", "[ABORT]"


def marker_outcome(text: str) -> tuple[str | None, str | None]:
    """(outcome, note): ``('done'|'abort', None)`` when the stripped reply ENDS with a
    marker; ``(None, 'marker_in_body')`` when a marker appears elsewhere; ``(None, None)``."""
    s = str(text or "").strip()
    # tolerate trailing punctuation / markdown emphasis after the marker ("[DONE]。", "**[DONE]**")
    s = s.rstrip(" \t\r\n.。!！*_`~>」』”\"'")      # never strip ']' — it closes the marker
    if s.endswith(DONE_MARK):
        return "done", None
    if s.endswith(ABORT_MARK):
        return "abort", None
    if DONE_MARK in s or ABORT_MARK in s:
        return None, "marker_in_body"
    return None, None


# ── the clock-pausing model port ────────────────────────────────────────────
@contextlib.contextmanager
def clock_pause_scope(clock):
    """Pause ``clock`` for the block. Prefers ``clock.paused()``; else ``pause()``/``resume()``;
    a clock without either is left alone (yields ``False`` so callers can record that)."""
    paused_cm = getattr(clock, "paused", None)
    if callable(paused_cm):
        with paused_cm():
            yield True
        return
    pause = getattr(clock, "pause", None)
    resume = getattr(clock, "resume", None)
    if callable(pause) and callable(resume):
        pause()
        try:
            yield True
        finally:
            resume()
        return
    yield False


def _to_port(model):
    """A LangChain chat model (has ``bind_tools``) becomes MAST's LangChainModelPort;
    anything with ``invoke``/``stream`` and no ``bind_tools`` already is a port."""
    if hasattr(model, "bind_tools"):
        from mast.agentruntime.model import LangChainModelPort
        return LangChainModelPort(model)
    return model


class ClockPausedPort:
    """``ChatModelPort`` proxy: the sim clock stands still while the model is called.

    Deliberately exposes only ``invoke`` and ``stream`` — no ``bind_tools`` — so
    ``ic_assembly._as_port`` recognises it as a port and does not re-wrap it.
    """

    def __init__(self, inner, clock):
        self._inner = _to_port(inner)
        self._clock = clock
        self.calls = 0
        self.paused_wall_s = 0.0          # wall seconds spent inside model calls
        self.clock_paused: bool | None = None   # did the clock support pausing?

    def _scope(self):
        return clock_pause_scope(self._clock)

    def invoke(self, request):
        t0 = time.perf_counter()
        self.calls += 1
        with self._scope() as supported:
            self.clock_paused = bool(supported)
            try:
                return self._inner.invoke(request)
            finally:
                self.paused_wall_s += time.perf_counter() - t0

    def stream(self, request):
        t0 = time.perf_counter()
        self.calls += 1
        with self._scope() as supported:
            self.clock_paused = bool(supported)
            try:
                yield from self._inner.stream(request)
            finally:
                self.paused_wall_s += time.perf_counter() - t0


def _budget_probe(host) -> tuple[float | None, int | None]:
    """(sim seconds, controller command count) as the host sees them; None where unreadable."""
    world = getattr(host, "world", None)
    clock = getattr(world, "clock", None)
    try:
        sim = float(clock.sim()) if clock is not None else None
    except Exception:  # noqa: BLE001
        sim = None
    log = getattr(getattr(host, "dispatcher", None), "call_log", None)
    # only commands issued while the sim clock was running count: MAST's background
    # polling (~18 cmd/s) keeps going while the model thinks and must not spend the budget
    cmds = sum(1 for c in log if len(c) < 5 or not c[4]) if isinstance(log, list) else None
    return sim, cmds


AUTONOMY_SUFFIX = (
    "\n\n【无人值守运行规则】本次是无人值守的自主运行：没有操作员在看屏幕，你的每条回复不会有人读。"
    "需要人决定的事用 ask_user 提问（值班策略会代为回答）。任务还没完成就**继续调用工具**，不要停下来汇报计划；"
    "用 ReportTipState 报告你对结面/针尖的判断（评分只看这个通道）。"
    "只有在任务完成时才停下，并在回复末尾写 [DONE]；确认无法完成时写 [ABORT] 并说明原因。"
    "标记必须是回复的最后一个词，写在句子中间不算。"
)
CONTINUE_TEXT = ("继续。没有操作员在线；按你的判断推进任务，需要决定的事用 ask_user。"
                 "完成时在回复末尾写 [DONE]，无法完成写 [ABORT]。")

#: appended for a paper scenario, so the claim ids are in the system prompt and not only in
#: the task text a long episode may have scrolled past
RESULT_SUFFIX_TMPL = (
    "\n\n【结果报告通道】本任务的结果只通过 ReportResult 报告，需要报告的编号是：{ids}。"
    "每条结果调用一次（claim_id + value，位置类另给 x_nm / y_nm，均为当前扫描坐标系）；"
    "可重复调用更新，以最后一次为准；未报告 = 未复现。"
)


def continue_text(missing: list[str] | None = None) -> str:
    """The between-turn nudge, naming the results still outstanding."""
    if missing:
        return CONTINUE_TEXT + f" 尚未用 ReportResult 报告：{', '.join(missing)}。"
    return CONTINUE_TEXT


# ── ledger: the sampling parameters MAST applies (episode.json sampling_params_json) ──
# ``make_chat_model`` is called with its defaults apart from ``request_timeout`` and
# ``usage_source``; these are those defaults, named here so the mirror below and the
# call in ``run_llm_episode`` cannot drift apart.
MODEL_CALL_DEFAULTS = {"max_tokens": 4096, "temperature": 0.2, "thinking_level": None}


def sampling_params_for(model_id: str | None, *, max_tokens: int = MODEL_CALL_DEFAULTS["max_tokens"],
                        temperature: float = MODEL_CALL_DEFAULTS["temperature"],
                        thinking_level: str | None = MODEL_CALL_DEFAULTS["thinking_level"],
                        request_timeout_s: float | None = None) -> dict:
    """What ``mast.agents._shared.models.make_chat_model`` would apply for ``model_id``.

    A mirror of the rules in ``models.py`` (read, not executed — no key, no client):
    reasoning models pin ``temperature`` to 1.0 and floor ``max_tokens`` at 16000; Claude /
    MiniMax with a thinking level get ``temperature`` 1.0 and an adaptive or budgeted
    ``thinking`` block; tool-carrying calls never stream. Every field is ``None`` where the
    module does not expose the rule (older MAST, unknown model id) — unknown is not a
    default. Never raises.
    """
    out: dict = {"model_id": model_id, "provider": None, "temperature": None, "max_tokens": None,
                 "thinking": None, "thinking_effective": None, "disable_streaming": None,
                 "base_url": None, "request_timeout_s": request_timeout_s,
                 "source": "mirror of mast.agents._shared.models.make_chat_model", "error": None}
    if not model_id:
        out["error"] = "no model id"
        return out
    try:
        from mast.agents._shared import models as M
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"MAST models not importable: {type(exc).__name__}: {exc}"
        return out
    try:
        provider = M.provider_for(model_id)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["provider"] = provider
    out["disable_streaming"] = getattr(M, "_DISABLE_STREAMING_FOR_TOOLS", None)
    forced1 = getattr(M, "_FORCED_TEMPERATURE_1", None)
    reasoning = getattr(M, "_ALWAYS_HIGH_REASONING", None)
    adaptive = getattr(M, "_ADAPTIVE_THINKING", None)
    budgets = getattr(M, "_THINKING_BUDGET_TOKENS", None)
    norm = getattr(M, "normalize_thinking_level", None)
    level = norm(thinking_level) if callable(norm) else None
    eff = getattr(M, "effective_thinking", None)
    out["thinking_effective"] = eff(model_id, thinking_level) if callable(eff) else None
    temp: float | None = float(temperature)
    mt: int | None = int(max_tokens)
    if provider in ("anthropic", "minimax"):
        thinking: dict | None = {"type": "off"} if level in (None, "off") else None
        if level and level != "off":
            temp = 1.0
            if adaptive is None or budgets is None:
                thinking, mt = None, None                       # rule not exposed: unknown
            elif model_id in adaptive:
                thinking = {"type": "adaptive", "effort": level}
                mt = max(mt, 16000)
            else:
                b = budgets.get(level)
                thinking = {"type": "enabled", "budget_tokens": b}
                if b is not None and mt <= b:
                    mt = b + 4096
        urls = getattr(M, "ANTHROPIC_COMPAT_BASE_URL", {}) or {}
        out["base_url"] = urls.get(provider) if provider == "minimax" else "https://api.anthropic.com"
    else:
        if forced1 is None or reasoning is None:
            temp, mt, thinking = None, None, None                # rule not exposed: unknown
        else:
            if model_id in forced1:
                temp = 1.0
            if model_id in reasoning:
                mt = max(mt, 16000)
                thinking = {"type": "intrinsic", "effort": "high"}    # always-on, no knob
            else:
                thinking = {"type": "off"}
        out["base_url"] = (getattr(M, "PROVIDER_BASE_URL", {}) or {}).get(provider)
    out.update({"temperature": temp, "max_tokens": mt, "thinking": thinking})
    return out


def observed_sampling_params(model: Any) -> dict:
    """The parameters the constructed chat model object actually carries (ChatOpenAI /
    ChatAnthropic attributes); ``{}`` for a model that exposes none (a test double)."""
    out: dict = {}
    for key, attrs in (("model_id", ("model_name", "model", "model_id")), ("temperature", ("temperature",)),
                       ("max_tokens", ("max_tokens",)), ("thinking", ("thinking",)),
                       ("output_config", ("output_config",)), ("disable_streaming", ("disable_streaming",)),
                       ("base_url", ("openai_api_base", "anthropic_api_url")),
                       ("request_timeout_s", ("request_timeout", "default_request_timeout"))):
        for a in attrs:
            try:
                v = getattr(model, a, None)
            except Exception:  # noqa: BLE001
                v = None
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool, dict, list)):
                out[key] = v
            else:
                out[key] = str(v)
            break
    return out


@dataclass
class DriverResult:
    # done | abort | budget | budget_sim | budget_cmds | wall | turns | error | aborted | stalled | handoff
    # ('limit' from MAST's loop is reported as 'budget': its per-run cap IS the episode remainder)
    outcome: str = ""
    stop_reason: str = ""
    final_text: str = ""
    model_calls: int = 0
    tool_calls: int = 0
    turns: int = 0
    wall_s: float = 0.0
    sim_s: float = 0.0           # sim seconds consumed by the episode (from episode start)
    wire_cmds: int = 0        # wire commands issued during the episode (from episode start)
    model_wall_s: float = 0.0    # wall seconds spent inside model calls (sim clock paused)
    clock_paused: bool | None = None   # None = no model call was made
    events: list = field(default_factory=list)
    turn_log: list = field(default_factory=list)
    hitl: list = field(default_factory=list)
    tools_removed: list = field(default_factory=list)
    skills_kept: int = 0
    skills_dropped: int = 0
    diagnosis: dict = field(default_factory=dict)   # see _diagnosis_from
    results: dict = field(default_factory=dict)     # see results.fold_results (paper claims)
    stub_calls: list = field(default_factory=list)  # calls to not-yet-visible tools (bind_all_names_mw)
    # ledger sampling_params_json: the mirror of make_chat_model's rules for this model id
    # plus ``observed`` = what the built model object carries (empty for an injected double)
    sampling_params: dict | None = None
    error: str | None = None


def _event_record(ev, turn: int) -> dict:
    d = asdict(ev) if is_dataclass(ev) else {"kind": getattr(ev, "kind", "?")}
    d["turn"] = turn
    for k in ("preview",):
        if isinstance(d.get(k), str) and len(d[k]) > 600:
            d[k] = d[k][:600] + "…"
    return d


def run_llm_episode(host, task_text: str, *, model_id: str, mode: str = "A",
                    episode_id: str = "stmbench", policy: str = "default",
                    max_model_calls: int = 400, max_tool_calls: int = 400,
                    max_turns: int = 40, max_wall_s: float | None = None,
                    max_sim_s: float | None = None, max_wire_cmds: int | None = None,
                    request_timeout_s: float | None = 300.0,
                    on_event=None, model=None, claims: list[dict] | None = None) -> DriverResult:
    """``model`` overrides the provider model (tests drive the loop with a ScriptedModel).

    ``max_sim_s`` / ``max_wire_cmds`` are the scenario budget, counted from the episode
    start; they are polled on every ``tool_end`` (see module docstring) and between turns.
    """
    from mast.agentruntime.context import RunContext
    from mast.agentruntime.pause import attach
    from mast.agents._shared.models import make_chat_model
    from langchain_core.messages import HumanMessage

    m: Mode = MODES[mode]
    res = DriverResult()
    t0 = time.perf_counter()
    sim0, cmds0 = _budget_probe(host)
    reports: list[dict] = []
    result_reports: list[dict] = []
    claims = list(claims or [])
    claim_ids = [str(c.get("id")) for c in claims]
    budget_hit: str | None = None          # 'budget_sim' | 'budget_cmds' once a budget is spent

    def _consumed() -> tuple[float, int]:
        sim, cmds = _budget_probe(host)
        sim_used = (sim - sim0) if (sim is not None and sim0 is not None) else 0.0
        cmds_used = (cmds - cmds0) if (cmds is not None and cmds0 is not None) else 0
        return sim_used, cmds_used

    def _budget_spent() -> str | None:
        sim_used, cmds_used = _consumed()
        if max_sim_s is not None and sim_used >= max_sim_s:
            return "budget_sim"
        if max_wire_cmds is not None and cmds_used >= max_wire_cmds:
            return "budget_cmds"
        return None

    app = host.app
    registry = filtered_registry(app._registry, m)
    res.skills_kept = len(getattr(registry, "_stmbench_kept", []) or []) or len(registry.list_skills())
    res.skills_dropped = len(getattr(registry, "_stmbench_dropped", []) or [])
    injected = model is not None
    if model is None:
        model = make_chat_model(model_id=model_id, usage_source=episode_id,
                                request_timeout=request_timeout_s, **MODEL_CALL_DEFAULTS)
    res.sampling_params = {**sampling_params_for(model_id, request_timeout_s=request_timeout_s,
                                                 **MODEL_CALL_DEFAULTS),
                           "injected_model": injected, "observed": observed_sampling_params(model)}
    port = ClockPausedPort(model, getattr(getattr(host, "world", None), "clock", None))
    suffix = (m.suffix or "") + AUTONOMY_SUFFIX
    if claim_ids:
        suffix += RESULT_SUFFIX_TMPL.format(ids=", ".join(claim_ids))
    loop = app.build_instrument_loop_for(registry=registry, model=port,
                                         system_suffix=suffix,
                                         max_model_calls=max_model_calls,
                                         max_tool_calls=max_tool_calls)
    res.tools_removed = apply_tool_deny(loop, m)
    inject_report_tool(loop, reports, lambda: res.turns)
    if claims:
        inject_result_tool(loop, result_reports, lambda: res.turns, claims=claims)
    # protocol: every tool NAME stays bound on every model call; a call to a not-yet-visible
    # tool loads its pack instead of executing (see bind_all_names_mw for the kimi-k3 hazard)
    from mast.agentruntime.middleware import MiddlewareStack

    from .bind_all_names_mw import BindAllToolNames

    bind_all = BindAllToolNames(loop.tools, registry=registry)
    loop.stack = MiddlewareStack(list(loop.stack) + [bind_all])
    res.stub_calls = bind_all.stub_calls        # same list: filled during the run, dumped at the end

    resolver = AutoResolver(policy=policy).start()
    ctx = RunContext(run_id=episode_id, conversation_id=episode_id, thread_id=episode_id,
                     agent_id="instrument_control")
    attach(ctx, resolver.store, owner="stmbench", thread_id=episode_id)
    convo: list = [HumanMessage(content=task_text)]
    try:
        while True:
            # per-turn caps = what is left of the episode budget; nothing left ⇒ no turn
            remaining_mc = max_model_calls - res.model_calls
            remaining_tc = max_tool_calls - res.tool_calls
            if remaining_mc <= 0 or remaining_tc <= 0:
                res.outcome = "budget"
                res.stop_reason = res.stop_reason or "episode call budget spent"
                break
            res.turns += 1
            loop.limits.max_model_calls = remaining_mc
            loop.limits.max_tool_calls = remaining_tc
            gen = loop.run(list(convo), ctx)
            while True:
                try:
                    ev = next(gen)
                except StopIteration as stop:
                    result = stop.value
                    break
                rec = _event_record(ev, res.turns)
                res.events.append(rec)
                if on_event is not None:
                    try:
                        on_event(rec)
                    except Exception:  # noqa: BLE001
                        pass
                kind = rec.get("kind")
                name = rec.get("name", "")
                if kind == "tool_start" and is_diagnosis_tool(name) and name not in loop.tools:
                    # a name guess issued as a non-existent tool: keep the statement for the
                    # ledger (the real ReportTipState records itself when it runs)
                    args = rec.get("args")
                    reports.append({"name": name, "turn": res.turns,
                                    "args": dict(args) if isinstance(args, dict) else {}})
                elif kind == "tool_start" and claims and is_result_tool(name) and name not in loop.tools:
                    args = rec.get("args")
                    result_reports.append({"name": name, "turn": res.turns,
                                           "args": dict(args) if isinstance(args, dict) else {}})
                elif kind == "tool_end" and budget_hit is None:
                    # poll the sim budget after every tool: one call can burn hours of sim
                    # time, and the loop only re-checks ctx.abort at its next hop
                    budget_hit = _budget_spent()
                    if budget_hit is not None:
                        ctx.abort.set()
            counters = getattr(result, "counters", None)
            # the loop counts the call that TRIPPED its limit (never made): clamp to the cap
            mc = min(int(getattr(counters, "model_calls", 0) or 0), remaining_mc)
            tc = min(int(getattr(counters, "tool_calls", 0) or 0), remaining_tc)
            res.model_calls += mc
            res.tool_calls += tc
            outcome = str(getattr(result, "outcome", "") or "")
            text = str(getattr(result, "final_text", "") or "")
            res.final_text = text
            res.stop_reason = str(getattr(result, "stop_reason", "") or "")
            marker, note = marker_outcome(text)
            res.turn_log.append({"turn": res.turns, "outcome": outcome, "model_calls": mc,
                                 "tool_calls": tc, "text": text[:800], "marker": marker,
                                 "note": note})
            convo = convo + list(getattr(result, "new_messages", []) or [])
            # carry run state (loaded tool packs …) into the next turn: the loop seeds its
            # state_delta from ctx.extra["state"], so a pack loaded on turn 1 stays visible
            delta = getattr(result, "state_delta", None)
            if isinstance(delta, dict) and delta:
                merged = dict(ctx.extra.get("state") or {})
                merged.update(delta)
                ctx.extra["state"] = merged
            if budget_hit is not None:
                # we set ctx.abort on a tool_end; the loop reported 'aborted' for it
                res.outcome = budget_hit
                break
            if outcome == "limit":
                # the loop's per-run cap was the episode remainder: that is the budget
                res.outcome = "budget"
                break
            if outcome != "final":
                res.outcome = outcome                      # error / aborted / stalled / handoff
                break
            if marker is not None:
                res.outcome = marker                       # done | abort (marker at the END)
                break
            if res.model_calls >= max_model_calls or res.tool_calls >= max_tool_calls:
                res.outcome = "budget"
                break
            budget_hit = _budget_spent()               # between turns (a turn with no tool)
            if budget_hit is not None:
                res.outcome = budget_hit
                break
            if res.turns >= max_turns:
                res.outcome = "turns"
                break
            if max_wall_s is not None and time.perf_counter() - t0 >= max_wall_s:
                res.outcome = "wall"
                break
            convo.append(HumanMessage(content=continue_text(
                missing_claims(fold_results(result_reports), claim_ids) if claims else None)))
    except Exception as exc:  # noqa: BLE001 — the episode still gets judged on the sim truth
        res.outcome = "error"
        res.error = f"{type(exc).__name__}: {exc}"
    finally:
        resolver.stop()
        res.hitl = resolver.as_records()
        res.wall_s = time.perf_counter() - t0
        res.sim_s, res.wire_cmds = _consumed()
        res.model_wall_s = port.paused_wall_s
        res.clock_paused = port.clock_paused
        res.diagnosis = _diagnosis_from(reports, res.final_text)
        res.results = fold_results(result_reports) if claims else {}
    return res
