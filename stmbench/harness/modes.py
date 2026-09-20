"""Benchmark modes (DESIGN.md §5.2): which tools the model sees and what it is told.

* ``A``  — the full MAST stack: every registered skill (composites included), the
           composite forge (draft/save/run_composite) and the knowledge tools.
* ``B0`` — primitives only: skills of composition level ≤ 2 (1:1 controller wrappers,
           short sequences, pure analysis) — no multi-step workflows, no composite
           forge, no knowledge tools. ``ask_user`` stays.
* ``B1`` — B0 + the knowledge tools (query_knowledge, get_workflow_advice,
           get_skill_guidance, get_fault_diagnosis, nanonis_manual, …).
* ``C``  — scripted baseline (no LLM; ``episode.py`` runs the family's composite).

"Primitives only" has no ready-made definition in MAST (see the plan §2.2): filtering
by ``CompositeSkillGraph`` keeps level-2 closed loops, filtering by level 0 drops
SaveScan/StartScan. Level ≤ 2 is the line drawn here, and it is a *deny* on the tool
surface rather than an allow-list so a new primitive is visible by default while a new
workflow is not. The system suffix tells the model what it has — the IC prompt names
ForgeAuTip / PrepareNobleTip / TipShapeWithReadback by name, and a model told to use a
tool that is not there wastes its budget looking for it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# tools built outside the skill registry (agents/_shared/tool_packs.py core pack)
COMPOSITE_FORGE_TOOLS = frozenset({"draft_composite", "save_composite", "run_composite",
                                   "propose_python_skill"})
KNOWLEDGE_TOOLS = frozenset({"query_knowledge", "get_workflow_advice", "get_skill_guidance",
                             "get_literature_parameters", "get_fault_diagnosis",
                             "get_noise_reference", "get_measurement_template",
                             "search_deep_reference", "read_reference_section",
                             "nanonis_manual"})
PRIMITIVE_MAX_LEVEL = 2


@dataclass(frozen=True)
class Mode:
    name: str
    llm: bool
    max_level: int | None            # None = every skill
    deny_tools: frozenset = field(default_factory=frozenset)
    suffix: str = ""


MODES: dict[str, Mode] = {
    "A": Mode("A", True, None, frozenset(), ""),
    "B0": Mode("B0", True, PRIMITIVE_MAX_LEVEL, COMPOSITE_FORGE_TOOLS | KNOWLEDGE_TOOLS,
               "【本次可用工具面】只有原语级技能（单条控制器读写、两三步的短序列、纯分析）与 ask_user。"
               "没有 ForgeAuTip / PrepareNobleTip / TipShapeWithReadback / AchieveAtomicResolution 之类的"
               "多步流程技能，也没有 draft/save/run_composite 与知识库工具。需要的流程请自己用原语一步步做，"
               "每一步之后读回仪器状态再决定下一步。"),
    "B1": Mode("B1", True, PRIMITIVE_MAX_LEVEL, COMPOSITE_FORGE_TOOLS,
               "【本次可用工具面】只有原语级技能（单条控制器读写、两三步的短序列、纯分析）、ask_user 与"
               "知识库工具（query_knowledge / get_workflow_advice / get_skill_guidance / get_fault_diagnosis /"
               " nanonis_manual）。没有多步流程技能，也没有 draft/save/run_composite。"
               "需要的流程请自己用原语一步步做，每一步之后读回仪器状态再决定下一步。"),
    "C": Mode("C", False, None, frozenset(), ""),
    # a person at the keyboard (stmbench.human): mode A's tool surface through the same
    # loop; the model port is the only thing that differs
    "H": Mode("H", True, None, frozenset(), ""),
}


def filtered_registry(registry, mode: Mode):
    """A fresh SkillRegistry holding only the skills the mode allows (same classes)."""
    if mode.max_level is None:
        return registry
    from mast.core.registry import SkillRegistry

    out = SkillRegistry()
    kept, dropped = [], []
    for meta in registry.list_skills():
        lvl = int(getattr(meta, "composition_level", 0) or 0)
        cat = getattr(getattr(meta, "category", None), "value", getattr(meta, "category", ""))
        # level 2 is "pure analysis, no controller writes" by the metadata's own definition,
        # yet PokeConditionTip / PulseConditionTip carry level 2 with category=composite —
        # closed conditioning loops that drive the instrument. A primitive surface keeps
        # level-2 analysis and drops level-2 composites (2026-08-28, run 9 of B0+kimi-k3).
        # …and CleanTipUntilBarrier carries level 2 with category=write (a closed loop that
        # pokes the tip until the barrier is clean). A primitive surface keeps level-2
        # skills only when they do not touch the instrument: category read / analysis.
        if lvl <= mode.max_level and not (lvl >= 2 and str(cat) not in ("read", "analysis")):
            try:
                out.register(registry.get(meta.name))
                kept.append(meta.name)
            except Exception:  # noqa: BLE001 — a skill that will not re-register is simply absent
                dropped.append(meta.name)
        else:
            dropped.append(meta.name)
    out._stmbench_kept = kept        # type: ignore[attr-defined]  (ledger provenance)
    out._stmbench_dropped = dropped  # type: ignore[attr-defined]
    return out


def apply_tool_deny(loop, mode: Mode) -> list[str]:
    """Drop denied tools from an assembled AgentLoop; returns what was removed."""
    removed = [n for n in list(loop.tools) if n in mode.deny_tools]
    for n in removed:
        loop.tools.pop(n, None)
    return removed
