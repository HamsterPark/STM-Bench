"""Track A prompts: one structured-output prompt per task, each with an explicit 「判不了」 option.

Every task's answer is a single JSON object. Two shapes are always legal:

* the task answer (schema in :data:`ANSWER_SCHEMAS`), or
* ``{"abstain": true, "reason": "..."}`` — the model says it cannot tell. The scoring rule:
  判不了 ≠ 没有 — abstentions are scored separately (abstain rate), never as a wrong answer.

Labels & conventions (must match ``stmbench.trackA.score`` and ``schema``):

* **T1** the frames are shown in time order; the earlier frames of the run are rendered in
  full (they have already ended), the LAST frame only down to its first
  ``schema.T1_HEAD_ROWS`` acquired rows. ``stop`` = *the operator stopped THAT last frame
  before it completed* (manifest ``stopped``); ``stop=false`` = they let it run to
  completion. ``p_stop`` is the probability of ``stop``.
* **T2** offsets are in nm relative to the centre of the shown frame, ``+x`` right, ``+y`` up.
* **T3** ``action`` ∈ {stay, relocate, long_stop} — three classes, the manifest's
  ``next_action`` (``score(..., t3_binary=True)`` folds relocate + long_stop into
  ``relocate``). ``probs`` carries the three class probabilities; ``p_relocate`` =
  ``probs.relocate``.
* **T4** ``bias_v`` (signed volts), ``setpoint_a`` (amperes, e.g. ``1e-10``); optional uncertainty
  ``bias_sigma_v`` and ``setpoint_log10_sigma`` feed the log-likelihood metric.

The context line a model reads (:func:`context_keys`) is derived from the schema: every
``input`` / ``strat`` column of the task that is not a file reference. Label and ``meta``
columns can therefore never reach the prompt, whatever a manifest row carries.

No model is called from this module; it only builds text + image lists and parses replies.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable

from . import schema as S

ABSTAIN = "判不了"
TASKS = ("T1", "T2", "T3", "T4")

SYSTEM = (
    "你是一位在扫描隧道显微镜（STM）前工作多年的操作员。你会看到原始 .sxm 帧的渲染图"
    "（平面校正 → 逐行中位对齐 → MAD 拉伸，未扫到的行是黑色）以及帧头部摘要。"
    "请只根据给出的信息作判断，并且只输出一个 JSON 对象，不要输出别的文字。"
    f"如果信息不足以判断，请如实回答「{ABSTAIN}」——输出 {{\"abstain\": true, \"reason\": \"...\"}}。"
    "「判不了」会单独计分，不算答错；错误回答仍按错误计分。"
)

_ABSTAIN_SHAPE = {"required": ["abstain"], "properties": {"abstain": {"const": True}, "reason": {"type": "string"}}}

ANSWER_SCHEMAS: dict[str, dict] = {
    "T1": {
        "type": "object",
        "oneOf": [
            {"required": ["stop", "p_stop"],
             "properties": {"stop": {"type": "boolean"},
                            "p_stop": {"type": "number", "minimum": 0, "maximum": 1},
                            "reason": {"type": "string"}}},
            _ABSTAIN_SHAPE,
        ],
    },
    "T2": {
        "type": "object",
        "oneOf": [
            {"required": ["dx_nm", "dy_nm"],
             "properties": {"dx_nm": {"type": "number"}, "dy_nm": {"type": "number"},
                            "size_nm": {"type": "number", "exclusiveMinimum": 0},
                            "candidates": {"type": "array", "items": {"type": "array", "items": {"type": "number"},
                                                                       "minItems": 2, "maxItems": 2}},
                            "reason": {"type": "string"}}},
            _ABSTAIN_SHAPE,
        ],
    },
    "T3": {
        "type": "object",
        "oneOf": [
            {"required": ["action"],
             "properties": {"action": {"type": "string", "enum": list(S.NEXT_ACTIONS)},
                            "p_relocate": {"type": "number", "minimum": 0, "maximum": 1},
                            "probs": {"type": "object",
                                      "properties": {c: {"type": "number", "minimum": 0, "maximum": 1}
                                                     for c in S.NEXT_ACTIONS},
                                      "additionalProperties": {"type": "number"}},
                            "reason": {"type": "string"}}},
            _ABSTAIN_SHAPE,
        ],
    },
    "T4": {
        "type": "object",
        "oneOf": [
            {"required": ["bias_v", "setpoint_a"],
             "properties": {"bias_v": {"type": "number"}, "setpoint_a": {"type": "number"},
                            "scan_speed_nm_per_s": {"type": "number"},
                            "bias_sigma_v": {"type": "number", "exclusiveMinimum": 0},
                            "setpoint_log10_sigma": {"type": "number", "exclusiveMinimum": 0},
                            "reason": {"type": "string"}}},
            _ABSTAIN_SHAPE,
        ],
    },
}

TASK_INSTRUCTIONS: dict[str, str] = {
    "T1": (
        "任务 T1「继续还是停」。你会按时间顺序看到同一位置的一串帧：前面的帧都已经结束——可能是扫完的完整帧，"
        "也可能是只扫了头几行就被停掉的「瞥视」，它们按实际扫到的行渲染，未扫到的行是黑色；"
        f"**最后一帧只显示它刚扫出来的前 {S.T1_HEAD_ROWS} 行**，它的结局就是要你判断的东西。"
        "请判断：**操作员会不会在这最后一帧扫完之前把它停掉**（stop=true），还是放手让它扫完（stop=false）。"
        "提示：这是序列判断——看最后一帧的头几行相对前一帧是否还在移动、高度是否还在漂；"
        "单帧的好坏几乎读不出停止的理由。\n"
        "输出：{\"stop\": true|false, \"p_stop\": 0~1, \"reason\": \"一句话\"}"
    ),
    "T2": (
        "任务 T2「下一步去哪」。你会看到一张大视野帧。操作员接下来在这张帧里选了一个子区域缩放进去。"
        "请给出你会选的子区域中心，相对于本帧中心的偏移（纳米；+x 向右，+y 向上），以及子区域边长 size_nm。"
        "可以再给最多 5 个备选中心 candidates（按优先级排序，第一个应等于你的主答案）。\n"
        "输出：{\"dx_nm\": 数, \"dy_nm\": 数, \"size_nm\": 数, \"candidates\": [[dx,dy],...], \"reason\": \"一句话\"}"
    ),
    "T3": (
        "任务 T3「换地方还是留下」。你会看到一帧（可能是完整帧，也可能是半途停掉的帧）和它之前的上下文。"
        "请判断操作员下一帧做了什么，三选一："
        "**stay** = 留在原地（重扫、缩放或平移不到一个视野）；"
        f"**relocate** = 换了位置（中心移动 ≥ {S.RELOCATE_MIN_MOVE_FRAC:g} 个视野）；"
        f"**long_stop** = 停下来了（超过 {S.LONG_STOP_S / 60:g} 分钟没有下一帧）。"
        "给出 action、三类概率 probs（和为 1）以及 p_relocate（= probs.relocate）。\n"
        "输出：{\"action\": \"stay\"|\"relocate\"|\"long_stop\", "
        "\"probs\": {\"stay\": 0~1, \"relocate\": 0~1, \"long_stop\": 0~1}, \"p_relocate\": 0~1, \"reason\": \"一句话\"}"
    ),
    "T4": (
        "任务 T4「工作点先验」。已知材料/体系、视野尺寸与意图（头部摘要里**没有**偏压与 setpoint，备注已去掉）。"
        "请给出你会用的偏压 bias_v（伏，带符号）和隧穿电流 setpoint_a（安培，例如 1e-10）。"
        "可选：bias_sigma_v（偏压不确定度，伏）与 setpoint_log10_sigma（setpoint 在 log10 空间的不确定度），"
        "用于对数似然计分；不给则用默认宽度。\n"
        "输出：{\"bias_v\": 数, \"setpoint_a\": 数, \"bias_sigma_v\": 数, \"setpoint_log10_sigma\": 数, \"reason\": \"一句话\"}"
    ),
}

_T4_HIDDEN_KEYS = ("bias_v", "setpoint_a", "comment")

# file references are shown as images, never as text
_FILE_REFERENCE_COLUMNS = frozenset({"path", "rel_path", "context_paths_json", "next_path", "first_path"})
# contract-shaped rows (sample_id / input_paths / label / group / split + free strat columns)
# are not declared in the schema; these are the strat names they have used so far
_CONTRACT_CONTEXT_KEYS = ("group", "era", "autosave_mode", "n_glances", "intent", "range_nm", "temperature_k")


def context_keys(task: str) -> tuple[str, ...]:
    """Row keys a model may read as text for ``task``: the schema's ``input`` + ``strat``
    columns minus file references, plus the contract-shape strat names. Derived, not listed —
    a label or ``meta`` column can never be added here by hand."""
    spec = S.SPECS[str(task).upper()]
    keys = [c.name for c in spec.cols
            if c.role in (S.ROLE_INPUT, S.ROLE_STRAT) and c.name not in _FILE_REFERENCE_COLUMNS]
    return tuple(keys) + tuple(k for k in _CONTRACT_CONTEXT_KEYS if k not in keys)


def _fmt_summary(task: str, summary: dict) -> str:
    s = dict(summary)
    if task == "T4":
        for k in _T4_HIDDEN_KEYS:
            s.pop(k, None)
    return json.dumps(s, ensure_ascii=False)


def _context_text(task: str, row: dict) -> str:
    """The row's readable columns (:func:`context_keys`) as one JSON line — never a label."""
    ctx = {k: row[k] for k in context_keys(task) if k in row and row[k] is not None and not _is_nan(row[k])}
    return json.dumps(ctx, ensure_ascii=False, default=str) if ctx else "{}"


def _is_nan(v: Any) -> bool:
    try:
        return v != v  # NaN is the only value unequal to itself
    except Exception:  # noqa: BLE001
        return False


def _frame_tag(i: int, n: int, summary: dict) -> str:
    tag = f"第 {i}/{n} 帧" if n > 1 else "这一帧"
    shown = summary.get("head_rows_shown")
    if shown is not None and not _is_nan(shown):
        # the judged frame (T1): only its head is shown, its outcome is the question
        return f"{tag}（只显示刚扫出的前 {int(shown)} 行，结局待你判断）"
    acq = summary.get("acq_frac")
    if isinstance(acq, (int, float)) and not isinstance(acq, bool) and not _is_nan(acq):
        return f"{tag}（已扫 {acq:.0%}）"
    return tag


def build_prompt(task: str, row: dict, renders: Iterable[Any]) -> dict:
    """Assemble ``{system, user, images, schema, task, sample_id}`` for one manifest row.

    ``renders`` are :class:`stmbench.trackA.render.RenderedFrame` objects (or anything with
    ``.png`` and ``.summary``), in the order the operator saw them. For T1 the caller renders
    the last one with ``head_rows`` (``cli.cmd_render`` does); its summary then carries
    ``head_rows_shown`` and the frame is tagged as the one being judged.
    """
    task = str(task).upper()
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}")
    renders = list(renders)
    lines = [TASK_INSTRUCTIONS[task], "", f"上下文：{_context_text(task, row)}", ""]
    for i, r in enumerate(renders, 1):
        summary = getattr(r, "summary", None) or (r.get("summary") if isinstance(r, dict) else {}) or {}
        lines.append(f"{_frame_tag(i, len(renders), summary)} 头部摘要：{_fmt_summary(task, summary)}")
    lines += ["", f"请只输出一个 JSON 对象；判不了就输出 {{\"abstain\": true, \"reason\": \"...\"}}。"]
    images = []
    for r in renders:
        png = getattr(r, "png", None) or (r.get("png") if isinstance(r, dict) else None)
        if png:
            images.append(png)
    return {"task": task, "sample_id": row.get("sample_id"), "system": SYSTEM,
            "user": "\n".join(lines), "images": images, "schema": ANSWER_SCHEMAS[task]}


# ── answer parsing ──────────────────────────────────────────────────────────

_JSON_OBJ = re.compile(r"\{.*\}", re.S)


def parse_answer(raw: Any) -> dict:
    """Turn a model reply (dict / JSON string / free text) into a dict.

    Guarantees a dict with ``abstain`` (bool) set. Unparseable text → ``{"abstain": True,
    "parse_error": True}`` so a garbled reply counts as an abstention, not as a guess.
    """
    if raw is None:
        return {"abstain": True, "parse_error": True, "raw": None}
    if isinstance(raw, dict):
        ans = dict(raw)
    else:
        text = str(raw).strip()
        ans = None
        try:
            ans = json.loads(text)
        except (ValueError, TypeError):
            m = _JSON_OBJ.search(text)
            if m:
                try:
                    ans = json.loads(m.group(0))
                except (ValueError, TypeError):
                    ans = None
        if not isinstance(ans, dict):
            if ABSTAIN in text:
                return {"abstain": True, "reason": text[:200]}
            return {"abstain": True, "parse_error": True, "raw": text[:200]}
    if ans.get("answer") == ABSTAIN or ans.get("abstain") in (True, "true", "True", 1):
        ans["abstain"] = True
    else:
        ans["abstain"] = False
    return ans


def is_abstain(ans: Any) -> bool:
    return bool(parse_answer(ans).get("abstain"))
