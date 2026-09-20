"""``ReportResult`` — the one channel a paper's numbers may arrive on.

The rule is the same one ``ReportTipState`` follows, for the same reason: a benchmark that
parses numbers out of free text is grading prose. One call per claim, the last call for a
claim wins, and a claim that is never reported counts as *not reproduced* — an unmeasured
result is not an abstention, it is a paper that did not come out.

This module deliberately imports nothing from MAST so the judge and the tests can use it.
"""
from __future__ import annotations

from typing import Any, Iterable

RESULT_TOOL = "ReportResult"

#: names a model might reach for instead of the real one (normalised: lower-case, alnum only)
RESULT_TOOL_NAMES = frozenset({"reportresult", "reportresults", "reportclaim", "reportvalue"})

RESULT_DESCRIPTION = (
    "逐条报告论文结果的数值。这是评分唯一读取的通道：一条结果调用一次，可重复调用更新，"
    "以最后一次为准；未报告的结果按未复现计。位置类结果必须给 x_nm / y_nm（当前扫描坐标系）。"
    "此工具不碰仪器。"
)


def result_schema(claim_ids: Iterable[str]) -> dict:
    ids = [str(c) for c in claim_ids]
    return {
        "type": "object",
        "properties": {
            "claim_id": {"type": "string", "enum": ids,
                         "description": "要报告的结果编号（见任务描述）"},
            "value": {"type": "number", "description": "数值，用该结果要求的单位"},
            "unit": {"type": "string", "description": "你使用的单位（用于核对，不参与判分）"},
            "sigma": {"type": "number", "description": "不确定度（可选）"},
            "x_nm": {"type": "number", "description": "位置类结果：x（当前扫描坐标系，nm）"},
            "y_nm": {"type": "number", "description": "位置类结果：y（当前扫描坐标系，nm）"},
            "note": {"type": "string", "description": "依据，一两句（可选）"},
        },
        "required": ["claim_id", "value"],
    }


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


def is_result_tool(name: str) -> bool:
    return _norm(name) in RESULT_TOOL_NAMES


def fold_results(reports: list[dict] | None) -> dict:
    """Last write wins, per claim id. Non-numeric values are dropped, not guessed."""
    claims: dict[str, dict] = {}
    unknown: list[str] = []
    for rec in reports or []:
        args = rec.get("args") if isinstance(rec, dict) else None
        if not isinstance(args, dict):
            continue
        cid = args.get("claim_id")
        if not isinstance(cid, str) or not cid:
            unknown.append(str(cid))
            continue
        val = args.get("value")
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        entry = {"value": float(val), "turn": rec.get("turn")}
        for k in ("unit", "note"):
            if isinstance(args.get(k), str):
                entry[k] = args[k]
        for k in ("sigma", "x_nm", "y_nm"):
            v = args.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                entry[k] = float(v)
        claims[cid] = entry
    return {"reports": list(reports or []), "claims": claims,
            "unknown_claim_ids": sorted(set(unknown))}


def missing_claims(folded: dict | None, claim_ids: Iterable[str]) -> list[str]:
    got = set((folded or {}).get("claims", {}))
    return [c for c in claim_ids if c not in got]


def ack_text(args: dict, folded: dict, claim_ids: Iterable[str]) -> str:
    left = missing_claims(folded, claim_ids)
    tail = f"尚未报告：{', '.join(left)}。" if left else "所有结果都已报告。"
    return f"已记录 {args.get('claim_id', '?')} = {args.get('value', '?')}。{tail}"


class ResultSink:
    """The scripted-baseline side of the same channel: mode C folds through this so the
    judge sees exactly the record shape an LLM episode produces."""

    def __init__(self):
        self.records: list[dict] = []

    def report(self, claim_id: str, value: float, unit: str = "", *, x_nm: float | None = None,
               y_nm: float | None = None, sigma: float | None = None, note: str = "") -> None:
        args: dict[str, Any] = {"claim_id": str(claim_id), "value": float(value)}
        if unit:
            args["unit"] = unit
        if x_nm is not None:
            args["x_nm"] = float(x_nm)
        if y_nm is not None:
            args["y_nm"] = float(y_nm)
        if sigma is not None:
            args["sigma"] = float(sigma)
        if note:
            args["note"] = note
        self.records.append({"name": RESULT_TOOL, "turn": 0, "args": args})

    def folded(self) -> dict:
        return fold_results(self.records)
