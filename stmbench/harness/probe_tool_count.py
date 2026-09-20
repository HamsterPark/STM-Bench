"""Does the provider honour a tool call to a name bound deep in a LONG tool list?

Binds two real-looking tools plus N stub tools (empty schema) and forces a call to a
stub name near the END of the list. If the returned name is the stub's, the provider
honours N tools; if it is one of the first tools, the list was silently truncated and
"every name bound" is no defence beyond that N.

    python -m stmbench.harness.probe_tool_count --model kimi-k3 --ns 60,128,200,300,520
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stmbench.paths import data_path


def _tool(name: str, desc: str, props: dict | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props or {}}}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="kimi-k3")
    ap.add_argument("--ns", default="60,128,200,300,520")
    ap.add_argument("--out", default=str(data_path("tmp_dbg", "tool_count.json")))
    ap.add_argument("--no-force", action="store_true",
                    help="do not set tool_choice='required' (thinking-mode providers reject it)")
    a = ap.parse_args(argv)
    from langchain_core.messages import HumanMessage, SystemMessage
    from mast.agents._shared.models import make_chat_model
    from stmbench.harness.runtime_host import _export_api_keys_from_repo

    _export_api_keys_from_repo()
    base = make_chat_model(model_id=a.model, usage_source="probe")
    rows = []
    for n in [int(x) for x in a.ns.split(",") if x.strip()]:
        tools = [_tool("HomeZController", "把 Z 压电移动到 home 位置。"),
                 _tool("SetScriptAutosave", "设置脚本自动保存开关。", {"on": {"type": "boolean"}})]
        stubs = [_tool(f"Stub{i:03d}Tool", f"[未加载·包 p{i % 7}] 桩工具 {i}") for i in range(n)]
        target = "TargetScanBufferGet"
        stubs.insert(max(0, len(stubs) - 3), _tool(target, "[未加载·包 scan] 读取扫描缓冲区设置。"))
        tools = tools + stubs
        try:
            if a.no_force:
                model = base.bind_tools(tools)
            else:
                model = base.bind_tools(tools, tool_choice=("any" if str(a.model).startswith("claude") else "required"))
            msg = model.invoke([SystemMessage(content="你是仪器控制助手，按指令调用工具。"),
                                HumanMessage(content=f"请调用名为 {target} 的工具（不带参数）。只做这一件事。")])
            calls = [c.get("name") for c in (getattr(msg, "tool_calls", None) or [])]
            rows.append({"n_tools": len(tools), "target_index": tools.index(next(t for t in tools if t["function"]["name"] == target)),
                         "returned": calls, "honoured": target in calls, "text": str(getattr(msg, "content", ""))[:160]})
        except Exception as exc:  # noqa: BLE001
            rows.append({"n_tools": len(tools), "error": f"{type(exc).__name__}: {exc}"[:300]})
        print(rows[-1], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"model": a.model, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
