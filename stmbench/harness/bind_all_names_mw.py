"""Keep every tool NAME bound on every model call (benchmark protocol).

MAST narrows the tool schemas the model sees to a core pack plus whatever packs the model
has loaded. Some providers constrain the *name* of a tool call to the bound list; when the
model reaches for an unloaded tool it may receive a call to a different bound tool with the
same prefix. That is an environment hazard, not a model failure the benchmark wants to measure.

This middleware sits innermost (after MAST's visibility narrowing) and

* appends a **stub** entry for every tool of the loop that is not visible this call —
  same name, one-line description prefixed with its pack, an empty schema — so every
  name the model can read in the catalog is a legal name to emit;
* intercepts a call to a tool that was *not visible* at the last model call: instead of
  executing it (possibly with defaults), it loads the tool's pack(s) by returning
  ``state_delta={"loaded_tool_packs": …}`` and tells the model to call again with
  arguments. The retry is then a normal visible call.

The same rule applies to every model and every mode, so it is part of the protocol, not
an accommodation for one provider. Stub calls are recorded (``stub_calls``) for the
ledger — they cost a tool call, which is fair: the model did not load the pack first.
"""
from __future__ import annotations

from mast.agentruntime.middleware import Middleware
from mast.agentruntime.tools import ToolResult, ToolSpec
from mast.agents._shared import tool_packs as _tp


class BindAllToolNames(Middleware):
    def __init__(self, all_tools: dict, registry=None, agent: str = "instrument_control",
                 desc_chars: int = 72):
        self._all = all_tools                       # the loop's live tool dict
        self._catalog = _tp.build_catalog(agent, list(all_tools.values()), registry)
        self._visible: set[str] = set()
        self._loaded: list[str] = []
        self._desc_chars = int(desc_chars)
        self.stub_calls: list[dict] = []

    # ── helpers ──
    def packs_of(self, name: str) -> list[str]:
        return sorted(p for p in self._catalog.packs_by_tool.get(name, ()) if p != _tp.CORE)

    def _stub(self, spec: ToolSpec) -> ToolSpec:
        first_line = (spec.description or "").strip().splitlines()[0] if (spec.description or "").strip() else ""
        packs = self.packs_of(spec.name)
        return ToolSpec(
            name=spec.name,
            description=f"[未加载·包 {'/'.join(packs) if packs else '?'}] {first_line[:self._desc_chars]}",
            schema={"type": "object", "properties": {}},
            fn=spec.fn, touches_instrument=spec.touches_instrument,
            wants_call_id=getattr(spec, "wants_call_id", False),
        )

    # ── hooks ──
    def wrap_model_call(self, request, call_next):
        visible = {getattr(t, "name", None) for t in request.tools}
        state = getattr(request, "state", None) or {}
        loaded = state.get("loaded_tool_packs") or []
        self._loaded = list(loaded)
        self._visible = {n for n in visible if n}
        stubs = [self._stub(spec) for name, spec in self._all.items() if name not in visible]
        if stubs:
            request.tools = list(request.tools) + stubs
        return call_next(request)

    def _merge_loaded(self, raw):
        """Keep ``loaded_tool_packs`` cumulative across tool results.

        MAST's ``load_tool_pack`` / ``search_tools`` return a state update containing the
        loaded pack. The loop applies a plain dictionary update, so a later load can replace
        the earlier list unless this middleware accumulates the entries. The stub path below
        already accumulates them; this makes the tool path follow the same rule.
        """
        from mast.agentruntime.tools import coerce_result

        res = coerce_result(raw)
        delta = getattr(res, "state_delta", None)
        packs = delta.get("loaded_tool_packs") if isinstance(delta, dict) else None
        if packs:
            loaded = list(self._loaded) + [p for p in packs if isinstance(p, str) and p not in self._loaded]
            self._loaded = loaded
            self._visible |= set(self._catalog.visible(loaded))
            delta["loaded_tool_packs"] = loaded
        return res

    def wrap_tool_call(self, call, ctx, call_next):
        name = getattr(call, "name", "")
        if name in self._visible or name not in self._all:
            return self._merge_loaded(call_next(call))
        packs = self.packs_of(name)
        loaded = list(self._loaded) + [p for p in packs if p not in self._loaded]
        self._loaded = loaded
        self._visible |= set(self._catalog.visible(loaded))
        self._visible.add(name)
        self.stub_calls.append({"name": name, "packs": packs, "args": dict(getattr(call, "args", {}) or {})})
        pk = "/".join(packs) if packs else "?"
        return ToolResult(
            text=(f"{name} 属于尚未加载的工具包 {pk}，这一次没有执行。已为你加载该包，"
                  f"它的完整参数会出现在下一步；请带上参数重新调用 {name}。"),
            state_delta={"loaded_tool_packs": loaded}, ok=True)
