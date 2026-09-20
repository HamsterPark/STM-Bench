"""Every tool name stays bound; a call to a not-yet-visible tool loads its pack instead
of executing (the kimi-k3 name-constraint hazard)."""
from __future__ import annotations

import pytest

from tests.conftest import requires_mast

pytestmark = requires_mast


class _Narrow:
    """Stand-in for MAST's visibility narrowing: keep only `keep` unless a pack is loaded."""

    name = "Narrow"

    def __init__(self, keep, catalog_visible):
        self.keep = set(keep)
        self.catalog_visible = catalog_visible

    def before_model(self, turn):
        pass

    def after_model(self, turn, response):
        pass

    def wrap_model_call(self, request, call_next):
        loaded = (getattr(request, "state", None) or {}).get("loaded_tool_packs") or []
        allowed = self.keep | set(self.catalog_visible(loaded))
        request.tools = [t for t in request.tools if t.name in allowed]
        return call_next(request)

    def wrap_tool_call(self, call, ctx, call_next):
        return call_next(call)


def test_stub_names_are_bound_and_first_call_loads_the_pack():
    from mast.agentruntime.middleware import MiddlewareStack, ToolCallView
    from mast.agentruntime.model import ModelRequest, ModelResponse
    from mast.agentruntime.tools import ToolResult, ToolSpec
    from mast.agents.instrument_control.tools import discover_instrument_skills

    from stmbench.harness.bind_all_names_mw import BindAllToolNames

    reg = discover_instrument_skills()
    executed = []
    tools = {n: ToolSpec(name=n, description=f"{n} does things", schema={"type": "object", "properties": {"x": {"type": "number"}}},
                         fn=lambda a, c, n=n: executed.append(n) or ToolResult(text="ran " + n))
             for n in ("GetBias", "StartScan", "SetScanBuffer")}
    mw = BindAllToolNames(tools, registry=reg)
    scan_packs = mw.packs_of("StartScan")
    assert scan_packs, "StartScan must belong to a non-core pack for this test to mean anything"
    narrow = _Narrow(keep={"GetBias"}, catalog_visible=mw._catalog.visible)
    stack = MiddlewareStack([narrow, mw])

    seen = {}

    def invoke(req):
        seen["names"] = [t.name for t in req.tools]
        seen["desc"] = {t.name: t.description for t in req.tools}
        seen["schema"] = {t.name: t.schema for t in req.tools}
        return ModelResponse(text="")

    stack.call_model(ModelRequest(tools=list(tools.values()), state={}), invoke)
    assert set(seen["names"]) == {"GetBias", "StartScan", "SetScanBuffer"}
    assert seen["desc"]["StartScan"].startswith("[未加载")
    assert seen["schema"]["StartScan"] == {"type": "object", "properties": {}}
    assert not seen["desc"]["GetBias"].startswith("[未加载")

    # a call to the hidden tool loads its pack instead of executing
    r = stack.call_tool(ToolCallView("StartScan", {"x": 1}), None, lambda v: tools[v.name].fn(v.args, None))
    assert executed == []
    assert isinstance(r, ToolResult) and r.ok and "StartScan" in r.text
    assert set(r.state_delta["loaded_tool_packs"]) >= set(scan_packs)
    assert mw.stub_calls and mw.stub_calls[0]["name"] == "StartScan"

    # visible tools execute normally; after the load the same name executes for real
    r2 = stack.call_tool(ToolCallView("GetBias", {}), None, lambda v: tools[v.name].fn(v.args, None))
    assert executed == ["GetBias"] and r2.text == "ran GetBias"
    r3 = stack.call_tool(ToolCallView("StartScan", {"x": 1}), None, lambda v: tools[v.name].fn(v.args, None))
    assert executed == ["GetBias", "StartScan"] and r3.text == "ran StartScan"

    # next model call with the pack loaded: StartScan is now a real (narrowing-visible) entry
    stack.call_model(ModelRequest(tools=list(tools.values()), state={"loaded_tool_packs": r.state_delta["loaded_tool_packs"]}), invoke)
    assert not seen["desc"]["StartScan"].startswith("[未加载")


def test_loading_a_second_pack_keeps_the_first_one_loaded():
    """``load_tool_pack`` returns ``{"loaded_tool_packs": [<that pack>]}``; the v2 loop
    applies it as a plain dict update, so a second load used to REPLACE the first and the
    first pack's tools fell back to stubs (two agents lost turns to it, 2026-09-11). The
    middleware merges, so packs only ever accumulate — as the stub path always did."""
    from mast.agentruntime.middleware import MiddlewareStack, ToolCallView
    from mast.agentruntime.model import ModelRequest, ModelResponse
    from mast.agentruntime.tools import ToolResult, ToolSpec
    from mast.agents.instrument_control.tools import discover_instrument_skills

    from stmbench.harness.bind_all_names_mw import BindAllToolNames

    reg = discover_instrument_skills()
    tools = {n: ToolSpec(name=n, description=f"{n} does things", schema={"type": "object", "properties": {}},
                         fn=lambda a, c: ToolResult(text="ran"))
             for n in ("GetBias", "StartScan", "AcquireSTS")}
    # a load_tool_pack look-alike: reports ONLY the pack it was asked for, as MAST's does
    tools["load_tool_pack"] = ToolSpec(
        name="load_tool_pack", description="load a pack", schema={"type": "object", "properties": {}},
        fn=lambda a, c: ToolResult(text=f"loaded {a['pack']}", state_delta={"loaded_tool_packs": [a["pack"]]}))
    mw = BindAllToolNames(tools, registry=reg)
    scan_pack, sts_pack = mw.packs_of("StartScan")[0], mw.packs_of("AcquireSTS")[0]
    assert scan_pack != sts_pack, (scan_pack, sts_pack)
    narrow = _Narrow(keep={"GetBias", "load_tool_pack"}, catalog_visible=mw._catalog.visible)
    stack = MiddlewareStack([narrow, mw])
    seen = {}

    def invoke(req):
        seen["stubs"] = {t.name for t in req.tools if t.description.startswith("[未加载")}
        return ModelResponse(text="")

    def run(name, args):
        return stack.call_tool(ToolCallView(name, args), None, lambda v: tools[v.name].fn(v.args, None))

    stack.call_model(ModelRequest(tools=list(tools.values()), state={}), invoke)
    assert {"StartScan", "AcquireSTS"} <= seen["stubs"]
    r1 = run("load_tool_pack", {"pack": scan_pack})
    r2 = run("load_tool_pack", {"pack": sts_pack})
    assert r1.state_delta["loaded_tool_packs"] == [scan_pack]
    assert r2.state_delta["loaded_tool_packs"] == [scan_pack, sts_pack]      # merged, not replaced
    # what the loop would carry into the next model call: both packs visible, neither a stub
    stack.call_model(ModelRequest(tools=list(tools.values()),
                                  state={"loaded_tool_packs": r2.state_delta["loaded_tool_packs"]}), invoke)
    assert not ({"StartScan", "AcquireSTS"} & seen["stubs"]), seen["stubs"]
    # and the first pack's tool executes for real instead of "loading" again
    assert run("StartScan", {}).text == "ran" and mw.stub_calls == []
