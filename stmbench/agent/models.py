"""Model ports for the agent loop (:mod:`stmbench.agent.loop`).

A port holds its own provider-shaped conversation, append-only, and plays one turn at a time::

    port.start(system, first_message)
    reply = port.turn(tools)              # ModelReply: text, thinking, tool calls, stop, usage
    port.add_tool_results(results)        # every call's result from that turn, in one message
    port.add_user_text(text)              # a nudge after a turn that called no tool

``tools`` are the instrument's MCP tool specs (``name``, ``description``, ``inputSchema``);
results carry MCP content blocks (``{"type": "text", "text"}`` and
``{"type": "image", "data": <base64>, "mimeType": "image/png"}``).

* :class:`AnthropicModel` — Claude through the official ``anthropic`` SDK (``pip install
  "stm-bench[agent]"``); adaptive thinking with summarised thoughts, an explicit effort,
  automatic prompt caching. Server-side refusal fallbacks are off by default: a turn answered
  by another model would no longer measure the model under test.
* :class:`OpenAICompatibleModel` — any model behind an OpenAI-compatible Chat Completions
  endpoint (``POST {base_url}/chat/completions``), over the standard library.
* :class:`ScriptedModel` — a fixed list of turns, for tests and dry runs.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict
    error: str | None = None          # the arguments could not be read (the call is refused)


@dataclass
class ToolResult:
    id: str
    blocks: list[dict]
    is_error: bool = False


@dataclass
class ModelReply:
    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop: str = ""
    usage: dict = field(default_factory=dict)
    served_by: str | None = None
    raw: list[dict] = field(default_factory=list)       # the turn's content, as plain JSON


def _plain(obj: Any) -> Any:
    """A response object as plain JSON (for the transcript), whatever the client returned."""
    for attr in ("model_dump", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn(exclude_none=True) if attr == "model_dump" else fn()
            except TypeError:
                return fn()
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: _plain(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return obj


# ── Claude ──────────────────────────────────────────────────────────────────
class AnthropicModel:
    provider = "anthropic"

    def __init__(self, model: str = "claude-opus-5-5", *, effort: str = "high", max_tokens: int = 64000,
                 thinking_display: str = "summarized", fallbacks: bool = False, max_retries: int = 5,
                 client: Any = None):
        self.model = model
        self.effort = effort
        self.max_tokens = int(max_tokens)
        self.thinking_display = thinking_display
        self.fallbacks = bool(fallbacks)
        self.max_retries = int(max_retries)
        self.client = client
        self.system = ""
        self.messages: list[dict] = []

    def describe(self) -> dict:
        return {"provider": self.provider, "model": self.model, "effort": self.effort,
                "max_tokens": self.max_tokens, "thinking": {"type": "adaptive", "display": self.thinking_display},
                "fallbacks": "default" if self.fallbacks else None, "prompt_caching": "automatic"}

    def _client(self):
        if self.client is None:
            import anthropic                        # the optional extra: stm-bench[agent]

            self.client = anthropic.Anthropic(max_retries=self.max_retries)
        return self.client

    def start(self, system: str, first_message: str) -> None:
        self.system = system
        self.messages = [{"role": "user", "content": first_message}]

    @staticmethod
    def tool_specs(tools: list[dict]) -> list[dict]:
        return [{"name": t["name"], "description": t.get("description", ""), "input_schema": t["inputSchema"]}
                for t in tools]

    def request(self, tools: list[dict]) -> dict:
        req = {"model": self.model, "max_tokens": self.max_tokens, "system": self.system,
               "messages": self.messages, "tools": self.tool_specs(tools),
               "thinking": {"type": "adaptive", "display": self.thinking_display},
               "output_config": {"effort": self.effort}, "cache_control": {"type": "ephemeral"}}
        if self.fallbacks:
            req.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        return req

    def turn(self, tools: list[dict]) -> ModelReply:
        client = self._client()
        api = client.beta.messages if self.fallbacks else client.messages
        with api.stream(**self.request(tools)) as stream:
            msg = stream.get_final_message()
        # the turn goes back exactly as it came (thinking blocks included): history is append-only
        self.messages.append({"role": "assistant", "content": msg.content})
        reply = ModelReply(stop=str(getattr(msg, "stop_reason", "") or ""),
                           served_by=getattr(msg, "model", None), raw=[_plain(b) for b in msg.content])
        text, thinking = [], []
        for b in msg.content:
            kind = getattr(b, "type", None)
            if kind == "text":
                text.append(b.text)
            elif kind == "thinking":
                thinking.append(getattr(b, "thinking", "") or "")
            elif kind == "tool_use":
                args = b.input if isinstance(b.input, dict) else {}
                err = None if isinstance(b.input, dict) else "the arguments are not an object"
                reply.tool_calls.append(ToolCall(str(b.id), str(b.name), dict(args), err))
        reply.text, reply.thinking = "\n".join(text), "\n".join(t for t in thinking if t)
        u = getattr(msg, "usage", None)
        reply.usage = {k: int(getattr(u, k, 0) or 0) for k in
                       ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
        if reply.stop == "refusal":
            det = getattr(msg, "stop_details", None)
            reply.usage["refusal_category"] = getattr(det, "category", None) if det is not None else None
        return reply

    def add_tool_results(self, results: list[ToolResult]) -> None:
        content = []
        for r in results:
            blocks = []
            for b in r.blocks:
                if b.get("type") == "image":
                    blocks.append({"type": "image", "source": {"type": "base64", "media_type": b.get("mimeType", "image/png"),
                                                               "data": b["data"]}})
                elif b.get("type") == "text":
                    blocks.append({"type": "text", "text": str(b.get("text", ""))})
            item = {"type": "tool_result", "tool_use_id": r.id, "content": blocks or [{"type": "text", "text": ""}]}
            if r.is_error:
                item["is_error"] = True
            content.append(item)
        self.messages.append({"role": "user", "content": content})

    def add_user_text(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})


# ── any OpenAI-compatible endpoint ─────────────────────────────────────────
class OpenAICompatibleModel:
    provider = "openai-compatible"

    def __init__(self, model: str, *, base_url: str = "https://api.openai.com/v1", api_key_env: str = "OPENAI_API_KEY",
                 max_tokens: int = 16000, vision: bool = True, timeout_s: float = 600.0,
                 extra_body: dict | None = None, post: Any = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.max_tokens = int(max_tokens)
        self.vision = bool(vision)
        self.timeout_s = float(timeout_s)
        self.extra_body = dict(extra_body or {})
        self._post = post or self._http_post
        self.messages: list[dict] = []

    def describe(self) -> dict:
        return {"provider": self.provider, "model": self.model, "base_url": self.base_url,
                "max_tokens": self.max_tokens, "vision": self.vision, "extra_body": self.extra_body or None}

    def _http_post(self, url: str, body: dict) -> dict:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise RuntimeError(f"no API key: set {self.api_key_env}")
        req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc

    def start(self, system: str, first_message: str) -> None:
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": first_message}]

    @staticmethod
    def tool_specs(tools: list[dict]) -> list[dict]:
        return [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                  "parameters": t["inputSchema"]}} for t in tools]

    def request(self, tools: list[dict]) -> dict:
        return {"model": self.model, "messages": self.messages, "tools": self.tool_specs(tools),
                "tool_choice": "auto", "max_tokens": self.max_tokens, **self.extra_body}

    def turn(self, tools: list[dict]) -> ModelReply:
        resp = self._post(f"{self.base_url}/chat/completions", self.request(tools))
        choice = (resp.get("choices") or [{}])[0]
        m = choice.get("message") or {}
        reply = ModelReply(stop=str(choice.get("finish_reason") or ""), served_by=resp.get("model"),
                           text=str(m.get("content") or ""), thinking=str(m.get("reasoning_content") or ""),
                           raw=[m])
        calls = m.get("tool_calls") or []
        for tc in calls:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
                err = None if isinstance(args, dict) else "the arguments are not an object"
            except json.JSONDecodeError as exc:
                args, err = {}, f"the arguments are not valid JSON ({exc.msg})"
            reply.tool_calls.append(ToolCall(str(tc.get("id")), str(fn.get("name")), args if isinstance(args, dict) else {}, err))
        hist: dict = {"role": "assistant", "content": m.get("content") or ""}
        if calls:
            hist["tool_calls"] = [{"id": tc.get("id"), "type": "function",
                                   "function": {"name": (tc.get("function") or {}).get("name"),
                                                "arguments": (tc.get("function") or {}).get("arguments") or "{}"}}
                                  for tc in calls]
        self.messages.append(hist)                  # reasoning_content is not sent back
        u = resp.get("usage") or {}
        cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or u.get("prompt_cache_hit_tokens") or 0
        reply.usage = {"input_tokens": int(u.get("prompt_tokens") or 0), "output_tokens": int(u.get("completion_tokens") or 0),
                       "cache_read_input_tokens": int(cached or 0), "cache_creation_input_tokens": 0}
        return reply

    def add_tool_results(self, results: list[ToolResult]) -> None:
        images: list[tuple[str, dict]] = []
        for r in results:
            texts = [str(b.get("text", "")) for b in r.blocks if b.get("type") == "text"]
            imgs = [b for b in r.blocks if b.get("type") == "image"]
            body = "\n".join(texts)
            if imgs and not self.vision:
                body += "\n(the image is not shown: this model runs without images; frame_values gives the numbers)"
            self.messages.append({"role": "tool", "tool_call_id": r.id, "content": body})
            images += [(r.id, b) for b in imgs]
        if images and self.vision:
            parts: list[dict] = [{"type": "text", "text": "Images returned by the tool calls above (" +
                                  ", ".join(i for i, _ in images) + "), in order."}]
            parts += [{"type": "image_url", "image_url": {"url": f"data:{b.get('mimeType', 'image/png')};base64,{b['data']}"}}
                      for _, b in images]
            self.messages.append({"role": "user", "content": parts})

    def add_user_text(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})


# ── scripted ────────────────────────────────────────────────────────────────
class ScriptedModel:
    """Plays fixed turns: each a list of ``(tool name, arguments)`` or a string (a turn with
    text and no tool call). Records what it was given, for tests."""

    provider = "scripted"

    def __init__(self, turns: list, model: str = "scripted"):
        self.turns = list(turns)
        self.model = model
        self.k = 0
        self.received: list[list[ToolResult]] = []
        self.nudges: list[str] = []
        self.system = ""

    def describe(self) -> dict:
        return {"provider": self.provider, "model": self.model}

    def start(self, system: str, first_message: str) -> None:
        self.system = system

    def turn(self, tools: list[dict]) -> ModelReply:
        if self.k >= len(self.turns):
            return ModelReply(text="(no more scripted turns)", stop="end_turn")
        t = self.turns[self.k]
        self.k += 1
        if isinstance(t, str):
            return ModelReply(text=t, stop="end_turn")
        calls = [ToolCall(f"call_{self.k}_{j}", name, dict(args)) for j, (name, args) in enumerate(t)]
        return ModelReply(text=f"turn {self.k}", tool_calls=calls, stop="tool_use",
                          usage={"input_tokens": 10, "output_tokens": 5})

    def add_tool_results(self, results: list[ToolResult]) -> None:
        self.received.append(list(results))

    def add_user_text(self, text: str) -> None:
        self.nudges.append(text)
