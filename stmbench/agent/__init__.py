"""A language model at the benchmark instrument, with the instrument's own tools and nothing else.

* :mod:`stmbench.agent.loop` — the episode loop, the prompt and the record (``driver.json``)
* :mod:`stmbench.agent.models` — model ports: Claude through the Anthropic SDK, any model behind an
  OpenAI-compatible Chat Completions endpoint, and a scripted port for tests
"""
from .loop import FIRST_MESSAGE, NUDGE, SYSTEM_PROMPT, AgentResult, prompt_sha256, run_agent_episode
from .models import AnthropicModel, ModelReply, OpenAICompatibleModel, ScriptedModel, ToolCall, ToolResult

__all__ = ["AgentResult", "AnthropicModel", "FIRST_MESSAGE", "ModelReply", "NUDGE", "OpenAICompatibleModel",
           "SYSTEM_PROMPT", "ScriptedModel", "ToolCall", "ToolResult", "prompt_sha256", "run_agent_episode"]
