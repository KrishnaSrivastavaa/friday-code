"""Shared test doubles and helpers for the OpenAI-SDK agent."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from src.my_agent.messages import AssistantMessage, ToolCall


def tool_call(name: str, args: dict[str, Any] | None = None, call_id: str = "call_1") -> ToolCall:
    """Build a :class:`ToolCall` requested by the model."""
    return ToolCall(id=call_id, name=name, arguments=args or {})


def assistant(content: str = "", tool_calls=None) -> AssistantMessage:
    """Build an :class:`AssistantMessage`."""
    return AssistantMessage(content=content, tool_calls=list(tool_calls or []))


def _openai_tool_calls(tool_calls):
    raw = []
    for call in tool_calls or []:
        if isinstance(call, ToolCall):
            arguments = call.raw_arguments if call.error and call.raw_arguments is not None else json.dumps(call.arguments)
            raw.append(
                SimpleNamespace(
                    id=call.id,
                    type="function",
                    function=SimpleNamespace(name=call.name, arguments=arguments),
                )
            )
        elif isinstance(call, (tuple, list)):
            call_id, name, args = call[0], call[1], call[2]
            raw.append(
                SimpleNamespace(
                    id=call_id,
                    type="function",
                    function=SimpleNamespace(name=name, arguments=json.dumps(args or {})),
                )
            )
        else:
            raise TypeError(f"Unsupported tool call spec: {type(call)}")
    return raw


def fake_response(
    content: str = "",
    tool_calls=None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
):
    """A fake OpenAI chat completion response object."""
    raw_calls = _openai_tool_calls(tool_calls)
    message = SimpleNamespace(
        content=content or None,
        tool_calls=raw_calls or None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, message=message, finish_reason="tool_calls" if raw_calls else "stop")],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


class ScriptedClient:
    """A deterministic stand-in for ``openai.OpenAI``.

    Queue response objects (or let it return a default ``done`` response) and
    record every ``chat.completions.create`` call so tests can assert on the
    request payload.
    """

    def __init__(self, responses=None):
        self._responses = list(responses or [])
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if self._responses:
            return self._responses.pop(0)
        return fake_response(content="done")


# Backwards-compatible names used by older tests/helpers
ai_tool_call = tool_call
ai_tool_calls = lambda *calls: assistant(tool_calls=[tool_call(c[1], c[2], c[0]) for c in calls])
ScriptedModel = ScriptedClient

