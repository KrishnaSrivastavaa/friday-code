"""Conversation message types used by the agent.

These are intentionally small, dependency-free dataclasses. They mirror the
OpenAI chat format closely, with two helpers to convert in each direction:

* :func:`to_openai` turns our messages into the dicts the API expects.
* :func:`from_openai` turns an API response message into an AssistantMessage.

Keeping our own types means the rest of the app (CLI, evals, tests) never has to
know about the OpenAI SDK's response objects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    # Preserved so an unparseable argument payload can still be echoed back to
    # the API (every tool call must get a matching tool message).
    raw_arguments: str | None = None
    # Set when the arguments were not valid JSON; `arguments` is then empty.
    error: str | None = None


@dataclass
class SystemMessage:
    content: str
    role: str = field(default="system", init=False)


@dataclass
class UserMessage:
    content: str
    role: str = field(default="user", init=False)


@dataclass
class AssistantMessage:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    role: str = field(default="assistant", init=False)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class ToolMessage:
    tool_call_id: str
    content: str
    name: str | None = None
    role: str = field(default="tool", init=False)


Message = SystemMessage | UserMessage | AssistantMessage | ToolMessage


@dataclass
class Usage:
    """Token usage accumulated across a run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        self.prompt_tokens += prompt_tokens or 0
        self.completion_tokens += completion_tokens or 0
        self.total_tokens += total_tokens or 0


def to_openai(message: Message) -> dict[str, Any]:
    """Serialize one of our messages into an OpenAI chat message dict."""
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}

    if isinstance(message, UserMessage):
        return {"role": "user", "content": message.content}

    if isinstance(message, AssistantMessage):
        payload: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or None,
        }
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": _arguments_json(call),
                    },
                }
                for call in message.tool_calls
            ]
        return payload

    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }

    raise TypeError(f"Unsupported message type: {type(message).__name__}")


def from_openai(message: Any) -> AssistantMessage:
    """Convert an OpenAI response message (object or dict) to AssistantMessage."""
    content = getattr(message, "content", None) or ""
    calls: list[ToolCall] = []

    for raw in getattr(message, "tool_calls", None) or []:
        call_id = getattr(raw, "id", "")
        function = getattr(raw, "function", None)
        name = getattr(function, "name", "")
        raw_arguments = getattr(function, "arguments", "") or ""

        try:
            parsed = json.loads(raw_arguments) if raw_arguments.strip() else {}
            if not isinstance(parsed, dict):
                raise ValueError("arguments must be a JSON object")
            calls.append(
                ToolCall(
                    id=call_id,
                    name=name,
                    arguments=parsed,
                    raw_arguments=raw_arguments,
                )
            )
        except (ValueError, TypeError) as exc:
            calls.append(
                ToolCall(
                    id=call_id,
                    name=name,
                    arguments={},
                    raw_arguments=raw_arguments,
                    error=str(exc),
                )
            )

    return AssistantMessage(content=content, tool_calls=calls)


def _arguments_json(call: ToolCall) -> str:
    """The original argument string for broken calls, else a fresh encoding."""
    if call.error is not None and call.raw_arguments is not None:
        return call.raw_arguments
    return json.dumps(call.arguments)
