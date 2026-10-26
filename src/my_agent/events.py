"""Events emitted by the agent loop.

The CLI (or any other frontend) can subscribe to these to show progress while
the agent is working, without knowing how the loop is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AssistantEvent:
    """The model produced text (and may also have requested tools)."""

    text: str


@dataclass
class ToolCallEvent:
    """The model asked to run a tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResultEvent:
    """A tool finished; `content` is what the model will see."""

    id: str
    name: str
    content: str


AgentEvent = AssistantEvent | ToolCallEvent | ToolResultEvent
