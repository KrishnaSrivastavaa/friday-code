"""Tests for the hand-rolled message types and OpenAI conversions."""

from __future__ import annotations

import json

from types import SimpleNamespace

from my_agent.messages import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
    from_openai,
    to_openai,
)


def test_to_openai_system_and_user():
    assert to_openai(SystemMessage("sys")) == {"role": "system", "content": "sys"}
    assert to_openai(UserMessage("hi")) == {"role": "user", "content": "hi"}


def test_to_openai_assistant_text():
    message = AssistantMessage(content="hello")
    payload = to_openai(message)
    assert payload == {"role": "assistant", "content": "hello"}


def test_to_openai_assistant_with_tool_calls():
    message = AssistantMessage(
        content="",
        tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "x.py"})],
    )
    payload = to_openai(message)
    assert payload["role"] == "assistant"
    assert payload["tool_calls"][0] == {
        "id": "c1",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path": "x.py"}'},
    }


def test_to_openai_tool_message():
    message = ToolMessage(tool_call_id="c1", content="ok", name="read_file")
    assert to_openai(message) == {"role": "tool", "tool_call_id": "c1", "content": "ok"}


def test_from_openai_parses_tool_calls():
    response_message = SimpleNamespace(
        content="thinking",
        tool_calls=[
            SimpleNamespace(
                id="c1",
                type="function",
                function=SimpleNamespace(name="list_files", arguments='{"path": "."}'),
            )
        ],
    )
    message = from_openai(response_message)
    assert isinstance(message, AssistantMessage)
    assert message.content == "thinking"
    assert message.tool_calls[0].name == "list_files"
    assert message.tool_calls[0].arguments == {"path": "."}
    assert message.tool_calls[0].error is None


def test_from_openai_handles_invalid_json():
    response_message = SimpleNamespace(
        content="",
        tool_calls=[
            SimpleNamespace(
                id="c1",
                type="function",
                function=SimpleNamespace(name="read_file", arguments="{bad json"),
            )
        ],
    )
    message = from_openai(response_message)
    assert message.tool_calls[0].arguments == {}
    assert message.tool_calls[0].error is not None


def test_from_openai_handles_non_object_arguments():
    response_message = SimpleNamespace(
        content="",
        tool_calls=[
            SimpleNamespace(
                id="c1",
                type="function",
                function=SimpleNamespace(name="read_file", arguments='["x"]'),
            )
        ],
    )
    message = from_openai(response_message)
    assert message.tool_calls[0].error is not None


def test_to_openai_preserves_raw_arguments_for_errored_call():
    call = ToolCall(
        id="c1",
        name="read_file",
        arguments={},
        raw_arguments="{bad",
        error="invalid JSON arguments: ...",
    )
    message = AssistantMessage(content="", tool_calls=[call])
    payload = to_openai(message)
    assert payload["tool_calls"][0]["function"]["arguments"] == "{bad"


def test_usage_accumulates():
    usage = Usage()
    usage.add(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    usage.add(3, 2)
    assert usage.prompt_tokens == 13
    assert usage.completion_tokens == 7
    assert usage.total_tokens == 15
