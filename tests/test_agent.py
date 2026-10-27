"""Tests for the hand-rolled agent loop (no LangGraph)."""

from __future__ import annotations

from my_agent.agent import Agent, MAX_ITERATIONS, build_agent, build_system_prompt, run
from my_agent.config.settings import settings
from my_agent.events import AssistantEvent, ToolCallEvent, ToolResultEvent
from my_agent.messages import AssistantMessage, ToolMessage, UserMessage
from my_agent.tools import CODING_TOOLS
from tests.helpers import ScriptedClient, assistant, fake_response, tool_call


# --------------------------------------------------------------------------- #
# System prompt
# --------------------------------------------------------------------------- #


def test_system_prompt_includes_workspace(workspace):
    prompt = build_system_prompt()
    assert str(workspace) in prompt


def test_system_prompt_guides_tool_usage(workspace):
    prompt = build_system_prompt().lower()
    for expected in ("read_file", "edit_file", "run_command", "verify"):
        assert expected in prompt


# --------------------------------------------------------------------------- #
# Agent loop
# --------------------------------------------------------------------------- #


def test_agent_sends_tools_schema():
    client = ScriptedClient([fake_response(content="done")])
    agent = Agent(client, model_name="gpt-4o-mini")

    result = agent.run([UserMessage(content="hi")])

    assert result.llm_calls == 1
    request = client.requests[0]
    tool_names = [t["function"]["name"] for t in request["tools"]]
    assert tool_names == [t.name for t in CODING_TOOLS]
    assert request["tool_choice"] == "auto"
    assert request["model"] == "gpt-4o-mini"
    assert result.final == "done"


def test_agent_prepends_system_prompt():
    client = ScriptedClient([fake_response(content="ok")])
    agent = Agent(client)

    agent.run([UserMessage(content="hi")])

    messages = client.requests[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_agent_runs_tool_then_answers(workspace):
    (workspace / "hello.py").write_text("print('hi')\n")
    client = ScriptedClient(
        [
            fake_response(tool_calls=[tool_call("read_file", {"path": "hello.py"}, "c1")]),
            fake_response(content="The file prints hi."),
        ]
    )
    agent = Agent(client)

    result = agent.run([UserMessage(content="what does hello.py do?")])

    types = [type(m).__name__ for m in result.messages]
    assert "ToolMessage" in types
    tool_msg = next(m for m in result.messages if isinstance(m, ToolMessage))
    assert "print('hi')" in tool_msg.content
    assert result.final == "The file prints hi."
    assert result.llm_calls == 2


def test_agent_executes_multiple_tool_calls_in_one_response(workspace):
    (workspace / "a.py").write_text("x")
    (workspace / "b.py").write_text("y")
    client = ScriptedClient(
        [
            fake_response(
                tool_calls=[
                    tool_call("read_file", {"path": "a.py"}, "c1"),
                    tool_call("read_file", {"path": "b.py"}, "c2"),
                ]
            ),
            fake_response(content="read both"),
        ]
    )
    agent = Agent(client)

    result = agent.run([UserMessage(content="go")])

    tool_messages = [m for m in result.messages if isinstance(m, ToolMessage)]
    assert [m.tool_call_id for m in tool_messages] == ["c1", "c2"]
    assert result.llm_calls == 2


def test_agent_handles_unknown_tool(workspace):
    client = ScriptedClient(
        [
            fake_response(tool_calls=[tool_call("nope", {}, "c1")]),
            fake_response(content="handled"),
        ]
    )
    agent = Agent(client)

    result = agent.run([UserMessage(content="go")])

    tool_msg = result.messages[1]
    assert "unknown tool" in tool_msg.content


def test_agent_turns_tool_exceptions_into_messages(workspace):
    client = ScriptedClient(
        [
            fake_response(tool_calls=[tool_call("read_file", {}, "c1")]),
            fake_response(content="recovered"),
        ]
    )
    agent = Agent(client)

    result = agent.run([UserMessage(content="go")])

    tool_msg = result.messages[1]
    assert "Error while running 'read_file'" in tool_msg.content


def test_agent_handles_invalid_json_arguments(workspace):
    import types

    # Build a response with malformed JSON arguments
    message = types.SimpleNamespace(
        content="",
        tool_calls=[
            types.SimpleNamespace(
                id="c1",
                type="function",
                function=types.SimpleNamespace(name="read_file", arguments="{bad"),
            )
        ],
    )
    raw = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=message)],
        usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    client = ScriptedClient([raw, fake_response(content="ok")])
    agent = Agent(client)

    result = agent.run([UserMessage(content="go")])

    tool_msg = result.messages[1]
    assert "could not parse arguments" in tool_msg.content


def test_agent_stops_at_max_iterations(workspace):
    responses = [fake_response(tool_calls=[tool_call("list_files", {"path": "."}, f"c{i}")]) for i in range(MAX_ITERATIONS + 5)]
    agent = Agent(ScriptedClient(responses))

    result = agent.run([UserMessage(content="loop")])

    assert result.llm_calls == MAX_ITERATIONS
    tool_messages = [m for m in result.messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == MAX_ITERATIONS


def test_agent_does_not_mutate_input_history(workspace):
    history = [UserMessage(content="hi")]
    client = ScriptedClient([fake_response(content="done")])
    agent = Agent(client)

    result = agent.run(history)

    assert history == [UserMessage(content="hi")]
    assert len(result.messages) == 1


def test_agent_accumulates_usage(workspace):
    client = ScriptedClient(
        [
            fake_response(content="", tool_calls=[tool_call("list_files", {"path": "."}, "c1")], prompt_tokens=10, completion_tokens=5),
            fake_response(content="done", prompt_tokens=7, completion_tokens=3),
        ]
    )
    agent = Agent(client)

    result = agent.run([UserMessage(content="go")])

    assert result.usage.prompt_tokens == 17
    assert result.usage.completion_tokens == 8
    assert result.usage.total_tokens == 25


def test_agent_emits_events(workspace):
    events = []
    client = ScriptedClient(
        [
            fake_response(content="Let me look.", tool_calls=[tool_call("list_files", {"path": "."}, "c1")]),
            fake_response(content="done"),
        ]
    )
    agent = Agent(client)

    agent.run([UserMessage(content="go")], on_event=events.append)

    kinds = [type(e).__name__ for e in events]
    assert kinds == ["AssistantEvent", "ToolCallEvent", "ToolResultEvent", "AssistantEvent"]
    assert events[0].text == "Let me look."
    assert events[1].name == "list_files"
    assert isinstance(events[2].content, str)


def test_run_helper(workspace):
    client = ScriptedClient([fake_response(content="hi")])
    agent = Agent(client)
    result = run(agent, "hello")
    assert result.final == "hi"


def test_build_agent_creates_client(monkeypatch):
    fake_client = ScriptedClient([fake_response(content="ok")])
    monkeypatch.setattr("my_agent.models.provider.get_client", lambda *a, **k: fake_client)
    agent = build_agent()
    result = run(agent, "x")
    assert result.final == "ok"


def test_agent_omits_tools_when_none():
    client = ScriptedClient([fake_response(content="ok")])
    agent = Agent(client, tools=[])
    agent.run([UserMessage(content="hi")])
    assert "tools" not in client.requests[0]
