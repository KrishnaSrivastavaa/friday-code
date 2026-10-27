"""Tests for the PyPilot CLI (helpers, setup flow and rendering)."""

from __future__ import annotations

import io
import sys
from argparse import Namespace

import pytest
from rich.console import Console

from my_agent import cli
from my_agent.agent import AgentResult
from my_agent.events import AssistantEvent, ToolCallEvent, ToolResultEvent
from my_agent.messages import AssistantMessage, ToolMessage, UserMessage
from tests.helpers import ScriptedClient, assistant


@pytest.fixture
def capture_console(monkeypatch):
    """Redirect the CLI's Rich console into a buffer we can assert on."""
    buffer = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=buffer, width=100))
    return buffer


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def test_content_text_from_string():
    assert cli._content_text(AssistantMessage(content="hi")) == "hi"


def test_content_text_from_block_list():
    class BlockMessage:
        content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]

    assert cli._content_text(BlockMessage()) == "ab"


def test_content_text_empty():
    assert cli._content_text(AssistantMessage(content="")) == ""


def test_brief_truncates():
    assert cli._brief({"a": 1}) == "{'a': 1}"
    assert cli._brief("x" * 100, limit=10).endswith("...")


def test_first_line():
    assert cli._first_line("a\nb") == "a"
    assert cli._first_line("") == ""
    assert cli._first_line("x" * 200, limit=10).endswith("...")


# --------------------------------------------------------------------------- #
# save_env
# --------------------------------------------------------------------------- #


def test_save_env_updates_and_appends(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("EXISTING=1\nMODEL_NAME=old\n")
    monkeypatch.setattr(cli, "ENV_FILE", env)

    cli.save_env({"MODEL_NAME": "new", "OPENAI_API_KEY": "k"})

    text = env.read_text()
    assert "MODEL_NAME=new" in text
    assert "OPENAI_API_KEY=k" in text
    assert "EXISTING=1" in text
    assert "MODEL_NAME=old" not in text


# --------------------------------------------------------------------------- #
# configure
# --------------------------------------------------------------------------- #


def test_configure_non_interactive(capture_console, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    args = Namespace(provider="openai", model="gpt-4o-mini", api_key="sk-test123456", save=False)
    provider, model, key = cli.configure(args)
    assert (provider, model, key) == ("openai", "gpt-4o-mini", "sk-test123456")


def test_configure_requires_key_when_not_interactive(monkeypatch, capture_console, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Use a clean home dir to avoid picking up saved config
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    args = Namespace(provider="openai", model="gpt-4o-mini", api_key=None, save=False)
    with pytest.raises(SystemExit):
        cli.configure(args)


def test_configure_defaults_provider_and_model(monkeypatch, capture_console, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    args = Namespace(provider=None, model=None, api_key="sk-test123456", save=False)
    provider, model, key = cli.configure(args)
    assert provider == "openai"
    assert model  # falls back to settings.model_name
    assert key == "sk-test123456"


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def test_handle_command_exit():
    assert cli.handle_command("/exit", [], "openai", "m")[0] == "exit"
    assert cli.handle_command("/quit", [], "openai", "m")[0] == "exit"


def test_handle_command_unknown_returns_ok(capture_console):
    assert cli.handle_command("/bogus", [], "openai", "m")[0] == "ok"


def test_handle_command_reset_clears_history(capture_console):
    history = [1, 2, 3]
    cli.handle_command("/reset", history, "openai", "m")
    assert history == []


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def test_print_helpers_render(capture_console):
    cli.print_banner("openai", "gpt-4o")
    cli.print_help()
    cli.print_tools()
    cli.print_model("openai", "gpt-4o")
    cli.print_tool_call("read_file", {"path": "x.py"})
    cli.print_tool_result("line1\nline2")
    cli.print_assistant("## Title\n\n```python\nx = 1\n```")

    output = capture_console.getvalue()
    assert "ARC-CODER" in output
    assert "read_file" in output


def test_run_turn_streams_and_updates_history(capture_console, workspace):
    class FakeAgent:
        def run(self, messages, on_event=None, **kwargs):
            if on_event:
                on_event(ToolCallEvent(id="c1", name="list_files", arguments={"path": "."}))
                on_event(ToolResultEvent(id="c1", name="list_files", content="a.py"))
                on_event(AssistantEvent(text="done"))
            return AgentResult(
                messages=[
                    AssistantMessage(
                        content="",
                        tool_calls=[__import__("my_agent.messages", fromlist=["ToolCall"]).ToolCall(id="c1", name="list_files", arguments={"path": "."})],
                    ),
                    ToolMessage(tool_call_id="c1", content="a.py", name="list_files"),
                    AssistantMessage(content="done"),
                ],
                llm_calls=2,
            )

    history: list = []
    cli.run_turn(FakeAgent(), history, "list files")

    output = capture_console.getvalue()
    assert "list_files" in output
    assert "done" in output
    # user + assistant + tool + assistant
    assert len(history) == 4
    assert isinstance(history[0], UserMessage)
    assert any(isinstance(m, ToolMessage) for m in history)


# --------------------------------------------------------------------------- #
# prompt_toolkit fallback
# --------------------------------------------------------------------------- #


def test_make_prompt_session_returns_none_when_not_a_tty(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    assert cli._make_prompt_session() is None


# --------------------------------------------------------------------------- #
# main wiring
# --------------------------------------------------------------------------- #


def test_main_one_shot(capture_console, workspace, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    class FakeAgent:
        def __init__(self, client=None, model_name=None):
            pass

        def run(self, messages, on_event=None, **kwargs):
            if on_event:
                on_event(AssistantEvent(text="task complete"))
            return AgentResult(messages=[AssistantMessage(content="task complete")], llm_calls=1)

    monkeypatch.setattr(cli, "get_client", lambda **kwargs: object())
    monkeypatch.setattr(cli, "Agent", FakeAgent)

    cli.main(["--api-key", "sk-test123456", "--model", "gpt-4o-mini", "do the thing"])

    assert "task complete" in capture_console.getvalue()


def test_main_without_key_exits(monkeypatch, capture_console):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        cli.main(["--model", "gpt-4o-mini"])


# --------------------------------------------------------------------------- #
# More command + rendering coverage
# --------------------------------------------------------------------------- #


def test_handle_command_info_commands_return_ok(capture_console):
    for command in ("/help", "/h", "/?", "/model", "/tools", "/workspace"):
        result = cli.handle_command(command, [], "openai", "gpt-4o")
        status = result[0] if isinstance(result, tuple) else result
        assert status == "ok"


def test_handle_command_clear_alias(capture_console):
    history = [1, 2]
    cli.handle_command("/clear", history, "openai", "gpt-4o")
    assert history == []


def test_print_assistant_ignores_empty(capture_console):
    cli.print_assistant("   ")
    assert capture_console.getvalue() == ""


def test_make_output_safe_is_idempotent():
    cli._make_output_safe()
    cli._make_output_safe()


def test_run_turn_reports_streaming_errors(capture_console, workspace):
    class BoomAgent:
        def run(self, messages, on_event=None, **kwargs):
            raise RuntimeError("kaboom")

    history: list = []
    cli.run_turn(BoomAgent(), history, "go")

    assert "kaboom" in capture_console.getvalue()
    # the user message was recorded even though the turn failed
    assert len(history) == 1


# --------------------------------------------------------------------------- #
# repl loop
# --------------------------------------------------------------------------- #


def test_repl_prints_bye_on_eof(capture_console, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a, **k: (_ for _ in ()).throw(EOFError))

    cli.repl(None, "openai", "gpt-4o")
    assert "bye from" in capture_console.getvalue()

    assert "bye from" in capture_console.getvalue()


def test_repl_runs_a_turn_then_exits(capture_console, monkeypatch, workspace):
    class FakeAgent:
        def run(self, messages, on_event=None, **kwargs):
            if on_event:
                on_event(AssistantEvent(text="hi there"))
            return AgentResult(messages=[AssistantMessage(content="hi there")], llm_calls=1)

    responses = iter(["hello", "/exit"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(responses))

    cli.repl(FakeAgent(), "openai", "gpt-4o")

    output = capture_console.getvalue()
    assert "hi there" in output
    assert "bye from" in output


def test_repl_ignores_blank_lines(capture_console, monkeypatch):
    responses = iter(["", "   ", "/exit"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(responses))

    cli.repl(None, "openai", "gpt-4o")

    assert "bye from" in capture_console.getvalue()
