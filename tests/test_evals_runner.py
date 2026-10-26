"""Integration tests for the eval runner plumbing -- no network involved."""

from __future__ import annotations

import types

from evals import run_evals
from evals.tasks import TASKS_BY_NAME
from src.my_agent.tools import _common
from tests.helpers import ScriptedClient, fake_response, tool_call


def test_run_task_passes_when_agent_solves_it(monkeypatch):
    original_root = _common.WORKSPACE_ROOT
    fake = ScriptedClient(
        [
            fake_response(
                tool_calls=[tool_call("write_file", {"path": "greet.py", "content": "print('Hello, World!')\n"}, "c1")]
            ),
            fake_response(content="Created greet.py."),
        ]
    )
    monkeypatch.setattr(run_evals, "get_client", lambda **kwargs: fake)

    outcome = run_evals.run_task(TASKS_BY_NAME["create_file"], "gpt-4o-mini", "sk-test")

    assert outcome["passed"] is True
    assert outcome["tool_calls"] == 1
    assert outcome["turns"] == 2
    assert _common.WORKSPACE_ROOT == original_root


def test_run_task_fails_when_agent_does_nothing(monkeypatch):
    original_root = _common.WORKSPACE_ROOT
    fake = ScriptedClient([fake_response(content="I cannot do that.")])
    monkeypatch.setattr(run_evals, "get_client", lambda **kwargs: fake)

    outcome = run_evals.run_task(TASKS_BY_NAME["create_file"], "gpt-4o-mini", "sk-test")

    assert outcome["passed"] is False
    assert _common.WORKSPACE_ROOT == original_root


def test_run_task_reports_agent_errors(monkeypatch):
    original_root = _common.WORKSPACE_ROOT

    class BoomClient:
        def __init__(self):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            raise RuntimeError("model exploded")

    monkeypatch.setattr(run_evals, "get_client", lambda **kwargs: BoomClient())

    outcome = run_evals.run_task(TASKS_BY_NAME["create_file"], "gpt-4o-mini", "sk-test")

    assert outcome["passed"] is False
    assert "model exploded" in outcome["detail"]
    assert _common.WORKSPACE_ROOT == original_root


def test_main_list_returns_zero(capsys):
    assert run_evals.main(["--list"]) == 0


def test_main_without_key_returns_one(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(run_evals.settings, "openai_api_key", None)
    assert run_evals.main([]) == 1
