"""Run PyPilot evaluations against a real (cheap) model.

Usage::

    uv run python -m evals.run_evals                 # all tasks, gpt-4o-mini
    uv run python -m evals.run_evals --only fix_bug  # a single task
    uv run python -m evals.run_evals --list          # list tasks and exit
    EVAL_MODEL=gpt-4o-mini uv run python -m evals.run_evals

Requires ``OPENAI_API_KEY`` (from the environment or ``.env``). Each task runs
the agent in a throwaway sandbox directory, so nothing in your project changes.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from evals.tasks import TASKS, TASKS_BY_NAME
from my_agent.agent import Agent, run
from my_agent.config.settings import settings
from my_agent.messages import ToolMessage
from my_agent.models.provider import get_client
from my_agent.tools import _common

DEFAULT_EVAL_MODEL = "gpt-4o-mini"

console = Console()


def run_task(task, model_name: str, api_key: str) -> dict:
    """Run one eval task in a sandbox and return its outcome."""
    original_root = _common.WORKSPACE_ROOT
    started = time.perf_counter()
    outcome = {
        "name": task.name,
        "passed": False,
        "detail": "",
        "turns": 0,
        "tool_calls": 0,
        "tokens": 0,
        "seconds": 0.0,
    }

    try:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = Path(tmp)
            task.setup(sandbox)
            _common.set_workspace(sandbox)

            client = get_client(api_key=api_key)
            agent = Agent(client, model_name=model_name)

            try:
                result = run(agent, task.prompt)
            except Exception as exc:  # noqa: BLE001
                outcome["detail"] = f"agent error: {exc}"
                return outcome

            messages = result.messages
            answer = result.final
            passed, detail = task.check(sandbox, answer)
            outcome.update(
                passed=passed,
                detail=detail,
                turns=result.llm_calls,
                tool_calls=len([m for m in messages if isinstance(m, ToolMessage)]),
                tokens=result.usage.total_tokens,
            )
    finally:
        _common.set_workspace(original_root)
        outcome["seconds"] = time.perf_counter() - started

    return outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals", description="Run PyPilot evaluations.")
    parser.add_argument(
        "--model",
        default=os.environ.get("EVAL_MODEL", DEFAULT_EVAL_MODEL),
        help=f"Model to evaluate (default: {DEFAULT_EVAL_MODEL}).",
    )
    parser.add_argument("--only", help="Run a single task by name.")
    parser.add_argument("--list", action="store_true", help="List tasks and exit.")
    args = parser.parse_args(argv)

    if args.list:
        for task in TASKS:
            console.print(f"  [cyan]{task.name}[/cyan]  {task.prompt}")
        return 0

    api_key = os.environ.get("OPENAI_API_KEY") or settings.openai_api_key
    if not api_key:
        console.print("[red]No OPENAI_API_KEY found.[/red] Set it or add it to .env.")
        return 1

    if args.only:
        if args.only not in TASKS_BY_NAME:
            console.print(f"[red]Unknown task:[/red] {args.only}")
            return 1
        selected = [TASKS_BY_NAME[args.only]]
    else:
        selected = TASKS

    console.print(
        f"\n[bold cyan]PyPilot evals[/bold cyan]  [dim]model={args.model}  "
        f"tasks={len(selected)}[/dim]\n"
    )

    outcomes = []
    for task in selected:
        console.print(f"[bold]running[/bold] {task.name} ...")
        outcome = run_task(task, args.model, api_key)
        outcomes.append(outcome)
        mark = "[green]PASS[/green]" if outcome["passed"] else "[red]FAIL[/red]"
        console.print(f"  {mark}  [dim]{outcome['detail']}[/dim]")

    table = Table(title="results", header_style="bold cyan")
    table.add_column("task")
    table.add_column("result")
    table.add_column("turns", justify="right")
    table.add_column("tools", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("secs", justify="right")
    for outcome in outcomes:
        table.add_row(
            outcome["name"],
            "[green]pass[/green]" if outcome["passed"] else "[red]fail[/red]",
            str(outcome["turns"]),
            str(outcome["tool_calls"]),
            str(outcome["tokens"]),
            f"{outcome['seconds']:.1f}",
        )
    console.print()
    console.print(table)

    passed = sum(1 for o in outcomes if o["passed"])
    console.print(f"\n[bold]{passed}/{len(outcomes)} passed[/bold]")
    return 0 if passed == len(outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
