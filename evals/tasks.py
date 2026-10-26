"""Deterministic evaluation tasks for PyPilot.

Each task is a small coding job plus a checker. The checker is a pure function
of the sandbox directory (and the agent's final answer), so a run either passes
or fails without any LLM-based grading -- which makes evals cheap and stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

CheckResult = tuple[bool, str]


@dataclass(frozen=True)
class EvalTask:
    name: str
    prompt: str
    setup: Callable[[Path], None]
    check: Callable[[Path, str], CheckResult]


def _run_python(path: Path, expression: str):
    """Execute a module in a fresh namespace and evaluate an expression in it."""
    namespace: dict = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return eval(expression, namespace)


# --------------------------------------------------------------------------- #
# Task 1: create a file from scratch
# --------------------------------------------------------------------------- #
def _setup_create_file(workspace: Path) -> None:
    return None


def _check_create_file(workspace: Path, answer: str) -> CheckResult:
    target = workspace / "greet.py"
    if not target.exists():
        return False, "greet.py was not created"
    if "Hello, World!" not in target.read_text(encoding="utf-8"):
        return False, "greet.py does not print 'Hello, World!'"
    return True, "greet.py created with the expected message"


# --------------------------------------------------------------------------- #
# Task 2: fix an off-by-operator bug
# --------------------------------------------------------------------------- #
def _setup_fix_bug(workspace: Path) -> None:
    (workspace / "calc.py").write_text("def add(a, b):\n    return a - b\n")


def _check_fix_bug(workspace: Path, answer: str) -> CheckResult:
    try:
        result = _run_python(workspace / "calc.py", "add(2, 3)")
    except Exception as exc:
        return False, f"could not run calc.py: {exc}"
    if result != 5:
        return False, f"add(2, 3) returned {result!r}, expected 5"
    return True, "add(2, 3) == 5"


# --------------------------------------------------------------------------- #
# Task 3: add a new function to an existing module
# --------------------------------------------------------------------------- #
def _setup_add_function(workspace: Path) -> None:
    (workspace / "strings_utils.py").write_text("def reverse(s):\n    return s[::-1]\n")


def _check_add_function(workspace: Path, answer: str) -> CheckResult:
    path = workspace / "strings_utils.py"
    if "def is_palindrome" not in path.read_text(encoding="utf-8"):
        return False, "is_palindrome was not added"
    try:
        good = _run_python(path, "is_palindrome('racecar')")
        bad = _run_python(path, "is_palindrome('abc')")
    except Exception as exc:
        return False, f"could not call is_palindrome: {exc}"
    if good is not True or bad is not False:
        return False, f"is_palindrome returned {good!r} then {bad!r}"
    return True, "is_palindrome behaves correctly"


# --------------------------------------------------------------------------- #
# Task 4: explore + report (checks the final answer)
# --------------------------------------------------------------------------- #
def _setup_explore_report(workspace: Path) -> None:
    (workspace / "alpha.py").write_text("x = 1\n")
    (workspace / "beta.py").write_text("y = 2\n")
    (workspace / "notes.txt").write_text("hello\n")


def _check_explore_report(workspace: Path, answer: str) -> CheckResult:
    lowered = answer.lower()
    if "alpha.py" in lowered and "beta.py" in lowered and "2" in answer:
        return True, "answer names both python files"
    return False, "answer did not name both python files"


# --------------------------------------------------------------------------- #
# Task 5: make a failing test pass
# --------------------------------------------------------------------------- #
def _setup_make_test_pass(workspace: Path) -> None:
    (workspace / "mod.py").write_text("def mul(a, b):\n    return a + b\n")
    (workspace / "test_mod.py").write_text(
        "from mod import mul\n\n\ndef test_mul():\n    assert mul(3, 4) == 12\n"
    )


def _check_make_test_pass(workspace: Path, answer: str) -> CheckResult:
    try:
        result = _run_python(workspace / "mod.py", "mul(3, 4)")
    except Exception as exc:
        return False, f"could not run mod.py: {exc}"
    if result != 12:
        return False, f"mul(3, 4) returned {result!r}, expected 12"
    return True, "mul(3, 4) == 12"


TASKS: list[EvalTask] = [
    EvalTask(
        name="create_file",
        prompt="Create a file called greet.py that prints exactly 'Hello, World!'.",
        setup=_setup_create_file,
        check=_check_create_file,
    ),
    EvalTask(
        name="fix_bug",
        prompt="calc.py has a bug: add() subtracts instead of adding. Fix it.",
        setup=_setup_fix_bug,
        check=_check_fix_bug,
    ),
    EvalTask(
        name="add_function",
        prompt=(
            "Add an is_palindrome(s) function to strings_utils.py that returns "
            "True if s reads the same forwards and backwards, otherwise False."
        ),
        setup=_setup_add_function,
        check=_check_add_function,
    ),
    EvalTask(
        name="explore_report",
        prompt="List the Python files in this project and tell me how many there are.",
        setup=_setup_explore_report,
        check=_check_explore_report,
    ),
    EvalTask(
        name="make_test_pass",
        prompt="mod.py has a failing test in test_mod.py. Make the test pass.",
        setup=_setup_make_test_pass,
        check=_check_make_test_pass,
    ),
]

TASKS_BY_NAME = {task.name: task for task in TASKS}
