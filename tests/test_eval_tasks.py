"""Validate the eval harness itself, without calling any model.

Every checker must reject the unsolved sandbox and accept a golden solution.
"""

from __future__ import annotations

import pytest

from evals.tasks import TASKS

GOLDEN_SOLUTIONS = {
    "create_file": lambda ws: (ws / "greet.py").write_text(
        "print('Hello, World!')\n"
    ),
    "fix_bug": lambda ws: (ws / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n"
    ),
    "add_function": lambda ws: (ws / "strings_utils.py").write_text(
        "def reverse(s):\n"
        "    return s[::-1]\n"
        "\n"
        "\n"
        "def is_palindrome(s):\n"
        "    return s == s[::-1]\n"
    ),
    "make_test_pass": lambda ws: (ws / "mod.py").write_text(
        "def mul(a, b):\n    return a * b\n"
    ),
}

GOLDEN_ANSWERS = {
    "explore_report": "There are 2 Python files: alpha.py and beta.py.",
}


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_checker_rejects_unsolved_task(task, tmp_path):
    task.setup(tmp_path)

    passed, _ = task.check(tmp_path, "")

    assert not passed


@pytest.mark.parametrize("task", TASKS, ids=lambda task: task.name)
def test_checker_accepts_golden_solution(task, tmp_path):
    task.setup(tmp_path)
    if task.name in GOLDEN_SOLUTIONS:
        GOLDEN_SOLUTIONS[task.name](tmp_path)

    passed, detail = task.check(tmp_path, GOLDEN_ANSWERS.get(task.name, ""))

    assert passed, detail
