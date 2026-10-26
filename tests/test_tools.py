"""Tests for every file/shell tool in ``src.my_agent.tools``.

Each tool is exercised through its LangChain ``.invoke`` surface (schema
validation + execution) against a real temporary workspace.
"""

from __future__ import annotations

import sys

from src.my_agent.tools import (
    edit_file,
    glob_files,
    list_files,
    read_file,
    run_command,
    search_files,
    write_file,
)
from src.my_agent.tools._common import MAX_RESULTS

# --------------------------------------------------------------------------- #
# list_files
# --------------------------------------------------------------------------- #


def test_list_files_lists_files_recursively(workspace):
    (workspace / "a.py").write_text("x")
    (workspace / "pkg").mkdir()
    (workspace / "pkg" / "b.py").write_text("y")

    result = list_files.invoke({"path": "."})

    assert "a.py" in result
    assert "pkg/b.py" in result


def test_list_files_ignores_noise_directories(workspace):
    for noise in (".git", "node_modules", "__pycache__", ".venv"):
        directory = workspace / noise
        directory.mkdir()
        (directory / "junk.py").write_text("x")
    (workspace / "keep.py").write_text("x")

    result = list_files.invoke({"path": "."})

    assert "keep.py" in result
    assert "junk.py" not in result
    assert ".git" not in result
    assert "node_modules" not in result


def test_list_files_missing_directory(workspace):
    result = list_files.invoke({"path": "nope"})
    assert result.startswith("Error:")
    assert "does not exist" in result


def test_list_files_path_is_a_file(workspace):
    (workspace / "f.py").write_text("x")
    result = list_files.invoke({"path": "f.py"})
    assert result.startswith("Error:")
    assert "not a directory" in result


def test_list_files_empty_directory(workspace):
    (workspace / "empty").mkdir()
    result = list_files.invoke({"path": "empty"})
    assert "No files found" in result


def test_list_files_blocks_path_escape(workspace):
    result = list_files.invoke({"path": "../"})
    assert result.startswith("Error:")
    assert "outside the workspace" in result


def test_list_files_truncates_large_listings(workspace):
    for index in range(MAX_RESULTS + 5):
        (workspace / f"file_{index}.txt").write_text("x")

    result = list_files.invoke({"path": "."})

    assert "more files not shown" in result
    assert len(result.splitlines()) <= MAX_RESULTS + 1


# --------------------------------------------------------------------------- #
# glob_files
# --------------------------------------------------------------------------- #


def test_glob_files_matches_pattern(workspace):
    (workspace / "a.py").write_text("x")
    (workspace / "b.txt").write_text("x")
    (workspace / "pkg").mkdir()
    (workspace / "pkg" / "c.py").write_text("x")

    result = glob_files.invoke({"pattern": "**/*.py", "path": "."})

    assert "a.py" in result
    assert "pkg/c.py" in result
    assert "b.txt" not in result


def test_glob_files_skips_ignored_directories(workspace):
    (workspace / "node_modules").mkdir()
    (workspace / "node_modules" / "z.py").write_text("x")
    (workspace / "main.py").write_text("x")

    result = glob_files.invoke({"pattern": "**/*.py", "path": "."})

    assert "main.py" in result
    assert "z.py" not in result


def test_glob_files_no_matches(workspace):
    result = glob_files.invoke({"pattern": "**/*.rs", "path": "."})
    assert "No files matched" in result


def test_glob_files_missing_directory(workspace):
    result = glob_files.invoke({"pattern": "**/*.py", "path": "nope"})
    assert result.startswith("Error:")


def test_glob_files_path_is_a_file(workspace):
    (workspace / "a.py").write_text("x")
    result = glob_files.invoke({"pattern": "**/*.py", "path": "a.py"})
    assert "not a directory" in result


# --------------------------------------------------------------------------- #
# search_files
# --------------------------------------------------------------------------- #


def test_search_files_reports_line_numbers(workspace):
    (workspace / "mod.py").write_text("alpha\nbeta\ngamma\n")

    result = search_files.invoke({"pattern": "beta", "path": "."})

    assert "mod.py:2: beta" in result


def test_search_files_extension_filter(workspace):
    (workspace / "a.py").write_text("needle")
    (workspace / "b.txt").write_text("needle")

    result = search_files.invoke(
        {"pattern": "needle", "path": ".", "file_extension": ".py"}
    )

    assert "a.py" in result
    assert "b.txt" not in result


def test_search_files_invalid_regex(workspace):
    result = search_files.invoke({"pattern": "([unclosed", "path": "."})
    assert result.startswith("Error:")
    assert "invalid regular expression" in result


def test_search_files_no_matches(workspace):
    (workspace / "a.py").write_text("hello")
    result = search_files.invoke({"pattern": "zzz", "path": "."})
    assert "No matches found" in result


def test_search_files_skips_binary_files(workspace):
    (workspace / "blob.bin").write_bytes(b"\x00\x01needle\x02")

    result = search_files.invoke({"pattern": "needle", "path": "."})

    assert "No matches found" in result


def test_search_files_missing_directory(workspace):
    result = search_files.invoke({"pattern": "x", "path": "nope"})
    assert result.startswith("Error:")


# --------------------------------------------------------------------------- #
# read_file
# --------------------------------------------------------------------------- #


def test_read_file_numbers_lines(workspace):
    (workspace / "a.py").write_text("first\nsecond\nthird")

    result = read_file.invoke({"path": "a.py"})

    assert result == "1: first\n2: second\n3: third"


def test_read_file_line_range(workspace):
    (workspace / "a.py").write_text("one\ntwo\nthree\nfour")

    result = read_file.invoke({"path": "a.py", "start_line": 2, "end_line": 3})

    assert result == "2: two\n3: three"


def test_read_file_range_out_of_bounds(workspace):
    (workspace / "a.py").write_text("one\ntwo")

    result = read_file.invoke({"path": "a.py", "start_line": 99})

    assert result.startswith("Error:")
    assert "outside the file" in result


def test_read_file_empty(workspace):
    (workspace / "empty.py").write_text("")
    assert "is empty" in read_file.invoke({"path": "empty.py"})


def test_read_file_missing(workspace):
    result = read_file.invoke({"path": "nope.py"})
    assert result.startswith("Error:")
    assert "does not exist" in result


def test_read_file_directory(workspace):
    (workspace / "pkg").mkdir()
    result = read_file.invoke({"path": "pkg"})
    assert "not a file" in result


def test_read_file_blocks_escape(workspace):
    result = read_file.invoke({"path": "../secret.txt"})
    assert "outside the workspace" in result


# --------------------------------------------------------------------------- #
# write_file
# --------------------------------------------------------------------------- #


def test_write_file_creates_parents(workspace):
    result = write_file.invoke(
        {"path": "pkg/sub/mod.py", "content": "print('hi')\n"}
    )

    assert (workspace / "pkg" / "sub" / "mod.py").read_text() == "print('hi')\n"
    assert "Wrote" in result


def test_write_file_overwrites(workspace):
    target = workspace / "a.txt"
    target.write_text("old")
    write_file.invoke({"path": "a.txt", "content": "new"})
    assert target.read_text() == "new"


def test_write_file_counts_lines(workspace):
    result = write_file.invoke({"path": "a.txt", "content": "a\nb"})
    assert "2 lines" in result


def test_write_file_blocks_escape(workspace):
    result = write_file.invoke({"path": "../evil.txt", "content": "x"})
    assert "outside the workspace" in result


# --------------------------------------------------------------------------- #
# edit_file
# --------------------------------------------------------------------------- #


def test_edit_file_replaces_unique_snippet(workspace):
    target = workspace / "a.py"
    target.write_text("def add(a, b):\n    return a + b\n")

    result = edit_file.invoke(
        {"path": "a.py", "old_str": "return a + b", "new_str": "return a - b"}
    )

    assert "return a - b" in target.read_text()
    assert "Edited" in result


def test_edit_file_not_found(workspace):
    (workspace / "a.py").write_text("hello")
    result = edit_file.invoke(
        {"path": "a.py", "old_str": "missing", "new_str": "x"}
    )
    assert "not found" in result


def test_edit_file_requires_unique_match(workspace):
    (workspace / "a.py").write_text("x = 1\nx = 1\n")
    result = edit_file.invoke({"path": "a.py", "old_str": "x = 1", "new_str": "x = 2"})
    assert "occurs 2 times" in result


def test_edit_file_empty_old_str(workspace):
    (workspace / "a.py").write_text("hello")
    result = edit_file.invoke({"path": "a.py", "old_str": "", "new_str": "x"})
    assert result.startswith("Error:")


def test_edit_file_identical_strings(workspace):
    (workspace / "a.py").write_text("hello")
    result = edit_file.invoke({"path": "a.py", "old_str": "hello", "new_str": "hello"})
    assert "identical" in result


def test_edit_file_missing_file(workspace):
    result = edit_file.invoke({"path": "nope.py", "old_str": "a", "new_str": "b"})
    assert "does not exist" in result


def test_edit_file_can_delete(workspace):
    target = workspace / "a.py"
    target.write_text("keep\nremove me\nkeep2")
    edit_file.invoke({"path": "a.py", "old_str": "remove me\n", "new_str": ""})
    assert target.read_text() == "keep\nkeep2"


# --------------------------------------------------------------------------- #
# run_command
# --------------------------------------------------------------------------- #


def _py(code: str) -> str:
    return f'"{sys.executable}" -c "{code}"'


def test_run_command_captures_stdout(workspace):
    result = run_command.invoke({"command": _py("print('hello-from-tool')")})
    assert "Exit code: 0" in result
    assert "hello-from-tool" in result


def test_run_command_reports_exit_code(workspace):
    result = run_command.invoke({"command": _py("import sys; sys.exit(3)")})
    assert "Exit code: 3" in result


def test_run_command_captures_stderr(workspace):
    result = run_command.invoke(
        {"command": _py("import sys; sys.stderr.write('boom')")}
    )
    assert "stderr:" in result
    assert "boom" in result


def test_run_command_runs_in_workspace(workspace):
    (workspace / "marker.txt").write_text("x")
    result = run_command.invoke(
        {"command": _py("import os; print(os.path.exists('marker.txt'))")}
    )
    assert "True" in result


def test_run_command_times_out(workspace):
    result = run_command.invoke(
        {"command": _py("import time; time.sleep(5)"), "timeout": 1}
    )
    assert "timed out" in result
