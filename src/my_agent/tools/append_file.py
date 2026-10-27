"""Append content to a file."""
from __future__ import annotations

from my_agent.tools._common import relative_to_workspace, resolve_in_workspace
from my_agent.tools.base import tool


@tool
def append_file(path: str, content: str) -> str:
    """Append content to a file.

    Args:
        path: Path to the file, relative to the workspace.
        content: Content to append.
    """
    try:
        target = resolve_in_workspace(path)
    except Exception as exc:
        return f"Error: {exc}"

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as f:
            f.write(content)
        return f"Appended {len(content)} characters to {relative_to_workspace(target)}"
    except Exception as exc:
        return f"Error appending to '{relative_to_workspace(target)}': {exc}"
