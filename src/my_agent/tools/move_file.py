"""Move or rename a file/directory."""
from __future__ import annotations

import shutil

from my_agent.tools._common import relative_to_workspace, resolve_in_workspace
from my_agent.tools.base import tool


@tool
def move_file(source: str, destination: str) -> str:
    """Move or rename a file/directory.

    Args:
        source: Source path relative to the workspace.
        destination: Destination path relative to the workspace.
    """
    try:
        src = resolve_in_workspace(source)
        dst = resolve_in_workspace(destination)
    except Exception as exc:
        return f"Error: {exc}"

    if not src.exists():
        return f"Error: '{relative_to_workspace(src)}' does not exist."

    try:
        # Create parent dirs for destination
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return f"Moved {relative_to_workspace(src)} -> {relative_to_workspace(dst)}"
    except Exception as exc:
        return f"Error moving '{relative_to_workspace(src)}' to '{relative_to_workspace(dst)}': {exc}"
