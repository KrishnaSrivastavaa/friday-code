"""Copy a file."""
from __future__ import annotations

import shutil

from my_agent.tools._common import relative_to_workspace, resolve_in_workspace
from my_agent.tools.base import tool


@tool
def copy_file(source: str, destination: str) -> str:
    """Copy a file.

    Args:
        source: Source file path relative to the workspace.
        destination: Destination file path relative to the workspace.
    """
    try:
        src = resolve_in_workspace(source)
        dst = resolve_in_workspace(destination)
    except Exception as exc:
        return f"Error: {exc}"

    if not src.exists():
        return f"Error: '{relative_to_workspace(src)}' does not exist."
    if not src.is_file():
        return f"Error: '{relative_to_workspace(src)}' is not a file."

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        return f"Copied {relative_to_workspace(src)} -> {relative_to_workspace(dst)}"
    except Exception as exc:
        return f"Error copying '{relative_to_workspace(src)}' to '{relative_to_workspace(dst)}': {exc}"
