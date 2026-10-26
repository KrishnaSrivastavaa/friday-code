"""Create a directory."""
from __future__ import annotations

from src.my_agent.tools._common import relative_to_workspace, resolve_in_workspace
from src.my_agent.tools.base import tool


@tool
def create_directory(path: str) -> str:
    """Create a directory.

    Args:
        path: Path to the directory to create, relative to the workspace.
    """
    try:
        target = resolve_in_workspace(path)
    except Exception as exc:
        return f"Error: {exc}"

    try:
        target.mkdir(parents=True, exist_ok=True)
        return f"Created directory {relative_to_workspace(target)}"
    except Exception as exc:
        return f"Error creating directory '{relative_to_workspace(target)}': {exc}"
