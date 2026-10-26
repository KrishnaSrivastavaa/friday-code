"""Delete a file or empty directory."""
from __future__ import annotations

from src.my_agent.tools._common import relative_to_workspace, resolve_in_workspace
from src.my_agent.tools.base import tool


@tool
def delete_file(path: str) -> str:
    """Delete a file or empty directory.

    Args:
        path: Path to the file or directory to delete, relative to the workspace.
    """
    try:
        target = resolve_in_workspace(path)
    except Exception as exc:
        return f"Error: {exc}"

    if not target.exists():
        return f"Error: '{relative_to_workspace(target)}' does not exist."

    try:
        if target.is_dir():
            # Only delete empty directories for safety
            if any(target.iterdir()):
                return f"Error: directory '{relative_to_workspace(target)}' is not empty. Use run_command for recursive deletion."
            target.rmdir()
            return f"Deleted directory {relative_to_workspace(target)}"
        else:
            target.unlink()
            return f"Deleted {relative_to_workspace(target)}"
    except Exception as exc:
        return f"Error deleting '{relative_to_workspace(target)}': {exc}"
