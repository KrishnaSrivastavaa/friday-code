from src.my_agent.tools.base import tool

from src.my_agent.tools._common import (
    MAX_RESULTS,
    iter_files,
    relative_to_workspace,
    resolve_in_workspace,
)


@tool
def list_files(path: str = ".") -> str:
    """List every file under a directory of the workspace, recursively.

    Ignores noise directories such as .git, node_modules, __pycache__ and
    virtualenvs. Paths are returned relative to the workspace root.

    Args:
        path: Directory to list, relative to the workspace root. Defaults to
            the workspace root itself.

    Returns:
        Newline-separated file paths, or a message if nothing was found.
    """
    try:
        root = resolve_in_workspace(path)
    except ValueError as exc:
        return f"Error: {exc}"

    if not root.exists():
        return f"Error: '{path}' does not exist."
    if not root.is_dir():
        return f"Error: '{path}' is not a directory. Use read_file to read a file."

    files = [relative_to_workspace(p) for p in iter_files(root)]
    if not files:
        return f"No files found under '{path}'."

    shown = files[:MAX_RESULTS]
    output = "\n".join(shown)
    if len(files) > MAX_RESULTS:
        output += f"\n... [{len(files) - MAX_RESULTS} more files not shown]"
    return output
