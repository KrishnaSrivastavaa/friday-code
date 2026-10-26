from src.my_agent.tools.base import tool

from src.my_agent.tools._common import (
    MAX_RESULTS,
    is_ignored,
    relative_to_workspace,
    resolve_in_workspace,
)


@tool
def glob_files(pattern: str, path: str = ".") -> str:
    """Find files whose paths match a glob pattern (e.g. '**/*.py', 'src/**/*.ts').

    Use '**' to match across directories. Ignored directories (.git,
    node_modules, __pycache__, ...) are skipped.

    Args:
        pattern: Glob pattern to match against workspace-relative paths.
        path: Directory to search in, relative to the workspace root.

    Returns:
        Newline-separated matching file paths, or a message if none matched.
    """
    try:
        root = resolve_in_workspace(path)
    except ValueError as exc:
        return f"Error: {exc}"

    if not root.exists():
        return f"Error: '{path}' does not exist."
    if not root.is_dir():
        return f"Error: '{path}' is not a directory."

    matches = sorted(
        relative_to_workspace(p)
        for p in root.glob(pattern)
        if p.is_file() and not is_ignored(p)
    )
    if not matches:
        return f"No files matched pattern '{pattern}' under '{path}'."

    shown = matches[:MAX_RESULTS]
    output = "\n".join(shown)
    if len(matches) > MAX_RESULTS:
        output += f"\n... [{len(matches) - MAX_RESULTS} more matches not shown]"
    return output
