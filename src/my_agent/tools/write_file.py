from my_agent.tools.base import tool

from my_agent.tools._common import relative_to_workspace, resolve_in_workspace


@tool
def write_file(path: str, content: str) -> str:
    """Create a new file or completely overwrite an existing one.

    Parent directories are created automatically. For small changes to an
    existing file prefer `edit_file`, which only replaces a unique snippet.

    Args:
        path: File to write, relative to the workspace root.
        content: Full new contents of the file.

    Returns:
        Confirmation with the number of lines and characters written.
    """
    try:
        file_path = resolve_in_workspace(path)
    except ValueError as exc:
        return f"Error: {exc}"

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"Error: could not write '{path}': {exc}"

    lines = content.count("\n") + 1 if content else 0
    return (
        f"Wrote {lines} lines ({len(content)} characters) "
        f"to {relative_to_workspace(file_path)}."
    )
