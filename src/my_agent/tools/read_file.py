from src.my_agent.tools.base import tool

from src.my_agent.tools._common import resolve_in_workspace


@tool
def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
    """Read a text file from the workspace.

    Each returned line is prefixed with its 1-based line number ("12: content")
    so lines can be referenced precisely when editing.

    Args:
        path: File to read, relative to the workspace root.
        start_line: First line to read (1-based, inclusive).
        end_line: Last line to read (1-based, inclusive). Omit to read to the
            end of the file.

    Returns:
        The numbered contents of the file, or an error message.
    """
    try:
        file_path = resolve_in_workspace(path)
    except ValueError as exc:
        return f"Error: {exc}"

    if not file_path.exists():
        return f"Error: '{path}' does not exist."
    if not file_path.is_file():
        return f"Error: '{path}' is not a file. Use list_files to list a directory."

    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"Error: could not read '{path}': {exc}"

    lines = text.splitlines()
    if not lines:
        return f"'{path}' is empty."

    start = max(start_line, 1)
    end = end_line if end_line is not None else len(lines)
    selected = lines[start - 1 : end]
    if not selected:
        return (
            f"Error: lines {start}-{end} are outside the file "
            f"(it has {len(lines)} lines)."
        )

    return "\n".join(f"{start + i}: {line}" for i, line in enumerate(selected))
