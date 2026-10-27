import re

from my_agent.tools.base import tool

from my_agent.tools._common import (
    MAX_RESULTS,
    iter_files,
    relative_to_workspace,
    resolve_in_workspace,
    truncate,
)

# Files larger than this are skipped when grepping.
MAX_SEARCH_FILE_BYTES = 1_000_000
# Long lines are cut so a single minified file can't flood the output.
MAX_LINE_CHARS = 300


@tool
def search_files(
    pattern: str, path: str = ".", file_extension: str | None = None
) -> str:
    """Search file contents with a regular expression (like grep).

    Args:
        pattern: Python regular expression to search for (case-sensitive).
        path: Directory to search, relative to the workspace root.
        file_extension: Optional extension filter, e.g. '.py' — only files
            ending with it are searched.

    Returns:
        Matches as 'relative/path:line_number: line_text', capped at 200
        matches, or a message if nothing matched.
    """
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"Error: invalid regular expression: {exc}"

    try:
        root = resolve_in_workspace(path)
    except ValueError as exc:
        return f"Error: {exc}"

    if not root.exists():
        return f"Error: '{path}' does not exist."
    if not root.is_dir():
        return f"Error: '{path}' is not a directory."

    results: list[str] = []
    truncated = False
    for file_path in iter_files(root):
        if file_extension and not file_path.name.endswith(file_extension):
            continue
        try:
            if file_path.stat().st_size > MAX_SEARCH_FILE_BYTES:
                continue
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "\x00" in text:  # binary file
            continue

        for line_no, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                rel = relative_to_workspace(file_path)
                results.append(f"{rel}:{line_no}: {line.strip()[:MAX_LINE_CHARS]}")
                if len(results) >= MAX_RESULTS:
                    truncated = True
                    break
        if truncated:
            break

    if not results:
        return f"No matches found for '{pattern}' under '{path}'."

    output = "\n".join(results)
    if truncated:
        output += f"\n... [stopped at {MAX_RESULTS} matches]"
    return truncate(output)
