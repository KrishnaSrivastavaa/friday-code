from my_agent.tools.base import tool

from my_agent.tools._common import resolve_in_workspace


@tool
def edit_file(path: str, old_str: str, new_str: str) -> str:
    """Replace an exact snippet in a file with a new one.

    `old_str` must appear exactly once in the file, so include enough
    surrounding lines to make it unique. Matching is exact: whitespace and
    case matter. Use an empty `new_str` to delete the snippet.

    Args:
        path: File to edit, relative to the workspace root.
        old_str: Exact text to replace (must be unique within the file).
        new_str: Replacement text; use "" to delete `old_str`.

    Returns:
        Confirmation of the change, or an error message.
    """
    if not old_str:
        return "Error: old_str must not be empty."
    if old_str == new_str:
        return "Error: old_str and new_str are identical."

    try:
        file_path = resolve_in_workspace(path)
    except ValueError as exc:
        return f"Error: {exc}"

    if not file_path.exists():
        return f"Error: '{path}' does not exist. Use write_file to create it."
    if not file_path.is_file():
        return f"Error: '{path}' is not a file."

    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"Error: could not read '{path}': {exc}"

    count = text.count(old_str)
    if count == 0:
        return (
            f"Error: the search string was not found in '{path}'. "
            "Read the file and check the exact whitespace."
        )
    if count > 1:
        return (
            f"Error: the search string occurs {count} times in '{path}'. "
            "Add surrounding context so it is unique."
        )

    file_path.write_text(text.replace(old_str, new_str), encoding="utf-8")
    return f"Edited {path}: replaced {len(old_str)} chars with {len(new_str)} chars."
