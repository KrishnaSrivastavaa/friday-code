"""Shared helpers for the agent's tools.

Every tool operates relative to a single workspace root. By default this is the
directory the agent was launched from, but it can be overridden with
`set_workspace()` (e.g. from the CLI or a FastAPI endpoint).
"""

import os
from pathlib import Path

WORKSPACE_ROOT: Path = Path.cwd().resolve()

# Directories that are never listed or searched through.
IGNORED_DIRS: set[str] = {
    ".git",
    ".hg",
    ".svn",
    ".tox",
    ".venv",
    ".idea",
    ".vscode",
    "__pycache__",
    "node_modules",
    "venv",
    "env",
    "dist",
    "build",
    "target",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".next",
    ".gradle",
}

# Hard cap on how much text a tool returns to the model.
MAX_OUTPUT_CHARS = 15_000
# Hard cap on the number of entries (files / matches) a tool returns.
MAX_RESULTS = 200


def set_workspace(path: str | Path) -> Path:
    """Point every tool at a different workspace root."""
    global WORKSPACE_ROOT
    WORKSPACE_ROOT = Path(path).resolve()
    return WORKSPACE_ROOT


def workspace_root() -> Path:
    """Current workspace root, read at call time so `set_workspace` takes effect."""
    return WORKSPACE_ROOT


def resolve_in_workspace(path: str) -> Path:
    """Resolve `path` (relative to the workspace root), refusing escapes.

    Raises ValueError if the resolved path falls outside the workspace.
    """
    candidate = (WORKSPACE_ROOT / path).resolve()
    if candidate != WORKSPACE_ROOT and WORKSPACE_ROOT not in candidate.parents:
        raise ValueError(f"Path '{path}' is outside the workspace.")
    return candidate


def relative_to_workspace(path: Path) -> str:
    """Workspace-relative path using forward slashes (portable for the LLM)."""
    return path.relative_to(WORKSPACE_ROOT).as_posix()


def is_ignored(path: Path) -> bool:
    """True if any directory part of `path` is an ignored directory."""
    try:
        parts = path.relative_to(WORKSPACE_ROOT).parts
    except ValueError:
        parts = path.parts
    return any(part in IGNORED_DIRS for part in parts)


def iter_files(root: Path):
    """Yield every file under `root`, pruning ignored directories."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for filename in filenames:
            yield Path(dirpath) / filename


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Cap text so huge outputs don't blow up the context window."""
    if len(text) <= limit:
        return text
    kept = text[:limit]
    return f"{kept}\n... [truncated: {len(text) - limit} more characters]"
