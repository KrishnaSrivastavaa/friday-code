import subprocess

from src.my_agent.tools.base import tool

from src.my_agent.tools._common import truncate, workspace_root

DEFAULT_TIMEOUT_SECONDS = 60
MAX_TIMEOUT_SECONDS = 300


@tool
def run_command(command: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> str:
    """Run a shell command in the workspace root and return its output.

    Use this for builds, tests, git, package managers, etc. The command runs in
    the OS shell ('cmd /c' on Windows, '/bin/sh -c' elsewhere). Interactive
    commands and anything needing a TTY will not work.

    Args:
        command: The shell command to execute.
        timeout: Seconds to wait before the command is killed (max 300).

    Returns:
        Exit code plus captured stdout and stderr.
    """
    timeout = min(max(timeout, 1), MAX_TIMEOUT_SECONDS)
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=workspace_root(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s: {command}"
    except OSError as exc:
        return f"Error: could not run command: {exc}"

    parts = [f"Exit code: {completed.returncode}"]
    if completed.stdout:
        parts.append(f"stdout:\n{completed.stdout}")
    if completed.stderr:
        parts.append(f"stderr:\n{completed.stderr}")
    if not completed.stdout and not completed.stderr:
        parts.append("(no output)")

    return truncate("\n".join(parts))
