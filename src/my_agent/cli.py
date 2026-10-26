"""FRIDAY - the terminal coding agent CLI.

Run it with:

    uv run python -m src.my_agent.cli

On first launch it asks for the provider, model and API key, builds the agent
with them, then drops into a chat loop. One-shot mode is also supported:

    uv run python -m src.my_agent.cli "list the python files in this project"
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.my_agent.agent import Agent, build_agent
from src.my_agent.config.global_config import get_stored_key, load_global_config, save_global_config, save_stored_key
from src.my_agent.config.settings import settings
from src.my_agent.events import AssistantEvent, ToolCallEvent, ToolResultEvent
from src.my_agent.messages import UserMessage
from src.my_agent.models.provider import get_client, get_model
from src.my_agent.tools import CODING_TOOLS
from src.my_agent.tools._common import workspace_root

APP_NAME = "FRIDAY"
TAGLINE = "AUTONOMOUS CODING AGENT"
VERSION = "0.1"

DEFAULT_PROVIDER = "openai"
SUPPORTED_PROVIDERS = ["openai", "openrouter", "gemini", "google", "anthropic"]
PROVIDER_MODELS = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"],
    "openrouter": ["anthropic/claude-sonnet-4", "google/gemini-2.5-flash", "deepseek/deepseek-chat-v3-0324", "meta-llama/llama-3.1-70b-instruct"],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
    "google": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
    "anthropic": ["claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022"],
}
SUGGESTED_MODELS = PROVIDER_MODELS["openai"]
COMMANDS = ["/help", "/model", "/tools", "/workspace", "/memory", "/reset", "/exit"]
ENV_FILE = Path(".env")

console = Console()


def _make_output_safe() -> None:
    """Never crash on redirects to a legacy code page (e.g. cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Setup: provider, model, API key
# --------------------------------------------------------------------------- #
def prompt_provider() -> str:
    console.print(f"[dim]Provider[/dim]  [cyan]openai[/cyan], [cyan]openrouter[/cyan], [cyan]gemini[/cyan], [cyan]anthropic[/cyan]")
    while True:
        try:
            answer = (input("  provider [openai]: ").strip() or DEFAULT_PROVIDER).lower()
        except EOFError:
            console.print()
            return DEFAULT_PROVIDER
        if answer == "google":
            answer = "gemini"
        if answer in SUPPORTED_PROVIDERS:
            return answer
        console.print(f"  [red]unsupported provider. choose from: {', '.join(SUPPORTED_PROVIDERS)}[/red]")


def prompt_model(provider: str = "openai") -> str:
    console.print("[dim]Choose a model[/dim]")
    models = PROVIDER_MODELS.get(provider, PROVIDER_MODELS["openai"])
    menu = Table(show_header=False, box=None, padding=(0, 2))
    menu.add_column(style="cyan", justify="right", no_wrap=True)
    menu.add_column(style="white")
    for index, name in enumerate(models, start=1):
        menu.add_row(str(index), name)
    custom_index = len(models) + 1
    menu.add_row(str(custom_index), "Other (type your own)")
    console.print(menu)
    console.print("[dim]Tip: You can also just type a full model name directly (e.g. cohere/north-mini-code:free)[/dim]")

    while True:
        try:
            answer = input("  model [1]: ").strip() or "1"
        except EOFError:
            console.print()
            return settings.model_name
        if not answer:
            continue
        if answer.isdigit():
            index = int(answer)
            if 1 <= index <= len(models):
                return models[index - 1]
            if index == custom_index:
                try:
                    custom = input("  model name: ").strip()
                except EOFError:
                    console.print()
                    return settings.model_name
                if custom:
                    return custom
                continue
            console.print("  [red]please pick a valid option[/red]")
            continue
        # If user types a custom model directly (common with openrouter)
        if "/" in answer or ":" in answer or "-" in answer or "." in answer:
            return answer
        # Allow any other non-numeric string as a custom model name
        return answer


def prompt_api_key(provider: str = "openai") -> str:
    labels = {
        "openai": "OpenAI",
        "openrouter": "OpenRouter",
        "gemini": "Google Gemini",
        "google": "Google Gemini",
        "anthropic": "Anthropic",
    }
    label = labels.get(provider, "API")
    while True:
        try:
            key = getpass.getpass(f"  {label} API key (hidden): ").strip()
        except EOFError:
            console.print()
            console.print("[red]error:[/red] no API key provided.")
            sys.exit(1)
        if key:
            return key
        console.print("  [red]API key cannot be empty[/red]")


def save_env(values: dict[str, str]) -> None:
    """Upsert KEY=VALUE pairs into .env without dropping existing entries."""
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    output: list[str] = []
    written: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if "=" in stripped and not stripped.startswith("#"):
            key = stripped.split("=", 1)[0].strip()
            if key in values:
                output.append(f"{key}={values[key]}")
                written.add(key)
                continue
        output.append(line)
    for key, value in values.items():
        if key not in written:
            output.append(f"{key}={value}")
    ENV_FILE.write_text("\n".join(output) + "\n", encoding="utf-8")


def configure(args: argparse.Namespace) -> tuple[str, str, str]:
    """Resolve provider / model / API key from flags, environment, or the user."""
    interactive = sys.stdin.isatty()
    global_cfg = load_global_config()

    console.print()
    console.print(f"[bold #55BDFF]{APP_NAME}[/bold #55BDFF] [#91A8C2]setup[/#91A8C2]")
    console.print()

    provider = args.provider
    if provider and provider == "google":
        provider = "gemini"

    # Check if we have a saved config
    has_saved = bool(global_cfg.get("provider") or global_cfg.get("api_keys"))

    if not provider:
        if has_saved and interactive:
            saved_provider = global_cfg.get("provider") or DEFAULT_PROVIDER
            if saved_provider == "google":
                saved_provider = "gemini"
            console.print(f"[dim]Saved provider:[/dim] [cyan]{saved_provider}[/cyan]")
            try:
                ans = input("  Use saved provider? [Y/n]: ").strip().lower()
            except EOFError:
                ans = "y"
            if ans in ("", "y", "yes"):
                provider = saved_provider
            else:
                provider = prompt_provider()
        else:
            provider = global_cfg.get("provider") or settings.model_provider or DEFAULT_PROVIDER
    provider = (provider or DEFAULT_PROVIDER).lower()
    if provider == "google":
        provider = "gemini"

    model_name = args.model
    if not model_name:
        stored_models = global_cfg.get("models", {}) or {}
        if has_saved and interactive and not args.provider:
            saved_model = stored_models.get(provider) or global_cfg.get("model_name") or settings.model_name
            console.print(f"[dim]Saved model for {provider}:[/dim] [cyan]{saved_model}[/cyan]")
            try:
                ans = input("  Use saved model? [Y/n]: ").strip().lower()
            except EOFError:
                ans = "y"
            if ans in ("", "y", "yes"):
                model_name = saved_model
            else:
                model_name = prompt_model(provider)
        else:
            model_name = stored_models.get(provider) or global_cfg.get("model_name") or settings.model_name

    env_key_map = {
        "openai": ["OPENAI_API_KEY"],
        "openrouter": ["OPENROUTER_API_KEY", "OPENAI_API_KEY"],
        "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"],
    }
    api_key = args.api_key
    source = None
    if api_key:
        source = "--api-key"
    else:
        # Check global config first (persisted across runs)
        api_key = get_stored_key(provider)
        if api_key:
            # Basic validation - reject obviously invalid keys
            if len(str(api_key).strip()) < 10 or str(api_key).strip() in ("k", "test", "key"):
                api_key = None
                # Clear bad key
                try:
                    cfg = global_cfg
                    keys = cfg.get("api_keys", {}) or {}
                    keys.pop(provider, None)
                    cfg["api_keys"] = keys
                    save_global_config(cfg)
                except Exception:
                    pass
            else:
                source = f"~/.pypilot/config.json ({provider})"
        if not api_key:
            for env_name in env_key_map.get(provider, ["OPENAI_API_KEY"]):
                api_key = os.environ.get(env_name)
                if api_key:
                    source = env_name
                    break

    # If still missing and interactive, prompt
    if not api_key and interactive:
        if has_saved and not args.provider:
            console.print(f"[dim]No saved API key for [cyan]{provider}[/cyan].[/dim]")
        api_key = prompt_api_key(provider)
        source = "prompt"
    elif not api_key:
        console.print("[red]error:[/red] no API key found for provider.")
        sys.exit(1)

    if api_key and source and source != "prompt":
        console.print(f"[dim]using API key from[/dim] [cyan]{source}[/cyan]")
    elif api_key and source == "prompt":
        console.print(f"[dim]using API key from[/dim] [cyan]prompt[/cyan]")

    # Final check - if no API key available, ask for it
    if not api_key or len(str(api_key).strip()) < 10:
        if interactive:
            api_key = prompt_api_key(provider)
            source = "prompt"
            console.print(f"[dim]using API key from[/dim] [cyan]prompt[/cyan]")
        else:
            console.print("[red]error:[/red] no API key found for provider.")
            sys.exit(1)

    # Persist configuration
    if args.save or interactive or not global_cfg:
        save_global_config(
            {
                "provider": provider,
                "model_name": model_name,
                "models": {**(global_cfg.get("models", {}) or {}), provider: model_name},
            }
        )
        if source == "prompt" or args.api_key or not get_stored_key(provider):
            save_stored_key(provider, api_key)
        if args.save:
            env_vals = {
                "MODEL_PROVIDER": provider,
                "MODEL_NAME": model_name,
            }
            if provider == "openrouter":
                env_vals["OPENROUTER_API_KEY"] = api_key
            elif provider == "gemini":
                env_vals["GEMINI_API_KEY"] = api_key
            elif provider == "anthropic":
                env_vals["ANTHROPIC_API_KEY"] = api_key
            else:
                env_vals["OPENAI_API_KEY"] = api_key
            save_env(env_vals)
            console.print(f"[green]saved settings to[/green] {ENV_FILE}")
        console.print(f"[green]saved configuration to[/green] ~/.pypilot/config.json")

    return provider, model_name, api_key


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_logo():
    """Render FRIDAY as a custom pixel-grid wordmark using Rich only."""
    # The glyphs are deliberately defined as individual pixel patterns rather
    # than loading an image or relying on a system font.
    glyphs = {
        "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
        "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
        "I": ("111", "010", "010", "010", "010", "010", "111"),
        "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
        "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
        "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    }
    colors = {
        "F": "#7C8796",
        "R": "#A7B1BF",
        "I": "#C6D0DD",
        "D": "#E7EDF5",
        "A": "#E7EDF5",
        "Y": "#55BDFF",
    }
    # Draw each pixel with a full block. A 5x7 grid keeps the wordmark
    # shorter than the previous version while preserving its proportions.
    # Rich bold styling makes the filled pixels appear stronger without
    # changing the cell width (which previously broke the glyphs).
    gap = 1
    rendered_rows = []
    for row_index in range(7):
        row = Text()
        for letter_index, letter in enumerate("FRIDAY"):
            pattern = glyphs[letter]
            if letter_index:
                row.append(" ")
            for pixel in pattern[row_index]:
                if pixel == "1":
                    row.append("█", style=f"bold {colors[letter]}")
                else:
                    row.append(" ")
        rendered_rows.append(Align.center(row))
    return Group(*rendered_rows)


def print_banner(provider: str, model_name: str) -> None:
    logo = render_logo()

    console.print()
    console.print(logo)
    console.print(Align.center(Text(TAGLINE, style="bold #91A8C2")))
    console.print(Align.center(Text(f"v{VERSION}  •  SYSTEM ONLINE", style="#55BDFF")))
    console.print()

    meta = Text()
    meta.append("provider ", style="#91A8C2")
    meta.append(provider, style="bold #55BDFF")
    meta.append("  •  model ", style="#91A8C2")
    meta.append(model_name, style="bold #E7EDF5")
    meta.append("  •  workspace ", style="#91A8C2")
    meta.append(str(workspace_root()), style="#E7EDF5")
    console.print(Align.center(meta))
    console.print()

    shortcuts = Text()
    shortcuts.append("type a task  ", style="#91A8C2")
    shortcuts.append("/help", style="bold #55BDFF")
    shortcuts.append("  •  ", style="#24415F")
    shortcuts.append("/model", style="bold #55BDFF")
    shortcuts.append("  •  ", style="#24415F")
    shortcuts.append("/provider", style="bold #55BDFF")
    shortcuts.append("  •  ", style="#24415F")
    shortcuts.append("/exit", style="bold #55BDFF")
    console.print(Align.center(shortcuts))
    console.print()
    console.print()


def print_assistant(text: str) -> None:
    if not text.strip():
        return
    console.print()
    header = Text()
    header.append("◈ ", style="bold #55BDFF")
    header.append(APP_NAME, style="bold #E7EDF5")
    header.append("  /  response", style="#91A8C2")
    console.print(header)
    console.print(Markdown(text))


def print_tool_call(name: str, args: dict) -> None:
    line = Text()
    line.append("  ◈ ", style="bold #55BDFF")
    line.append(name, style="bold #E7EDF5")
    line.append("  " + _brief(args), style="#91A8C2")
    console.print(line)


def print_tool_result(content) -> None:
    line = Text()
    line.append("    └─ ", style="#24415F")
    line.append(_first_line(content), style="#91A8C2")
    console.print(line)


def _content_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def _brief(value, limit: int = 90) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _first_line(value, limit: int = 110) -> str:
    text = str(value).strip()
    line = text.splitlines()[0] if text else ""
    return line if len(line) <= limit else line[: limit - 3] + "..."




# --------------------------------------------------------------------------- #
# Agent turn
# --------------------------------------------------------------------------- #
def run_turn(agent, history: list, user_text: str) -> None:
    """Run one turn, showing a spinner, tool activity and the final answer."""
    history.append(UserMessage(content=user_text))

    status = console.status(f"[cyan]{APP_NAME}[/cyan] [dim]thinking…[/dim]", spinner="dots")
    status.start()

    def on_event(event):
        status.stop()
        if isinstance(event, AssistantEvent):
            print_assistant(event.text)
        elif isinstance(event, ToolCallEvent):
            print_tool_call(event.name, event.arguments)
        elif isinstance(event, ToolResultEvent):
            print_tool_result(event.content)
        status.start()

    try:
        from src.my_agent.tools._common import workspace_root

        result = agent.run(history, on_event=on_event, workspace=str(workspace_root()))
        history.extend(result.messages)
    except KeyboardInterrupt:
        status.stop()
        console.print("[red]  (interrupted)[/red]")
    except Exception as exc:  # noqa: BLE001 - keep the REPL alive
        status.stop()
        console.print(f"[red]  error:[/red] {exc}")
    finally:
        status.stop()

    console.print()


def run_once(agent, task: str) -> None:
    run_turn(agent, [], task)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def print_help() -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(style="white")
    table.add_row("/help", "show this help")
    table.add_row("/model", "show the active provider and model")
    table.add_row("/model <name>", "change model for current provider")
    table.add_row("/provider <name> [model]", "switch provider (will prompt for key if missing)")
    table.add_row("/tools", "list the tools the agent can use")
    table.add_row("/workspace", "show the workspace directory")
    table.add_row("/memory", "show memory/conversation info")
    table.add_row("/reset", "clear the conversation history")
    table.add_row("/exit", "leave")
    console.print(Panel(table, title=f"[cyan]{APP_NAME}[/cyan] [dim]commands[/dim]", border_style="cyan", padding=(1, 2)))


def print_tools() -> None:
    table = Table(box=box.SIMPLE, header_style="bold cyan", padding=(0, 1))
    table.add_column("tool", style="cyan", no_wrap=True)
    table.add_column("description", style="white")
    for tool in CODING_TOOLS:
        summary = (tool.description or "").strip().splitlines()[0]
        table.add_row(tool.name, summary)
    console.print(table)


def print_model(provider: str, model_name: str) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column(style="bold white")
    table.add_row("provider", provider)
    table.add_row("model", model_name)
    console.print(table)


def handle_command(command: str, history: list, provider: str, model_name: str, agent=None):
    parts = command.strip().split()
    name = parts[0].lower() if parts else ""
    if name in ("/exit", "/quit", "/q"):
        return "exit", provider, model_name
    if name in ("/help", "/h", "/?"):
        print_help()
        return "ok", provider, model_name
    elif name == "/model":
        if len(parts) == 1:
            print_model(provider, model_name)
            return "ok", provider, model_name
        elif len(parts) >= 2 and agent is not None:
            new_model = " ".join(parts[1:])
            agent.model_name = new_model
            try:
                from src.my_agent.config.global_config import load_global_config, save_global_config

                cfg = load_global_config()
                cfg["model_name"] = new_model
                models = cfg.get("models", {}) or {}
                models[provider] = new_model
                cfg["models"] = models
                save_global_config(cfg)
            except Exception:
                pass
            console.print(f"[green]Switched model to[/green] [cyan]{new_model}[/cyan]")
            return "ok", provider, new_model
        else:
            console.print("[red]Usage:[/red] /model <model_name>")
            return "ok", provider, model_name
    elif name == "/provider":
        if len(parts) < 2:
            console.print("[red]Usage:[/red] /provider <name> [model]")
            return "ok", provider, model_name
        new_provider = parts[1].lower()
        if new_provider == "google":
            new_provider = "gemini"
        if new_provider not in SUPPORTED_PROVIDERS and new_provider != "google":
            console.print(f"[red]Unsupported provider.[/red] Choose from: {', '.join(SUPPORTED_PROVIDERS)}")
            return "ok", provider, model_name
        new_model = " ".join(parts[2:]) if len(parts) > 2 else None
        if agent is None:
            console.print("[red]Cannot switch provider in this context[/red]")
            return "ok", provider, model_name
        try:
            from src.my_agent.config.global_config import get_stored_key, load_global_config, save_global_config, save_stored_key
            from src.my_agent.models.provider import get_client

            global_cfg = load_global_config()
            api_key = get_stored_key(new_provider)
            source = None
            if api_key:
                if len(str(api_key).strip()) < 10:
                    api_key = None
                else:
                    source = "saved"
            env_key_map = {
                "openai": ["OPENAI_API_KEY"],
                "openrouter": ["OPENROUTER_API_KEY", "OPENAI_API_KEY"],
                "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"],
                "anthropic": ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"],
            }
            if not api_key:
                for env_name in env_key_map.get(new_provider, ["OPENAI_API_KEY"]):
                    api_key = os.environ.get(env_name)
                    if api_key:
                        source = env_name
                        break
            if not api_key and sys.stdin.isatty():
                console.print(f"[dim]No saved API key for [cyan]{new_provider}[/cyan].[/dim]")
                api_key = prompt_api_key(new_provider)
                source = "prompt"
            if not api_key:
                console.print("[red]No API key found for provider.[/red]")
                return "ok", provider, model_name
            client = get_client(api_key=api_key, provider=new_provider)
            agent.client = client
            if new_model:
                agent.model_name = new_model
            else:
                stored_models = global_cfg.get("models", {}) or {}
                agent.model_name = stored_models.get(new_provider) or PROVIDER_MODELS.get(new_provider, [agent.model_name])[0]
            save_global_config(
                {
                    "provider": new_provider,
                    "model_name": agent.model_name,
                    "models": {**(global_cfg.get("models", {}) or {}), new_provider: agent.model_name},
                }
            )
            if source in ("prompt", None) or not get_stored_key(new_provider):
                save_stored_key(new_provider, api_key)
            console.print(f"[green]Switched to[/green] [cyan]{new_provider}[/cyan] / [cyan]{agent.model_name}[/cyan]")
            return "ok", new_provider, agent.model_name
        except SystemExit:
            raise
        except Exception as exc:
            console.print(f"[red]Error switching provider:[/red] {exc}")
            return "ok", provider, model_name
    elif name == "/tools":
        print_tools()
    elif name == "/workspace":
        console.print(f"  {workspace_root()}")
    elif name == "/memory":
        try:
            from src.my_agent.memory import get_store

            store = get_store()
            convs = store.list_conversations(workspace_root())
            if convs:
                console.print(f"  [cyan]Conversations for this workspace:[/cyan]")
                for c in convs[:5]:
                    console.print(f"    • {c['id'][:8]}... - {c['title']} ({c['total_tokens']} tokens)")
            else:
                console.print("  [dim]No conversations stored yet[/dim]")
        except Exception as exc:
            console.print(f"  [red]Error:[/red] {exc}")
    elif name in ("/reset", "/clear"):
        history.clear()
        console.print("[dim]conversation history cleared[/dim]")
    else:
        console.print(f"[red]unknown command:[/red] {name} [dim](try /help)[/dim]")
    return "ok", provider, model_name


# --------------------------------------------------------------------------- #
# REPL
# --------------------------------------------------------------------------- #
def _make_prompt_session():
    """A prompt_toolkit session with history + command completion, if possible."""
    if not sys.stdin.isatty():
        return None
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.styles import Style as PromptStyle
    except Exception:
        return None

    style = PromptStyle.from_dict({"prompt": "bold #55BDFF"})
    history_path = Path.home() / ".pypilot_history"
    try:
        return PromptSession(
            message=HTML("<prompt>◈ FRIDAY ❯ </prompt>"),
            completer=WordCompleter(COMMANDS, ignore_case=True),
            history=FileHistory(str(history_path)),
            style=style,
        )
    except Exception:
        return None


def repl(agent, provider: str, model_name: str) -> None:
    print_banner(provider, model_name)
    session = _make_prompt_session()
    history: list = []
    current_provider = provider
    current_model = model_name

    while True:
        try:
            if session is not None:
                line = session.prompt()
            else:
                line = input("FRIDAY > ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        line = line.strip()
        if not line:
            continue
        if line.startswith("/"):
            result = handle_command(line, history, current_provider, current_model, agent=agent)
            if isinstance(result, tuple):
                status, current_provider, current_model = result
                if status == "exit":
                    break
            elif result == "exit":
                break
            continue
        run_turn(agent, history, line)

    console.print("[#91A8C2]bye from FRIDAY[/#91A8C2]")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="friday",
        description="FRIDAY - an autonomous terminal coding agent powered by the OpenAI SDK.",
    )
    parser.add_argument("prompt", nargs="*", help="Run a single task and exit.")
    parser.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        help="Model provider (openai, openrouter, gemini, anthropic).",
    )
    parser.add_argument("--model", help="Model name, e.g. gpt-4o.")
    parser.add_argument("--api-key", help="API key (else from env or prompted).")
    parser.add_argument("--save", action="store_true", help="Save settings to .env.")
    args = parser.parse_args(argv)

    _make_output_safe()
    try:
        provider, model_name, api_key = configure(args)
        client = get_client(api_key=api_key, provider=provider)
        agent = Agent(client, model_name=model_name)
        if args.prompt:
            run_once(agent, " ".join(args.prompt))
        else:
            repl(agent, provider, model_name)
    except KeyboardInterrupt:
        console.print()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]error:[/red] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
