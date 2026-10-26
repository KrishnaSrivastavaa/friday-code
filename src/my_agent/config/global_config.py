"""Global configuration persisted in the user's home directory.

Used so users don't have to re-enter provider/model/API keys on every run.
Keys are stored per provider. Switching providers will only prompt for the
new provider's API key if it's not already stored.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def get_config_dir() -> Path:
    config_dir = Path.home() / ".pypilot"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_path() -> Path:
    return get_config_dir() / "config.json"


def load_global_config() -> dict[str, Any]:
    path = get_config_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_global_config(config: dict[str, Any]) -> None:
    path = get_config_path()
    existing = load_global_config()
    existing.update(config)
    with path.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    # Make it user-readable only on unix-like systems if possible
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def get_stored_key(provider: str) -> str | None:
    config = load_global_config()
    keys = config.get("api_keys", {}) or {}
    key = keys.get(provider)
    if key:
        return str(key).strip() or None
    return None


def save_stored_key(provider: str, api_key: str) -> None:
    config = load_global_config()
    keys = config.get("api_keys", {}) or {}
    keys[provider] = api_key.strip()
    config["api_keys"] = keys
    save_global_config(config)
