from __future__ import annotations

import os

from openai import OpenAI

from src.my_agent.config.settings import settings


def get_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    max_retries: int | None = None,
    provider: str | None = None,
) -> OpenAI:
    """Create an OpenAI SDK client configured from settings/arguments.

    Supports OpenAI, OpenRouter, Google Gemini, and Anthropic via OpenAI-compatible endpoints.
    """
    provider = (provider or settings.model_provider or "openai").lower()
    retries = settings.max_retries if max_retries is None else max_retries

    if provider == "openrouter":
        key = (
            api_key
            or settings.openrouter_api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not key:
            raise ValueError(
                "OpenRouter API key is required. Pass --api-key, set OPENROUTER_API_KEY, "
                "or run `pypilot` and save it to .env."
            )
        return OpenAI(
            api_key=key,
            base_url=base_url or settings.openrouter_base_url,
            max_retries=retries,
        )
    elif provider == "gemini" or provider == "google":
        key = (
            api_key
            or settings.gemini_api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not key:
            raise ValueError(
                "Gemini API key is required. Pass --api-key, set GEMINI_API_KEY, "
                "or run `pypilot` and save it to .env."
            )
        return OpenAI(
            api_key=key,
            base_url=base_url or settings.gemini_base_url,
            max_retries=retries,
        )
    elif provider == "anthropic":
        key = (
            api_key
            or settings.anthropic_api_key
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not key:
            raise ValueError(
                "Anthropic API key is required. Pass --api-key, set ANTHROPIC_API_KEY, "
                "or run `pypilot` and save it to .env."
            )
        return OpenAI(
            api_key=key,
            base_url=base_url or settings.anthropic_base_url,
            max_retries=retries,
        )
    else:
        key = api_key or settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key is required. Pass --api-key, set OPENAI_API_KEY, "
                "or run `pypilot` and save it to .env."
            )
        return OpenAI(
            api_key=key,
            base_url=base_url or settings.openai_base_url,
            max_retries=retries,
        )


# Backwards-compatible alias
get_model = get_client

