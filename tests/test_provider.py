"""Tests for the OpenAI client factory."""

from __future__ import annotations

import os

from openai import OpenAI

from src.my_agent.config.settings import settings
from src.my_agent.models.provider import get_client


def test_get_client_constructs_openai_client():
    client = get_client(api_key="sk-test")
    assert isinstance(client, OpenAI)


def test_get_client_respects_max_retries():
    client = get_client(api_key="sk-test", max_retries=5)
    assert isinstance(client, OpenAI)


def test_get_client_raises_without_key(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    try:
        get_client()
    except ValueError as exc:
        assert "API key" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_settings_have_sane_defaults():
    assert settings.model_provider == "openai"
    assert settings.temperature == 0.0
    assert settings.max_tokens > 0
    assert settings.max_retries >= 0
