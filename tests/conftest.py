"""Pytest fixtures shared across the test-suite."""

from __future__ import annotations

import pytest

from src.my_agent.tools import _common


@pytest.fixture
def workspace(tmp_path):
    """Point the tools at a fresh temp directory for the duration of a test.

    Restores the previous workspace root afterwards so tests never leak state
    into one another.
    """
    original = _common.WORKSPACE_ROOT
    _common.set_workspace(tmp_path)
    try:
        yield tmp_path
    finally:
        _common.set_workspace(original)
