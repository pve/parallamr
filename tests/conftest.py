"""Shared pytest fixtures and test utilities for all provider tests.

This module provides reusable test helpers that eliminate boilerplate code
and ensure consistent mocking patterns across all provider tests.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any, Optional, List


# ============================================================================
# MOCK HELPERS
# ============================================================================

def create_mock_session() -> AsyncMock:
    """Create properly configured mock aiohttp.ClientSession."""
    import aiohttp
    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_session.close = AsyncMock()
    return mock_session


def create_mock_response(status: int, json_data: Dict[str, Any]) -> AsyncMock:
    """Create mock HTTP response with given status and JSON data."""
    mock_response = AsyncMock()
    mock_response.status = status
    mock_response.json = AsyncMock(return_value=json_data)
    mock_response.text = AsyncMock(return_value=str(json_data))
    mock_response.raise_for_status = MagicMock()
    return mock_response


def create_mock_context(response: AsyncMock) -> AsyncMock:
    """Create mock async context manager for HTTP response."""
    ctx = MagicMock() # Use MagicMock for the context manager itself
    ctx.__aenter__ = AsyncMock(return_value=response)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


def setup_mock_post(
    mock_session: AsyncMock,
    status: int,
    json_data: Dict[str, Any]
) -> None:
    """Setup mock session for POST request."""
    response = create_mock_response(status, json_data)
    ctx = create_mock_context(response)
    mock_session.post.return_value = ctx
    mock_session.request.return_value = ctx


def setup_mock_get(
    mock_session: AsyncMock,
    status: int,
    json_data: Dict[str, Any]
) -> None:
    """Setup mock session for GET request."""
    response = create_mock_response(status, json_data)
    ctx = create_mock_context(response)
    mock_session.get.return_value = ctx
    mock_session.request.return_value = ctx


def setup_mock_error(
    mock_session: AsyncMock,
    exception: Exception
) -> None:
    """Setup mock session to raise an exception."""
    mock_session.post.side_effect = exception
    mock_session.get.side_effect = exception
    mock_session.request.side_effect = exception


def setup_mock_sequential_responses(
    mock_session: AsyncMock,
    responses_data: List[tuple[int, Dict[str, Any]]]
) -> None:
    """Setup mock session to return different responses for sequential calls."""
    contexts = []
    for status, json_data in responses_data:
        response = create_mock_response(status, json_data)
        ctx = create_mock_context(response)
        contexts.append(ctx)

    mock_session.post.side_effect = contexts
    mock_session.get.side_effect = contexts
    mock_session.request.side_effect = contexts


# ============================================================================
# PYTEST FIXTURES
# ============================================================================

@pytest.fixture
def mock_session():
    """Provide clean mock session for each test."""
    return create_mock_session()


@pytest.fixture
def mock_env_no_keys(monkeypatch):
    """Remove all provider API keys from environment."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)


@pytest.fixture
def mock_openai_key(monkeypatch):
    """Set OPENAI_API_KEY environment variable."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key-123")


@pytest.fixture
def mock_openrouter_key(monkeypatch):
    """Set OPENROUTER_API_KEY environment variable."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key-456")


@pytest.fixture
def mock_ollama_url(monkeypatch):
    """Set OLLAMA_BASE_URL environment variable."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://test-ollama:8080")


# ============================================================================
# TEST ASSERTION HELPERS
# ============================================================================

def assert_provider_response_valid(response, success: bool = True):
    """Assert that ProviderResponse has all required fields."""
    from parallamr.models import ProviderResponse

    assert isinstance(response, ProviderResponse)
    assert response.success == success

    if success:
        assert len(response.output) > 0
        assert response.output_tokens > 0
    else:
        assert response.error_message is not None and len(response.error_message) > 0


def assert_session_not_closed(mock_session: AsyncMock):
    """Assert that injected session was not closed by provider."""
    mock_session.close.assert_not_called()
