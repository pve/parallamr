"""Shared pytest fixtures and test utilities for all provider tests.

This module provides reusable test helpers that eliminate boilerplate code
and ensure consistent mocking patterns across all provider tests.
"""

import pytest
from unittest.mock import AsyncMock
from typing import Dict, Any, Optional
import aiohttp


# ============================================================================
# MOCK HELPERS - Eliminate 90% of test boilerplate
# ============================================================================

def create_mock_session() -> AsyncMock:
    """Create properly configured mock aiohttp.ClientSession.

    Returns:
        AsyncMock configured to behave like aiohttp.ClientSession

    Example:
        >>> session = create_mock_session()
        >>> provider = OpenAIProvider(api_key="test", session=session)
    """
    mock_session = AsyncMock(spec=aiohttp.ClientSession)
    mock_session.close = AsyncMock()
    return mock_session


def create_mock_response(status: int, json_data: Dict[str, Any]) -> AsyncMock:
    """Create mock HTTP response with given status and JSON data.

    Args:
        status: HTTP status code (e.g., 200, 404, 500)
        json_data: Dictionary to return from response.json()

    Returns:
        AsyncMock configured as HTTP response

    Example:
        >>> response = create_mock_response(200, {"result": "success"})
        >>> data = await response.json()  # Returns {"result": "success"}
    """
    mock_response = AsyncMock()
    mock_response.status = status
    mock_response.json = AsyncMock(return_value=json_data)
    mock_response.raise_for_status = AsyncMock()
    return mock_response


def create_mock_context(response: AsyncMock) -> AsyncMock:
    """Create mock async context manager for HTTP response.

    Args:
        response: Mock response object from create_mock_response()

    Returns:
        AsyncMock that can be used with 'async with' statements

    Example:
        >>> response = create_mock_response(200, {"data": "test"})
        >>> ctx = create_mock_context(response)
        >>> async with ctx as resp:
        ...     data = await resp.json()
    """
    ctx = AsyncMock()
    ctx.__aenter__.return_value = response
    ctx.__aexit__.return_value = None
    return ctx


def setup_mock_post(
    mock_session: AsyncMock,
    status: int,
    json_data: Dict[str, Any]
) -> None:
    """Setup mock session for POST request (one-liner setup).

    This is the most common setup pattern - use this to reduce 20 lines
    of boilerplate to a single function call.

    Args:
        mock_session: Mock session from create_mock_session()
        status: HTTP status code to return
        json_data: JSON data to return from API

    Example:
        >>> from fixtures.ollama_responses import SUCCESSFUL_COMPLETION
        >>> session = create_mock_session()
        >>> setup_mock_post(session, 200, SUCCESSFUL_COMPLETION)
        >>> provider = OllamaProvider(session=session)
        >>> result = await provider.get_completion("test", "llama3.1")
    """
    response = create_mock_response(status, json_data)
    ctx = create_mock_context(response)
    mock_session.post.return_value = ctx
    mock_session.request.return_value = ctx  # Some providers use .request()


def setup_mock_get(
    mock_session: AsyncMock,
    status: int,
    json_data: Dict[str, Any]
) -> None:
    """Setup mock session for GET request (one-liner setup).

    Args:
        mock_session: Mock session from create_mock_session()
        status: HTTP status code to return
        json_data: JSON data to return from API

    Example:
        >>> from fixtures.ollama_responses import MODELS_LIST_RESPONSE
        >>> session = create_mock_session()
        >>> setup_mock_get(session, 200, MODELS_LIST_RESPONSE)
        >>> provider = OllamaProvider(session=session)
        >>> models = await provider.list_models()
    """
    response = create_mock_response(status, json_data)
    ctx = create_mock_context(response)
    mock_session.get.return_value = ctx


def setup_mock_error(
    mock_session: AsyncMock,
    exception: Exception
) -> None:
    """Setup mock session to raise an exception.

    Args:
        mock_session: Mock session from create_mock_session()
        exception: Exception to raise (e.g., asyncio.TimeoutError())

    Example:
        >>> session = create_mock_session()
        >>> setup_mock_error(session, asyncio.TimeoutError())
        >>> provider = OllamaProvider(session=session)
        >>> result = await provider.get_completion("test", "llama3.1")
        >>> assert result.success is False
        >>> assert "timeout" in result.error_message.lower()
    """
    mock_session.post.side_effect = exception
    mock_session.get.side_effect = exception
    mock_session.request.side_effect = exception


def setup_mock_sequential_responses(
    mock_session: AsyncMock,
    responses: list[tuple[int, Dict[str, Any]]]
) -> None:
    """Setup mock session to return different responses for sequential calls.

    Args:
        mock_session: Mock session from create_mock_session()
        responses: List of (status_code, json_data) tuples

    Example:
        >>> session = create_mock_session()
        >>> setup_mock_sequential_responses(session, [
        ...     (200, {"response": "First"}),
        ...     (200, {"response": "Second"}),
        ...     (200, {"response": "Third"})
        ... ])
        >>> provider = OllamaProvider(session=session)
        >>> r1 = await provider.get_completion("1", "llama3.1")
        >>> r2 = await provider.get_completion("2", "llama3.1")
        >>> r3 = await provider.get_completion("3", "llama3.1")
    """
    contexts = []
    for status, json_data in responses:
        response = create_mock_response(status, json_data)
        ctx = create_mock_context(response)
        contexts.append(ctx)

    mock_session.post.side_effect = contexts
    mock_session.request.side_effect = contexts


# ============================================================================
# PYTEST FIXTURES - Reusable across all provider tests
# ============================================================================

@pytest.fixture
def mock_session():
    """Provide clean mock session for each test.

    Usage:
        @pytest.mark.asyncio
        async def test_completion(mock_session):
            setup_mock_post(mock_session, 200, SUCCESSFUL_COMPLETION)
            provider = OllamaProvider(session=mock_session)
            result = await provider.get_completion("test", "llama3.1")
            assert result.success is True
    """
    return create_mock_session()


@pytest.fixture
def mock_env_no_keys(monkeypatch):
    """Remove all provider API keys from environment.

    Useful for testing default behavior when no keys are configured.

    Usage:
        def test_init_without_keys(mock_env_no_keys):
            provider = OpenAIProvider()
            assert provider.api_key is None
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)


@pytest.fixture
def mock_openai_key(monkeypatch):
    """Set OPENAI_API_KEY environment variable.

    Usage:
        def test_with_openai_key(mock_openai_key):
            provider = OpenAIProvider()
            assert provider.api_key == "test-openai-key-123"
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key-123")


@pytest.fixture
def mock_openrouter_key(monkeypatch):
    """Set OPENROUTER_API_KEY environment variable.

    Usage:
        def test_with_openrouter_key(mock_openrouter_key):
            provider = OpenRouterProvider()
            assert provider.api_key == "test-openrouter-key-456"
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key-456")


@pytest.fixture
def mock_ollama_url(monkeypatch):
    """Set OLLAMA_BASE_URL environment variable.

    Usage:
        def test_with_ollama_url(mock_ollama_url):
            provider = OllamaProvider()
            assert provider.base_url == "http://test-ollama:8080"
    """
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://test-ollama:8080")


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Register custom pytest markers.

    Markers:
        integration: Tests that require real API access
        slow: Tests that take >1 second to execute
    """
    config.addinivalue_line(
        "markers",
        "integration: integration tests requiring real APIs (skip by default)"
    )
    config.addinivalue_line(
        "markers",
        "slow: slow tests taking >1 second (skip with -m 'not slow')"
    )


# ============================================================================
# TEST ASSERTION HELPERS
# ============================================================================

def assert_provider_response_valid(response, success: bool = True):
    """Assert that ProviderResponse has all required fields.

    Args:
        response: ProviderResponse object to validate
        success: Expected success status

    Raises:
        AssertionError: If response is invalid

    Example:
        >>> result = await provider.get_completion("test", "gpt-4")
        >>> assert_provider_response_valid(result, success=True)
    """
    from parallamr.models import ProviderResponse

    assert isinstance(response, ProviderResponse), \
        f"Expected ProviderResponse, got {type(response)}"

    assert response.success == success, \
        f"Expected success={success}, got {response.success}"

    assert hasattr(response, 'output'), "Response missing 'output' field"
    assert hasattr(response, 'output_tokens'), "Response missing 'output_tokens' field"
    assert hasattr(response, 'error_message'), "Response missing 'error_message' field"

    if success:
        assert len(response.output) > 0, "Successful response should have non-empty output"
        assert response.output_tokens > 0, "Successful response should have token count"
        assert response.error_message is None or response.error_message == "", \
            "Successful response should not have error message"
    else:
        assert response.error_message is not None and len(response.error_message) > 0, \
            "Failed response must have error message"


def assert_session_not_closed(mock_session: AsyncMock):
    """Assert that injected session was not closed by provider.

    Providers should never close injected sessions - the caller owns them.

    Args:
        mock_session: Mock session to check

    Example:
        >>> session = create_mock_session()
        >>> setup_mock_post(session, 200, SUCCESSFUL_COMPLETION)
        >>> provider = OllamaProvider(session=session)
        >>> await provider.get_completion("test", "llama3.1")
        >>> assert_session_not_closed(session)
    """
    mock_session.close.assert_not_called()
