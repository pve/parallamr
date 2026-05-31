"""Comprehensive tests for OpenRouter provider implementation."""

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from parallamr.models import ProviderResponse
from parallamr.providers.base import (
    AuthenticationError,
    ContextWindowExceededError,
    ModelNotAvailableError,
    Provider,
    ProviderError,
    RateLimitError,
)
from tests.fixtures.openrouter_responses import (
    COMPLETION_EMPTY_CHOICES,
    COMPLETION_EMPTY_RESPONSE,
    COMPLETION_GPT4,
    COMPLETION_LENGTH_FINISH,
    COMPLETION_LLAMA31,
    COMPLETION_MISSING_CHOICES,
    COMPLETION_MISSING_MESSAGE,
    COMPLETION_NO_USAGE,
    CONTEXT_WINDOWS,
    ERROR_400_BAD_REQUEST,
    ERROR_401_UNAUTHORIZED,
    ERROR_403_FORBIDDEN,
    ERROR_404_MODEL_NOT_FOUND,
    ERROR_413_CONTEXT_LENGTH_EXCEEDED,
    ERROR_429_CREDITS_EXHAUSTED,
    ERROR_429_RATE_LIMIT,
    ERROR_500_INTERNAL_SERVER,
    ERROR_502_BAD_GATEWAY,
    ERROR_503_SERVICE_UNAVAILABLE,
    MODELS_LIST_EMPTY,
    MODELS_LIST_MISSING_CONTEXT,
    MODELS_LIST_RESPONSE,
    MODELS_LIST_SINGLE,
    SUCCESSFUL_COMPLETION,
    create_completion_response,
    create_error_response,
    create_models_list,
)

# Import the OpenRouter provider
from parallamr.providers.openrouter import OpenRouterProvider
from tests.conftest import (
    setup_mock_post,
    setup_mock_get,
    setup_mock_error,
    setup_mock_sequential_responses,
    assert_session_not_closed,
    create_mock_response,
    create_mock_context,
    assert_provider_response_valid
)


class TestOpenRouterProviderInit:
    """Test OpenRouter provider initialization (11 tests)."""

    def test_init_with_api_key(self):
        """Provider accepts API key directly."""
        provider = OpenRouterProvider(api_key="test-openrouter-key-123")
        assert provider.api_key == "test-openrouter-key-123"

    def test_init_with_env_getter(self):
        """Provider accepts custom env_getter for API key."""
        def mock_env_getter(key: str) -> Optional[str]:
            if key == "OPENROUTER_API_KEY":
                return "env-test-key-456"
            return None

        provider = OpenRouterProvider(env_getter=mock_env_getter)
        assert provider.api_key == "env-test-key-456"

    def test_init_without_api_key(self, monkeypatch):
        """Provider handles missing API key."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        provider = OpenRouterProvider()
        assert provider.api_key is None

    def test_init_with_custom_base_url(self):
        """Provider accepts custom base URL."""
        provider = OpenRouterProvider(
            api_key="test-key",
            base_url="https://custom-openrouter.example.com/api/v1"
        )
        assert provider.base_url == "https://custom-openrouter.example.com/api/v1"

    def test_init_default_base_url(self):
        """Provider uses default OpenRouter base URL."""
        provider = OpenRouterProvider(api_key="test-key")
        assert provider.base_url == "https://openrouter.ai/api/v1"

    def test_init_with_timeout(self):
        """Provider accepts custom timeout."""
        provider = OpenRouterProvider(api_key="test-key", timeout=600)
        assert provider.timeout == 600

    def test_init_default_timeout(self):
        """Provider uses default timeout."""
        provider = OpenRouterProvider(api_key="test-key")
        assert provider.timeout == 300

    def test_init_with_session(self):
        """Provider accepts injected aiohttp session."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        assert provider._session is mock_session

    def test_init_without_session(self):
        """Provider initializes with None session by default."""
        provider = OpenRouterProvider(api_key="test-key")
        assert provider._session is None

    def test_init_model_cache_empty(self):
        """Provider initializes with empty model cache."""
        provider = OpenRouterProvider(api_key="test-key")
        assert provider._model_cache is None

    def test_init_env_key_precedence(self):
        """Direct API key takes precedence over env getter."""
        def mock_env_getter(key: str) -> Optional[str]:
            return "env-key-should-not-be-used"

        provider = OpenRouterProvider(
            api_key="direct-key-123",
            env_getter=mock_env_getter
        )
        assert provider.api_key == "direct-key-123"


class TestOpenRouterProviderCompletion:
    """Test OpenRouter provider completion requests (15 tests)."""

    @pytest.mark.asyncio
    async def test_successful_completion(self, mock_session):
        """Provider returns successful completion response."""
        setup_mock_sequential_responses(mock_session, [
            (200, SUCCESSFUL_COMPLETION),
            (200, MODELS_LIST_RESPONSE)
        ])
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        assert_provider_response_valid(result, success=True)
        assert result.output == "This is a test response from Claude 3.5 Sonnet via OpenRouter."
        assert result.output_tokens == 25

    @pytest.mark.asyncio
    async def test_completion_without_api_key(self):
        """Provider handles missing API key gracefully."""
        provider = OpenRouterProvider(api_key=None)
        result = await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        assert result.success is False
        assert "API key" in result.error_message

    @pytest.mark.asyncio
    async def test_completion_with_kwargs(self, mock_session):
        """Provider passes additional kwargs to API."""
        setup_mock_sequential_responses(mock_session, [
            (200, SUCCESSFUL_COMPLETION),
            (200, MODELS_LIST_RESPONSE)
        ])
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion(
            "test prompt",
            "anthropic/claude-3.5-sonnet",
            temperature=0.7,
            max_tokens=100
        )

        call_args = mock_session.request.call_args_list[0]
        payload = call_args[1]["json"]
        assert payload.get("temperature") == 0.7
        assert payload.get("max_tokens") == 100

    @pytest.mark.asyncio
    async def test_completion_request_format(self, mock_session):
        """Provider formats request correctly."""
        setup_mock_sequential_responses(mock_session, [
            (200, SUCCESSFUL_COMPLETION),
            (200, MODELS_LIST_RESPONSE)
        ])
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        call_args = mock_session.request.call_args_list[0]
        assert call_args[0][1] == "https://openrouter.ai/api/v1/chat/completions"
        assert call_args[1]["headers"]["Authorization"] == "Bearer test-key"

        payload = call_args[1]["json"]
        assert payload["model"] == "anthropic/claude-3.5-sonnet"
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "test prompt"

    @pytest.mark.asyncio
    async def test_completion_extracts_tokens(self, mock_session):
        """Provider extracts token counts from response."""
        setup_mock_sequential_responses(mock_session, [
            (200, SUCCESSFUL_COMPLETION),
            (200, MODELS_LIST_RESPONSE)
        ])
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")
        assert result.output_tokens == 25

    @pytest.mark.asyncio
    async def test_completion_timeout(self, mock_session):
        """Provider handles request timeout."""
        setup_mock_error(mock_session, asyncio.TimeoutError())
        provider = OpenRouterProvider(api_key="test-key", session=mock_session, timeout=10)
        result = await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        assert result.success is False
        assert "timeout" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_completion_network_error(self, mock_session):
        """Provider handles network errors."""
        setup_mock_error(mock_session, aiohttp.ClientError("Connection failed"))
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        assert result.success is False
        assert "network error" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_completion_session_not_closed(self, mock_session):
        """Provider does not close injected session."""
        setup_mock_sequential_responses(mock_session, [
            (200, SUCCESSFUL_COMPLETION),
            (200, MODELS_LIST_RESPONSE)
        ])
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")
        assert_session_not_closed(mock_session)

    @pytest.mark.asyncio
    async def test_completion_malformed_response(self, mock_session):
        """Provider handles malformed response (missing choices)."""
        setup_mock_post(mock_session, 200, {"id": "test"})
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test", "model")
        assert result.success is False
        assert "Malformed" in result.error_message

    @pytest.mark.asyncio
    async def test_completion_empty_choices(self, mock_session):
        """Provider handles empty choices."""
        setup_mock_post(mock_session, 200, {"choices": []})
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test", "model")
        assert result.success is False
        assert "choices" in result.error_message

    @pytest.mark.asyncio
    async def test_completion_no_content(self, mock_session):
        """Provider handles missing message content."""
        setup_mock_post(mock_session, 200, {"choices": [{"message": {}}]})
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test", "model")
        assert result.success is False
        assert "message" in result.error_message

    @pytest.mark.asyncio
    async def test_completion_with_model_cache(self, mock_session):
        """Provider uses model cache if available."""
        setup_mock_post(mock_session, 200, SUCCESSFUL_COMPLETION)
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        provider._model_cache = {"anthropic/claude-3.5-sonnet": {"context_length": 200000}}

        result = await provider.get_completion("test", "anthropic/claude-3.5-sonnet")
        assert result.success is True
        assert result.context_window == 200000
        # Should NOT call /models
        assert mock_session.request.call_count == 1

    @pytest.mark.asyncio
    async def test_completion_invalid_json(self, mock_session):
        """Provider handles invalid JSON response."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.side_effect = ValueError()
        mock_response.text.return_value = "Not JSON"
        setup_mock_post(mock_session, 200, {})
        mock_session.request.return_value.__aenter__.return_value = mock_response

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test", "model")
        assert result.success is False
        assert "Invalid JSON" in result.error_message

    @pytest.mark.asyncio
    async def test_completion_unexpected_exception(self, mock_session):
        """Provider handles unexpected exceptions."""
        setup_mock_error(mock_session, RuntimeError("Unexpected"))
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test", "model")
        assert result.success is False
        assert "unexpected error" in result.error_message.lower()


class TestOpenRouterProviderModels:
    """Test OpenRouter provider model management (10 tests)."""

    @pytest.mark.asyncio
    async def test_list_models_success(self, mock_session):
        """Provider lists available models."""
        setup_mock_get(mock_session, 200, MODELS_LIST_RESPONSE)
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        models = await provider.list_models()

        assert len(models) == 4
        assert "anthropic/claude-3.5-sonnet" in models

    @pytest.mark.asyncio
    async def test_get_context_window_success(self, mock_session):
        """Provider retrieves context window for model."""
        setup_mock_get(mock_session, 200, MODELS_LIST_RESPONSE)
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        context = await provider.get_context_window("anthropic/claude-3.5-sonnet")
        assert context == 200000

    @pytest.mark.asyncio
    async def test_get_context_window_unknown(self, mock_session):
        """Provider handles unknown model context window."""
        setup_mock_get(mock_session, 200, MODELS_LIST_RESPONSE)
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        context = await provider.get_context_window("unknown/model")
        assert context is None

    @pytest.mark.asyncio
    async def test_list_models_caching(self, mock_session):
        """Provider caches model list."""
        setup_mock_get(mock_session, 200, MODELS_LIST_RESPONSE)
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        await provider.list_models()
        await provider.list_models()
        assert mock_session.request.call_count == 1

    @pytest.mark.asyncio
    async def test_list_models_error(self, mock_session):
        """Provider handles error listing models."""
        setup_mock_error(mock_session, aiohttp.ClientError("API Error"))
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        models = await provider.list_models()
        assert models == []

    def test_is_model_available_with_cache(self):
        """Provider checks model availability using cache."""
        provider = OpenRouterProvider()
        provider._model_cache = {"anthropic/claude-3.5-sonnet": {}}
        assert provider.is_model_available("anthropic/claude-3.5-sonnet") is True
        assert provider.is_model_available("nonexistent") is False

    def test_is_model_available_optimistic(self):
        """Provider is optimistic when cache is empty."""
        provider = OpenRouterProvider()
        assert provider.is_model_available("any-model") is True

    @pytest.mark.asyncio
    async def test_list_models_empty(self, mock_session):
        """Provider handles empty model list."""
        setup_mock_get(mock_session, 200, {"data": []})
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        models = await provider.list_models()
        assert models == []

    @pytest.mark.asyncio
    async def test_get_models_info_failure(self, mock_session):
        """Provider handles failure to fetch models info."""
        setup_mock_get(mock_session, 500, {})
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        info = await provider._get_models_info()
        assert info is None

    def test_get_provider_name(self):
        """Provider returns correct provider name."""
        provider = OpenRouterProvider()
        assert provider.get_provider_name() == "openrouter"


class TestOpenRouterProviderErrorHandling:
    """Test OpenRouter provider error handling (12 tests)."""

    @pytest.mark.asyncio
    async def test_error_401_unauthorized(self, mock_session):
        """Provider handles 401 unauthorized error."""
        setup_mock_post(mock_session, 401, ERROR_401_UNAUTHORIZED)
        provider = OpenRouterProvider(api_key="invalid-key", session=mock_session)
        result = await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        assert result.success is False
        assert "authentication" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_429_rate_limit(self, mock_session):
        """Provider handles 429 rate limit error."""
        setup_mock_post(mock_session, 429, ERROR_429_RATE_LIMIT)
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        assert result.success is False
        assert "rate limit" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_429_credits_exhausted(self, mock_session):
        """Provider handles 429 credits exhausted error."""
        setup_mock_post(mock_session, 429, ERROR_429_CREDITS_EXHAUSTED)
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        assert result.success is False
        assert "credits" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_500_internal_server(self, mock_session):
        """Provider handles 500 internal server error."""
        setup_mock_post(mock_session, 500, ERROR_500_INTERNAL_SERVER)
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        assert result.success is False
        assert "server error" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_413_context_length(self, mock_session):
        """Provider handles 413 context length exceeded."""
        setup_mock_post(mock_session, 413, ERROR_413_CONTEXT_LENGTH_EXCEEDED)
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("very long prompt", "model")
        assert result.success is False
        assert "context" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_403_forbidden(self, mock_session):
        """Provider handles 403 forbidden."""
        setup_mock_post(mock_session, 403, ERROR_403_FORBIDDEN)
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test", "model")
        assert result.success is False
        assert "forbidden" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_404_not_found(self, mock_session):
        """Provider handles 404 not found."""
        setup_mock_post(mock_session, 404, ERROR_404_MODEL_NOT_FOUND)
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test", "model")
        assert result.success is False
        assert "not found" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_non_json(self, mock_session):
        """Provider handles non-JSON error response."""
        mock_response = AsyncMock()
        mock_response.status = 502
        mock_response.json.side_effect = ValueError()
        mock_response.text.return_value = "Bad Gateway"
        setup_mock_post(mock_session, 502, {})
        mock_session.request.return_value.__aenter__.return_value = mock_response

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test", "model")
        assert "Bad Gateway" in result.error_message

    @pytest.mark.asyncio
    async def test_error_empty_body(self, mock_session):
        """Provider handles error with empty body."""
        mock_response = AsyncMock()
        mock_response.status = 503
        mock_response.json.side_effect = ValueError()
        mock_response.text.side_effect = Exception()
        setup_mock_post(mock_session, 503, {})
        mock_session.request.return_value.__aenter__.return_value = mock_response

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test", "model")
        assert "Could not read response body" in result.error_message

    @pytest.mark.asyncio
    async def test_error_generic_message(self, mock_session):
        """Provider extracts error message from JSON."""
        setup_mock_post(mock_session, 400, {"error": {"message": "Custom error"}})
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test", "model")
        assert "Custom error" in result.error_message
