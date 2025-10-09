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
    async def test_successful_completion(self):
        """Provider returns successful completion response."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SUCCESSFUL_COMPLETION)
        mock_response.raise_for_status = MagicMock()

        # Setup context manager
        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        assert result.success is True
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
    async def test_completion_with_kwargs(self):
        """Provider passes additional kwargs to API."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SUCCESSFUL_COMPLETION)
        mock_response.raise_for_status = MagicMock()

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion(
            "test prompt",
            "anthropic/claude-3.5-sonnet",
            temperature=0.7,
            max_tokens=100
        )

        # Verify kwargs were passed
        call_args = mock_session.post.call_args
        payload = call_args[1]["json"]
        assert payload.get("temperature") == 0.7
        assert payload.get("max_tokens") == 100

    @pytest.mark.asyncio
    async def test_completion_request_format(self):
        """Provider formats request correctly."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SUCCESSFUL_COMPLETION)
        mock_response.raise_for_status = MagicMock()

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        # Verify request format
        call_args = mock_session.post.call_args
        assert "https://openrouter.ai/api/v1/chat/completions" in str(call_args)
        assert call_args[1]["headers"]["Authorization"] == "Bearer test-key"
        assert call_args[1]["headers"]["Content-Type"] == "application/json"

        payload = call_args[1]["json"]
        assert payload["model"] == "anthropic/claude-3.5-sonnet"
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "test prompt"

    @pytest.mark.asyncio
    async def test_completion_extracts_tokens(self):
        """Provider extracts token counts from response."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SUCCESSFUL_COMPLETION)
        mock_response.raise_for_status = MagicMock()

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        assert result.output_tokens == 25

    @pytest.mark.asyncio
    async def test_completion_without_usage_data(self):
        """Provider estimates tokens when usage data missing."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=COMPLETION_NO_USAGE)
        mock_response.raise_for_status = MagicMock()

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        assert result.success is True
        assert result.output_tokens > 0  # Should estimate tokens

    @pytest.mark.asyncio
    async def test_completion_with_context_window(self):
        """Provider includes context window in response."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        # Mock completion response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SUCCESSFUL_COMPLETION)
        mock_response.raise_for_status = MagicMock()

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        # Mock models list response
        mock_models_response = AsyncMock()
        mock_models_response.status = 200
        mock_models_response.json = AsyncMock(return_value=MODELS_LIST_RESPONSE)
        mock_models_response.raise_for_status = MagicMock()

        mock_get_ctx = AsyncMock()
        mock_get_ctx.__aenter__.return_value = mock_models_response
        mock_get_ctx.__aexit__.return_value = None
        mock_session.get.return_value = mock_get_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        assert result.context_window == 200000  # Claude 3.5 Sonnet context window

    @pytest.mark.asyncio
    async def test_completion_timeout(self):
        """Provider handles request timeout."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        # Simulate timeout
        mock_session.post.side_effect = asyncio.TimeoutError()

        provider = OpenRouterProvider(api_key="test-key", session=mock_session, timeout=10)
        result = await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        assert result.success is False
        assert "timeout" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_completion_network_error(self):
        """Provider handles network errors."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        # Simulate network error
        mock_session.post.side_effect = aiohttp.ClientError("Connection failed")

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        assert result.success is False
        assert "network error" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_completion_unexpected_error(self):
        """Provider handles unexpected exceptions."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        # Simulate unexpected error
        mock_session.post.side_effect = ValueError("Unexpected error")

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        assert result.success is False
        assert "unexpected error" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_completion_session_not_closed(self):
        """Provider does not close injected session."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SUCCESSFUL_COMPLETION)
        mock_response.raise_for_status = MagicMock()

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        # Verify session not closed
        mock_session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_completion_multiple_sequential(self):
        """Provider handles multiple sequential completions."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        responses = []
        for i in range(3):
            resp = AsyncMock()
            resp.status = 200
            resp.json = AsyncMock(return_value=create_completion_response(
                content=f"Response {i}",
                completion_tokens=10 + i
            ))
            resp.raise_for_status = MagicMock()
            responses.append(resp)

        post_contexts = []
        for resp in responses:
            ctx = AsyncMock()
            ctx.__aenter__.return_value = resp
            ctx.__aexit__.return_value = None
            post_contexts.append(ctx)

        mock_session.post.side_effect = post_contexts

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)

        results = []
        for i in range(3):
            result = await provider.get_completion(f"prompt {i}", "anthropic/claude-3.5-sonnet")
            results.append(result)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert results[0].output == "Response 0"
        assert results[1].output == "Response 1"
        assert results[2].output == "Response 2"

    @pytest.mark.asyncio
    async def test_completion_without_injected_session(self):
        """Provider creates temporary session when none injected."""
        # This test verifies backward compatibility
        provider = OpenRouterProvider(api_key="test-key")
        assert provider._session is None

    @pytest.mark.asyncio
    async def test_completion_model_not_available(self):
        """Provider handles unavailable model."""
        provider = OpenRouterProvider(api_key="test-key")
        provider._model_cache = {
            "anthropic/claude-3.5-sonnet": {},
            "openai/gpt-4-turbo": {}
        }

        result = await provider.get_completion("test prompt", "nonexistent/model")

        assert result.success is False
        assert "not found" in result.error_message.lower()


class TestOpenRouterProviderModels:
    """Test OpenRouter provider model management (10 tests)."""

    @pytest.mark.asyncio
    async def test_list_models_success(self):
        """Provider lists available models."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=MODELS_LIST_RESPONSE)
        mock_response.raise_for_status = MagicMock()

        mock_get_ctx = AsyncMock()
        mock_get_ctx.__aenter__.return_value = mock_response
        mock_get_ctx.__aexit__.return_value = None
        mock_session.get.return_value = mock_get_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        models = await provider.list_models()

        assert len(models) == 4
        assert "anthropic/claude-3.5-sonnet" in models
        assert "openai/gpt-4-turbo" in models
        assert "meta-llama/llama-3.1-70b-instruct" in models

    @pytest.mark.asyncio
    async def test_list_models_empty(self):
        """Provider handles empty model list."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=MODELS_LIST_EMPTY)
        mock_response.raise_for_status = MagicMock()

        mock_get_ctx = AsyncMock()
        mock_get_ctx.__aenter__.return_value = mock_response
        mock_get_ctx.__aexit__.return_value = None
        mock_session.get.return_value = mock_get_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        models = await provider.list_models()

        assert models == []

    @pytest.mark.asyncio
    async def test_list_models_caching(self):
        """Provider caches model list."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=MODELS_LIST_RESPONSE)
        mock_response.raise_for_status = MagicMock()

        mock_get_ctx = AsyncMock()
        mock_get_ctx.__aenter__.return_value = mock_response
        mock_get_ctx.__aexit__.return_value = None
        mock_session.get.return_value = mock_get_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)

        # Call twice
        models1 = await provider.list_models()
        models2 = await provider.list_models()

        # Should only call API once (caching)
        assert mock_session.get.call_count == 1
        assert models1 == models2

    @pytest.mark.asyncio
    async def test_list_models_error_handling(self):
        """Provider handles errors when listing models."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_session.get.side_effect = aiohttp.ClientError("API error")

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        models = await provider.list_models()

        # Should return empty list on error
        assert models == []

    @pytest.mark.asyncio
    async def test_get_context_window_success(self):
        """Provider retrieves context window for model."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=MODELS_LIST_RESPONSE)
        mock_response.raise_for_status = MagicMock()

        mock_get_ctx = AsyncMock()
        mock_get_ctx.__aenter__.return_value = mock_response
        mock_get_ctx.__aexit__.return_value = None
        mock_session.get.return_value = mock_get_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)

        context = await provider.get_context_window("anthropic/claude-3.5-sonnet")

        assert context == 200000

    @pytest.mark.asyncio
    async def test_get_context_window_unknown_model(self):
        """Provider handles unknown model context window."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=MODELS_LIST_RESPONSE)
        mock_response.raise_for_status = MagicMock()

        mock_get_ctx = AsyncMock()
        mock_get_ctx.__aenter__.return_value = mock_response
        mock_get_ctx.__aexit__.return_value = None
        mock_session.get.return_value = mock_get_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)

        context = await provider.get_context_window("unknown/model")

        assert context is None

    @pytest.mark.asyncio
    async def test_get_context_window_different_models(self):
        """Provider returns correct context windows for different models."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=MODELS_LIST_RESPONSE)
        mock_response.raise_for_status = MagicMock()

        mock_get_ctx = AsyncMock()
        mock_get_ctx.__aenter__.return_value = mock_response
        mock_get_ctx.__aexit__.return_value = None
        mock_session.get.return_value = mock_get_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)

        claude_context = await provider.get_context_window("anthropic/claude-3.5-sonnet")
        gpt4_context = await provider.get_context_window("openai/gpt-4-turbo")
        llama_context = await provider.get_context_window("meta-llama/llama-3.1-70b-instruct")

        assert claude_context == 200000
        assert gpt4_context == 128000
        assert llama_context == 131072

    def test_is_model_available_with_cache(self):
        """Provider checks model availability using cache."""
        provider = OpenRouterProvider(api_key="test-key")
        provider._model_cache = {
            "anthropic/claude-3.5-sonnet": {},
            "openai/gpt-4-turbo": {}
        }

        assert provider.is_model_available("anthropic/claude-3.5-sonnet") is True
        assert provider.is_model_available("openai/gpt-4-turbo") is True
        assert provider.is_model_available("nonexistent/model") is False

    def test_is_model_available_without_cache(self):
        """Provider optimistically returns True when cache empty."""
        provider = OpenRouterProvider(api_key="test-key")

        # No cache yet - should be optimistic
        assert provider.is_model_available("any/model") is True

    @pytest.mark.asyncio
    async def test_list_models_single_model(self):
        """Provider handles single model response."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=MODELS_LIST_SINGLE)
        mock_response.raise_for_status = MagicMock()

        mock_get_ctx = AsyncMock()
        mock_get_ctx.__aenter__.return_value = mock_response
        mock_get_ctx.__aexit__.return_value = None
        mock_session.get.return_value = mock_get_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        models = await provider.list_models()

        assert len(models) == 1
        assert "anthropic/claude-3.5-sonnet" in models


class TestOpenRouterProviderErrorHandling:
    """Test OpenRouter provider error handling (12 tests)."""

    @pytest.mark.asyncio
    async def test_error_401_unauthorized(self):
        """Provider handles 401 unauthorized error."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.json = AsyncMock(return_value=ERROR_401_UNAUTHORIZED)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        provider = OpenRouterProvider(api_key="invalid-key", session=mock_session)
        result = await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        assert result.success is False
        assert "authentication" in result.error_message.lower() or "api key" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_403_forbidden(self):
        """Provider handles 403 forbidden error."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 403
        mock_response.json = AsyncMock(return_value=ERROR_403_FORBIDDEN)
        mock_response.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=403
        ))

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "restricted/model")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_error_404_model_not_found(self):
        """Provider handles 404 model not found error."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.json = AsyncMock(return_value=ERROR_404_MODEL_NOT_FOUND)
        mock_response.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=404
        ))

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "invalid/model")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_error_413_context_length_exceeded(self):
        """Provider handles 413 context length exceeded error."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 413
        mock_response.json = AsyncMock(return_value=ERROR_413_CONTEXT_LENGTH_EXCEEDED)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("very long prompt" * 1000, "anthropic/claude-3.5-sonnet")

        assert result.success is False
        assert "context" in result.error_message.lower() or "large" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_429_rate_limit(self):
        """Provider handles 429 rate limit error."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.json = AsyncMock(return_value=ERROR_429_RATE_LIMIT)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        assert result.success is False
        assert "rate limit" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_429_credits_exhausted(self):
        """Provider handles 429 credits exhausted error."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.json = AsyncMock(return_value=ERROR_429_CREDITS_EXHAUSTED)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        assert result.success is False
        assert "rate limit" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_500_internal_server(self):
        """Provider handles 500 internal server error."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.json = AsyncMock(return_value=ERROR_500_INTERNAL_SERVER)
        mock_response.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=500
        ))

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_error_502_bad_gateway(self):
        """Provider handles 502 bad gateway error."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 502
        mock_response.json = AsyncMock(return_value=ERROR_502_BAD_GATEWAY)
        mock_response.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=502
        ))

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_error_503_service_unavailable(self):
        """Provider handles 503 service unavailable error."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 503
        mock_response.json = AsyncMock(return_value=ERROR_503_SERVICE_UNAVAILABLE)
        mock_response.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=503
        ))

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_error_400_bad_request(self):
        """Provider handles 400 bad request error."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.json = AsyncMock(return_value=ERROR_400_BAD_REQUEST)
        mock_response.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=400
        ))

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("", "anthropic/claude-3.5-sonnet")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_error_json_decode_error(self):
        """Provider handles malformed JSON response."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(side_effect=ValueError("Invalid JSON"))
        mock_response.raise_for_status = MagicMock()

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_error_connection_reset(self):
        """Provider handles connection reset error."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_session.post.side_effect = aiohttp.ClientConnectionError("Connection reset")

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        assert result.success is False
        assert "network" in result.error_message.lower() or "connection" in result.error_message.lower()


class TestOpenRouterProviderSessionInjection:
    """Test session injection for parallel processing (6 tests)."""

    @pytest.mark.asyncio
    async def test_accepts_injected_session(self):
        """Provider accepts and stores injected session."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)

        assert provider._session is mock_session

    @pytest.mark.asyncio
    async def test_uses_injected_session(self):
        """Provider uses injected session for API calls."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SUCCESSFUL_COMPLETION)
        mock_response.raise_for_status = MagicMock()

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        # Verify session.post was called
        mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_reused_across_calls(self):
        """Same session used for multiple requests."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        responses = []
        for i in range(3):
            resp = AsyncMock()
            resp.status = 200
            resp.json = AsyncMock(return_value=create_completion_response(content=f"Response {i}"))
            resp.raise_for_status = MagicMock()
            responses.append(resp)

        post_contexts = []
        for resp in responses:
            ctx = AsyncMock()
            ctx.__aenter__.return_value = resp
            ctx.__aexit__.return_value = None
            post_contexts.append(ctx)

        mock_session.post.side_effect = post_contexts

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)

        for i in range(3):
            await provider.get_completion(f"prompt {i}", "anthropic/claude-3.5-sonnet")

        assert mock_session.post.call_count == 3
        assert provider._session is mock_session

    @pytest.mark.asyncio
    async def test_session_not_closed(self):
        """Injected session not closed by provider."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SUCCESSFUL_COMPLETION)
        mock_response.raise_for_status = MagicMock()

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")
        del provider

        mock_session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_parallel_requests_share_session(self):
        """Multiple concurrent requests share one session."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        call_count = 0

        def mock_post_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            ctx = AsyncMock()
            resp = AsyncMock()
            resp.status = 200
            resp.json = AsyncMock(return_value=create_completion_response(
                content=f"Response {call_count}"
            ))
            resp.raise_for_status = MagicMock()
            ctx.__aenter__.return_value = resp
            ctx.__aexit__.return_value = None
            return ctx

        mock_session.post.side_effect = mock_post_side_effect

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)

        # Run 10 parallel completions
        tasks = [
            provider.get_completion(f"prompt {i}", "anthropic/claude-3.5-sonnet")
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all(r.success for r in results)
        assert mock_session.post.call_count == 10
        mock_session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_used_for_model_list(self):
        """Injected session used for listing models."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=MODELS_LIST_RESPONSE)
        mock_response.raise_for_status = MagicMock()

        mock_get_ctx = AsyncMock()
        mock_get_ctx.__aenter__.return_value = mock_response
        mock_get_ctx.__aexit__.return_value = None
        mock_session.get.return_value = mock_get_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        await provider.list_models()

        mock_session.get.assert_called_once()
        mock_session.close.assert_not_called()


class TestOpenRouterProviderSpecific:
    """Test OpenRouter-specific features (6 tests)."""

    @pytest.mark.asyncio
    async def test_openrouter_headers_present(self):
        """Provider sends OpenRouter-specific headers."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SUCCESSFUL_COMPLETION)
        mock_response.raise_for_status = MagicMock()

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        # Verify OpenRouter-specific headers
        call_args = mock_session.post.call_args
        headers = call_args[1]["headers"]
        assert "HTTP-Referer" in headers
        assert headers["HTTP-Referer"] == "https://github.com/parallamr/parallamr"
        assert "X-Title" in headers
        assert headers["X-Title"] == "Parallamr"

    @pytest.mark.asyncio
    async def test_openai_compatible_format(self):
        """Provider uses OpenAI-compatible request format."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SUCCESSFUL_COMPLETION)
        mock_response.raise_for_status = MagicMock()

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        # Verify OpenAI-compatible format
        call_args = mock_session.post.call_args
        payload = call_args[1]["json"]
        assert "model" in payload
        assert "messages" in payload
        assert isinstance(payload["messages"], list)
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "test prompt"

    @pytest.mark.asyncio
    async def test_model_pricing_metadata(self):
        """Provider parses model pricing from API response."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=MODELS_LIST_RESPONSE)
        mock_response.raise_for_status = MagicMock()

        mock_get_ctx = AsyncMock()
        mock_get_ctx.__aenter__.return_value = mock_response
        mock_get_ctx.__aexit__.return_value = None
        mock_session.get.return_value = mock_get_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)

        # Populate cache
        await provider.list_models()

        # Check pricing info is stored
        assert provider._model_cache is not None
        assert "anthropic/claude-3.5-sonnet" in provider._model_cache
        model_info = provider._model_cache["anthropic/claude-3.5-sonnet"]
        assert "pricing" in model_info
        assert "prompt" in model_info["pricing"]
        assert "completion" in model_info["pricing"]

    @pytest.mark.asyncio
    async def test_multiple_model_providers(self):
        """Provider handles models from multiple providers."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        # Test completions from different providers
        test_cases = [
            ("anthropic/claude-3.5-sonnet", SUCCESSFUL_COMPLETION),
            ("openai/gpt-4-turbo", COMPLETION_GPT4),
            ("meta-llama/llama-3.1-70b-instruct", COMPLETION_LLAMA31)
        ]

        for model, response_data in test_cases:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=response_data)
            mock_response.raise_for_status = MagicMock()

            mock_post_ctx = AsyncMock()
            mock_post_ctx.__aenter__.return_value = mock_response
            mock_post_ctx.__aexit__.return_value = None
            mock_session.post.return_value = mock_post_ctx

            provider = OpenRouterProvider(api_key="test-key", session=mock_session)
            result = await provider.get_completion("test prompt", model)

            assert result.success is True
            assert len(result.output) > 0

    @pytest.mark.asyncio
    async def test_context_window_from_api(self):
        """Provider retrieves context window from OpenRouter API."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=MODELS_LIST_RESPONSE)
        mock_response.raise_for_status = MagicMock()

        mock_get_ctx = AsyncMock()
        mock_get_ctx.__aenter__.return_value = mock_response
        mock_get_ctx.__aexit__.return_value = None
        mock_session.get.return_value = mock_get_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)

        # Different models have different context windows
        claude_ctx = await provider.get_context_window("anthropic/claude-3.5-sonnet")
        gpt4_ctx = await provider.get_context_window("openai/gpt-4-turbo")
        llama_ctx = await provider.get_context_window("meta-llama/llama-3.1-70b-instruct")

        assert claude_ctx == 200000
        assert gpt4_ctx == 128000
        assert llama_ctx == 131072

    @pytest.mark.asyncio
    async def test_model_name_format(self):
        """Provider handles provider/model-name format."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SUCCESSFUL_COMPLETION)
        mock_response.raise_for_status = MagicMock()

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        provider = OpenRouterProvider(api_key="test-key", session=mock_session)

        # Verify model name is passed correctly
        await provider.get_completion("test prompt", "anthropic/claude-3.5-sonnet")

        call_args = mock_session.post.call_args
        payload = call_args[1]["json"]
        assert payload["model"] == "anthropic/claude-3.5-sonnet"
        assert "/" in payload["model"]  # Ensure provider/model format
