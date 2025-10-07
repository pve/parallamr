"""Comprehensive tests for OpenAI provider implementation."""

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
from fixtures.openai_responses import (
    AZURE_COMPLETION_RESPONSE,
    COMPLETION_MULTIPLE_CHOICES,
    COMPLETION_NO_STREAM,
    COMPLETION_NO_USAGE,
    CONTEXT_WINDOWS,
    ERROR_400_BAD_REQUEST,
    ERROR_401_UNAUTHORIZED,
    ERROR_403_FORBIDDEN,
    ERROR_404_NOT_FOUND,
    ERROR_413_PAYLOAD_TOO_LARGE,
    ERROR_429_RATE_LIMIT,
    ERROR_500_INTERNAL_SERVER,
    ERROR_502_BAD_GATEWAY,
    ERROR_503_SERVICE_UNAVAILABLE,
    LOCALAI_COMPLETION_RESPONSE,
    MODELS_LIST_EMPTY,
    MODELS_LIST_RESPONSE,
    SUCCESSFUL_COMPLETION,
    TOGETHER_COMPLETION_RESPONSE,
    create_completion_response,
    create_error_response,
    create_models_list_response,
)

# Import the OpenAI provider
from parallamr.providers.openai import OpenAIProvider


class TestOpenAIProviderInit:
    """Test OpenAI provider initialization (10 tests)."""

    def test_init_with_api_key(self):
        """Provider accepts API key directly."""
        provider = OpenAIProvider(api_key="test-api-key-123")
        assert provider.api_key == "test-api-key-123"

    def test_init_with_env_getter(self):
        """Provider accepts custom env_getter for API key."""
        def mock_env_getter(key: str) -> Optional[str]:
            if key == "OPENAI_API_KEY":
                return "env-test-key-456"
            return None

        provider = OpenAIProvider(env_getter=mock_env_getter)
        assert provider.api_key == "env-test-key-456"

    def test_init_without_api_key(self, monkeypatch):
        """Provider handles missing API key."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = OpenAIProvider()
        assert provider.api_key is None

    def test_init_with_custom_base_url(self):
        """Provider accepts custom base URL."""
        provider = OpenAIProvider(
            api_key="test-key",
            base_url="https://custom-openai.example.com/v1"
        )
        assert provider.base_url == "https://custom-openai.example.com/v1"

    def test_init_default_base_url(self):
        """Provider uses default OpenAI base URL."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider.base_url == "https://api.openai.com/v1"

    def test_init_with_timeout(self):
        """Provider accepts custom timeout."""
        provider = OpenAIProvider(api_key="test-key", timeout=600)
        assert provider.timeout == 600

    def test_init_default_timeout(self):
        """Provider uses default timeout."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider.timeout == 300

    def test_init_with_session(self):
        """Provider accepts injected aiohttp session."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        assert provider._session is mock_session

    def test_init_without_session(self):
        """Provider initializes with None session by default."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider._session is None

    def test_init_model_cache_empty(self):
        """Provider initializes with empty model cache."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider._model_cache is None


class TestOpenAIProviderCompletion:
    """Test OpenAI provider completion requests (15 tests)."""

    @pytest.mark.asyncio
    async def test_successful_completion(self):
        """Provider returns successful completion response."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SUCCESSFUL_COMPLETION)

        # Setup context manager
        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "gpt-4")

        assert result.success is True
        assert result.output == "This is a test response from GPT-4."
        assert result.output_tokens == 20

    @pytest.mark.asyncio
    async def test_completion_without_api_key(self):
        """Provider handles missing API key gracefully."""
        provider = OpenAIProvider(api_key=None)
        result = await provider.get_completion("test prompt", "gpt-4")

        assert result.success is False
        assert "API key" in result.error_message

    @pytest.mark.asyncio
    async def test_completion_with_kwargs(self):
        """Provider passes additional kwargs to API."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SUCCESSFUL_COMPLETION)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion(
            "test prompt",
            "gpt-4",
            temperature=0.7,
            max_tokens=100
        )

        # Verify kwargs were passed
        call_args = mock_session.request.call_args
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

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        await provider.get_completion("test prompt", "gpt-4")

        # Verify request format
        call_args = mock_session.request.call_args
        assert "https://api.openai.com/v1/chat/completions" in str(call_args)
        assert call_args[1]["headers"]["Authorization"] == "Bearer test-key"
        assert call_args[1]["headers"]["Content-Type"] == "application/json"

        payload = call_args[1]["json"]
        assert payload["model"] == "gpt-4"
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "test prompt"

    @pytest.mark.asyncio
    async def test_completion_extracts_tokens(self):
        """Provider extracts token counts from response."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SUCCESSFUL_COMPLETION)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "gpt-4")

        assert result.output_tokens == 20

    @pytest.mark.asyncio
    async def test_completion_without_usage_data(self):
        """Provider estimates tokens when usage data missing."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=COMPLETION_NO_USAGE)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "gpt-4")

        assert result.success is True
        assert result.output_tokens > 0  # Should estimate tokens

    @pytest.mark.asyncio
    async def test_completion_with_context_window(self):
        """Provider includes context window in response."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SUCCESSFUL_COMPLETION)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "gpt-4")

        assert result.context_window == 8192  # GPT-4 context window

    @pytest.mark.asyncio
    async def test_completion_timeout(self):
        """Provider handles request timeout."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        # Simulate timeout
        mock_session.request.side_effect = asyncio.TimeoutError()

        provider = OpenAIProvider(api_key="test-key", session=mock_session, timeout=10)
        result = await provider.get_completion("test prompt", "gpt-4")

        assert result.success is False
        assert "timeout" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_completion_network_error(self):
        """Provider handles network errors."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        # Simulate network error
        mock_session.request.side_effect = aiohttp.ClientError("Connection failed")

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "gpt-4")

        assert result.success is False
        assert "network error" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_completion_unexpected_error(self):
        """Provider handles unexpected exceptions."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        # Simulate unexpected error
        mock_session.request.side_effect = ValueError("Unexpected error")

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "gpt-4")

        assert result.success is False
        assert "unexpected error" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_completion_session_not_closed(self):
        """Provider does not close injected session."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SUCCESSFUL_COMPLETION)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        await provider.get_completion("test prompt", "gpt-4")

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
            responses.append(resp)

        post_contexts = []
        for resp in responses:
            ctx = AsyncMock()
            ctx.__aenter__.return_value = resp
            ctx.__aexit__.return_value = None
            post_contexts.append(ctx)

        mock_session.request.side_effect = post_contexts

        provider = OpenAIProvider(api_key="test-key", session=mock_session)

        results = []
        for i in range(3):
            result = await provider.get_completion(f"prompt {i}", "gpt-4")
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
        provider = OpenAIProvider(api_key="test-key")
        assert provider._session is None

        # Note: Actual API call would require mocking at aiohttp.ClientSession level
        # This test just verifies the session is None initially

    @pytest.mark.asyncio
    async def test_completion_model_not_available(self):
        """Provider handles unavailable model."""
        provider = OpenAIProvider(api_key="test-key")
        provider._model_cache = {"gpt-4": {}, "gpt-3.5-turbo": {}}

        result = await provider.get_completion("test prompt", "nonexistent-model")

        assert result.success is False
        assert "not found" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_completion_custom_parameters(self):
        """Provider handles custom API parameters."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SUCCESSFUL_COMPLETION)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion(
            "test prompt",
            "gpt-4",
            temperature=0.5,
            top_p=0.9,
            presence_penalty=0.1,
            frequency_penalty=0.2
        )

        call_args = mock_session.request.call_args
        payload = call_args[1]["json"]
        assert payload["temperature"] == 0.5
        assert payload["top_p"] == 0.9
        assert payload["presence_penalty"] == 0.1
        assert payload["frequency_penalty"] == 0.2


class TestOpenAIProviderModels:
    """Test OpenAI provider model management (10 tests)."""

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

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        models = await provider.list_models()

        assert len(models) == 4
        assert "gpt-4" in models
        assert "gpt-3.5-turbo" in models

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

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
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

        provider = OpenAIProvider(api_key="test-key", session=mock_session)

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

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        models = await provider.list_models()

        # Should return empty list on error
        assert models == []

    @pytest.mark.asyncio
    async def test_get_context_window_success(self):
        """Provider retrieves context window for model."""
        provider = OpenAIProvider(api_key="test-key")

        context = await provider.get_context_window("gpt-4")

        assert context == 8192

    @pytest.mark.asyncio
    async def test_get_context_window_unknown_model(self):
        """Provider handles unknown model context window."""
        provider = OpenAIProvider(api_key="test-key")

        context = await provider.get_context_window("unknown-model")

        assert context is None

    @pytest.mark.asyncio
    async def test_get_context_window_different_models(self):
        """Provider returns correct context windows for different models."""
        provider = OpenAIProvider(api_key="test-key")

        gpt4_context = await provider.get_context_window("gpt-4")
        gpt35_context = await provider.get_context_window("gpt-3.5-turbo")
        gpt35_16k_context = await provider.get_context_window("gpt-3.5-turbo-16k")

        assert gpt4_context == 8192
        assert gpt35_context == 4096
        assert gpt35_16k_context == 16384

    def test_is_model_available_with_cache(self):
        """Provider checks model availability using cache."""
        provider = OpenAIProvider(api_key="test-key")
        provider._model_cache = {
            "gpt-4": {},
            "gpt-3.5-turbo": {}
        }

        assert provider.is_model_available("gpt-4") is True
        assert provider.is_model_available("gpt-3.5-turbo") is True
        assert provider.is_model_available("nonexistent") is False

    def test_is_model_available_without_cache(self):
        """Provider optimistically returns True when cache empty."""
        provider = OpenAIProvider(api_key="test-key")

        # No cache yet - should be optimistic
        assert provider.is_model_available("any-model") is True

    def test_get_provider_name(self):
        """Provider returns correct provider name."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider.get_provider_name() == "openai"


class TestOpenAIProviderErrorHandling:
    """Test OpenAI provider error handling (12 tests)."""

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
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(api_key="invalid-key", session=mock_session)
        result = await provider.get_completion("test prompt", "gpt-4")

        assert result.success is False
        assert "authentication" in result.error_message.lower() or "api key" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_403_forbidden(self):
        """Provider handles 403 forbidden error."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 403
        mock_response.json = AsyncMock(return_value=ERROR_403_FORBIDDEN)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "restricted-model")

        assert result.success is False
        assert "access" in result.error_message.lower() or "forbidden" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_404_not_found(self):
        """Provider handles 404 model not found error."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.json = AsyncMock(return_value=ERROR_404_NOT_FOUND)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "invalid-model")

        assert result.success is False
        assert "not found" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_413_payload_too_large(self):
        """Provider handles 413 payload too large error."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 413
        mock_response.json = AsyncMock(return_value=ERROR_413_PAYLOAD_TOO_LARGE)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("very long prompt" * 1000, "gpt-4")

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
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "gpt-4")

        assert result.success is False
        assert "rate limit" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_500_internal_server(self):
        """Provider handles 500 internal server error."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.json = AsyncMock(return_value=ERROR_500_INTERNAL_SERVER)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "gpt-4")

        assert result.success is False
        assert "server" in result.error_message.lower() or "error" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_502_bad_gateway(self):
        """Provider handles 502 bad gateway error."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 502
        mock_response.json = AsyncMock(return_value=ERROR_502_BAD_GATEWAY)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "gpt-4")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_error_503_service_unavailable(self):
        """Provider handles 503 service unavailable error."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 503
        mock_response.json = AsyncMock(return_value=ERROR_503_SERVICE_UNAVAILABLE)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "gpt-4")

        assert result.success is False
        assert "unavailable" in result.error_message.lower() or "server" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_400_bad_request(self):
        """Provider handles 400 bad request error."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.json = AsyncMock(return_value=ERROR_400_BAD_REQUEST)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("", "gpt-4")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_error_json_decode_error(self):
        """Provider handles malformed JSON response."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(side_effect=ValueError("Invalid JSON"))

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "gpt-4")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_error_missing_choices(self):
        """Provider handles response with missing choices."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"id": "test", "object": "chat.completion"})

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "gpt-4")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_error_connection_reset(self):
        """Provider handles connection reset error."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_session.request.side_effect = aiohttp.ClientConnectionError("Connection reset")

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        result = await provider.get_completion("test prompt", "gpt-4")

        assert result.success is False
        assert "network" in result.error_message.lower() or "connection" in result.error_message.lower()


class TestOpenAIProviderSessionInjection:
    """Test session injection for parallel processing (8 tests)."""

    @pytest.mark.asyncio
    async def test_accepts_injected_session(self):
        """Provider accepts and stores injected session."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        provider = OpenAIProvider(api_key="test-key", session=mock_session)

        assert provider._session is mock_session

    @pytest.mark.asyncio
    async def test_uses_injected_session(self):
        """Provider uses injected session for API calls."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SUCCESSFUL_COMPLETION)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        await provider.get_completion("test prompt", "gpt-4")

        # Verify session.request was called
        mock_session.request.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_reused_across_calls(self):
        """Same session used for multiple requests."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        responses = []
        for i in range(3):
            resp = AsyncMock()
            resp.status = 200
            resp.json = AsyncMock(return_value=create_completion_response(content=f"Response {i}"))
            responses.append(resp)

        post_contexts = []
        for resp in responses:
            ctx = AsyncMock()
            ctx.__aenter__.return_value = resp
            ctx.__aexit__.return_value = None
            post_contexts.append(ctx)

        mock_session.request.side_effect = post_contexts

        provider = OpenAIProvider(api_key="test-key", session=mock_session)

        for i in range(3):
            await provider.get_completion(f"prompt {i}", "gpt-4")

        assert mock_session.request.call_count == 3
        assert provider._session is mock_session

    @pytest.mark.asyncio
    async def test_session_not_closed(self):
        """Injected session not closed by provider."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SUCCESSFUL_COMPLETION)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        await provider.get_completion("test prompt", "gpt-4")
        del provider

        mock_session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_parallel_requests_share_session(self):
        """Multiple concurrent requests share one session."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        call_count = 0

        def mock_request_side_effect(method, url, **kwargs):
            nonlocal call_count
            call_count += 1

            ctx = AsyncMock()
            resp = AsyncMock()
            resp.status = 200
            resp.json = AsyncMock(return_value=create_completion_response(
                content=f"Response {call_count}"
            ))
            ctx.__aenter__.return_value = resp
            ctx.__aexit__.return_value = None
            return ctx

        mock_session.request.side_effect = mock_request_side_effect

        provider = OpenAIProvider(api_key="test-key", session=mock_session)

        # Run 10 parallel completions
        tasks = [
            provider.get_completion(f"prompt {i}", "gpt-4")
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all(r.success for r in results)
        assert mock_session.request.call_count == 10
        mock_session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_survives_provider_deletion(self):
        """Session lifetime independent of provider."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SUCCESSFUL_COMPLETION)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        provider1 = OpenAIProvider(api_key="test-key", session=mock_session)
        await provider1.get_completion("test", "gpt-4")
        del provider1

        provider2 = OpenAIProvider(api_key="test-key", session=mock_session)
        await provider2.get_completion("test", "gpt-4")
        del provider2

        mock_session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_backward_compatibility_no_session(self):
        """Provider works without session injection."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider._session is None

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

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        await provider.list_models()

        mock_session.get.assert_called_once()
        mock_session.close.assert_not_called()


class TestOpenAIProviderCompatibility:
    """Test compatibility with OpenAI-compatible APIs (8 tests)."""

    @pytest.mark.asyncio
    async def test_azure_openai_compatibility(self):
        """Provider works with Azure OpenAI endpoints."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=AZURE_COMPLETION_RESPONSE)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(
            api_key="azure-key",
            base_url="https://myresource.openai.azure.com/openai/deployments/gpt-4",
            session=mock_session
        )
        result = await provider.get_completion("test prompt", "gpt-4")

        assert result.success is True
        assert result.output == "Response from Azure OpenAI"

    @pytest.mark.asyncio
    async def test_localai_compatibility(self):
        """Provider works with LocalAI."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=LOCALAI_COMPLETION_RESPONSE)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(
            api_key="localai-key",
            base_url="http://localhost:8080/v1",
            session=mock_session
        )
        result = await provider.get_completion("test prompt", "llama-2-7b")

        assert result.success is True
        assert result.output == "Response from LocalAI"

    @pytest.mark.asyncio
    async def test_together_ai_compatibility(self):
        """Provider works with Together AI."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=TOGETHER_COMPLETION_RESPONSE)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(
            api_key="together-key",
            base_url="https://api.together.xyz/v1",
            session=mock_session
        )
        result = await provider.get_completion("test prompt", "mistralai/Mixtral-8x7B-Instruct-v0.1")

        assert result.success is True
        assert result.output == "Response from Together AI"

    @pytest.mark.asyncio
    async def test_custom_base_url_used(self):
        """Provider uses custom base URL for requests."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SUCCESSFUL_COMPLETION)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        custom_url = "https://custom-api.example.com/v1"
        provider = OpenAIProvider(
            api_key="test-key",
            base_url=custom_url,
            session=mock_session
        )
        await provider.get_completion("test prompt", "gpt-4")

        # Verify custom URL was used
        call_args = mock_session.request.call_args
        assert custom_url in call_args[0][0]

    @pytest.mark.asyncio
    async def test_openai_compatible_headers(self):
        """Provider sends compatible headers."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SUCCESSFUL_COMPLETION)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        await provider.get_completion("test prompt", "gpt-4")

        # Verify headers
        call_args = mock_session.request.call_args
        headers = call_args[1]["headers"]
        assert "Authorization" in headers
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_openai_compatible_request_format(self):
        """Provider sends OpenAI-compatible request format."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SUCCESSFUL_COMPLETION)

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.request.return_value = mock_post_ctx

        provider = OpenAIProvider(api_key="test-key", session=mock_session)
        await provider.get_completion("test prompt", "gpt-4")

        # Verify request format
        call_args = mock_session.request.call_args
        payload = call_args[1]["json"]
        assert "model" in payload
        assert "messages" in payload
        assert isinstance(payload["messages"], list)
        assert payload["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_openai_compatible_response_parsing(self):
        """Provider parses OpenAI-compatible responses."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        # Test with various compatible response formats
        compatible_responses = [
            SUCCESSFUL_COMPLETION,
            AZURE_COMPLETION_RESPONSE,
            LOCALAI_COMPLETION_RESPONSE,
            TOGETHER_COMPLETION_RESPONSE
        ]

        for resp_data in compatible_responses:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=resp_data)

            mock_post_ctx = AsyncMock()
            mock_post_ctx.__aenter__.return_value = mock_response
            mock_post_ctx.__aexit__.return_value = None
            mock_session.post.return_value = mock_post_ctx

            provider = OpenAIProvider(api_key="test-key", session=mock_session)
            result = await provider.get_completion("test prompt", "gpt-4")

            assert result.success is True
            assert len(result.output) > 0
            assert result.output_tokens > 0

    @pytest.mark.asyncio
    async def test_env_var_compatibility(self):
        """Provider reads OPENAI_API_KEY environment variable."""
        def mock_env_getter(key: str) -> Optional[str]:
            if key == "OPENAI_API_KEY":
                return "env-api-key-123"
            return None

        provider = OpenAIProvider(env_getter=mock_env_getter)
        assert provider.api_key == "env-api-key-123"


class TestOpenAIProviderIntegration:
    """Integration tests for end-to-end workflows (2 tests)."""

    @pytest.mark.skip(reason="Requires actual OpenAI API key")
    @pytest.mark.asyncio
    async def test_real_api_integration(self):
        """Test with real OpenAI API (skipped by default)."""
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        provider = OpenAIProvider(api_key=api_key)
        result = await provider.get_completion("Say hello", "gpt-3.5-turbo")

        assert result.success is True
        assert len(result.output) > 0

    @pytest.mark.skip(reason="Requires actual OpenAI API key")
    @pytest.mark.asyncio
    async def test_real_api_parallel_processing(self):
        """Test parallel processing with real API (skipped by default)."""
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        async with aiohttp.ClientSession() as session:
            provider = OpenAIProvider(api_key=api_key, session=session)

            tasks = [
                provider.get_completion(f"Count to {i}", "gpt-3.5-turbo")
                for i in range(5)
            ]
            results = await asyncio.gather(*tasks)

            assert len(results) == 5
            assert all(r.success for r in results)
