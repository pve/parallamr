"""Comprehensive tests for Ollama provider implementation."""

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from parallamr.models import ProviderResponse
from parallamr.providers.base import (
    ModelNotAvailableError,
    Provider,
    ProviderError,
    TimeoutError,
)
from tests.fixtures.ollama_responses import (
    COMPLETION_CODELLAMA,
    COMPLETION_EMPTY_RESPONSE,
    COMPLETION_EXTRA_FIELDS,
    COMPLETION_MALFORMED_TIMESTAMP,
    COMPLETION_MINIMAL,
    COMPLETION_MISTRAL,
    COMPLETION_MISSING_DONE,
    COMPLETION_NEGATIVE_TIMING,
    COMPLETION_NO_TIMING,
    CONTEXT_WINDOWS,
    ERROR_400_CONTEXT_EXCEEDED,
    ERROR_400_INVALID_REQUEST,
    ERROR_404_MODEL_NOT_FOUND,
    ERROR_500_MODEL_NOT_LOADED,
    ERROR_500_OUT_OF_MEMORY,
    ERROR_502_BAD_GATEWAY,
    ERROR_503_SERVICE_UNAVAILABLE,
    MODEL_INFO_CODELLAMA,
    MODEL_INFO_LLAMA31,
    MODEL_INFO_MISTRAL,
    MODEL_INFO_NO_CONTEXT,
    MODELS_LIST_EMPTY,
    MODELS_LIST_MALFORMED,
    MODELS_LIST_RESPONSE,
    MODELS_LIST_SINGLE,
    SUCCESSFUL_COMPLETION,
    create_completion_response,
    create_error_response,
    create_model_show_response,
    create_models_list,
)

# Import the Ollama provider
from parallamr.providers.ollama import OllamaProvider
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


class TestOllamaProviderInit:
    """Test Ollama provider initialization (10 tests)."""

    def test_init_with_base_url(self):
        """Provider accepts custom base URL directly."""
        provider = OllamaProvider(base_url="http://custom-ollama:8080")
        assert provider.base_url == "http://custom-ollama:8080"

    def test_init_with_env_getter(self):
        """Provider accepts custom env_getter for base URL."""
        def mock_env_getter(key: str, default: str = "") -> Optional[str]:
            if key == "OLLAMA_BASE_URL":
                return "http://env-ollama:11434"
            return default

        provider = OllamaProvider(env_getter=mock_env_getter)
        assert provider.base_url == "http://env-ollama:11434"

    def test_init_default_base_url(self):
        """Provider uses default localhost URL."""
        provider = OllamaProvider()
        assert provider.base_url == "http://localhost:11434"

    def test_init_with_timeout(self):
        """Provider accepts custom timeout."""
        provider = OllamaProvider(timeout=600)
        assert provider.timeout == 600

    def test_init_default_timeout(self):
        """Provider uses default timeout."""
        provider = OllamaProvider()
        assert provider.timeout == 300

    def test_init_with_session(self):
        """Provider accepts injected aiohttp session."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        provider = OllamaProvider(session=mock_session)
        assert provider._session is mock_session

    def test_init_without_session(self):
        """Provider initializes with None session by default."""
        provider = OllamaProvider()
        assert provider._session is None

    def test_init_model_cache_empty(self):
        """Provider initializes with empty model cache."""
        provider = OllamaProvider()
        assert provider._model_cache is None

    def test_init_env_getter_with_default(self):
        """Provider env_getter handles default value correctly."""
        def mock_env_getter(key: str, default: str = "") -> Optional[str]:
            return default

        provider = OllamaProvider(env_getter=mock_env_getter)
        assert provider.base_url == "http://localhost:11434"

    def test_init_invalid_url(self):
        """Provider handles invalid base URL."""
        provider = OllamaProvider(base_url="ftp://invalid")
        result, error = provider._validate_configuration()
        assert result is False
        assert "Invalid Ollama base URL" in error


class TestOllamaProviderCompletion:
    """Test Ollama provider completion requests (15 tests)."""

    @pytest.mark.asyncio
    async def test_successful_completion(self, mock_session):
        """Provider returns successful completion response."""
        setup_mock_sequential_responses(mock_session, [
            (200, SUCCESSFUL_COMPLETION),
            (200, MODEL_INFO_LLAMA31)
        ])
        provider = OllamaProvider(session=mock_session)

        result = await provider.get_completion("test prompt", "llama3.1:latest")

        assert_provider_response_valid(result, success=True)
        assert result.output == "This is a test response from Llama 3.1."
        assert result.context_window == 131072

    @pytest.mark.asyncio
    async def test_completion_with_kwargs(self, mock_session):
        """Provider passes additional kwargs to API."""
        setup_mock_sequential_responses(mock_session, [
            (200, SUCCESSFUL_COMPLETION),
            (200, MODEL_INFO_LLAMA31)
        ])
        provider = OllamaProvider(session=mock_session)

        result = await provider.get_completion(
            "test prompt",
            "llama3.1:latest",
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
            (200, MODEL_INFO_LLAMA31)
        ])
        provider = OllamaProvider(base_url="http://test:11434", session=mock_session)

        await provider.get_completion("test prompt", "llama3.1:latest")

        call_args = mock_session.request.call_args_list[0]
        assert call_args[0][1] == "http://test:11434/api/generate"
        payload = call_args[1]["json"]
        assert payload["model"] == "llama3.1:latest"
        assert payload["prompt"] == "test prompt"
        assert payload["stream"] is False

    @pytest.mark.asyncio
    async def test_completion_estimates_tokens(self, mock_session):
        """Provider estimates token counts when not provided."""
        setup_mock_sequential_responses(mock_session, [
            (200, SUCCESSFUL_COMPLETION),
            (200, MODEL_INFO_LLAMA31)
        ])
        provider = OllamaProvider(session=mock_session)

        result = await provider.get_completion("test prompt", "llama3.1:latest")
        assert result.output_tokens > 0

    @pytest.mark.asyncio
    async def test_completion_timeout(self, mock_session):
        """Provider handles request timeout."""
        setup_mock_error(mock_session, asyncio.TimeoutError())
        provider = OllamaProvider(session=mock_session, timeout=10)

        result = await provider.get_completion("test prompt", "llama3.1:latest")
        assert result.success is False
        assert "timeout" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_completion_connection_error(self, mock_session):
        """Provider handles connection errors."""
        setup_mock_error(mock_session, aiohttp.ClientConnectorError(None, OSError("Connection refused")))
        provider = OllamaProvider(session=mock_session)

        result = await provider.get_completion("test prompt", "llama3.1:latest")
        assert result.success is False
        assert "connect" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_completion_network_error(self, mock_session):
        """Provider handles network errors."""
        setup_mock_error(mock_session, aiohttp.ClientError("Network failure"))
        provider = OllamaProvider(session=mock_session)

        result = await provider.get_completion("test prompt", "llama3.1:latest")
        assert result.success is False
        assert "network error" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_completion_unexpected_error(self, mock_session):
        """Provider handles unexpected exceptions."""
        setup_mock_error(mock_session, ValueError("Unexpected error"))
        provider = OllamaProvider(session=mock_session)

        result = await provider.get_completion("test prompt", "llama3.1:latest")
        assert result.success is False
        assert "unexpected error" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_completion_session_not_closed(self, mock_session):
        """Provider does not close injected session."""
        setup_mock_sequential_responses(mock_session, [
            (200, SUCCESSFUL_COMPLETION),
            (200, MODEL_INFO_LLAMA31)
        ])
        provider = OllamaProvider(session=mock_session)

        await provider.get_completion("test prompt", "llama3.1:latest")
        assert_session_not_closed(mock_session)

    @pytest.mark.asyncio
    async def test_completion_model_not_available(self, mock_session):
        """Provider handles unavailable model."""
        provider = OllamaProvider(session=mock_session)
        provider._model_cache = ["available-model"]

        result = await provider.get_completion("test prompt", "nonexistent-model")
        assert result.success is False
        assert "not found" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_completion_malformed_json(self, mock_session):
        """Provider handles malformed JSON response."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.text.return_value = "<html>Not JSON</html>"
        setup_mock_post(mock_session, 200, {}) # Just to set up context
        mock_session.request.return_value.__aenter__.return_value = mock_response

        provider = OllamaProvider(session=mock_session)
        result = await provider.get_completion("test", "llama3.1")
        assert result.success is False
        assert "Invalid JSON" in result.error_message

    @pytest.mark.asyncio
    async def test_completion_empty_response(self, mock_session):
        """Provider handles empty completion response."""
        setup_mock_sequential_responses(mock_session, [
            (200, COMPLETION_EMPTY_RESPONSE),
            (200, MODEL_INFO_LLAMA31)
        ])
        provider = OllamaProvider(session=mock_session)
        result = await provider.get_completion("test", "llama3.1")
        assert result.success is True
        assert result.output == ""

    @pytest.mark.asyncio
    async def test_completion_with_model_tags(self, mock_session):
        """Provider handles model tags correctly."""
        setup_mock_sequential_responses(mock_session, [
            (200, COMPLETION_CODELLAMA),
            (200, MODEL_INFO_CODELLAMA)
        ])
        provider = OllamaProvider(session=mock_session)
        result = await provider.get_completion("test", "codellama:13b")
        assert result.success is True
        assert "def hello_world" in result.output

    @pytest.mark.asyncio
    async def test_completion_minimal_response(self, mock_session):
        """Provider handles minimal response format."""
        setup_mock_sequential_responses(mock_session, [
            (200, COMPLETION_MINIMAL),
            (200, MODEL_INFO_LLAMA31)
        ])
        provider = OllamaProvider(session=mock_session)
        result = await provider.get_completion("test", "llama3.1")
        assert result.success is True
        assert result.output == "Short reply."

    @pytest.mark.asyncio
    async def test_completion_with_variables(self, mock_session):
        """Provider passes variables (not standard for Ollama but for compatibility)."""
        setup_mock_sequential_responses(mock_session, [
            (200, SUCCESSFUL_COMPLETION),
            (200, MODEL_INFO_LLAMA31)
        ])
        provider = OllamaProvider(session=mock_session)
        result = await provider.get_completion("test", "llama3.1", variables={"a": 1})
        assert result.success is True


class TestOllamaProviderModels:
    """Test Ollama provider model management (10 tests)."""

    @pytest.mark.asyncio
    async def test_list_models_success(self, mock_session):
        """Provider lists available models."""
        setup_mock_get(mock_session, 200, MODELS_LIST_RESPONSE)
        provider = OllamaProvider(session=mock_session)

        models = await provider.list_models()
        assert len(models) == 4
        assert "llama3.1:latest" in models

    @pytest.mark.asyncio
    async def test_list_models_caching(self, mock_session):
        """Provider caches model list."""
        setup_mock_get(mock_session, 200, MODELS_LIST_RESPONSE)
        provider = OllamaProvider(session=mock_session)

        await provider.list_models()
        await provider.list_models()
        assert mock_session.request.call_count == 1

    @pytest.mark.asyncio
    async def test_list_models_empty(self, mock_session):
        """Provider handles empty model list."""
        setup_mock_get(mock_session, 200, MODELS_LIST_EMPTY)
        provider = OllamaProvider(session=mock_session)
        models = await provider.list_models()
        assert models == []

    @pytest.mark.asyncio
    async def test_list_models_error(self, mock_session):
        """Provider handles error when listing models."""
        setup_mock_error(mock_session, aiohttp.ClientError("API Error"))
        provider = OllamaProvider(session=mock_session)
        models = await provider.list_models()
        assert models == []

    def test_is_model_available_with_cache(self):
        """Provider checks model availability using cache."""
        provider = OllamaProvider()
        provider._model_cache = ["llama3.1:latest"]
        assert provider.is_model_available("llama3.1:latest") is True
        assert provider.is_model_available("nonexistent") is False

    def test_is_model_available_optimistic(self):
        """Provider is optimistic when cache is empty."""
        provider = OllamaProvider()
        assert provider.is_model_available("anything") is True

    @pytest.mark.asyncio
    async def test_get_context_window_success(self, mock_session):
        """Provider retrieves context window."""
        setup_mock_post(mock_session, 200, MODEL_INFO_LLAMA31)
        provider = OllamaProvider(session=mock_session)
        context = await provider.get_context_window("llama3.1")
        assert context == 131072

    @pytest.mark.asyncio
    async def test_get_context_window_mistral(self, mock_session):
        """Provider retrieves context window for Mistral."""
        setup_mock_post(mock_session, 200, MODEL_INFO_MISTRAL)
        provider = OllamaProvider(session=mock_session)
        context = await provider.get_context_window("mistral")
        assert context == 8192

    @pytest.mark.asyncio
    async def test_get_context_window_not_found(self, mock_session):
        """Provider handles 404 for context window."""
        setup_mock_post(mock_session, 404, {"error": "not found"})
        provider = OllamaProvider(session=mock_session)
        context = await provider.get_context_window("unknown")
        assert context is None

    def test_get_provider_name(self):
        """Provider returns correct provider name."""
        provider = OllamaProvider()
        assert provider.get_provider_name() == "ollama"


class TestOllamaProviderErrorHandling:
    """Test Ollama provider error handling (10 tests)."""

    @pytest.mark.asyncio
    async def test_error_404_model_not_found(self, mock_session):
        """Provider handles 404 model not found error."""
        setup_mock_post(mock_session, 404, {"error": "model not found"})
        provider = OllamaProvider(session=mock_session)

        result = await provider.get_completion("test prompt", "llama3.1:latest")
        assert result.success is False
        assert "not found" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_500_server_error(self, mock_session):
        """Provider handles 500 server error."""
        setup_mock_post(mock_session, 500, {"error": "internal error"})
        provider = OllamaProvider(session=mock_session)

        result = await provider.get_completion("test prompt", "llama3.1:latest")
        assert result.success is False
        assert "error" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_400_invalid_request(self, mock_session):
        """Provider handles 400 invalid request."""
        setup_mock_post(mock_session, 400, {"error": "invalid format"})
        provider = OllamaProvider(session=mock_session)
        result = await provider.get_completion("test", "llama3.1")
        assert result.success is False
        assert "Invalid request" in result.error_message

    @pytest.mark.asyncio
    async def test_error_with_message_field(self, mock_session):
        """Provider handles errors using 'message' field."""
        setup_mock_post(mock_session, 400, {"message": "Custom error message"})
        provider = OllamaProvider(session=mock_session)
        result = await provider.get_completion("test", "llama3.1")
        assert "Custom error message" in result.error_message

    @pytest.mark.asyncio
    async def test_error_network_timeout(self, mock_session):
        """Provider handles network timeout."""
        setup_mock_error(mock_session, asyncio.TimeoutError())
        provider = OllamaProvider(session=mock_session)
        result = await provider.get_completion("test", "llama3.1")
        assert "timeout" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_connection_refused(self, mock_session):
        """Provider handles connection refused."""
        setup_mock_error(mock_session, aiohttp.ClientConnectorError(None, OSError()))
        provider = OllamaProvider(session=mock_session)
        result = await provider.get_completion("test", "llama3.1")
        assert "connect" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_client_error(self, mock_session):
        """Provider handles generic client error."""
        setup_mock_error(mock_session, aiohttp.ClientError("generic error"))
        provider = OllamaProvider(session=mock_session)
        result = await provider.get_completion("test", "llama3.1")
        assert "network error" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_malformed_response_no_json(self, mock_session):
        """Provider handles non-json response body on error."""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.json.side_effect = ValueError()
        mock_response.text.return_value = "Internal Server Error"
        setup_mock_post(mock_session, 500, {})
        mock_session.request.return_value.__aenter__.return_value = mock_response

        provider = OllamaProvider(session=mock_session)
        result = await provider.get_completion("test", "llama3.1")
        assert "Internal Server Error" in result.error_message

    @pytest.mark.asyncio
    async def test_error_empty_error_response(self, mock_session):
        """Provider handles error status with empty body."""
        mock_response = AsyncMock()
        mock_response.status = 502
        mock_response.json.side_effect = ValueError()
        mock_response.text.side_effect = Exception()
        setup_mock_post(mock_session, 502, {})
        mock_session.request.return_value.__aenter__.return_value = mock_response

        provider = OllamaProvider(session=mock_session)
        result = await provider.get_completion("test", "llama3.1")
        assert "Could not read response body" in result.error_message or "502" in result.error_message

    @pytest.mark.asyncio
    async def test_error_unexpected_exception(self, mock_session):
        """Provider handles completely unexpected exceptions."""
        setup_mock_error(mock_session, RuntimeError("Doom"))
        provider = OllamaProvider(session=mock_session)
        result = await provider.get_completion("test", "llama3.1")
        assert "unexpected error" in result.error_message.lower()


class TestOllamaProviderSessionInjection:
    """Test session injection for parallel processing (6 tests)."""

    @pytest.mark.asyncio
    async def test_parallel_requests_share_session(self, mock_session):
        """Multiple concurrent requests share one session."""
        def mock_request_side_effect(method, url, **kwargs):
            if "/api/generate" in url:
                resp = create_mock_response(200, SUCCESSFUL_COMPLETION)
            else:
                resp = create_mock_response(200, MODEL_INFO_LLAMA31)
            return create_mock_context(resp)

        mock_session.request.side_effect = mock_request_side_effect
        provider = OllamaProvider(session=mock_session)

        tasks = [provider.get_completion(f"prompt {i}", "llama3.1:latest") for i in range(5)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r.success for r in results)
        assert mock_session.request.call_count == 10

    @pytest.mark.asyncio
    async def test_session_reused(self, mock_session):
        """Session is reused for multiple calls."""
        setup_mock_sequential_responses(mock_session, [
            (200, SUCCESSFUL_COMPLETION),
            (200, MODEL_INFO_LLAMA31),
            (200, SUCCESSFUL_COMPLETION),
            (200, MODEL_INFO_LLAMA31)
        ])
        provider = OllamaProvider(session=mock_session)
        await provider.get_completion("test1", "llama3.1")
        await provider.get_completion("test2", "llama3.1")
        assert mock_session.request.call_count == 4

    @pytest.mark.asyncio
    async def test_session_not_closed_by_provider(self, mock_session):
        """Injected session not closed by provider."""
        setup_mock_sequential_responses(mock_session, [
            (200, SUCCESSFUL_COMPLETION),
            (200, MODEL_INFO_LLAMA31)
        ])
        provider = OllamaProvider(session=mock_session)
        await provider.get_completion("test", "llama3.1")
        assert_session_not_closed(mock_session)

    def test_init_accepts_session(self):
        """Provider accepts session in init."""
        mock_session = AsyncMock()
        provider = OllamaProvider(session=mock_session)
        assert provider._session is mock_session

    @pytest.mark.asyncio
    async def test_request_uses_injected_session(self, mock_session):
        """Provider actually uses the injected session."""
        setup_mock_sequential_responses(mock_session, [
            (200, SUCCESSFUL_COMPLETION),
            (200, MODEL_INFO_LLAMA31)
        ])
        provider = OllamaProvider(session=mock_session)
        await provider.get_completion("test", "llama3.1")
        assert mock_session.request.called

    @pytest.mark.asyncio
    async def test_parallel_context_window_fetch(self, mock_session):
        """Parallel context window fetches share session."""
        setup_mock_post(mock_session, 200, MODEL_INFO_LLAMA31)
        provider = OllamaProvider(session=mock_session)
        tasks = [provider.get_context_window(f"model-{i}") for i in range(5)]
        results = await asyncio.gather(*tasks)
        assert len(results) == 5
        assert mock_session.request.call_count == 5


class TestOllamaProviderPullModel:
    """Test Ollama pull_model functionality (6 tests)."""

    @pytest.mark.asyncio
    async def test_pull_model_success(self, mock_session):
        """Provider successfully pulls a model."""
        setup_mock_post(mock_session, 200, {"status": "success"})
        provider = OllamaProvider(session=mock_session)
        result = await provider.pull_model("llama3.1")
        assert result is True
        assert provider._model_cache is None

    @pytest.mark.asyncio
    async def test_pull_model_failure(self, mock_session):
        """Provider handles pull failure."""
        setup_mock_post(mock_session, 500, {"error": "download failed"})
        provider = OllamaProvider(session=mock_session)
        result = await provider.pull_model("llama3.1")
        assert result is False

    @pytest.mark.asyncio
    async def test_pull_model_timeout(self, mock_session):
        """Provider handles timeout during pull."""
        setup_mock_error(mock_session, asyncio.TimeoutError())
        provider = OllamaProvider(session=mock_session)
        result = await provider.pull_model("llama3.1")
        assert result is False

    @pytest.mark.asyncio
    async def test_pull_model_clears_cache(self, mock_session):
        """Provider clears model cache after pulling."""
        setup_mock_post(mock_session, 200, {"status": "success"})
        provider = OllamaProvider(session=mock_session)
        provider._model_cache = ["old"]
        await provider.pull_model("new")
        assert provider._model_cache is None

    @pytest.mark.asyncio
    async def test_pull_model_request_params(self, mock_session):
        """Provider sends correct params to pull endpoint."""
        setup_mock_post(mock_session, 200, {"status": "success"})
        provider = OllamaProvider(session=mock_session)
        await provider.pull_model("llama3.1")
        call_args = mock_session.request.call_args
        assert "/api/pull" in call_args[0][1]
        assert call_args[1]["json"]["name"] == "llama3.1"

    @pytest.mark.asyncio
    async def test_pull_model_extended_timeout(self, mock_session):
        """Provider uses extended timeout for pull."""
        setup_mock_post(mock_session, 200, {"status": "success"})
        provider = OllamaProvider(session=mock_session)
        await provider.pull_model("llama3.1")
        call_args = mock_session.request.call_args
        assert call_args[1]["timeout"].total == 600
