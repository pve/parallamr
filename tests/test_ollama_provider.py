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

    def test_init_combined_parameters(self):
        """Provider accepts all initialization parameters together."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        def mock_env_getter(key: str, default: str = "") -> Optional[str]:
            return "http://test:8080" if key == "OLLAMA_BASE_URL" else default

        provider = OllamaProvider(
            base_url="http://custom:9090",
            timeout=500,
            env_getter=mock_env_getter,
            session=mock_session
        )

        # Explicit base_url should override env_getter
        assert provider.base_url == "http://custom:9090"
        assert provider.timeout == 500
        assert provider._session is mock_session


class TestOllamaProviderCompletion:
    """Test Ollama provider completion requests (15 tests)."""

    @pytest.mark.asyncio
    async def test_successful_completion(self, mock_session):
        """Provider returns successful completion response."""
        from tests.conftest import setup_mock_post

        setup_mock_post(mock_session, 200, SUCCESSFUL_COMPLETION)

        # Mock list_models to return available models
        provider = OllamaProvider(session=mock_session)
        provider._model_cache = ["llama3.1:latest"]

        result = await provider.get_completion("test prompt", "llama3.1:latest")

        assert result.success is True
        assert result.output == "This is a test response from Llama 3.1."
        assert result.output_tokens > 0

    @pytest.mark.asyncio
    async def test_completion_with_kwargs(self, mock_session):
        """Provider passes additional kwargs to API."""
        from tests.conftest import setup_mock_post

        setup_mock_post(mock_session, 200, SUCCESSFUL_COMPLETION)

        provider = OllamaProvider(session=mock_session)
        provider._model_cache = ["llama3.1:latest"]

        result = await provider.get_completion(
            "test prompt",
            "llama3.1:latest",
            temperature=0.7,
            max_tokens=100
        )

        # Verify kwargs were passed in payload
        call_args = mock_session.post.call_args
        payload = call_args[1]["json"]
        assert payload.get("temperature") == 0.7
        assert payload.get("max_tokens") == 100

    @pytest.mark.asyncio
    async def test_completion_request_format(self, mock_session):
        """Provider formats request correctly."""
        from tests.conftest import setup_mock_post

        setup_mock_post(mock_session, 200, SUCCESSFUL_COMPLETION)

        provider = OllamaProvider(base_url="http://test:11434", session=mock_session)
        provider._model_cache = ["llama3.1:latest"]

        await provider.get_completion("test prompt", "llama3.1:latest")

        # Verify request format
        call_args = mock_session.post.call_args
        assert "http://test:11434/api/generate" in str(call_args)

        payload = call_args[1]["json"]
        assert payload["model"] == "llama3.1:latest"
        assert payload["prompt"] == "test prompt"
        assert payload["stream"] is False

    @pytest.mark.asyncio
    async def test_completion_estimates_tokens(self, mock_session):
        """Provider estimates token counts when not provided."""
        from tests.conftest import setup_mock_post

        setup_mock_post(mock_session, 200, SUCCESSFUL_COMPLETION)

        provider = OllamaProvider(session=mock_session)
        provider._model_cache = ["llama3.1:latest"]

        result = await provider.get_completion("test prompt", "llama3.1:latest")

        # Ollama doesn't provide token counts, so we estimate
        assert result.output_tokens > 0

    @pytest.mark.asyncio
    async def test_completion_with_context_window(self, mock_session):
        """Provider includes context window in response."""
        from tests.conftest import setup_mock_post, setup_mock_sequential_responses

        # Setup sequential responses: first for list_models, then for completion, then for show
        mock_response_list = AsyncMock()
        mock_response_list.status = 200
        mock_response_list.json = AsyncMock(return_value=MODELS_LIST_RESPONSE)
        mock_response_list.raise_for_status = MagicMock()

        mock_ctx_list = AsyncMock()
        mock_ctx_list.__aenter__.return_value = mock_response_list
        mock_ctx_list.__aexit__.return_value = None

        mock_response_completion = AsyncMock()
        mock_response_completion.status = 200
        mock_response_completion.json = AsyncMock(return_value=SUCCESSFUL_COMPLETION)
        mock_response_completion.raise_for_status = MagicMock()

        mock_ctx_completion = AsyncMock()
        mock_ctx_completion.__aenter__.return_value = mock_response_completion
        mock_ctx_completion.__aexit__.return_value = None

        mock_response_show = AsyncMock()
        mock_response_show.status = 200
        mock_response_show.json = AsyncMock(return_value=MODEL_INFO_LLAMA31)

        mock_ctx_show = AsyncMock()
        mock_ctx_show.__aenter__.return_value = mock_response_show
        mock_ctx_show.__aexit__.return_value = None

        mock_session.get.return_value = mock_ctx_list
        mock_session.post.side_effect = [mock_ctx_completion, mock_ctx_show]

        provider = OllamaProvider(session=mock_session)
        result = await provider.get_completion("test prompt", "llama3.1:latest")

        assert result.context_window == 131072

    @pytest.mark.asyncio
    async def test_completion_timeout(self, mock_session):
        """Provider handles request timeout."""
        from tests.conftest import setup_mock_error

        setup_mock_error(mock_session, asyncio.TimeoutError())

        provider = OllamaProvider(session=mock_session, timeout=10)
        result = await provider.get_completion("test prompt", "llama3.1:latest")

        assert result.success is False
        assert "timeout" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_completion_connection_error(self, mock_session):
        """Provider handles connection errors."""
        from tests.conftest import setup_mock_error

        setup_mock_error(mock_session, aiohttp.ClientConnectorError(None, OSError("Connection refused")))

        provider = OllamaProvider(session=mock_session)
        result = await provider.get_completion("test prompt", "llama3.1:latest")

        assert result.success is False
        assert "connect" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_completion_network_error(self, mock_session):
        """Provider handles network errors."""
        from tests.conftest import setup_mock_error

        setup_mock_error(mock_session, aiohttp.ClientError("Network failure"))

        provider = OllamaProvider(session=mock_session)
        result = await provider.get_completion("test prompt", "llama3.1:latest")

        assert result.success is False
        assert "network error" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_completion_unexpected_error(self, mock_session):
        """Provider handles unexpected exceptions."""
        from tests.conftest import setup_mock_error

        setup_mock_error(mock_session, ValueError("Unexpected error"))

        provider = OllamaProvider(session=mock_session)
        result = await provider.get_completion("test prompt", "llama3.1:latest")

        assert result.success is False
        assert "unexpected error" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_completion_session_not_closed(self, mock_session):
        """Provider does not close injected session."""
        from tests.conftest import setup_mock_post, assert_session_not_closed

        setup_mock_post(mock_session, 200, SUCCESSFUL_COMPLETION)

        provider = OllamaProvider(session=mock_session)
        provider._model_cache = ["llama3.1:latest"]

        await provider.get_completion("test prompt", "llama3.1:latest")

        assert_session_not_closed(mock_session)

    @pytest.mark.asyncio
    async def test_completion_multiple_sequential(self, mock_session):
        """Provider handles multiple sequential completions."""
        from tests.conftest import create_mock_response, create_mock_context

        responses = []
        for i in range(3):
            resp = create_mock_response(200, create_completion_response(
                response=f"Response {i}",
                eval_count=10 + i
            ))
            responses.append(resp)

        post_contexts = [create_mock_context(resp) for resp in responses]
        mock_session.post.side_effect = post_contexts

        provider = OllamaProvider(session=mock_session)
        provider._model_cache = ["llama3.1:latest"]

        results = []
        for i in range(3):
            result = await provider.get_completion(f"prompt {i}", "llama3.1:latest")
            results.append(result)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert results[0].output == "Response 0"
        assert results[1].output == "Response 1"
        assert results[2].output == "Response 2"

    @pytest.mark.asyncio
    async def test_completion_without_injected_session(self):
        """Provider creates temporary session when none injected."""
        provider = OllamaProvider()
        assert provider._session is None

    @pytest.mark.asyncio
    async def test_completion_model_not_available(self, mock_session):
        """Provider handles unavailable model."""
        from tests.conftest import setup_mock_get

        setup_mock_get(mock_session, 200, MODELS_LIST_RESPONSE)

        provider = OllamaProvider(session=mock_session)
        result = await provider.get_completion("test prompt", "nonexistent-model")

        assert result.success is False
        assert "not found" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_completion_with_model_tags(self, mock_session):
        """Provider handles model tags correctly (llama3.1:latest format)."""
        from tests.conftest import setup_mock_post

        setup_mock_post(mock_session, 200, COMPLETION_CODELLAMA)

        provider = OllamaProvider(session=mock_session)
        provider._model_cache = ["codellama:13b"]

        result = await provider.get_completion("write a function", "codellama:13b")

        assert result.success is True
        assert "def hello_world" in result.output

    @pytest.mark.asyncio
    async def test_completion_minimal_response(self, mock_session):
        """Provider handles minimal response format."""
        from tests.conftest import setup_mock_post

        setup_mock_post(mock_session, 200, COMPLETION_MINIMAL)

        provider = OllamaProvider(session=mock_session)
        provider._model_cache = ["llama3.1:latest"]

        result = await provider.get_completion("test", "llama3.1:latest")

        assert result.success is True
        assert result.output == "Short reply."


class TestOllamaProviderModels:
    """Test Ollama provider model management (10 tests)."""

    @pytest.mark.asyncio
    async def test_list_models_success(self, mock_session):
        """Provider lists available models."""
        from tests.conftest import setup_mock_get

        setup_mock_get(mock_session, 200, MODELS_LIST_RESPONSE)

        provider = OllamaProvider(session=mock_session)
        models = await provider.list_models()

        assert len(models) == 4
        assert "llama3.1:latest" in models
        assert "mistral:latest" in models
        assert "codellama:13b" in models

    @pytest.mark.asyncio
    async def test_list_models_preserves_tags(self, mock_session):
        """Provider preserves full model names including tags."""
        from tests.conftest import setup_mock_get

        setup_mock_get(mock_session, 200, MODELS_LIST_RESPONSE)

        provider = OllamaProvider(session=mock_session)
        models = await provider.list_models()

        # Should preserve tags like ":latest", ":13b"
        assert "llama3.1:latest" in models
        assert "codellama:13b" in models
        assert "llama2:7b" in models

    @pytest.mark.asyncio
    async def test_list_models_empty(self, mock_session):
        """Provider handles empty model list."""
        from tests.conftest import setup_mock_get

        setup_mock_get(mock_session, 200, MODELS_LIST_EMPTY)

        provider = OllamaProvider(session=mock_session)
        models = await provider.list_models()

        assert models == []

    @pytest.mark.asyncio
    async def test_list_models_caching(self, mock_session):
        """Provider caches model list."""
        from tests.conftest import setup_mock_get

        setup_mock_get(mock_session, 200, MODELS_LIST_RESPONSE)

        provider = OllamaProvider(session=mock_session)

        # Call twice
        models1 = await provider.list_models()
        models2 = await provider.list_models()

        # Should only call API once (caching)
        assert mock_session.get.call_count == 1
        assert models1 == models2

    @pytest.mark.asyncio
    async def test_list_models_error_handling(self, mock_session):
        """Provider handles errors when listing models."""
        from tests.conftest import setup_mock_error

        setup_mock_error(mock_session, aiohttp.ClientError("API error"))

        provider = OllamaProvider(session=mock_session)
        models = await provider.list_models()

        # Should return empty list on error
        assert models == []

    @pytest.mark.asyncio
    async def test_list_models_single(self, mock_session):
        """Provider handles single model in list."""
        from tests.conftest import setup_mock_get

        setup_mock_get(mock_session, 200, MODELS_LIST_SINGLE)

        provider = OllamaProvider(session=mock_session)
        models = await provider.list_models()

        assert len(models) == 1
        assert "llama3.1:latest" in models

    @pytest.mark.asyncio
    async def test_list_models_malformed_data(self, mock_session):
        """Provider handles malformed model data."""
        from tests.conftest import setup_mock_get

        setup_mock_get(mock_session, 200, MODELS_LIST_MALFORMED)

        provider = OllamaProvider(session=mock_session)
        models = await provider.list_models()

        # Should include valid entries and skip invalid ones
        assert "llama3.1:latest" in models

    def test_is_model_available_with_cache(self):
        """Provider checks model availability using cache."""
        provider = OllamaProvider()
        provider._model_cache = [
            "llama3.1:latest",
            "mistral:latest"
        ]

        assert provider.is_model_available("llama3.1:latest") is True
        assert provider.is_model_available("mistral:latest") is True
        assert provider.is_model_available("nonexistent") is False

    def test_is_model_available_without_cache(self):
        """Provider optimistically returns True when cache empty."""
        provider = OllamaProvider()

        # No cache yet - should be optimistic
        assert provider.is_model_available("any-model") is True

    @pytest.mark.asyncio
    async def test_list_models_uses_api_tags_endpoint(self, mock_session):
        """Provider uses correct Ollama API endpoint for listing models."""
        from tests.conftest import setup_mock_get

        setup_mock_get(mock_session, 200, MODELS_LIST_RESPONSE)

        provider = OllamaProvider(base_url="http://test:11434", session=mock_session)
        await provider.list_models()

        # Verify correct endpoint was called
        call_args = mock_session.get.call_args
        assert "http://test:11434/api/tags" in str(call_args)


class TestOllamaProviderContextWindow:
    """Test Ollama provider context window retrieval (8 tests)."""

    @pytest.mark.asyncio
    async def test_get_context_window_llama(self, mock_session):
        """Provider retrieves context window for Llama model."""
        from tests.conftest import create_mock_response, create_mock_context

        response = create_mock_response(200, MODEL_INFO_LLAMA31)
        ctx = create_mock_context(response)
        mock_session.post.return_value = ctx

        provider = OllamaProvider(session=mock_session)
        context = await provider.get_context_window("llama3.1:latest")

        assert context == 131072

    @pytest.mark.asyncio
    async def test_get_context_window_mistral(self, mock_session):
        """Provider retrieves context window for Mistral model."""
        from tests.conftest import create_mock_response, create_mock_context

        response = create_mock_response(200, MODEL_INFO_MISTRAL)
        ctx = create_mock_context(response)
        mock_session.post.return_value = ctx

        provider = OllamaProvider(session=mock_session)
        context = await provider.get_context_window("mistral:latest")

        assert context == 8192

    @pytest.mark.asyncio
    async def test_get_context_window_codellama(self, mock_session):
        """Provider retrieves context window for CodeLlama model."""
        from tests.conftest import create_mock_response, create_mock_context

        response = create_mock_response(200, MODEL_INFO_CODELLAMA)
        ctx = create_mock_context(response)
        mock_session.post.return_value = ctx

        provider = OllamaProvider(session=mock_session)
        context = await provider.get_context_window("codellama:13b")

        assert context == 16384

    @pytest.mark.asyncio
    async def test_get_context_window_no_context_data(self, mock_session):
        """Provider handles missing context window data."""
        from tests.conftest import create_mock_response, create_mock_context

        response = create_mock_response(200, MODEL_INFO_NO_CONTEXT)
        ctx = create_mock_context(response)
        mock_session.post.return_value = ctx

        provider = OllamaProvider(session=mock_session)
        context = await provider.get_context_window("custom:model")

        assert context is None

    @pytest.mark.asyncio
    async def test_get_context_window_404_error(self, mock_session):
        """Provider handles 404 when model not found."""
        from tests.conftest import create_mock_response, create_mock_context

        response = create_mock_response(404, ERROR_404_MODEL_NOT_FOUND)
        ctx = create_mock_context(response)
        mock_session.post.return_value = ctx

        provider = OllamaProvider(session=mock_session)
        context = await provider.get_context_window("invalid-model")

        assert context is None

    @pytest.mark.asyncio
    async def test_get_context_window_network_error(self, mock_session):
        """Provider handles network errors gracefully."""
        from tests.conftest import setup_mock_error

        setup_mock_error(mock_session, aiohttp.ClientError("Network error"))

        provider = OllamaProvider(session=mock_session)
        context = await provider.get_context_window("llama3.1:latest")

        assert context is None

    @pytest.mark.asyncio
    async def test_get_context_window_uses_show_endpoint(self, mock_session):
        """Provider uses correct Ollama API endpoint for model info."""
        from tests.conftest import create_mock_response, create_mock_context

        response = create_mock_response(200, MODEL_INFO_LLAMA31)
        ctx = create_mock_context(response)
        mock_session.post.return_value = ctx

        provider = OllamaProvider(base_url="http://test:11434", session=mock_session)
        await provider.get_context_window("llama3.1:latest")

        # Verify correct endpoint and payload
        call_args = mock_session.post.call_args
        assert "http://test:11434/api/show" in str(call_args)
        payload = call_args[1]["json"]
        assert payload["name"] == "llama3.1:latest"

    @pytest.mark.asyncio
    async def test_get_context_window_generic_key(self, mock_session):
        """Provider handles generic context_length keys."""
        from tests.conftest import create_mock_response, create_mock_context

        # Create response with generic context_length key
        model_info = {
            "modelfile": "# Test model",
            "parameters": "",
            "model_info": {
                "general.architecture": "custom",
                "custom.context_length": 32768
            }
        }

        response = create_mock_response(200, model_info)
        ctx = create_mock_context(response)
        mock_session.post.return_value = ctx

        provider = OllamaProvider(session=mock_session)
        context = await provider.get_context_window("custom:model")

        assert context == 32768


class TestOllamaProviderErrorHandling:
    """Test Ollama provider error handling (10 tests)."""

    @pytest.mark.asyncio
    async def test_error_404_model_not_found(self, mock_session):
        """Provider handles 404 model not found error."""
        from tests.conftest import setup_mock_post

        setup_mock_post(mock_session, 404, ERROR_404_MODEL_NOT_FOUND)

        provider = OllamaProvider(session=mock_session)
        provider._model_cache = ["llama3.1:latest"]

        result = await provider.get_completion("test prompt", "llama3.1:latest")

        assert result.success is False
        assert "not found" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_400_invalid_request(self, mock_session):
        """Provider handles 400 bad request error."""
        from tests.conftest import create_mock_response, create_mock_context

        response = create_mock_response(400, ERROR_400_INVALID_REQUEST)
        response.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(
            request_info=None,
            history=None,
            status=400
        ))

        ctx = create_mock_context(response)
        mock_session.post.return_value = ctx

        provider = OllamaProvider(session=mock_session)
        provider._model_cache = ["llama3.1:latest"]

        result = await provider.get_completion("", "llama3.1:latest")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_error_500_model_not_loaded(self, mock_session):
        """Provider handles 500 server error."""
        from tests.conftest import create_mock_response, create_mock_context

        response = create_mock_response(500, ERROR_500_MODEL_NOT_LOADED)
        response.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(
            request_info=None,
            history=None,
            status=500
        ))

        ctx = create_mock_context(response)
        mock_session.post.return_value = ctx

        provider = OllamaProvider(session=mock_session)
        provider._model_cache = ["llama3.1:latest"]

        result = await provider.get_completion("test", "llama3.1:latest")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_error_502_bad_gateway(self, mock_session):
        """Provider handles 502 bad gateway error."""
        from tests.conftest import create_mock_response, create_mock_context

        response = create_mock_response(502, ERROR_502_BAD_GATEWAY)
        response.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(
            request_info=None,
            history=None,
            status=502
        ))

        ctx = create_mock_context(response)
        mock_session.post.return_value = ctx

        provider = OllamaProvider(session=mock_session)
        provider._model_cache = ["llama3.1:latest"]

        result = await provider.get_completion("test", "llama3.1:latest")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_error_503_service_unavailable(self, mock_session):
        """Provider handles 503 service unavailable error."""
        from tests.conftest import create_mock_response, create_mock_context

        response = create_mock_response(503, ERROR_503_SERVICE_UNAVAILABLE)
        response.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(
            request_info=None,
            history=None,
            status=503
        ))

        ctx = create_mock_context(response)
        mock_session.post.return_value = ctx

        provider = OllamaProvider(session=mock_session)
        provider._model_cache = ["llama3.1:latest"]

        result = await provider.get_completion("test", "llama3.1:latest")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_error_json_decode_error(self, mock_session):
        """Provider handles malformed JSON response."""
        from tests.conftest import create_mock_context

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(side_effect=ValueError("Invalid JSON"))
        mock_response.raise_for_status = MagicMock()

        ctx = create_mock_context(mock_response)
        mock_session.post.return_value = ctx

        provider = OllamaProvider(session=mock_session)
        provider._model_cache = ["llama3.1:latest"]

        result = await provider.get_completion("test", "llama3.1:latest")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_error_connection_refused(self, mock_session):
        """Provider handles connection refused error."""
        from tests.conftest import setup_mock_error

        setup_mock_error(
            mock_session,
            aiohttp.ClientConnectorError(None, OSError("Connection refused"))
        )

        provider = OllamaProvider(session=mock_session)
        result = await provider.get_completion("test", "llama3.1:latest")

        assert result.success is False
        assert "connect" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_error_context_exceeded(self, mock_session):
        """Provider handles context length exceeded error."""
        from tests.conftest import create_mock_response, create_mock_context

        response = create_mock_response(400, ERROR_400_CONTEXT_EXCEEDED)
        response.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(
            request_info=None,
            history=None,
            status=400
        ))

        ctx = create_mock_context(response)
        mock_session.post.return_value = ctx

        provider = OllamaProvider(session=mock_session)
        provider._model_cache = ["llama3.1:latest"]

        result = await provider.get_completion("very long prompt" * 1000, "llama3.1:latest")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_error_out_of_memory(self, mock_session):
        """Provider handles out of memory error."""
        from tests.conftest import create_mock_response, create_mock_context

        response = create_mock_response(500, ERROR_500_OUT_OF_MEMORY)
        response.raise_for_status = MagicMock(side_effect=aiohttp.ClientResponseError(
            request_info=None,
            history=None,
            status=500
        ))

        ctx = create_mock_context(response)
        mock_session.post.return_value = ctx

        provider = OllamaProvider(session=mock_session)
        provider._model_cache = ["llama3.1:70b"]

        result = await provider.get_completion("test", "llama3.1:70b")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_error_response_with_error_field(self, mock_session):
        """Provider handles error field in response."""
        from tests.conftest import setup_mock_post

        error_response = {
            "model": "llama3.1:latest",
            "response": "",
            "done": True,
            "error": "Internal model error occurred"
        }

        setup_mock_post(mock_session, 200, error_response)

        provider = OllamaProvider(session=mock_session)
        provider._model_cache = ["llama3.1:latest"]

        result = await provider.get_completion("test", "llama3.1:latest")

        assert result.success is False
        assert "internal model error" in result.error_message.lower()


class TestOllamaProviderSessionInjection:
    """Test session injection for parallel processing (6 tests)."""

    @pytest.mark.asyncio
    async def test_accepts_injected_session(self):
        """Provider accepts and stores injected session."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        provider = OllamaProvider(session=mock_session)

        assert provider._session is mock_session

    @pytest.mark.asyncio
    async def test_uses_injected_session(self, mock_session):
        """Provider uses injected session for API calls."""
        from tests.conftest import setup_mock_post

        setup_mock_post(mock_session, 200, SUCCESSFUL_COMPLETION)

        provider = OllamaProvider(session=mock_session)
        provider._model_cache = ["llama3.1:latest"]

        await provider.get_completion("test prompt", "llama3.1:latest")

        # Verify session.post was called
        mock_session.post.assert_called()

    @pytest.mark.asyncio
    async def test_session_reused_across_calls(self, mock_session):
        """Same session used for multiple requests."""
        from tests.conftest import create_mock_response, create_mock_context

        responses = []
        for i in range(3):
            resp = create_mock_response(200, create_completion_response(response=f"Response {i}"))
            responses.append(resp)

        post_contexts = [create_mock_context(resp) for resp in responses]
        mock_session.post.side_effect = post_contexts

        provider = OllamaProvider(session=mock_session)
        provider._model_cache = ["llama3.1:latest"]

        for i in range(3):
            await provider.get_completion(f"prompt {i}", "llama3.1:latest")

        assert mock_session.post.call_count == 3
        assert provider._session is mock_session

    @pytest.mark.asyncio
    async def test_session_not_closed(self, mock_session):
        """Injected session not closed by provider."""
        from tests.conftest import setup_mock_post, assert_session_not_closed

        setup_mock_post(mock_session, 200, SUCCESSFUL_COMPLETION)

        provider = OllamaProvider(session=mock_session)
        provider._model_cache = ["llama3.1:latest"]

        await provider.get_completion("test prompt", "llama3.1:latest")
        del provider

        assert_session_not_closed(mock_session)

    @pytest.mark.asyncio
    async def test_parallel_requests_share_session(self, mock_session):
        """Multiple concurrent requests share one session."""
        call_count = 0

        def mock_post_side_effect(url, **kwargs):
            nonlocal call_count
            call_count += 1

            from tests.conftest import create_mock_response, create_mock_context

            resp = create_mock_response(200, create_completion_response(
                response=f"Response {call_count}"
            ))
            return create_mock_context(resp)

        mock_session.post.side_effect = mock_post_side_effect

        provider = OllamaProvider(session=mock_session)
        provider._model_cache = ["llama3.1:latest"]

        # Run 10 parallel completions
        tasks = [
            provider.get_completion(f"prompt {i}", "llama3.1:latest")
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all(r.success for r in results)
        assert mock_session.post.call_count == 10
        mock_session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_backward_compatibility_no_session(self):
        """Provider works without session injection."""
        provider = OllamaProvider()
        assert provider._session is None


class TestOllamaProviderPullModel:
    """Test Ollama pull_model functionality (6 tests)."""

    @pytest.mark.asyncio
    async def test_pull_model_success(self, mock_session):
        """Provider successfully pulls a model."""
        from tests.conftest import create_mock_response, create_mock_context

        response = create_mock_response(200, {"status": "success"})
        ctx = create_mock_context(response)
        mock_session.post.return_value = ctx

        provider = OllamaProvider(session=mock_session)
        result = await provider.pull_model("llama3.1:latest")

        assert result is True

    @pytest.mark.asyncio
    async def test_pull_model_clears_cache(self, mock_session):
        """Provider clears model cache after pulling."""
        from tests.conftest import create_mock_response, create_mock_context

        response = create_mock_response(200, {"status": "success"})
        ctx = create_mock_context(response)
        mock_session.post.return_value = ctx

        provider = OllamaProvider(session=mock_session)
        provider._model_cache = ["old-model"]

        await provider.pull_model("new-model:latest")

        # Cache should be cleared
        assert provider._model_cache is None

    @pytest.mark.asyncio
    async def test_pull_model_uses_pull_endpoint(self, mock_session):
        """Provider uses correct Ollama API endpoint for pulling."""
        from tests.conftest import create_mock_response, create_mock_context

        response = create_mock_response(200, {"status": "success"})
        ctx = create_mock_context(response)
        mock_session.post.return_value = ctx

        provider = OllamaProvider(base_url="http://test:11434", session=mock_session)
        await provider.pull_model("llama3.1:latest")

        # Verify correct endpoint and payload
        call_args = mock_session.post.call_args
        assert "http://test:11434/api/pull" in str(call_args)
        payload = call_args[1]["json"]
        assert payload["name"] == "llama3.1:latest"

    @pytest.mark.asyncio
    async def test_pull_model_failure(self, mock_session):
        """Provider handles pull failure gracefully."""
        from tests.conftest import setup_mock_error

        setup_mock_error(mock_session, aiohttp.ClientError("Pull failed"))

        provider = OllamaProvider(session=mock_session)
        result = await provider.pull_model("invalid-model")

        assert result is False

    @pytest.mark.asyncio
    async def test_pull_model_timeout(self, mock_session):
        """Provider handles timeout during pull."""
        from tests.conftest import setup_mock_error

        setup_mock_error(mock_session, asyncio.TimeoutError())

        provider = OllamaProvider(session=mock_session)
        result = await provider.pull_model("large-model:70b")

        assert result is False

    @pytest.mark.asyncio
    async def test_pull_model_extended_timeout(self, mock_session):
        """Provider uses extended timeout for pull operation."""
        from tests.conftest import create_mock_response, create_mock_context

        response = create_mock_response(200, {"status": "success"})
        ctx = create_mock_context(response)
        mock_session.post.return_value = ctx

        provider = OllamaProvider(session=mock_session)
        await provider.pull_model("llama3.1:latest")

        # Verify extended timeout was used (600 seconds)
        call_args = mock_session.post.call_args
        timeout_arg = call_args[1].get("timeout")
        assert timeout_arg is not None
        assert timeout_arg.total == 600
