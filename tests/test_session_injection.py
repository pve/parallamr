"""Tests for HTTP session dependency injection in providers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from parallamr.providers import OllamaProvider, OpenRouterProvider


class TestOpenRouterSessionInjection:
    """Test session injection for OpenRouter provider."""

    @pytest.mark.asyncio
    async def test_accepts_injected_session(self):
        """Provider accepts and stores injected session."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)

        assert provider._session is mock_session

    @pytest.mark.asyncio
    async def test_default_session_is_none(self):
        """No session injection defaults to None (lazy creation)."""
        provider = OpenRouterProvider(api_key="test-key")
        assert provider._session is None

    @pytest.mark.asyncio
    async def test_uses_injected_session_for_completion(self):
        """Provider uses injected session for API calls."""
        # Setup: Create mock session
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "test output"}}],
            "usage": {"completion_tokens": 5}
        })

        # Setup context manager
        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        # Action: Create provider with injected session
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)

        # Make completion request
        result = await provider.get_completion(
            prompt="test prompt",
            model="gpt-3.5-turbo"
        )

        # Assertions
        assert result.success is True
        assert result.output == "test output"

        # Verify session.post was called
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args

        # Verify correct URL
        assert call_args[0][0] == "https://openrouter.ai/api/v1/chat/completions"

        # Verify headers
        assert "Authorization" in call_args[1]["headers"]
        assert call_args[1]["headers"]["Authorization"] == "Bearer test-key"

        # Verify payload
        payload = call_args[1]["json"]
        assert payload["model"] == "gpt-3.5-turbo"
        assert payload["messages"][0]["content"] == "test prompt"

        # Verify session not closed (injected sessions stay open)
        mock_session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_reused_across_multiple_calls(self):
        """Verify same session used for multiple requests."""
        # Setup
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        # Create mock responses for multiple calls
        responses = []
        for i in range(3):
            resp = AsyncMock()
            resp.status = 200
            resp.json = AsyncMock(return_value={
                "choices": [{"message": {"content": f"response {i}"}}],
                "usage": {"completion_tokens": 5}
            })
            responses.append(resp)

        # Setup post to return different responses
        post_contexts = []
        for resp in responses:
            ctx = AsyncMock()
            ctx.__aenter__.return_value = resp
            ctx.__aexit__.return_value = None
            post_contexts.append(ctx)

        mock_session.post.side_effect = post_contexts

        # Action
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)

        results = []
        for i in range(3):
            result = await provider.get_completion(
                prompt=f"prompt {i}",
                model="gpt-3.5-turbo"
            )
            results.append(result)

        # Assertions
        assert len(results) == 3
        assert all(r.success for r in results)
        assert mock_session.post.call_count == 3

        # Verify it's the same session object for all calls
        assert provider._session is mock_session

        # Session never closed
        mock_session.close.assert_not_called()


class TestOllamaSessionInjection:
    """Test session injection for Ollama provider."""

    @pytest.mark.asyncio
    async def test_accepts_injected_session(self):
        """Provider accepts and stores injected session."""
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        provider = OllamaProvider(
            base_url="http://test:11434",
            session=mock_session
        )

        assert provider._session is mock_session

    @pytest.mark.asyncio
    async def test_default_session_is_none(self):
        """No session injection defaults to None (lazy creation)."""
        provider = OllamaProvider(base_url="http://test:11434")
        assert provider._session is None

    @pytest.mark.asyncio
    async def test_uses_injected_session_for_completion(self):
        """Provider uses injected session for generate endpoint."""
        # Setup: Create mock session
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        # Mock list_models response (GET to /api/tags)
        get_response = AsyncMock()
        get_response.status = 200
        get_response.json = AsyncMock(return_value={
            "models": [{"name": "llama3.1"}]
        })
        get_response.raise_for_status = MagicMock()

        get_ctx = AsyncMock()
        get_ctx.__aenter__.return_value = get_response
        get_ctx.__aexit__.return_value = None
        mock_session.get.return_value = get_ctx

        # Mock completion response (POST to /api/generate)
        post_response = AsyncMock()
        post_response.status = 200
        post_response.json = AsyncMock(return_value={
            "response": "test output",
            "done": True
        })
        post_response.raise_for_status = MagicMock()

        # Mock context window response (POST to /api/show)
        show_response = AsyncMock()
        show_response.status = 200
        show_response.json = AsyncMock(return_value={
            "model_info": {"llama.context_length": 131072}
        })

        # Setup POST to return different responses
        def mock_post_side_effect(url, **kwargs):
            ctx = AsyncMock()
            if "/api/generate" in url:
                ctx.__aenter__.return_value = post_response
            elif "/api/show" in url:
                ctx.__aenter__.return_value = show_response
            ctx.__aexit__.return_value = None
            return ctx

        mock_session.post.side_effect = mock_post_side_effect

        # Action: Create provider with injected session
        provider = OllamaProvider(
            base_url="http://test:11434",
            session=mock_session
        )

        # Make completion request
        result = await provider.get_completion(
            prompt="test prompt",
            model="llama3.1"
        )

        # Assertions
        assert result.success is True
        assert result.output == "test output"

        # Verify session.post was called
        assert mock_session.post.call_count >= 1  # May include context window call

        # Verify list_models was called
        assert mock_session.get.called

        # Verify generate was called
        assert mock_session.post.called

        # Verify session not closed
        mock_session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_reused_across_operations(self):
        """Session reused across different operation types."""
        # Setup
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        # Mock GET response (for tags)
        get_response = AsyncMock()
        get_response.status = 200
        get_response.json = AsyncMock(return_value={
            "models": [
                {"name": "llama3.1"},
                {"name": "llama2"}
            ]
        })
        get_response.raise_for_status = MagicMock()

        get_ctx = AsyncMock()
        get_ctx.__aenter__.return_value = get_response
        get_ctx.__aexit__.return_value = None
        mock_session.get.return_value = get_ctx

        # Mock POST responses
        def create_post_response(data):
            resp = AsyncMock()
            resp.status = 200
            resp.json = AsyncMock(return_value=data)
            resp.raise_for_status = MagicMock()
            return resp

        def mock_post_side_effect(url, **kwargs):
            ctx = AsyncMock()
            if "/api/generate" in url:
                ctx.__aenter__.return_value = create_post_response({
                    "response": "completion",
                    "done": True
                })
            elif "/api/show" in url:
                ctx.__aenter__.return_value = create_post_response({
                    "model_info": {"llama.context_length": 131072}
                })
            ctx.__aexit__.return_value = None
            return ctx

        mock_session.post.side_effect = mock_post_side_effect

        # Action
        provider = OllamaProvider(
            base_url="http://test:11434",
            session=mock_session
        )

        # Call different operations
        models = await provider.list_models()
        completion = await provider.get_completion("test", "llama3.1")
        context = await provider.get_context_window("llama3.1")

        # Assertions
        assert len(models) == 2
        assert completion.success is True
        assert context == 131072

        # Verify session used for all operations
        assert provider._session is mock_session
        assert mock_session.get.called
        assert mock_session.post.called

        # Session never closed
        mock_session.close.assert_not_called()


class TestSessionLifecycleManagement:
    """Test session lifecycle and ownership."""

    @pytest.mark.asyncio
    async def test_injected_session_not_closed_after_request(self):
        """Provider never closes injected sessions."""
        # Setup
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "test"}}],
            "usage": {"completion_tokens": 5}
        })

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        # Action: Create provider, make request, delete provider
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)
        await provider.get_completion("test", "gpt-3.5-turbo")
        del provider

        # Assertion: Session still open
        mock_session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_injected_session_survives_multiple_providers(self):
        """Session lifetime independent of provider instances."""
        # Setup
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "test"}}],
            "usage": {"completion_tokens": 5}
        })

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response
        mock_post_ctx.__aexit__.return_value = None
        mock_session.post.return_value = mock_post_ctx

        # Action: Create multiple providers with same session
        provider1 = OpenRouterProvider(api_key="test-key", session=mock_session)
        await provider1.get_completion("test", "gpt-3.5-turbo")
        del provider1

        provider2 = OpenRouterProvider(api_key="test-key", session=mock_session)
        await provider2.get_completion("test", "gpt-3.5-turbo")
        del provider2

        # Assertion: Session still open after both providers destroyed
        mock_session.close.assert_not_called()


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    @pytest.mark.asyncio
    async def test_openrouter_works_without_session_injection(self):
        """Existing code continues to work (no breaking changes)."""
        # Action: Create provider without session injection (old way)
        provider = OpenRouterProvider(api_key="test-key")

        # Verify no session stored initially
        assert provider._session is None

        # This test just verifies the provider can be created without session injection
        # and has the correct initial state. Actual API calls would require a real API key.

    @pytest.mark.asyncio
    async def test_ollama_works_without_session_injection(self):
        """Default behavior unchanged."""
        # Action: Create provider without session injection
        provider = OllamaProvider(base_url="http://test:11434")

        # Verify no session stored initially
        assert provider._session is None

        # This test just verifies the provider can be created without session injection
        # and has the correct initial state. Actual API calls would require a running Ollama server.


class TestParallelProcessing:
    """Test session handling in parallel scenarios."""

    @pytest.mark.asyncio
    async def test_parallel_requests_share_session(self):
        """Multiple concurrent requests share one session."""
        # Setup
        mock_session = AsyncMock(spec=aiohttp.ClientSession)

        # Create counter for concurrent calls
        call_count = 0

        def mock_post_side_effect(url, **kwargs):
            nonlocal call_count
            call_count += 1

            ctx = AsyncMock()
            resp = AsyncMock()
            resp.status = 200
            resp.json = AsyncMock(return_value={
                "choices": [{"message": {"content": f"response {call_count}"}}],
                "usage": {"completion_tokens": 5}
            })
            resp.raise_for_status = MagicMock()
            ctx.__aenter__.return_value = resp
            ctx.__aexit__.return_value = None
            return ctx

        mock_session.post.side_effect = mock_post_side_effect

        # Action
        provider = OpenRouterProvider(api_key="test-key", session=mock_session)

        # Run 10 parallel completions
        tasks = [
            provider.get_completion(f"prompt {i}", "gpt-3.5-turbo")
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)

        # Assertions
        assert len(results) == 10
        assert all(r.success for r in results)

        # All requests used the same session
        assert provider._session is mock_session

        # Session called 10 times
        assert mock_session.post.call_count == 10

        # Session not closed
        mock_session.close.assert_not_called()
