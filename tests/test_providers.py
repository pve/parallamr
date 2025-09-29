"""Tests for provider implementations."""

import pytest

from parallaxr.models import ProviderResponse
from parallaxr.providers import MockProvider


class TestMockProvider:
    """Test MockProvider implementation."""

    @pytest.mark.asyncio
    async def test_get_completion(self):
        """Test getting completion from mock provider."""
        provider = MockProvider()
        prompt = "Test prompt"

        response = await provider.get_completion(prompt, "mock")

        assert isinstance(response, ProviderResponse)
        assert response.success is True
        assert "MOCK RESPONSE" in response.output
        assert "Test prompt" in response.output
        assert response.output_tokens > 0
        assert response.context_window is None

    @pytest.mark.asyncio
    async def test_get_completion_with_variables(self):
        """Test getting completion with variables."""
        provider = MockProvider()
        prompt = "Test prompt"
        variables = {"topic": "AI", "source": "Wikipedia"}

        response = await provider.get_completion(
            prompt, "mock", variables=variables
        )

        assert response.success is True
        assert "AI" in response.output
        assert "Wikipedia" in response.output

    @pytest.mark.asyncio
    async def test_get_context_window(self):
        """Test getting context window (should be None for mock)."""
        provider = MockProvider()

        context_window = await provider.get_context_window("mock")

        assert context_window is None

    @pytest.mark.asyncio
    async def test_list_models(self):
        """Test listing available models."""
        provider = MockProvider()

        models = await provider.list_models()

        assert models == ["mock"]

    def test_is_model_available(self):
        """Test checking if model is available."""
        provider = MockProvider()

        assert provider.is_model_available("mock") is True
        assert provider.is_model_available("nonexistent") is False

    def test_get_provider_name(self):
        """Test getting provider name."""
        provider = MockProvider()

        assert provider.get_provider_name() == "mock"


# Note: Testing OpenRouter and Ollama providers would require actual API access
# or extensive mocking. In a real project, you would add integration tests
# that can be run with actual API keys or local Ollama instances.

class TestProviderIntegration:
    """Integration tests for providers (require actual services)."""

    @pytest.mark.skip(reason="Requires actual OpenRouter API key")
    @pytest.mark.asyncio
    async def test_openrouter_integration(self):
        """Test OpenRouter provider with real API (skipped by default)."""
        # This test would require a real API key and should be run separately
        # from unit tests, perhaps in an integration test suite
        pass

    @pytest.mark.skip(reason="Requires running Ollama instance")
    @pytest.mark.asyncio
    async def test_ollama_integration(self):
        """Test Ollama provider with real instance (skipped by default)."""
        # This test would require a running Ollama instance
        pass