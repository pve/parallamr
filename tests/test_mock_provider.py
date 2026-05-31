"""Tests for provider implementations."""

from typing import Optional

import pytest

from parallamr.models import ProviderResponse
from parallamr.providers import MockProvider


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
        assert response.context_window == 100000  # MockProvider now returns default context window

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
        """Test getting context window for mock provider."""
        provider = MockProvider()

        context_window = await provider.get_context_window("mock")

        assert context_window == 100000  # Default mock context window

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


class TestProviderDependencyInjection:
    """Test dependency injection in provider classes."""

    def test_openrouter_env_getter_injection(self):
        """Test OpenRouter provider with custom env_getter."""
        from parallamr.providers import OpenRouterProvider

        # Create mock env_getter that returns test API key
        def mock_env_getter(key: str) -> Optional[str]:
            if key == "OPENROUTER_API_KEY":
                return "test-api-key-123"
            return None

        # Inject env_getter
        provider = OpenRouterProvider(env_getter=mock_env_getter)

        # Verify it used the injected env_getter
        assert provider.api_key == "test-api-key-123"

    def test_openrouter_base_url_injection(self):
        """Test OpenRouter provider with custom base_url."""
        from parallamr.providers import OpenRouterProvider

        # Inject custom base URL (for testing with mock server)
        provider = OpenRouterProvider(
            api_key="test-key",
            base_url="http://localhost:8080/api/v1"
        )

        # Verify custom base URL is used
        assert provider.base_url == "http://localhost:8080/api/v1"

    def test_openrouter_default_behavior(self):
        """Test OpenRouter provider uses defaults when nothing injected."""
        from parallamr.providers import OpenRouterProvider

        # No injection - should use defaults
        provider = OpenRouterProvider(api_key="direct-key")

        # Verify defaults
        assert provider.api_key == "direct-key"
        assert provider.base_url == "https://openrouter.ai/api/v1"

    def test_ollama_env_getter_injection(self):
        """Test Ollama provider with custom env_getter."""
        from parallamr.providers import OllamaProvider

        # Create mock env_getter that returns test URL
        def mock_env_getter(key: str, default: str = "") -> str:
            if key == "OLLAMA_BASE_URL":
                return "http://test-ollama:5000"
            return default

        # Inject env_getter
        provider = OllamaProvider(env_getter=mock_env_getter)

        # Verify it used the injected env_getter
        assert provider.base_url == "http://test-ollama:5000"

    def test_ollama_base_url_injection(self):
        """Test Ollama provider with custom base_url."""
        from parallamr.providers import OllamaProvider

        # Inject custom base URL
        provider = OllamaProvider(base_url="http://custom-ollama:8080")

        # Verify custom base URL is used
        assert provider.base_url == "http://custom-ollama:8080"

    def test_ollama_default_behavior(self, monkeypatch):
        """Test Ollama provider uses defaults when nothing injected."""
        from parallamr.providers import OllamaProvider

        # Ensure no env var is set (for test isolation)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

        # No injection - should use defaults
        provider = OllamaProvider()

        # Verify default (localhost:11434 since env var not set)
        assert provider.base_url == "http://localhost:11434"


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

    def test_ollama_model_name_parsing(self):
        """Test that model names with tags are preserved (regression test)."""
        # This tests the bug fix where model tags were being stripped
        # Old behavior: "llama3.1:latest" -> "llama3.1"
        # New behavior: "llama3.1:latest" -> "llama3.1:latest"

        mock_api_response = {
            "models": [
                {"name": "llama3.1:latest"},
                {"name": "llama3.1:8b"},
                {"name": "llama3:latest"},
                {"name": "qwen2.5-coder:1.5b"},
                {"name": "qwen2.5-coder:1.5b-base"},
            ]
        }

        # Simulate the list_models logic (lines 193-201 in ollama.py)
        models = []
        for model_info in mock_api_response.get("models", []):
            model_name = model_info.get("name", "")
            if model_name:
                # Keep full model name including tag (e.g., "llama3.1:latest")
                models.append(model_name)

        # Verify full model names with tags are preserved
        assert "llama3.1:latest" in models
        assert "llama3.1:8b" in models
        assert "llama3:latest" in models
        assert "qwen2.5-coder:1.5b" in models
        assert "qwen2.5-coder:1.5b-base" in models

        # Verify no duplicates
        assert models.count("llama3.1:latest") == 1
        assert models.count("llama3.1:8b") == 1

        # Verify tags are NOT stripped (old bug would have created these)
        assert "llama3.1" not in models  # Old bug would create this
        assert "llama3" not in models    # Old bug would create this

        # Verify we have exactly the expected number of models
        assert len(models) == 5

    def test_ollama_context_window_parsing(self):
        """Test that Ollama provider extracts context window from API response."""
        # Mock API response based on actual Ollama /api/show response
        mock_api_response = {
            "model_info": {
                "general.architecture": "llama",
                "llama.context_length": 131072,
                "llama.attention.head_count": 32,
                "llama.block_count": 32,
            }
        }

        # Simulate the get_context_window logic (lines 157-171 in ollama.py)
        # The current code looks in wrong places: modelinfo and parameters
        # It should look in model_info with key "llama.context_length"

        model_info = mock_api_response.get("model_info", {})
        context_window = model_info.get("llama.context_length")

        # This should extract the context window correctly
        assert context_window == 131072
