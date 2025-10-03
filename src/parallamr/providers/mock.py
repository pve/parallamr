"""Mock provider for testing purposes."""

import json
from typing import Optional

from ..models import ProviderResponse
from ..token_counter import estimate_tokens
from .base import Provider


class MockProvider(Provider):
    """Mock provider that returns formatted test responses."""

    def __init__(self, timeout: int = 300):
        """Initialize the mock provider."""
        super().__init__(timeout)

    async def get_completion(
        self,
        prompt: str,
        model: str,
        **kwargs
    ) -> ProviderResponse:
        """
        Get a mock completion response.

        Args:
            prompt: Input prompt text
            model: Model identifier (should be "mock")
            **kwargs: Additional parameters (included in response)

        Returns:
            ProviderResponse with formatted mock data
        """
        input_tokens = estimate_tokens(prompt)

        # Extract variables from kwargs if present
        variables = kwargs.get("variables", {})

        # Create mock response with metadata
        mock_output = (
            "MOCK RESPONSE\n"
            f"Input tokens: {input_tokens}\n"
            f"Model: {model}\n"
            f"Variables: {json.dumps(variables, indent=2)}\n"
            "--- Original Input ---\n"
            f"{prompt}"
        )

        output_tokens = estimate_tokens(mock_output)

        return ProviderResponse(
            output=mock_output,
            output_tokens=output_tokens,
            success=True,
            context_window=None  # Mock provider doesn't specify context window
        )

    async def get_context_window(self, model: str) -> Optional[int]:
        """
        Get context window for mock model.

        Args:
            model: Model identifier

        Returns:
            None (mock provider doesn't specify context window)
        """
        return None

    async def list_models(self) -> list[str]:
        """
        List available mock models.

        Returns:
            List containing only "mock"
        """
        return ["mock"]

    def is_model_available(self, model: str) -> bool:
        """
        Check if model is available.

        Args:
            model: Model identifier

        Returns:
            True only if model is "mock"
        """
        return model == "mock"