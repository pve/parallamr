"""Base provider interface for LLM services."""

from abc import ABC, abstractmethod
from typing import Optional

from ..models import ProviderResponse


class Provider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, timeout: int = 300):
        """
        Initialize the provider.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout

    @abstractmethod
    async def get_completion(
        self,
        prompt: str,
        model: str,
        **kwargs
    ) -> ProviderResponse:
        """
        Get completion from the provider.

        Args:
            prompt: Input prompt text
            model: Model identifier
            **kwargs: Additional provider-specific parameters

        Returns:
            ProviderResponse containing the completion result
        """
        pass

    @abstractmethod
    async def get_context_window(self, model: str) -> Optional[int]:
        """
        Get model's context window size.

        Args:
            model: Model identifier

        Returns:
            Context window size in tokens, or None if unknown
        """
        pass

    @abstractmethod
    async def list_models(self) -> list[str]:
        """
        List available models for this provider.

        Returns:
            List of model identifiers
        """
        pass

    @abstractmethod
    def is_model_available(self, model: str) -> bool:
        """
        Check if a model is available for this provider.

        Args:
            model: Model identifier

        Returns:
            True if model is available, False otherwise
        """
        pass

    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        return self.__class__.__name__.replace("Provider", "").lower()


class ProviderError(Exception):
    """Base exception for provider-related errors."""
    pass


class ModelNotAvailableError(ProviderError):
    """Raised when a requested model is not available."""
    pass


class AuthenticationError(ProviderError):
    """Raised when authentication fails."""
    pass


class RateLimitError(ProviderError):
    """Raised when rate limits are exceeded."""
    pass


class TimeoutError(ProviderError):
    """Raised when requests timeout."""
    pass


class ContextWindowExceededError(ProviderError):
    """Raised when input exceeds model's context window."""
    pass