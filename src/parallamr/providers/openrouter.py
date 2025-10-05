"""OpenRouter API provider implementation."""

import asyncio
import os
from typing import Any, Callable, Dict, Optional

import aiohttp

from ..models import ProviderResponse
from ..token_counter import estimate_tokens
from .base import (
    AuthenticationError,
    ContextWindowExceededError,
    ModelNotAvailableError,
    Provider,
    ProviderError,
    RateLimitError,
    TimeoutError,
)


class OpenRouterProvider(Provider):
    """OpenRouter API provider for accessing multiple LLM models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 300,
        base_url: Optional[str] = None,
        env_getter: Optional[Callable[[str], Optional[str]]] = None,
        session: Optional[aiohttp.ClientSession] = None
    ):
        """
        Initialize the OpenRouter provider.

        Args:
            api_key: OpenRouter API key (if None, reads from OPENROUTER_API_KEY env var)
            timeout: Request timeout in seconds
            base_url: API base URL (for testing with mock servers)
            env_getter: Function to get env vars (defaults to os.getenv)
            session: Optional aiohttp.ClientSession for connection reuse (defaults to None)
        """
        super().__init__(timeout)

        # Use injected env_getter for testability
        _env_getter = env_getter or os.getenv
        self.api_key = api_key or _env_getter("OPENROUTER_API_KEY")

        self.base_url = base_url or "https://openrouter.ai/api/v1"
        self._session = session
        self._model_cache: Optional[Dict[str, Any]] = None

    async def get_completion(
        self,
        prompt: str,
        model: str,
        **kwargs
    ) -> ProviderResponse:
        """
        Get completion from OpenRouter API.

        Args:
            prompt: Input prompt text
            model: Model identifier
            **kwargs: Additional parameters

        Returns:
            ProviderResponse containing the completion result
        """
        if not self.api_key:
            return ProviderResponse(
                output="",
                output_tokens=0,
                success=False,
                error_message="OpenRouter API key not provided"
            )

        if not self.is_model_available(model):
            return ProviderResponse(
                output="",
                output_tokens=0,
                success=False,
                error_message=f"Model {model} not found or unavailable"
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/parallamr/parallamr",
            "X-Title": "Parallamr"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            **kwargs
        }

        try:
            # Use injected session if available, otherwise create temporary session
            if self._session:
                async with self._session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 401:
                        return ProviderResponse(
                            output="",
                            output_tokens=0,
                            success=False,
                            error_message="Authentication failed - invalid API key"
                        )
                    elif response.status == 429:
                        return ProviderResponse(
                            output="",
                            output_tokens=0,
                            success=False,
                            error_message="Rate limit exceeded"
                        )
                    elif response.status == 413:
                        return ProviderResponse(
                            output="",
                            output_tokens=0,
                            success=False,
                            error_message="Request too large - input exceeds model context window"
                        )

                    response.raise_for_status()
                    data = await response.json()

                    # Extract response content
                    output = data["choices"][0]["message"]["content"]
                    output_tokens = data.get("usage", {}).get("completion_tokens", estimate_tokens(output))

                    # Get context window for this model
                    context_window = await self.get_context_window(model)

                    return ProviderResponse(
                        output=output,
                        output_tokens=output_tokens,
                        success=True,
                        context_window=context_window
                    )
            else:
                # No injected session - use temporary session (backward compatibility)
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload
                    ) as response:
                        if response.status == 401:
                            return ProviderResponse(
                                output="",
                                output_tokens=0,
                                success=False,
                                error_message="Authentication failed - invalid API key"
                            )
                        elif response.status == 429:
                            return ProviderResponse(
                                output="",
                                output_tokens=0,
                                success=False,
                                error_message="Rate limit exceeded"
                            )
                        elif response.status == 413:
                            return ProviderResponse(
                                output="",
                                output_tokens=0,
                                success=False,
                                error_message="Request too large - input exceeds model context window"
                            )

                        response.raise_for_status()
                        data = await response.json()

                        # Extract response content
                        output = data["choices"][0]["message"]["content"]
                        output_tokens = data.get("usage", {}).get("completion_tokens", estimate_tokens(output))

                        # Get context window for this model
                        context_window = await self.get_context_window(model)

                        return ProviderResponse(
                            output=output,
                            output_tokens=output_tokens,
                            success=True,
                            context_window=context_window
                        )

        except asyncio.TimeoutError:
            return ProviderResponse(
                output="",
                output_tokens=0,
                success=False,
                error_message=f"Request timeout after {self.timeout} seconds"
            )
        except aiohttp.ClientError as e:
            return ProviderResponse(
                output="",
                output_tokens=0,
                success=False,
                error_message=f"Network error: {str(e)}"
            )
        except Exception as e:
            return ProviderResponse(
                output="",
                output_tokens=0,
                success=False,
                error_message=f"Unexpected error: {str(e)}"
            )

    async def get_context_window(self, model: str) -> Optional[int]:
        """
        Get model's context window size from OpenRouter API.

        Args:
            model: Model identifier

        Returns:
            Context window size in tokens, or None if unknown
        """
        models_info = await self._get_models_info()
        if models_info and model in models_info:
            return models_info[model].get("context_length")
        return None

    async def list_models(self) -> list[str]:
        """
        List available models from OpenRouter API.

        Returns:
            List of model identifiers
        """
        models_info = await self._get_models_info()
        return list(models_info.keys()) if models_info else []

    def is_model_available(self, model: str) -> bool:
        """
        Check if a model is available (synchronous check using cache).

        Args:
            model: Model identifier

        Returns:
            True if model is available, False otherwise
        """
        # For synchronous check, we rely on cached data
        # If cache is empty, we assume the model might be available
        if self._model_cache is None:
            return True  # Optimistic - will be validated in get_completion

        return model in self._model_cache

    async def _get_models_info(self) -> Optional[Dict[str, Any]]:
        """
        Fetch and cache models information from OpenRouter API.

        Returns:
            Dictionary mapping model names to their info, or None on error
        """
        if self._model_cache is not None:
            return self._model_cache

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            # Use injected session if available, otherwise create temporary session
            if self._session:
                async with self._session.get(
                    f"{self.base_url}/models",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

                    # Convert list of models to dictionary
                    models_dict = {}
                    for model_info in data.get("data", []):
                        model_id = model_info.get("id")
                        if model_id:
                            models_dict[model_id] = model_info

                    self._model_cache = models_dict
                    return self._model_cache
            else:
                # No injected session - use temporary session
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                    async with session.get(
                        f"{self.base_url}/models",
                        headers=headers
                    ) as response:
                        response.raise_for_status()
                        data = await response.json()

                        # Convert list of models to dictionary
                        models_dict = {}
                        for model_info in data.get("data", []):
                            model_id = model_info.get("id")
                            if model_id:
                                models_dict[model_id] = model_info

                        self._model_cache = models_dict
                        return self._model_cache

        except Exception:
            # If we can't fetch models, return None and let individual requests handle errors
            return None