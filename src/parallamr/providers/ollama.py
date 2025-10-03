"""Ollama API provider implementation."""

import asyncio
import os
from typing import Any, Dict, List, Optional

import aiohttp

from ..models import ProviderResponse
from ..token_counter import estimate_tokens
from .base import (
    AuthenticationError,
    ModelNotAvailableError,
    Provider,
    ProviderError,
    TimeoutError,
)


class OllamaProvider(Provider):
    """Ollama API provider for local LLM models."""

    def __init__(self, base_url: Optional[str] = None, timeout: int = 300):
        """
        Initialize the Ollama provider.

        Args:
            base_url: Ollama server URL (if None, reads from OLLAMA_BASE_URL env var)
            timeout: Request timeout in seconds
        """
        super().__init__(timeout)
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._model_cache: Optional[List[str]] = None

    async def get_completion(
        self,
        prompt: str,
        model: str,
        **kwargs
    ) -> ProviderResponse:
        """
        Get completion from Ollama API.

        Args:
            prompt: Input prompt text
            model: Model identifier
            **kwargs: Additional parameters

        Returns:
            ProviderResponse containing the completion result
        """
        # Check if model is available
        available_models = await self.list_models()
        if model not in available_models:
            return ProviderResponse(
                output="",
                output_tokens=0,
                success=False,
                error_message=f"Model {model} not found or unavailable"
            )

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 404:
                        return ProviderResponse(
                            output="",
                            output_tokens=0,
                            success=False,
                            error_message=f"Model {model} not found on Ollama server"
                        )

                    response.raise_for_status()
                    data = await response.json()

                    output = data.get("response", "")

                    # Ollama doesn't always provide token counts, so we estimate
                    output_tokens = estimate_tokens(output)

                    # Get context window for this model
                    context_window = await self.get_context_window(model)

                    # Check for any errors in the response
                    error_message = None
                    if data.get("error"):
                        error_message = data["error"]

                    return ProviderResponse(
                        output=output,
                        output_tokens=output_tokens,
                        success=not bool(error_message),
                        error_message=error_message,
                        context_window=context_window
                    )

        except asyncio.TimeoutError:
            return ProviderResponse(
                output="",
                output_tokens=0,
                success=False,
                error_message=f"Request timeout after {self.timeout} seconds"
            )
        except aiohttp.ClientConnectorError:
            return ProviderResponse(
                output="",
                output_tokens=0,
                success=False,
                error_message=f"Cannot connect to Ollama server at {self.base_url}"
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
        Get model's context window size from Ollama API.

        Args:
            model: Model identifier

        Returns:
            Context window size in tokens, or None if unknown
        """
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(
                    f"{self.base_url}/api/show",
                    json={"name": model}
                ) as response:
                    if response.status != 200:
                        return None

                    data = await response.json()

                    # Try to extract context window from model info
                    # Ollama may provide this in different formats
                    model_info = data.get("modelinfo", {})

                    # Common places where context window might be specified
                    for key in ["context_length", "max_context_length", "context_window"]:
                        if key in model_info:
                            return model_info[key]

                    # Some models might have it in parameters
                    parameters = data.get("parameters", {})
                    if "num_ctx" in parameters:
                        return parameters["num_ctx"]

                    return None

        except Exception:
            # If we can't get model info, return None
            return None

    async def list_models(self) -> list[str]:
        """
        List available models from Ollama API.

        Returns:
            List of model identifiers
        """
        if self._model_cache is not None:
            return self._model_cache

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    response.raise_for_status()
                    data = await response.json()

                    models = []
                    for model_info in data.get("models", []):
                        model_name = model_info.get("name", "")
                        if model_name:
                            # Remove tag suffix if present (e.g., "llama2:latest" -> "llama2")
                            base_name = model_name.split(":")[0]
                            models.append(base_name)

                    self._model_cache = models
                    return models

        except Exception:
            # If we can't fetch models, return empty list
            return []

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

    async def pull_model(self, model: str) -> bool:
        """
        Pull/download a model to the Ollama server.

        Args:
            model: Model identifier to pull

        Returns:
            True if successful, False otherwise
        """
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
                async with session.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model}
                ) as response:
                    response.raise_for_status()

                    # Clear cache so it will be refreshed
                    self._model_cache = None

                    return True

        except Exception:
            return False