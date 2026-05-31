"""OpenRouter API provider implementation."""

import asyncio
import os
from typing import Any, Callable, Dict, Optional, List

import aiohttp

from .. import __version__
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

        self.base_url = (base_url or "https://openrouter.ai/api/v1").rstrip("/")
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
        # Step 1: Validate configuration
        valid, error = self._validate_configuration()
        if not valid:
            return ProviderResponse(
                output="", output_tokens=0, success=False, error_message=error
            )

        # Step 2: Check if model is available (optimistic if cache empty)
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
            "X-Title": "Parallamr",
            "User-Agent": f"Parallamr/{__version__} (https://github.com/parallamr/parallamr)"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            **kwargs
        }

        # Step 3: Make API request with error handling
        status, data, error = await self._make_request(
            "POST", "/chat/completions", headers=headers, json_data=payload
        )

        # Step 4: Handle errors
        if error or status >= 400:
            return self._map_error_response(status, data, error)

        # Step 5: Transform successful response
        return await self._transform_api_response(data, model)

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict] = None,
        timeout_override: Optional[int] = None,
    ) -> tuple[int, Optional[Dict], Optional[str]]:
        """Make authenticated HTTP request to OpenRouter API."""
        url = f"{self.base_url}{endpoint}"
        timeout = timeout_override or self.timeout

        try:
            if self._session:
                async with self._session.request(
                    method,
                    url,
                    headers=headers,
                    json=json_data,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    return await self._process_response(response)
            else:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as session:
                    async with session.request(
                        method, url, headers=headers, json=json_data
                    ) as response:
                        return await self._process_response(response)

        except asyncio.TimeoutError:
            return 0, None, f"Request timeout after {timeout} seconds"
        except aiohttp.ClientError as e:
            return 0, None, f"Network error: {str(e)}"
        except Exception as e:
            return 0, None, f"Unexpected error: {str(e)}"

    async def _process_response(
        self, response: aiohttp.ClientResponse
    ) -> tuple[int, Optional[Dict], Optional[str]]:
        """Process HTTP response and extract data."""
        status = response.status

        try:
            data = await response.json()
        except Exception:
            try:
                text = await response.text()
                return status, None, f"Invalid JSON response: {text[:200]}"
            except Exception:
                return status, None, "Could not read response body"

        # Check for API errors in response body
        if isinstance(data, dict) and "error" in data:
            error_msg = data["error"].get("message", "Unknown error")
            return status, data, error_msg

        if status == 200:
            return status, data, None

        return status, data, f"HTTP {status} error"

    def _map_error_response(
        self, status: int, error_data: Optional[Dict], error_msg: Optional[str]
    ) -> ProviderResponse:
        """Map API errors to ProviderResponse."""
        if status == 401:
            message = "Authentication failed - invalid API key"
        elif status == 429:
            # Check for specific credit exhaustion vs rate limit
            message = "Rate limit exceeded"
            if error_data and "error" in error_data:
                msg = error_data["error"].get("message", "").lower()
                if "credits" in msg or "insufficient" in msg:
                    message = "OpenRouter credits exhausted"
                elif "rate" in msg:
                    message = "Rate limit exceeded"
        elif status == 413:
            message = "Request too large - input exceeds model context window"
        elif status == 403:
            message = "Access forbidden - check API key permissions"
        elif status == 404:
            message = "Model or endpoint not found"
        elif status >= 500:
            message = f"OpenRouter server error: {error_msg}"
        else:
            message = error_msg or f"HTTP {status} error"

        return ProviderResponse(
            output="", output_tokens=0, success=False, error_message=message
        )

    async def _transform_api_response(
        self, data: Dict[str, Any], model: str
    ) -> ProviderResponse:
        """Transform OpenRouter API response to ProviderResponse."""
        try:
            if "choices" not in data or not data["choices"]:
                return ProviderResponse(
                    output="",
                    output_tokens=0,
                    success=False,
                    error_message="Malformed API response: missing 'choices'"
                )

            choice = data["choices"][0]
            if "message" not in choice or "content" not in choice["message"]:
                 return ProviderResponse(
                    output="",
                    output_tokens=0,
                    success=False,
                    error_message="Malformed API response: missing 'message' or 'content'"
                )

            output = choice["message"]["content"]
            output_tokens = data.get("usage", {}).get("completion_tokens", estimate_tokens(output))
            context_window = await self.get_context_window(model)

            return ProviderResponse(
                output=output,
                output_tokens=output_tokens,
                success=True,
                context_window=context_window
            )
        except (KeyError, IndexError, TypeError) as e:
            return ProviderResponse(
                output="",
                output_tokens=0,
                success=False,
                error_message=f"Malformed API response: {str(e)}"
            )

    async def get_context_window(self, model: str) -> Optional[int]:
        """Get model's context window size from OpenRouter API."""
        models_info = await self._get_models_info()
        if models_info and model in models_info:
            return models_info[model].get("context_length")
        return None

    async def list_models(self) -> list[str]:
        """List available models from OpenRouter API."""
        models_info = await self._get_models_info()
        return list(models_info.keys()) if models_info else []

    def is_model_available(self, model: str) -> bool:
        """Check if a model is available (synchronous check using cache)."""
        if self._model_cache is None:
            return True  # Optimistic

        return model in self._model_cache

    async def _get_models_info(self) -> Optional[Dict[str, Any]]:
        """Fetch and cache models information from OpenRouter API."""
        if self._model_cache is not None:
            return self._model_cache

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        status, data, error = await self._make_request(
            "GET", "/models", headers=headers, timeout_override=30
        )

        if status == 200 and data:
            models_dict = {}
            for model_info in data.get("data", []):
                model_id = model_info.get("id")
                if model_id:
                    models_dict[model_id] = model_info

            self._model_cache = models_dict
            return self._model_cache

        return None

    def _validate_configuration(self) -> tuple[bool, Optional[str]]:
        """Validate provider configuration."""
        if not self.api_key:
            return False, "OpenRouter API key not provided"
        if not self.base_url:
            return False, "Base URL is required."
        return True, None
