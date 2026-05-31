"""Ollama API provider implementation."""

import asyncio
import os
from typing import Any, Callable, Dict, List, Optional

import aiohttp

from .. import __version__
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

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 300,
        env_getter: Optional[Callable[[str, str], Optional[str]]] = None,
        session: Optional[aiohttp.ClientSession] = None
    ):
        """
        Initialize the Ollama provider.

        Args:
            base_url: Ollama server URL (if None, reads from OLLAMA_BASE_URL env var)
            timeout: Request timeout in seconds
            env_getter: Function to get env vars with default (defaults to os.getenv)
            session: Optional aiohttp.ClientSession for connection reuse (defaults to None)
        """
        super().__init__(timeout)

        # Use injected env_getter for testability
        _env_getter = env_getter or os.getenv
        raw_url = base_url or _env_getter("OLLAMA_BASE_URL", "http://localhost:11434")
        self.base_url = raw_url.rstrip("/")

        self._session = session
        self._model_cache: Optional[List[str]] = None

    async def get_completion(
        self,
        prompt: str,
        model: str,
        **kwargs
    ) -> ProviderResponse:
        """
        Get completion from Ollama API.
        """
        # Step 1: Validate configuration
        valid, error = self._validate_configuration()
        if not valid:
            return ProviderResponse(
                output="", output_tokens=0, success=False, error_message=error
            )

        # Step 2: Check if model is available
        if not self.is_model_available(model):
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

        # Step 3: Make API request
        status, data, error = await self._make_request(
            "POST", "/api/generate", json_data=payload
        )

        # Step 4: Handle errors
        if error or status >= 400:
            return self._map_error_response(status, data, error, model)

        # Step 5: Transform successful response
        return await self._transform_api_response(data, model)

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict] = None,
        timeout_override: Optional[int] = None,
    ) -> tuple[int, Optional[Dict], Optional[str]]:
        """Make authenticated HTTP request to Ollama API."""
        url = f"{self.base_url}{endpoint}"
        timeout = timeout_override or self.timeout

        headers = {
            "User-Agent": f"Parallamr/{__version__} (https://github.com/parallamr/parallamr)",
            "Content-Type": "application/json"
        }

        try:
            if self._session:
                async with self._session.request(
                    method,
                    url,
                    json=json_data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    return await self._process_response(response)
            else:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as session:
                    async with session.request(
                        method, url, json=json_data, headers=headers
                    ) as response:
                        return await self._process_response(response)

        except asyncio.TimeoutError:
            return 0, None, f"Request timeout after {timeout} seconds"
        except aiohttp.ClientConnectorError:
            return 0, None, f"Cannot connect to Ollama server at {self.base_url}"
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

        if isinstance(data, dict) and "error" in data:
            return status, data, data["error"]

        if isinstance(data, dict) and "message" in data and status >= 400:
            return status, data, data["message"]

        if status == 200:
            return status, data, None

        return status, data, f"HTTP {status} error"

    def _map_error_response(
        self, status: int, error_data: Optional[Dict], error_msg: Optional[str], model: str
    ) -> ProviderResponse:
        """Map API errors to ProviderResponse."""
        if status == 404:
            message = f"Model {model} not found on Ollama server"
        elif status == 400:
            message = f"Invalid request to Ollama: {error_msg}"
        elif status >= 500:
            message = f"Ollama server error: {error_msg}"
        else:
            message = error_msg or f"HTTP {status} error"

        return ProviderResponse(
            output="", output_tokens=0, success=False, error_message=message
        )

    async def _transform_api_response(
        self, data: Dict[str, Any], model: str
    ) -> ProviderResponse:
        """Transform Ollama API response to ProviderResponse."""
        try:
            output = data.get("response", "")
            output_tokens = estimate_tokens(output)
            context_window = await self.get_context_window(model)

            return ProviderResponse(
                output=output,
                output_tokens=output_tokens,
                success=True,
                context_window=context_window
            )
        except Exception as e:
            return ProviderResponse(
                output="",
                output_tokens=0,
                success=False,
                error_message=f"Malformed API response: {str(e)}"
            )

    async def get_context_window(self, model: str) -> Optional[int]:
        """Get model's context window size from Ollama API."""
        try:
            status, data, error = await self._make_request(
                "POST", "/api/show", json_data={"name": model}, timeout_override=30
            )

            if status != 200 or not data:
                return None

            model_info = data.get("model_info", {})
            for key in ["llama.context_length", "context_length", "context_window"]:
                if key in model_info:
                    return model_info[key]

            for key, value in model_info.items():
                if "context_length" in key:
                    return value

            return None
        except Exception:
            return None

    async def list_models(self) -> list[str]:
        """List available models from Ollama API."""
        if self._model_cache is not None:
            return self._model_cache

        status, data, error = await self._make_request(
            "GET", "/api/tags", timeout_override=30
        )

        if status == 200 and data:
            models = []
            for model_info in data.get("models", []):
                model_name = model_info.get("name", "")
                if model_name:
                    models.append(model_name)

            self._model_cache = models
            return models

        return []

    def is_model_available(self, model: str) -> bool:
        """Check if a model is available (synchronous check using cache)."""
        if self._model_cache is None:
            return True  # Optimistic

        return model in self._model_cache

    async def pull_model(self, model: str) -> bool:
        """Pull/download a model to the Ollama server."""
        status, data, error = await self._make_request(
            "POST", "/api/pull", json_data={"name": model}, timeout_override=600
        )

        if status == 200:
            self._model_cache = None  # Clear cache
            return True

        return False

    def _validate_configuration(self) -> tuple[bool, Optional[str]]:
        """Validate provider configuration."""
        if not self.base_url:
            return False, "Ollama base URL not provided."
        if not (self.base_url.startswith("http://") or self.base_url.startswith("https://")):
            return False, f"Invalid Ollama base URL: {self.base_url}"
        return True, None
