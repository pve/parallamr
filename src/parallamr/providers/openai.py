"""OpenAI API provider implementation."""

import asyncio
import os
from typing import Any, Callable, Dict, List, Optional

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


class OpenAIProvider(Provider):
    """OpenAI API provider for GPT models and compatible endpoints."""

    # Class-level model metadata cache (shared across instances)
    _MODEL_METADATA: Dict[str, Dict[str, Any]] = {
        # GPT-4 Turbo
        "gpt-4-turbo": {"context_length": 128000, "family": "gpt-4"},
        "gpt-4-turbo-preview": {"context_length": 128000, "family": "gpt-4"},
        "gpt-4-0125-preview": {"context_length": 128000, "family": "gpt-4"},
        "gpt-4-1106-preview": {"context_length": 128000, "family": "gpt-4"},
        # GPT-4o (Omni)
        "gpt-4o": {"context_length": 128000, "family": "gpt-4o"},
        "gpt-4o-2024-08-06": {"context_length": 128000, "family": "gpt-4o"},
        "gpt-4o-2024-05-13": {"context_length": 128000, "family": "gpt-4o"},
        "gpt-4o-mini": {"context_length": 128000, "family": "gpt-4o"},
        "gpt-4o-mini-2024-07-18": {"context_length": 128000, "family": "gpt-4o"},
        # GPT-4 (Original)
        "gpt-4": {"context_length": 8192, "family": "gpt-4"},
        "gpt-4-0613": {"context_length": 8192, "family": "gpt-4"},
        "gpt-4-32k": {"context_length": 32768, "family": "gpt-4"},
        "gpt-4-32k-0613": {"context_length": 32768, "family": "gpt-4"},
        # GPT-3.5 Turbo
        "gpt-3.5-turbo": {"context_length": 16385, "family": "gpt-3.5"},
        "gpt-3.5-turbo-0125": {"context_length": 16385, "family": "gpt-3.5"},
        "gpt-3.5-turbo-1106": {"context_length": 16385, "family": "gpt-3.5"},
        "gpt-3.5-turbo-16k": {"context_length": 16385, "family": "gpt-3.5"},
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: int = 300,
        base_url: Optional[str] = None,
        env_getter: Optional[Callable[[str], Optional[str]]] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            organization: Optional OpenAI organization ID (reads from OPENAI_ORG_ID env var)
            timeout: Request timeout in seconds (default: 300)
            base_url: API base URL for OpenAI-compatible endpoints (default: official API)
            env_getter: Function to get env vars (defaults to os.getenv) - for testing
            session: Optional aiohttp.ClientSession for connection reuse
        """
        super().__init__(timeout)

        # Use injected env_getter for testability
        _env_getter = env_getter or os.getenv

        # Authentication configuration
        self.api_key = api_key or _env_getter("OPENAI_API_KEY")
        self.organization = organization or _env_getter("OPENAI_ORG_ID")

        # Endpoint configuration
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")

        # HTTP session management
        self._session = session

        # Runtime caches
        self._model_cache: Optional[List[str]] = None
        self._model_details_cache: Optional[Dict[str, Any]] = None

    async def get_completion(
        self, prompt: str, model: str, **kwargs
    ) -> ProviderResponse:
        """
        Get completion from OpenAI API.

        Args:
            prompt: Input prompt text
            model: Model identifier (e.g., "gpt-4", "gpt-3.5-turbo")
            **kwargs: Additional parameters passed through to API

        Returns:
            ProviderResponse containing the completion result
        """
        # Step 1: Validate configuration (Configuration Failure Mode)
        valid, error = self._validate_configuration()
        if not valid:
            return ProviderResponse(
                output="", output_tokens=0, success=False, error_message=error
            )

        # Step 2: Validate parameters (Configuration Failure Mode)
        valid, error = self._validate_parameters(kwargs)
        if not valid:
            return ProviderResponse(
                output="",
                output_tokens=0,
                success=False,
                error_message=f"Invalid parameters: {error}",
            )

        # Step 3: Validate model availability (Operational Failure Mode - Model)
        if not self.is_model_available(model):
            return ProviderResponse(
                output="",
                output_tokens=0,
                success=False,
                error_message=f"Model {model} not found or unavailable",
            )

        # Step 4: Build request payload
        try:
            payload = self._build_completion_payload(prompt, model, kwargs)
        except Exception as e:
            return ProviderResponse(
                output="",
                output_tokens=0,
                success=False,
                error_message=f"Failed to build request: {str(e)}",
            )

        # Step 5: Make API request with error handling (Operational Failure Mode - Network/API)
        status, data, error = await self._make_request(
            "POST", "/chat/completions", json_data=payload
        )

        # Step 6: Handle errors
        if error or status >= 400:
            return self._map_error_response(status, data, error)

        # Step 7: Transform successful response (Operational Failure Mode - Response Parsing)
        return await self._transform_api_response(data, model)

    async def get_context_window(self, model: str) -> Optional[int]:
        """
        Get model's context window size.
        """
        # First check static metadata
        static_window = self._get_static_context_window(model)
        if static_window is not None:
            return static_window

        # Fall back to API query
        models_info = await self._get_models_info()
        if models_info and model in models_info:
            return models_info[model].get("context_window") or models_info[model].get("context_length")

        return None

    async def list_models(self) -> List[str]:
        """
        List available models from OpenAI API.
        """
        models_info = await self._get_models_info()
        if models_info:
            return list(models_info.keys())

        # Fallback to static metadata if API call fails
        return list(self._MODEL_METADATA.keys())

    def is_model_available(self, model: str) -> bool:
        """
        Check if a model is available (synchronous check using cache).
        """
        if self._model_cache is None:
            return True # Optimistic

        return model in self._model_cache

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers with authentication."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"Parallamr/{__version__} (https://github.com/parallamr/parallamr)"
        }

        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        return headers

    def _build_completion_payload(
        self, prompt: str, model: str, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build OpenAI API request payload."""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        # Map supported parameters
        api_params = [
            "temperature", "max_tokens", "top_p", "frequency_penalty",
            "presence_penalty", "stop", "n", "user", "logit_bias",
            "logprobs", "top_logprobs", "response_format", "seed",
            "tools", "tool_choice",
        ]

        for param in api_params:
            if param in kwargs:
                payload[param] = kwargs[param]

        return payload

    async def _transform_api_response(
        self, api_response: Dict[str, Any], model: str
    ) -> ProviderResponse:
        """Transform OpenAI API response to ProviderResponse."""
        try:
            # Extract completion content
            if "choices" not in api_response or not api_response["choices"]:
                 return ProviderResponse(
                    output="",
                    output_tokens=0,
                    success=False,
                    error_message="Malformed API response: missing 'choices'",
                )

            choice = api_response["choices"][0]
            if "message" not in choice or "content" not in choice["message"]:
                 return ProviderResponse(
                    output="",
                    output_tokens=0,
                    success=False,
                    error_message="Malformed API response: missing 'message' or 'content'",
                )

            output = choice["message"]["content"]

            # Extract token usage
            usage = api_response.get("usage", {})
            output_tokens = usage.get("completion_tokens", estimate_tokens(output))

            # Get context window for this model
            context_window = await self.get_context_window(model)

            return ProviderResponse(
                output=output,
                output_tokens=output_tokens,
                success=True,
                context_window=context_window,
            )

        except (KeyError, IndexError, TypeError) as e:
            return ProviderResponse(
                output="",
                output_tokens=0,
                success=False,
                error_message=f"Malformed API response: {str(e)}",
            )

    def _map_error_response(
        self, status: int, error_data: Optional[Dict], error_msg: Optional[str]
    ) -> ProviderResponse:
        """Map API errors to ProviderResponse."""
        if error_data and "error" in error_data:
            error_type = error_data["error"].get("type", "unknown_error")
            error_message = error_data["error"].get("message", "Unknown error")
            error_code = error_data["error"].get("code", None)
        else:
            error_type = "unknown_error"
            error_message = error_msg or f"HTTP {status} error"
            error_code = None

        if status == 401:
            message = "Authentication failed - invalid API key"
        elif status == 403:
            message = "Access forbidden - check API key permissions"
        elif status == 404:
            message = "Model or endpoint not found"
        elif status == 429:
            message = "Rate limit exceeded - please wait and retry"
        elif status == 413:
            message = "Request too large - input exceeds model context window"
        elif status >= 500:
            message = "OpenAI server error - please retry"
        else:
            message = f"{error_type}: {error_message}"

        if error_code:
            message = f"{message} (code: {error_code})"

        return ProviderResponse(
            output="", output_tokens=0, success=False, error_message=message
        )

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict] = None,
        timeout_override: Optional[int] = None,
    ) -> tuple[int, Optional[Dict], Optional[str]]:
        """Make authenticated HTTP request to OpenAI API."""
        url = f"{self.base_url}{endpoint}"
        headers = self._build_headers()
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

        if isinstance(data, dict) and "error" in data:
            error_msg = data["error"].get("message", "Unknown error")
            error_type = data["error"].get("type", "unknown")
            return status, data, f"{error_type}: {error_msg}"

        if status == 200:
            return status, data, None

        return status, data, f"HTTP {status} error"

    def _validate_configuration(self) -> tuple[bool, Optional[str]]:
        """Validate provider configuration."""
        if not self.api_key:
            return False, "OpenAI API key not provided. Set OPENAI_API_KEY or pass api_key parameter."
        if not self.base_url:
            return False, "Base URL is required."
        return True, None

    def _validate_parameters(self, kwargs: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate API parameters."""
        if "temperature" in kwargs:
            temp = kwargs["temperature"]
            if not isinstance(temp, (int, float)) or not (0.0 <= temp <= 2.0):
                return False, "temperature must be between 0.0 and 2.0"

        if "max_tokens" in kwargs:
            max_tok = kwargs["max_tokens"]
            if not isinstance(max_tok, int) or max_tok < 1:
                return False, "max_tokens must be a positive integer"

        if "top_p" in kwargs:
            top_p = kwargs["top_p"]
            if not isinstance(top_p, (int, float)) or not (0.0 <= top_p <= 1.0):
                return False, "top_p must be between 0.0 and 1.0"

        return True, None

    def _get_static_context_window(self, model: str) -> Optional[int]:
        """Get context window from static metadata."""
        metadata = self._MODEL_METADATA.get(model)
        if metadata:
            return metadata.get("context_length")
        return None

    async def _get_models_info(self) -> Optional[Dict[str, Any]]:
        """Fetch and cache models information from OpenAI API."""
        if self._model_details_cache is not None:
            return self._model_details_cache

        valid, error = self._validate_configuration()
        if not valid:
            return None

        try:
            status, data, error = await self._make_request(
                "GET", "/models", timeout_override=30
            )

            if status == 200 and data:
                models_dict = {}
                for model_info in data.get("data", []):
                    model_id = model_info.get("id")
                    if model_id:
                        models_dict[model_id] = model_info

                self._model_details_cache = models_dict
                self._model_cache = list(models_dict.keys())
                return self._model_details_cache

        except Exception:
            pass

        return None
