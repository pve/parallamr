"""OpenAI API provider implementation."""

import asyncio
import os
from typing import Any, Callable, Dict, List, Optional

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
        self.base_url = base_url or "https://api.openai.com/v1"

        # HTTP session management
        self._session = session

        # Runtime caches
        self._models_cache: Optional[List[str]] = None
        self._model_details_cache: Optional[Dict[str, Any]] = None

    async def get_completion(
        self, prompt: str, model: str, **kwargs
    ) -> ProviderResponse:
        """
        Get completion from OpenAI API.

        Args:
            prompt: Input prompt text
            model: Model identifier (e.g., "gpt-4", "gpt-3.5-turbo")
            **kwargs: Additional parameters passed through to API:
                - temperature: Sampling temperature (0.0 to 2.0)
                - max_tokens: Maximum tokens to generate
                - top_p: Nucleus sampling parameter
                - frequency_penalty: Frequency penalty (-2.0 to 2.0)
                - presence_penalty: Presence penalty (-2.0 to 2.0)
                - stop: Stop sequences (string or list)
                - n: Number of completions to generate
                - user: End-user identifier for abuse monitoring

        Returns:
            ProviderResponse containing the completion result
        """
        # Step 1: Validate configuration
        valid, error = self._validate_configuration()
        if not valid:
            return ProviderResponse(
                output="", output_tokens=0, success=False, error_message=error
            )

        # Step 2: Validate parameters
        valid, error = self._validate_parameters(kwargs)
        if not valid:
            return ProviderResponse(
                output="",
                output_tokens=0,
                success=False,
                error_message=f"Invalid parameters: {error}",
            )

        # Step 3: Build request payload
        try:
            payload = self._build_completion_payload(prompt, model, kwargs)
        except Exception as e:
            return ProviderResponse(
                output="",
                output_tokens=0,
                success=False,
                error_message=f"Failed to build request: {str(e)}",
            )

        # Step 4: Make API request with error handling
        status, data, error = await self._make_request(
            "POST", "/chat/completions", json_data=payload
        )

        # Step 5: Handle errors
        if error or status >= 400:
            return self._map_error_response(status, data, error)

        # Step 6: Transform successful response
        return self._transform_api_response(data, model)

    async def get_context_window(self, model: str) -> Optional[int]:
        """
        Get model's context window size.

        Uses static metadata first, falls back to API query for unknown models.

        Args:
            model: Model identifier

        Returns:
            Context window size in tokens, or None if unknown
        """
        # First check static metadata
        static_window = self._get_static_context_window(model)
        if static_window is not None:
            return static_window

        # Fall back to API query
        models_info = await self._get_models_info()
        if models_info and model in models_info:
            return models_info[model].get("context_window")

        return None

    async def list_models(self) -> List[str]:
        """
        List available models from OpenAI API.

        Returns:
            List of model identifiers
        """
        models_info = await self._get_models_info()
        if models_info:
            return list(models_info.keys())

        # Fallback to static metadata if API call fails
        return list(self._MODEL_METADATA.keys())

    def is_model_available(self, model: str) -> bool:
        """
        Check if a model is available (synchronous check using cache).

        Args:
            model: Model identifier

        Returns:
            True if model is available, False otherwise
        """
        # For synchronous check, we rely on cached data
        # If cache is empty, check static metadata or assume available
        if self._models_cache is None:
            return model in self._MODEL_METADATA or True

        return model in self._models_cache

    def _build_headers(self) -> Dict[str, str]:
        """
        Build request headers with authentication.

        Returns:
            Dictionary of HTTP headers
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Add organization header if configured
        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        # Add user agent for tracking
        headers["User-Agent"] = "Parallamr/0.6.0 (https://github.com/parallamr/parallamr)"

        return headers

    def _build_completion_payload(
        self, prompt: str, model: str, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build OpenAI API request payload with parameter mapping.

        Args:
            prompt: User prompt text
            model: Model identifier
            kwargs: Additional parameters from experiments CSV

        Returns:
            API request payload dictionary
        """
        # Base payload with messages format
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Map supported parameters (filter out non-API params)
        api_params = [
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "n",
            "user",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "response_format",
            "seed",
            "tools",
            "tool_choice",
        ]

        for param in api_params:
            if param in kwargs:
                payload[param] = kwargs[param]

        # Always disable streaming in non-streaming mode
        payload["stream"] = False

        return payload

    def _transform_api_response(
        self, api_response: Dict[str, Any], model: str
    ) -> ProviderResponse:
        """
        Transform OpenAI API response to ProviderResponse.

        Args:
            api_response: Raw API response dictionary
            model: Model identifier

        Returns:
            Standardized ProviderResponse
        """
        try:
            # Extract completion content
            output = api_response["choices"][0]["message"]["content"]

            # Extract token usage
            usage = api_response.get("usage", {})
            output_tokens = usage.get("completion_tokens", estimate_tokens(output))

            # Get context window for this model
            context_window = self._get_static_context_window(model)

            return ProviderResponse(
                output=output,
                output_tokens=output_tokens,
                success=True,
                context_window=context_window,
            )

        except (KeyError, IndexError) as e:
            # Malformed API response
            return ProviderResponse(
                output="",
                output_tokens=0,
                success=False,
                error_message=f"Malformed API response: {str(e)}",
            )

    def _map_error_response(
        self, status: int, error_data: Optional[Dict], error_msg: Optional[str]
    ) -> ProviderResponse:
        """
        Map API errors to ProviderResponse.

        Args:
            status: HTTP status code
            error_data: Parsed error response
            error_msg: Direct error message

        Returns:
            ProviderResponse with appropriate error message
        """
        # Extract error details
        if error_data and "error" in error_data:
            error_type = error_data["error"].get("type", "unknown_error")
            error_message = error_data["error"].get("message", "Unknown error")
            error_code = error_data["error"].get("code", None)
        else:
            error_type = "unknown_error"
            error_message = error_msg or f"HTTP {status} error"
            error_code = None

        # Map status codes to user-friendly messages
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
        elif status == 500:
            message = "OpenAI server error - please retry"
        elif status == 503:
            message = "OpenAI service unavailable - server overloaded"
        else:
            message = f"{error_type}: {error_message}"

        # Add error code if present
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
        """
        Make authenticated HTTP request to OpenAI API.

        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint path (e.g., "/chat/completions")
            json_data: Request body data
            timeout_override: Override default timeout

        Returns:
            Tuple of (status_code, response_data, error_message)
        """
        url = f"{self.base_url}{endpoint}"
        headers = self._build_headers()
        timeout = timeout_override or self.timeout

        try:
            # Use injected session if available, otherwise create temporary session
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
                # No injected session - use temporary session (backward compatibility)
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
        """
        Process HTTP response and extract data.

        Args:
            response: aiohttp response object

        Returns:
            Tuple of (status_code, response_data, error_message)
        """
        status = response.status

        try:
            data = await response.json()
        except Exception:
            # Failed to parse JSON
            text = await response.text()
            return status, None, f"Invalid JSON response: {text[:200]}"

        # Check for API errors in response body
        if "error" in data:
            error_msg = data["error"].get("message", "Unknown error")
            error_type = data["error"].get("type", "unknown")
            return status, data, f"{error_type}: {error_msg}"

        # For successful responses
        if status == 200:
            return status, data, None

        # For error status codes without error field
        return status, data, f"HTTP {status} error"

    def _validate_configuration(self) -> tuple[bool, Optional[str]]:
        """
        Validate provider configuration.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.api_key:
            return (
                False,
                "OpenAI API key not provided. Set OPENAI_API_KEY or pass api_key parameter.",
            )

        if not self.base_url:
            return False, "Base URL is required."

        return True, None

    def _validate_parameters(self, kwargs: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate API parameters.

        Args:
            kwargs: Parameters to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Temperature validation
        if "temperature" in kwargs:
            temp = kwargs["temperature"]
            if not isinstance(temp, (int, float)) or not (0.0 <= temp <= 2.0):
                return False, "temperature must be between 0.0 and 2.0"

        # Max tokens validation
        if "max_tokens" in kwargs:
            max_tok = kwargs["max_tokens"]
            if not isinstance(max_tok, int) or max_tok < 1:
                return False, "max_tokens must be a positive integer"

        # Top_p validation
        if "top_p" in kwargs:
            top_p = kwargs["top_p"]
            if not isinstance(top_p, (int, float)) or not (0.0 <= top_p <= 1.0):
                return False, "top_p must be between 0.0 and 1.0"

        # Frequency penalty validation
        if "frequency_penalty" in kwargs:
            freq_pen = kwargs["frequency_penalty"]
            if not isinstance(freq_pen, (int, float)) or not (-2.0 <= freq_pen <= 2.0):
                return False, "frequency_penalty must be between -2.0 and 2.0"

        # Presence penalty validation
        if "presence_penalty" in kwargs:
            pres_pen = kwargs["presence_penalty"]
            if not isinstance(pres_pen, (int, float)) or not (-2.0 <= pres_pen <= 2.0):
                return False, "presence_penalty must be between -2.0 and 2.0"

        return True, None

    def _get_static_context_window(self, model: str) -> Optional[int]:
        """
        Get context window from static metadata.

        Args:
            model: Model identifier

        Returns:
            Context window size in tokens, or None if not in metadata
        """
        metadata = self._MODEL_METADATA.get(model)
        if metadata:
            return metadata.get("context_length")
        return None

    async def _get_models_info(self) -> Optional[Dict[str, Any]]:
        """
        Fetch and cache models information from OpenAI API.

        Returns:
            Dictionary mapping model names to their info, or None on error
        """
        if self._model_details_cache is not None:
            return self._model_details_cache

        # Validate configuration first
        valid, error = self._validate_configuration()
        if not valid:
            return None

        try:
            status, data, error = await self._make_request(
                "GET", "/models", timeout_override=30
            )

            if status == 200 and data:
                # Convert list of models to dictionary
                models_dict = {}
                for model_info in data.get("data", []):
                    model_id = model_info.get("id")
                    if model_id:
                        models_dict[model_id] = model_info

                self._model_details_cache = models_dict
                self._models_cache = list(models_dict.keys())
                return self._model_details_cache

        except Exception:
            # If we can't fetch models, return None and let individual requests handle errors
            pass

        return None
