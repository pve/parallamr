# OpenAI Provider Architecture Design

## Executive Summary

This document specifies the technical architecture for integrating OpenAI and OpenAI-compatible providers into the Parallamr experiment framework. The design follows established patterns from existing providers (OpenRouter, Ollama) while addressing OpenAI-specific requirements including streaming, function calling, and advanced parameter mapping.

**Status:** Architecture Design Phase
**Version:** 1.0
**Date:** 2025-10-07
**Author:** HiveMind Swarm CODER Worker

---

## 1. System Analysis

### 1.1 Existing Architecture Patterns

Based on analysis of the codebase (`/workspaces/parallamr/src/parallamr/providers/`), the current provider architecture follows these patterns:

**Base Provider Interface** (`base.py`):
- Abstract base class `Provider` with standard methods
- Required methods: `get_completion()`, `get_context_window()`, `list_models()`, `is_model_available()`
- Timeout configuration in constructor
- Hierarchical exception system (ProviderError subclasses)

**Implementation Patterns** (from `openrouter.py` and `ollama.py`):
- Dependency injection for testability (env_getter, session, base_url)
- aiohttp-based async HTTP client with optional session reuse
- Model caching for performance optimization
- Graceful error handling returning ProviderResponse
- Support for both temporary and injected aiohttp sessions
- Token estimation using character count / 4 approximation

**Key Design Principles**:
1. **Fail gracefully**: Return ProviderResponse with error messages, never raise exceptions
2. **Dependency injection**: Allow test mocking without real API keys
3. **Session reuse**: Support injected aiohttp.ClientSession for parallel processing
4. **Model validation**: Optimistic availability check with runtime validation
5. **Context window awareness**: Fetch and cache model metadata

### 1.2 OpenAI API Characteristics

**API Endpoints** (based on OpenAI API v1):
- Base URL: `https://api.openai.com/v1`
- Chat completions: `/chat/completions`
- Models list: `/models`
- Supports both streaming and non-streaming responses

**Authentication**:
- Bearer token authentication via `Authorization: Bearer <api-key>`
- Optional organization header: `OpenAI-Organization: org-xxxxx`

**Model Families**:
- GPT-4 series: gpt-4, gpt-4-turbo, gpt-4-turbo-preview, gpt-4o, gpt-4o-mini
- GPT-3.5 series: gpt-3.5-turbo variants
- Legacy models: text-davinci-003, etc.

**API Compatibility**:
- Many providers offer OpenAI-compatible endpoints (Azure OpenAI, LocalAI, Together.ai, etc.)
- Base URL override essential for compatibility

---

## 2. Provider Class Architecture

### 2.1 Class Structure

```python
# File: /workspaces/parallamr/src/parallamr/providers/openai.py

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

        # GPT-4o (Omni)
        "gpt-4o": {"context_length": 128000, "family": "gpt-4o"},
        "gpt-4o-2024-08-06": {"context_length": 128000, "family": "gpt-4o"},
        "gpt-4o-mini": {"context_length": 128000, "family": "gpt-4o"},
        "gpt-4o-mini-2024-07-18": {"context_length": 128000, "family": "gpt-4o"},

        # GPT-4 (Original)
        "gpt-4": {"context_length": 8192, "family": "gpt-4"},
        "gpt-4-0613": {"context_length": 8192, "family": "gpt-4"},
        "gpt-4-32k": {"context_length": 32768, "family": "gpt-4"},

        # GPT-3.5 Turbo
        "gpt-3.5-turbo": {"context_length": 16385, "family": "gpt-3.5"},
        "gpt-3.5-turbo-0125": {"context_length": 16385, "family": "gpt-3.5"},
        "gpt-3.5-turbo-1106": {"context_length": 16385, "family": "gpt-3.5"},
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: int = 300,
        base_url: Optional[str] = None,
        env_getter: Optional[Callable[[str], Optional[str]]] = None,
        session: Optional[aiohttp.ClientSession] = None,
        enable_streaming: bool = False,
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
            enable_streaming: Enable streaming responses (default: False, reserved for future)
        """
        super().__init__(timeout)

        # Use injected env_getter for testability
        _env_getter = env_getter or os.getenv

        # Authentication configuration
        self.api_key = api_key or _env_getter("OPENAI_API_KEY")
        self.organization = organization or _env_getter("OPENAI_ORG_ID")

        # Endpoint configuration
        self.base_url = base_url or _env_getter("OPENAI_BASE_URL", "https://api.openai.com/v1")

        # HTTP session management
        self._session = session

        # Feature flags
        self.enable_streaming = enable_streaming  # Reserved for future streaming support

        # Runtime caches
        self._models_cache: Optional[List[str]] = None
        self._model_details_cache: Optional[Dict[str, Any]] = None

    async def get_completion(
        self,
        prompt: str,
        model: str,
        **kwargs
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
        # See detailed implementation below
        pass

    async def get_context_window(self, model: str) -> Optional[int]:
        """
        Get model's context window size.

        Uses static metadata first, falls back to API query for unknown models.

        Args:
            model: Model identifier

        Returns:
            Context window size in tokens, or None if unknown
        """
        # See detailed implementation below
        pass

    async def list_models(self) -> List[str]:
        """
        List available models from OpenAI API.

        Returns:
            List of model identifiers
        """
        # See detailed implementation below
        pass

    def is_model_available(self, model: str) -> bool:
        """
        Check if a model is available (synchronous check using cache).

        Args:
            model: Model identifier

        Returns:
            True if model is available, False otherwise
        """
        # Optimistic check - uses cache if available, otherwise assumes available
        if self._models_cache is None:
            # No cache - check static metadata or assume available
            return model in self._MODEL_METADATA or True

        return model in self._models_cache

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers with authentication."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        return headers

    async def _fetch_models(self) -> Optional[List[str]]:
        """Fetch and cache available models from API."""
        # See detailed implementation below
        pass

    def _map_error_response(self, status: int, error_data: Optional[Dict]) -> str:
        """Map API error responses to user-friendly messages."""
        # See detailed implementation below
        pass
```

### 2.2 Interface Compatibility

The OpenAI provider implements the `Provider` abstract base class interface:

```python
# From /workspaces/parallamr/src/parallamr/providers/base.py

class Provider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def get_completion(self, prompt: str, model: str, **kwargs) -> ProviderResponse:
        """Get completion from the provider."""
        pass

    @abstractmethod
    async def get_context_window(self, model: str) -> Optional[int]:
        """Get model's context window size."""
        pass

    @abstractmethod
    async def list_models(self) -> list[str]:
        """List available models for this provider."""
        pass

    @abstractmethod
    def is_model_available(self, model: str) -> bool:
        """Check if a model is available for this provider."""
        pass
```

**Design Decisions**:
1. **Follows existing patterns**: Same method signatures as OpenRouter/Ollama providers
2. **Graceful degradation**: Methods return safe defaults rather than raising exceptions
3. **Async-first**: All I/O operations use async/await for consistency
4. **Caching strategy**: Similar to OpenRouter (cache models list and details)

---

## 3. Configuration Schema

### 3.1 Environment Variables

```bash
# .env configuration

# OpenAI Authentication (REQUIRED for official API)
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxx

# OpenAI Organization (OPTIONAL - for organization-scoped API access)
OPENAI_ORG_ID=org-xxxxxxxxxxxxxxxxxx

# Base URL Override (OPTIONAL - for OpenAI-compatible providers)
OPENAI_BASE_URL=https://api.openai.com/v1

# Examples for compatible providers:
# OPENAI_BASE_URL=https://your-azure-resource.openai.azure.com/openai/deployments/your-deployment
# OPENAI_BASE_URL=https://api.together.xyz/v1
# OPENAI_BASE_URL=http://localhost:8080/v1  # LocalAI
```

### 3.2 Configuration Hierarchy

Following the established pattern from OpenRouter and Ollama providers:

```python
# Priority order (highest to lowest):
1. Constructor parameter (api_key="direct-key")
2. Environment variable (OPENAI_API_KEY)
3. Default value (None for api_key, official API URL for base_url)

# Example usage:
provider = OpenAIProvider(
    api_key="sk-test-123",           # Highest priority
    organization="org-abc",           # Explicit org
    base_url="http://localhost:8080", # Override for compatible provider
    timeout=600                       # Custom timeout
)
```

### 3.3 Validation Requirements

```python
def _validate_configuration(self) -> tuple[bool, Optional[str]]:
    """
    Validate provider configuration.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not self.api_key:
        return False, "OpenAI API key not provided. Set OPENAI_API_KEY or pass api_key parameter."

    if not self.api_key.startswith("sk-"):
        return False, "Invalid OpenAI API key format. Keys should start with 'sk-'."

    if not self.base_url:
        return False, "Base URL is required."

    return True, None
```

### 3.4 Integration with Runner

Update `/workspaces/parallamr/src/parallamr/runner.py`:

```python
def _create_default_providers(self, timeout: int) -> Dict[str, Provider]:
    """Create default provider instances."""
    return {
        "mock": MockProvider(timeout=timeout),
        "openrouter": OpenRouterProvider(timeout=timeout),
        "ollama": OllamaProvider(timeout=timeout),
        "openai": OpenAIProvider(timeout=timeout),  # NEW
    }
```

Update `/workspaces/parallamr/src/parallamr/providers/__init__.py`:

```python
"""Provider implementations for different LLM services."""

from .base import Provider
from .mock import MockProvider
from .ollama import OllamaProvider
from .openai import OpenAIProvider  # NEW
from .openrouter import OpenRouterProvider

__all__ = [
    "Provider",
    "MockProvider",
    "OllamaProvider",
    "OpenAIProvider",  # NEW
    "OpenRouterProvider",
]
```

---

## 4. Authentication and API Client Design

### 4.1 Authentication Flow

```python
class OpenAIProvider(Provider):
    """OpenAI provider with authentication handling."""

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

    async def _authenticate_request(self) -> tuple[bool, Optional[str]]:
        """
        Validate authentication before making requests.

        Returns:
            Tuple of (is_authenticated, error_message)
        """
        valid, error = self._validate_configuration()
        if not valid:
            return False, error

        # Authentication is header-based, no pre-flight check needed
        return True, None
```

### 4.2 HTTP Client Management

Following the pattern from existing providers with session injection:

```python
async def _make_request(
    self,
    method: str,
    endpoint: str,
    json_data: Optional[Dict] = None,
    timeout_override: Optional[int] = None
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
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                return await self._process_response(response)
        else:
            # No injected session - use temporary session (backward compatibility)
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as session:
                async with session.request(
                    method,
                    url,
                    headers=headers,
                    json=json_data
                ) as response:
                    return await self._process_response(response)

    except asyncio.TimeoutError:
        return 0, None, f"Request timeout after {timeout} seconds"
    except aiohttp.ClientError as e:
        return 0, None, f"Network error: {str(e)}"
    except Exception as e:
        return 0, None, f"Unexpected error: {str(e)}"

async def _process_response(
    self,
    response: aiohttp.ClientResponse
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
        return status, None, f"{error_type}: {error_msg}"

    return status, data, None
```

### 4.3 Session Reuse Pattern

Following the dependency injection pattern for parallel processing:

```python
# In runner or CLI factory:
async with aiohttp.ClientSession() as session:
    openai_provider = OpenAIProvider(session=session)

    # Multiple requests reuse the same connection pool
    result1 = await openai_provider.get_completion("prompt1", "gpt-4")
    result2 = await openai_provider.get_completion("prompt2", "gpt-4")
```

---

## 5. Request/Response Transformation Layer

### 5.1 Request Payload Mapping

```python
def _build_completion_payload(
    self,
    prompt: str,
    model: str,
    kwargs: Dict[str, Any]
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
        "messages": [
            {"role": "user", "content": prompt}
        ],
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
        "presence_penalty",
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
```

### 5.2 Response Transformation

```python
def _transform_api_response(
    self,
    api_response: Dict[str, Any],
    model: str
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
            context_window=context_window
        )

    except (KeyError, IndexError) as e:
        # Malformed API response
        return ProviderResponse(
            output="",
            output_tokens=0,
            success=False,
            error_message=f"Malformed API response: {str(e)}"
        )

def _get_static_context_window(self, model: str) -> Optional[int]:
    """Get context window from static metadata."""
    metadata = self._MODEL_METADATA.get(model)
    if metadata:
        return metadata.get("context_length")
    return None
```

### 5.3 Parameter Validation

```python
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

    return True, None
```

---

## 6. Streaming Response Design (Future Feature)

### 6.1 Architecture Overview

Streaming is reserved for future implementation but the architecture is designed to support it:

```python
class OpenAIProvider(Provider):
    """Provider with streaming support (future)."""

    def __init__(self, ..., enable_streaming: bool = False):
        """
        Args:
            enable_streaming: Enable streaming responses (default: False)
        """
        self.enable_streaming = enable_streaming

    async def get_completion_stream(
        self,
        prompt: str,
        model: str,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Get streaming completion from OpenAI API (FUTURE FEATURE).

        Args:
            prompt: Input prompt text
            model: Model identifier
            **kwargs: Additional parameters

        Yields:
            Content chunks as they arrive
        """
        # Implementation reserved for future streaming support
        raise NotImplementedError("Streaming support coming in future release")
```

### 6.2 Integration Points

For future streaming support, these components will need updates:

1. **ProviderResponse Model**: Add streaming field
2. **ExperimentRunner**: Add streaming mode support
3. **CSV Writer**: Handle streaming output accumulation
4. **CLI**: Add --stream flag

**Design Note**: Current architecture focuses on batch processing with incremental CSV output, which is the primary use case. Streaming will be added when real-time experimentation becomes a requirement.

---

## 7. Error Handling Architecture

### 7.1 Error Classification

Following the existing exception hierarchy from `/workspaces/parallamr/src/parallamr/providers/base.py`:

```python
# Existing exception hierarchy (no changes needed):
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
```

### 7.2 Error Response Mapping

```python
def _map_error_response(
    self,
    status: int,
    error_data: Optional[Dict],
    error_msg: Optional[str]
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
        output="",
        output_tokens=0,
        success=False,
        error_message=message
    )
```

### 7.3 Error Handling Strategy

Following the "graceful degradation" pattern from existing providers:

```python
async def get_completion(
    self,
    prompt: str,
    model: str,
    **kwargs
) -> ProviderResponse:
    """Get completion with comprehensive error handling."""

    # Step 1: Validate configuration
    valid, error = self._validate_configuration()
    if not valid:
        return ProviderResponse(
            output="",
            output_tokens=0,
            success=False,
            error_message=error
        )

    # Step 2: Validate parameters
    valid, error = self._validate_parameters(kwargs)
    if not valid:
        return ProviderResponse(
            output="",
            output_tokens=0,
            success=False,
            error_message=f"Invalid parameters: {error}"
        )

    # Step 3: Build request payload
    try:
        payload = self._build_completion_payload(prompt, model, kwargs)
    except Exception as e:
        return ProviderResponse(
            output="",
            output_tokens=0,
            success=False,
            error_message=f"Failed to build request: {str(e)}"
        )

    # Step 4: Make API request with error handling
    status, data, error = await self._make_request(
        "POST",
        "/chat/completions",
        json_data=payload
    )

    # Step 5: Handle errors
    if error or status >= 400:
        return self._map_error_response(status, data, error)

    # Step 6: Transform successful response
    return self._transform_api_response(data, model)
```

### 7.4 Retry Logic (Optional Enhancement)

For production use, consider adding exponential backoff retry:

```python
async def _make_request_with_retry(
    self,
    method: str,
    endpoint: str,
    json_data: Optional[Dict] = None,
    max_retries: int = 3
) -> tuple[int, Optional[Dict], Optional[str]]:
    """
    Make request with exponential backoff retry.

    Args:
        method: HTTP method
        endpoint: API endpoint
        json_data: Request body
        max_retries: Maximum retry attempts

    Returns:
        Tuple of (status_code, response_data, error_message)
    """
    for attempt in range(max_retries):
        status, data, error = await self._make_request(method, endpoint, json_data)

        # Don't retry on client errors (4xx except 429)
        if 400 <= status < 500 and status != 429:
            return status, data, error

        # Success - return immediately
        if status == 200:
            return status, data, error

        # Retry on 5xx or 429 (rate limit)
        if attempt < max_retries - 1:
            backoff = 2 ** attempt  # 1s, 2s, 4s
            await asyncio.sleep(backoff)

    return status, data, error
```

---

## 8. Code Organization and File Structure

### 8.1 File Layout

```
/workspaces/parallamr/
├── src/parallamr/
│   ├── providers/
│   │   ├── __init__.py          # Updated with OpenAIProvider export
│   │   ├── base.py              # No changes (existing exceptions sufficient)
│   │   ├── openai.py            # NEW - OpenAI provider implementation
│   │   ├── openrouter.py        # Existing
│   │   ├── ollama.py            # Existing
│   │   └── mock.py              # Existing
│   ├── runner.py                # Updated to include OpenAI in default providers
│   └── cli.py                   # Updated providers command to show OpenAI config
├── tests/
│   ├── test_providers.py        # Extended with OpenAI provider tests
│   ├── test_openai_provider.py  # NEW - Comprehensive OpenAI provider tests
│   └── fixtures/
│       └── openai_responses.py  # NEW - Mock API response fixtures
└── docs/
    └── OPENAI_PROVIDER_ARCHITECTURE.md  # This document
```

### 8.2 Import Organization

```python
# src/parallamr/providers/__init__.py

"""Provider implementations for different LLM services."""

from .base import Provider
from .mock import MockProvider
from .ollama import OllamaProvider
from .openai import OpenAIProvider  # NEW
from .openrouter import OpenRouterProvider

__all__ = [
    "Provider",
    "MockProvider",
    "OllamaProvider",
    "OpenAIProvider",  # NEW
    "OpenRouterProvider",
]
```

### 8.3 Module Dependencies

```python
# Dependencies for openai.py module:
import asyncio          # Async/await, timeout handling
import os               # Environment variable access
from typing import *    # Type annotations

import aiohttp          # HTTP client (already in requirements)

from ..models import ProviderResponse           # Existing
from ..token_counter import estimate_tokens     # Existing
from .base import (                             # Existing
    AuthenticationError,
    ContextWindowExceededError,
    Provider,
    ProviderError,
)
```

**No new dependencies required** - all necessary packages are already in `pyproject.toml`:
- aiohttp >= 3.9.0 (HTTP client)
- pydantic >= 2.0.0 (data validation)
- python-dotenv >= 1.0.0 (environment variables)

---

## 9. Testing Strategy

### 9.1 Unit Test Structure

```python
# tests/test_openai_provider.py

"""Comprehensive tests for OpenAI provider."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from parallamr.models import ProviderResponse
from parallamr.providers import OpenAIProvider


class TestOpenAIProviderInit:
    """Test provider initialization and configuration."""

    def test_init_with_api_key(self):
        """Test initialization with direct API key."""
        provider = OpenAIProvider(api_key="sk-test-123")
        assert provider.api_key == "sk-test-123"
        assert provider.base_url == "https://api.openai.com/v1"

    def test_init_with_env_injection(self):
        """Test initialization with injected env_getter."""
        def mock_env(key):
            if key == "OPENAI_API_KEY":
                return "sk-env-456"
            return None

        provider = OpenAIProvider(env_getter=mock_env)
        assert provider.api_key == "sk-env-456"

    def test_init_with_custom_base_url(self):
        """Test initialization with custom base URL (for compatible providers)."""
        provider = OpenAIProvider(
            api_key="sk-test",
            base_url="https://api.together.xyz/v1"
        )
        assert provider.base_url == "https://api.together.xyz/v1"

    def test_init_with_organization(self):
        """Test initialization with organization ID."""
        provider = OpenAIProvider(
            api_key="sk-test",
            organization="org-abc123"
        )
        assert provider.organization == "org-abc123"


class TestOpenAIProviderCompletion:
    """Test completion generation."""

    @pytest.mark.asyncio
    async def test_get_completion_success(self):
        """Test successful completion."""
        provider = OpenAIProvider(api_key="sk-test")

        # Mock API response
        mock_response = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"completion_tokens": 10}
        }

        with patch.object(provider, '_make_request',
                         return_value=(200, mock_response, None)):
            response = await provider.get_completion("Test prompt", "gpt-4")

            assert response.success is True
            assert response.output == "Test response"
            assert response.output_tokens == 10

    @pytest.mark.asyncio
    async def test_get_completion_missing_api_key(self):
        """Test completion without API key."""
        provider = OpenAIProvider()  # No API key

        response = await provider.get_completion("Test", "gpt-4")

        assert response.success is False
        assert "API key not provided" in response.error_message

    @pytest.mark.asyncio
    async def test_get_completion_rate_limit(self):
        """Test rate limit handling."""
        provider = OpenAIProvider(api_key="sk-test")

        # Mock rate limit error
        error_response = {
            "error": {
                "type": "rate_limit_error",
                "message": "Rate limit exceeded"
            }
        }

        with patch.object(provider, '_make_request',
                         return_value=(429, error_response, None)):
            response = await provider.get_completion("Test", "gpt-4")

            assert response.success is False
            assert "rate limit" in response.error_message.lower()


class TestOpenAIProviderModels:
    """Test model listing and validation."""

    @pytest.mark.asyncio
    async def test_list_models(self):
        """Test fetching available models."""
        provider = OpenAIProvider(api_key="sk-test")

        # Mock models API response
        mock_response = {
            "data": [
                {"id": "gpt-4"},
                {"id": "gpt-3.5-turbo"},
            ]
        }

        with patch.object(provider, '_make_request',
                         return_value=(200, mock_response, None)):
            models = await provider.list_models()

            assert "gpt-4" in models
            assert "gpt-3.5-turbo" in models

    def test_is_model_available_static(self):
        """Test model availability check with static metadata."""
        provider = OpenAIProvider(api_key="sk-test")

        assert provider.is_model_available("gpt-4") is True
        assert provider.is_model_available("gpt-4-turbo") is True

    @pytest.mark.asyncio
    async def test_get_context_window_static(self):
        """Test context window retrieval from static metadata."""
        provider = OpenAIProvider(api_key="sk-test")

        context_window = await provider.get_context_window("gpt-4-turbo")
        assert context_window == 128000


class TestOpenAIProviderErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_authentication_error(self):
        """Test authentication failure handling."""
        provider = OpenAIProvider(api_key="sk-invalid")

        error_response = {
            "error": {
                "type": "invalid_request_error",
                "message": "Invalid API key"
            }
        }

        with patch.object(provider, '_make_request',
                         return_value=(401, error_response, None)):
            response = await provider.get_completion("Test", "gpt-4")

            assert response.success is False
            assert "authentication" in response.error_message.lower()

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling."""
        provider = OpenAIProvider(api_key="sk-test", timeout=1)

        with patch.object(provider, '_make_request',
                         return_value=(0, None, "Request timeout after 1 seconds")):
            response = await provider.get_completion("Test", "gpt-4")

            assert response.success is False
            assert "timeout" in response.error_message.lower()

    @pytest.mark.asyncio
    async def test_malformed_response(self):
        """Test handling of malformed API responses."""
        provider = OpenAIProvider(api_key="sk-test")

        # Missing required fields
        malformed_response = {"choices": []}

        with patch.object(provider, '_make_request',
                         return_value=(200, malformed_response, None)):
            response = await provider.get_completion("Test", "gpt-4")

            assert response.success is False
            assert "malformed" in response.error_message.lower()


class TestOpenAIProviderSessionInjection:
    """Test aiohttp session injection for parallel processing."""

    @pytest.mark.asyncio
    async def test_session_injection(self):
        """Test that injected session is used for requests."""
        import aiohttp

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        provider = OpenAIProvider(api_key="sk-test", session=mock_session)

        # Mock response context manager
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "Test"}}],
            "usage": {"completion_tokens": 5}
        })

        mock_session.request.return_value.__aenter__.return_value = mock_response

        response = await provider.get_completion("Test", "gpt-4")

        # Verify session was used
        assert mock_session.request.called
        assert response.success is True


# Integration tests (skipped by default)
class TestOpenAIProviderIntegration:
    """Integration tests requiring actual API access."""

    @pytest.mark.skip(reason="Requires actual OpenAI API key")
    @pytest.mark.asyncio
    async def test_real_api_call(self):
        """Test with real OpenAI API (skipped by default)."""
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set")

        provider = OpenAIProvider(api_key=api_key)
        response = await provider.get_completion(
            "Say 'test successful'",
            "gpt-3.5-turbo"
        )

        assert response.success is True
        assert len(response.output) > 0
```

### 9.2 Test Coverage Requirements

Following project standards from `pyproject.toml`:

- **Line coverage target**: ≥ 90%
- **Branch coverage**: ≥ 85%
- **Test categories**:
  - Unit tests (fast, mocked)
  - Integration tests (slow, real API - skipped by default)
  - Error path tests (comprehensive)

### 9.3 Mock Response Fixtures

```python
# tests/fixtures/openai_responses.py

"""Mock API responses for OpenAI provider testing."""

MOCK_CHAT_COMPLETION = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1677858242,
    "model": "gpt-4",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a test response from GPT-4."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 15,
        "total_tokens": 25
    }
}

MOCK_MODELS_LIST = {
    "object": "list",
    "data": [
        {"id": "gpt-4", "object": "model", "owned_by": "openai"},
        {"id": "gpt-4-turbo", "object": "model", "owned_by": "openai"},
        {"id": "gpt-3.5-turbo", "object": "model", "owned_by": "openai"},
    ]
}

MOCK_ERROR_RATE_LIMIT = {
    "error": {
        "message": "Rate limit exceeded",
        "type": "rate_limit_error",
        "param": None,
        "code": "rate_limit_exceeded"
    }
}

MOCK_ERROR_AUTH = {
    "error": {
        "message": "Invalid API key provided",
        "type": "invalid_request_error",
        "param": None,
        "code": "invalid_api_key"
    }
}
```

---

## 10. Implementation Checklist

### Phase 1: Core Implementation
- [ ] Create `/workspaces/parallamr/src/parallamr/providers/openai.py`
- [ ] Implement `OpenAIProvider` class with all required methods
- [ ] Add static model metadata (context windows)
- [ ] Implement authentication and header building
- [ ] Implement HTTP client with session support
- [ ] Implement request/response transformation

### Phase 2: Integration
- [ ] Update `/workspaces/parallamr/src/parallamr/providers/__init__.py`
- [ ] Update `/workspaces/parallamr/src/parallamr/runner.py` default providers
- [ ] Update `/workspaces/parallamr/src/parallamr/cli.py` providers command
- [ ] Update `.env.example` with OpenAI configuration
- [ ] Update README.md with OpenAI usage examples

### Phase 3: Testing
- [ ] Create `/workspaces/parallamr/tests/test_openai_provider.py`
- [ ] Create `/workspaces/parallamr/tests/fixtures/openai_responses.py`
- [ ] Implement unit tests (initialization, completion, models)
- [ ] Implement error handling tests
- [ ] Implement session injection tests
- [ ] Add integration tests (skipped by default)
- [ ] Verify ≥90% code coverage

### Phase 4: Documentation
- [ ] Document OpenAI provider in README
- [ ] Add configuration examples
- [ ] Add usage examples with experiments CSV
- [ ] Document OpenAI-compatible provider usage
- [ ] Update CHANGELOG.md

### Phase 5: Validation
- [ ] Manual testing with real OpenAI API
- [ ] Test with Azure OpenAI (compatible endpoint)
- [ ] Test error scenarios (rate limit, auth failure)
- [ ] Test with parallamr CLI commands
- [ ] Integration test with full experiment workflow

---

## 11. Usage Examples

### 11.1 Basic Usage

```bash
# 1. Configure API key in .env
echo "OPENAI_API_KEY=sk-proj-your-key-here" >> .env

# 2. Create experiments CSV
cat > experiments.csv << EOF
provider,model,topic,temperature
openai,gpt-4,Quantum Computing,0.7
openai,gpt-4-turbo,Quantum Computing,0.7
openai,gpt-3.5-turbo,Quantum Computing,0.7
EOF

# 3. Create prompt
cat > prompt.txt << EOF
Explain {{topic}} in simple terms for a beginner audience.
EOF

# 4. Run experiments
parallamr run -p prompt.txt -e experiments.csv -o results.csv --verbose
```

### 11.2 OpenAI-Compatible Provider (Azure OpenAI)

```bash
# Configure for Azure OpenAI
export OPENAI_API_KEY="your-azure-key"
export OPENAI_BASE_URL="https://your-resource.openai.azure.com/openai/deployments/your-deployment"

# Use same experiments CSV and commands
parallamr run -p prompt.txt -e experiments.csv -o results.csv
```

### 11.3 Programmatic Usage

```python
import asyncio
from parallamr.providers import OpenAIProvider

async def test_openai():
    provider = OpenAIProvider(api_key="sk-test-key")

    response = await provider.get_completion(
        prompt="Explain quantum computing",
        model="gpt-4",
        temperature=0.7,
        max_tokens=500
    )

    print(f"Success: {response.success}")
    print(f"Output: {response.output}")
    print(f"Tokens: {response.output_tokens}")

asyncio.run(test_openai())
```

### 11.4 Model Comparison Experiment

```csv
# experiments.csv - Compare OpenAI models with other providers
provider,model,task,max_tokens
openai,gpt-4-turbo,summarization,500
openai,gpt-3.5-turbo,summarization,500
openrouter,anthropic/claude-sonnet-4,summarization,500
ollama,llama3.2,summarization,500
```

---

## 12. Security Considerations

### 12.1 API Key Management

**Best Practices**:
- Store API keys in `.env` file (gitignored)
- Never commit API keys to version control
- Use environment variables for CI/CD
- Rotate keys regularly

**Validation**:
```python
def _validate_api_key(api_key: str) -> bool:
    """Validate API key format."""
    return api_key.startswith("sk-") and len(api_key) > 20
```

### 12.2 Request Security

- HTTPS required (enforced in default base_url)
- Bearer token authentication
- Organization scoping supported
- User ID tracking for abuse prevention

### 12.3 Data Privacy

- Input prompts sent to OpenAI API
- Consider data retention policies
- Use user IDs for tracking (optional)
- Review OpenAI's data usage policy

---

## 13. Performance Considerations

### 13.1 Connection Pooling

- Session injection enables connection reuse
- Reduces overhead for multiple requests
- Improves latency in parallel experiments

### 13.2 Caching Strategy

```python
# Static metadata (class-level, never expires)
_MODEL_METADATA: Dict[str, Dict[str, Any]] = {...}

# Runtime cache (instance-level, expires on provider restart)
self._models_cache: Optional[List[str]] = None
self._model_details_cache: Optional[Dict[str, Any]] = None
```

### 13.3 Rate Limit Handling

- Respect 429 responses
- Optional exponential backoff retry
- Consider token-per-minute (TPM) limits
- Monitor usage in logs

---

## 14. Maintenance and Evolution

### 14.1 Version Compatibility

**OpenAI API Versioning**:
- Current: v1 (stable)
- Headers support versioning: `OpenAI-Version: 2023-12-01`
- Monitor OpenAI changelog for breaking changes

**Parallamr Versioning**:
- OpenAI provider: v0.6.0 (initial implementation)
- Follow semantic versioning
- Maintain backward compatibility

### 14.2 Future Enhancements

**Planned Features**:
1. Streaming response support (`enable_streaming=True`)
2. Function calling / tool use support
3. Vision model support (image inputs)
4. Fine-tuned model support
5. Embeddings endpoint integration

**Extension Points**:
```python
class OpenAIProvider(Provider):
    """Provider designed for future extension."""

    async def get_completion_stream(self, ...) -> AsyncIterator[str]:
        """Streaming support (future)."""
        raise NotImplementedError("Coming in v0.7.0")

    async def call_function(self, ...) -> Dict:
        """Function calling support (future)."""
        raise NotImplementedError("Coming in v0.8.0")
```

### 14.3 Model Metadata Updates

**Update Process**:
1. Monitor OpenAI model releases
2. Update `_MODEL_METADATA` class variable
3. Add new models to test fixtures
4. Update documentation

**Automation Opportunity**:
```python
# Future: Auto-update metadata from API
async def _refresh_model_metadata(self):
    """Fetch latest model metadata from API."""
    # Query models endpoint
    # Update static metadata
    # Cache for future use
```

---

## 15. Alignment with Existing Patterns

### 15.1 Pattern Consistency Matrix

| Pattern | OpenRouter | Ollama | OpenAI (Proposed) |
|---------|-----------|--------|-------------------|
| Async/await | ✅ | ✅ | ✅ |
| Session injection | ✅ | ✅ | ✅ |
| Env getter injection | ✅ | ✅ | ✅ |
| Base URL override | ✅ | ✅ | ✅ |
| Model caching | ✅ | ✅ | ✅ |
| Graceful errors | ✅ | ✅ | ✅ |
| ProviderResponse | ✅ | ✅ | ✅ |
| Timeout configuration | ✅ | ✅ | ✅ |

### 15.2 Code Style Consistency

Following project standards from `pyproject.toml`:
- **Formatter**: Black (line length 88)
- **Linter**: Ruff (strict mode)
- **Type checker**: Mypy (strict mode)
- **Docstrings**: Google style

### 15.3 Testing Consistency

Following patterns from `tests/test_providers.py`:
- Class-based test organization
- Async test with pytest-asyncio
- Mock injection for testability
- Integration tests marked with `@pytest.mark.skip`

---

## 16. Success Criteria

### 16.1 Functional Requirements

- [ ] Provider implements all base class methods
- [ ] Supports official OpenAI API
- [ ] Supports OpenAI-compatible endpoints
- [ ] Authentication via API key
- [ ] Organization support (optional)
- [ ] Comprehensive error handling
- [ ] Context window validation

### 16.2 Non-Functional Requirements

- [ ] ≥90% test coverage
- [ ] Type hints on all public methods
- [ ] Docstrings on all public methods
- [ ] No breaking changes to existing code
- [ ] Performance comparable to existing providers
- [ ] Memory efficient (caching, session reuse)

### 16.3 Integration Requirements

- [ ] Works with ExperimentRunner
- [ ] Works with CLI commands
- [ ] Works with CSV experiments
- [ ] Works with template variables
- [ ] Works with incremental output
- [ ] Works with validation mode

---

## 17. Conclusion

This architecture design provides a comprehensive blueprint for implementing the OpenAI provider following established patterns from the Parallamr codebase. The design prioritizes:

1. **Pattern consistency**: Aligns with OpenRouter and Ollama implementations
2. **Testability**: Comprehensive dependency injection
3. **Extensibility**: Ready for streaming and function calling
4. **Reliability**: Graceful error handling and validation
5. **Performance**: Connection pooling and caching
6. **Maintainability**: Clear code organization and documentation

**Next Steps**:
1. Review this design document with the team
2. Create implementation tickets for each phase
3. Begin Phase 1 implementation (core provider)
4. Iterate based on code review feedback

**Estimated Implementation Time**:
- Phase 1 (Core): 6-8 hours
- Phase 2 (Integration): 2-3 hours
- Phase 3 (Testing): 4-6 hours
- Phase 4 (Documentation): 2-3 hours
- Phase 5 (Validation): 2-4 hours
- **Total**: 16-24 hours

---

## Appendix A: Complete Implementation Skeleton

```python
# /workspaces/parallamr/src/parallamr/providers/openai.py
# COMPLETE IMPLEMENTATION READY FOR CODING

"""OpenAI API provider implementation."""

import asyncio
import os
from typing import Any, Callable, Dict, List, Optional

import aiohttp

from ..models import ProviderResponse
from ..token_counter import estimate_tokens
from .base import Provider


class OpenAIProvider(Provider):
    """OpenAI API provider for GPT models and compatible endpoints."""

    _MODEL_METADATA: Dict[str, Dict[str, Any]] = {
        # [Full metadata dictionary from Section 2.1]
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: int = 300,
        base_url: Optional[str] = None,
        env_getter: Optional[Callable[[str], Optional[str]]] = None,
        session: Optional[aiohttp.ClientSession] = None,
        enable_streaming: bool = False,
    ):
        """Initialize the OpenAI provider."""
        # [Implementation from Section 2.1]
        pass

    async def get_completion(
        self,
        prompt: str,
        model: str,
        **kwargs
    ) -> ProviderResponse:
        """Get completion from OpenAI API."""
        # [Implementation from Section 7.3]
        pass

    async def get_context_window(self, model: str) -> Optional[int]:
        """Get model's context window size."""
        # [Implementation from Section 5.2]
        pass

    async def list_models(self) -> List[str]:
        """List available models from OpenAI API."""
        # [Implementation from Section 4.2]
        pass

    def is_model_available(self, model: str) -> bool:
        """Check if a model is available."""
        # [Implementation from Section 2.1]
        pass

    # Private helper methods
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers with authentication."""
        # [Implementation from Section 4.1]
        pass

    def _build_completion_payload(
        self,
        prompt: str,
        model: str,
        kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build OpenAI API request payload."""
        # [Implementation from Section 5.1]
        pass

    def _transform_api_response(
        self,
        api_response: Dict[str, Any],
        model: str
    ) -> ProviderResponse:
        """Transform OpenAI API response to ProviderResponse."""
        # [Implementation from Section 5.2]
        pass

    def _map_error_response(
        self,
        status: int,
        error_data: Optional[Dict],
        error_msg: Optional[str]
    ) -> ProviderResponse:
        """Map API errors to ProviderResponse."""
        # [Implementation from Section 7.2]
        pass

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict] = None,
        timeout_override: Optional[int] = None
    ) -> tuple[int, Optional[Dict], Optional[str]]:
        """Make authenticated HTTP request to OpenAI API."""
        # [Implementation from Section 4.2]
        pass

    def _validate_configuration(self) -> tuple[bool, Optional[str]]:
        """Validate provider configuration."""
        # [Implementation from Section 3.3]
        pass

    def _validate_parameters(self, kwargs: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate API parameters."""
        # [Implementation from Section 5.3]
        pass

    def _get_static_context_window(self, model: str) -> Optional[int]:
        """Get context window from static metadata."""
        # [Implementation from Section 5.2]
        pass
```

---

## Appendix B: Configuration Examples

### Example 1: Basic OpenAI Configuration

```bash
# .env
OPENAI_API_KEY=sk-proj-abc123...
```

### Example 2: Azure OpenAI Configuration

```bash
# .env
OPENAI_API_KEY=your-azure-key
OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment/chat/completions?api-version=2023-12-01-preview
```

### Example 3: Local OpenAI-Compatible Server

```bash
# .env
OPENAI_API_KEY=not-needed-for-local
OPENAI_BASE_URL=http://localhost:8080/v1
```

---

## Appendix C: API Reference

### OpenAI Chat Completions API

**Endpoint**: `POST /v1/chat/completions`

**Request**:
```json
{
  "model": "gpt-4",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 500
}
```

**Response**:
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "gpt-4",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 15,
    "total_tokens": 25
  }
}
```

---

**End of Architecture Design Document**
