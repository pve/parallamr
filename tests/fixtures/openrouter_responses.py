"""Mock API response fixtures for OpenRouter provider testing.

This module provides comprehensive test fixtures for the OpenRouter API provider,
including successful responses, error cases, and edge cases based on the
actual OpenRouter API format (OpenAI-compatible with extensions).

OpenRouter API Documentation: https://openrouter.ai/docs
"""

from typing import Any, Dict, List


# ============================================================================
# SUCCESSFUL COMPLETION RESPONSES
# ============================================================================

# Standard successful completion response
SUCCESSFUL_COMPLETION = {
    "id": "gen-1234567890abcdef",
    "model": "anthropic/claude-3.5-sonnet",
    "object": "chat.completion",
    "created": 1728475200,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a test response from Claude 3.5 Sonnet via OpenRouter."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 15,
        "completion_tokens": 25,
        "total_tokens": 40
    }
}

# GPT-4 completion via OpenRouter
COMPLETION_GPT4 = {
    "id": "gen-abcdef1234567890",
    "model": "openai/gpt-4-turbo",
    "object": "chat.completion",
    "created": 1728475201,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Response from GPT-4 Turbo via OpenRouter."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 18,
        "total_tokens": 30
    }
}

# Llama 3.1 completion via OpenRouter
COMPLETION_LLAMA31 = {
    "id": "gen-fedcba0987654321",
    "model": "meta-llama/llama-3.1-70b-instruct",
    "object": "chat.completion",
    "created": 1728475202,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Response from Llama 3.1 70B Instruct."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 30,
        "total_tokens": 50
    }
}

# Completion without usage data (fallback to token estimation)
COMPLETION_NO_USAGE = {
    "id": "gen-xyz9876543210abc",
    "model": "anthropic/claude-3.5-sonnet",
    "object": "chat.completion",
    "created": 1728475204,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Response without usage data."
            },
            "finish_reason": "stop"
        }
    ]
}

# Completion with length finish reason (hit max tokens)
COMPLETION_LENGTH_FINISH = {
    "id": "gen-length123456789",
    "model": "openai/gpt-3.5-turbo",
    "object": "chat.completion",
    "created": 1728475205,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This response was cut off because it reached the maximum token limit..."
            },
            "finish_reason": "length"
        }
    ],
    "usage": {
        "prompt_tokens": 50,
        "completion_tokens": 100,
        "total_tokens": 150
    }
}

# Empty response (edge case)
COMPLETION_EMPTY_RESPONSE = {
    "id": "gen-empty000000000",
    "model": "anthropic/claude-3.5-sonnet",
    "object": "chat.completion",
    "created": 1728475207,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": ""
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 0,
        "total_tokens": 5
    }
}


# ============================================================================
# MODELS LIST RESPONSES (/api/v1/models)
# ============================================================================

# Standard models list with popular models
MODELS_LIST_RESPONSE = {
    "data": [
        {
            "id": "anthropic/claude-3.5-sonnet",
            "name": "Claude 3.5 Sonnet",
            "created": 1719792000,
            "description": "Anthropic's most intelligent model",
            "context_length": 200000,
            "pricing": {
                "prompt": "0.000003",
                "completion": "0.000015"
            },
            "top_provider": {
                "context_length": 200000,
                "max_completion_tokens": 8096,
                "is_moderated": False
            }
        },
        {
            "id": "openai/gpt-4-turbo",
            "name": "GPT-4 Turbo",
            "created": 1706745600,
            "description": "OpenAI's flagship model with 128K context",
            "context_length": 128000,
            "pricing": {
                "prompt": "0.00001",
                "completion": "0.00003"
            },
            "top_provider": {
                "context_length": 128000,
                "max_completion_tokens": 4096,
                "is_moderated": True
            }
        },
        {
            "id": "meta-llama/llama-3.1-70b-instruct",
            "name": "Llama 3.1 70B Instruct",
            "created": 1721692800,
            "description": "Meta's powerful open-source model",
            "context_length": 131072,
            "pricing": {
                "prompt": "0.0000004",
                "completion": "0.0000008"
            },
            "top_provider": {
                "context_length": 131072,
                "max_completion_tokens": 8192,
                "is_moderated": False
            }
        },
        {
            "id": "mistralai/mistral-large",
            "name": "Mistral Large",
            "created": 1709251200,
            "description": "Mistral AI's flagship model",
            "context_length": 32768,
            "pricing": {
                "prompt": "0.000003",
                "completion": "0.000009"
            },
            "top_provider": {
                "context_length": 32768,
                "max_completion_tokens": 8192,
                "is_moderated": False
            }
        }
    ]
}

# Empty models list
MODELS_LIST_EMPTY = {
    "data": []
}

# Models list with single model
MODELS_LIST_SINGLE = {
    "data": [
        {
            "id": "anthropic/claude-3.5-sonnet",
            "name": "Claude 3.5 Sonnet",
            "created": 1719792000,
            "description": "Anthropic's most intelligent model",
            "context_length": 200000,
            "pricing": {
                "prompt": "0.000003",
                "completion": "0.000015"
            }
        }
    ]
}


# ============================================================================
# ERROR RESPONSES
# ============================================================================

# Authentication error (401)
ERROR_401_UNAUTHORIZED = {
    "error": {
        "message": "Invalid API key provided",
        "type": "invalid_request_error",
        "code": "invalid_api_key"
    }
}

# Forbidden - no access to model (403)
ERROR_403_FORBIDDEN = {
    "error": {
        "message": "You do not have access to this model",
        "type": "permission_error",
        "code": "model_access_denied"
    }
}

# Model not found (404)
ERROR_404_MODEL_NOT_FOUND = {
    "error": {
        "message": "The model 'invalid/model-name' does not exist",
        "type": "invalid_request_error",
        "code": "model_not_found"
    }
}

# Rate limit exceeded (429)
ERROR_429_RATE_LIMIT = {
    "error": {
        "message": "Rate limit exceeded. Please try again later.",
        "type": "rate_limit_error",
        "code": "rate_limit_exceeded",
        "metadata": {
            "retry_after": 60
        }
    }
}

# Rate limit with credits information (429)
ERROR_429_CREDITS_EXHAUSTED = {
    "error": {
        "message": "Insufficient credits. Please add more credits to your account.",
        "type": "insufficient_quota",
        "code": "insufficient_credits"
    }
}

# Bad request - missing required field (400)
ERROR_400_BAD_REQUEST = {
    "error": {
        "message": "Invalid request: missing required parameter 'messages'",
        "type": "invalid_request_error",
        "code": "missing_required_parameter"
    }
}

# Context length exceeded (413)
ERROR_413_CONTEXT_LENGTH_EXCEEDED = {
    "error": {
        "message": "This model's maximum context length is 8192 tokens. Your request used 10000 tokens.",
        "type": "invalid_request_error",
        "code": "context_length_exceeded"
    }
}

# Internal server error (500)
ERROR_500_INTERNAL_SERVER = {
    "error": {
        "message": "The server encountered an error processing your request",
        "type": "server_error",
        "code": "internal_error"
    }
}

# Bad gateway (502)
ERROR_502_BAD_GATEWAY = {
    "error": {
        "message": "Bad gateway: upstream provider is unavailable",
        "type": "server_error",
        "code": "bad_gateway"
    }
}

# Service unavailable (503)
ERROR_503_SERVICE_UNAVAILABLE = {
    "error": {
        "message": "Service temporarily unavailable. Please try again later.",
        "type": "server_error",
        "code": "service_unavailable"
    }
}


# ============================================================================
# EDGE CASES & MALFORMED RESPONSES
# ============================================================================

# Response with missing choices
COMPLETION_MISSING_CHOICES = {
    "id": "gen-missing123456789",
    "model": "anthropic/claude-3.5-sonnet",
    "object": "chat.completion",
    "created": 1728475208,
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    }
}

# Response with empty choices array
COMPLETION_EMPTY_CHOICES = {
    "id": "gen-empty987654321",
    "model": "openai/gpt-4",
    "object": "chat.completion",
    "created": 1728475209,
    "choices": [],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 0,
        "total_tokens": 10
    }
}

# Response with missing message in choice
COMPLETION_MISSING_MESSAGE = {
    "id": "gen-nomsg123456789",
    "model": "anthropic/claude-3.5-sonnet",
    "object": "chat.completion",
    "created": 1728475210,
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    }
}

# Models list with missing context_length
MODELS_LIST_MISSING_CONTEXT = {
    "data": [
        {
            "id": "custom/model-without-context",
            "name": "Custom Model",
            "created": 1728475200,
            "description": "Model without context length specified",
            "pricing": {
                "prompt": "0.000001",
                "completion": "0.000002"
            }
        }
    ]
}


# ============================================================================
# CONTEXT WINDOW MAPPING
# ============================================================================

# Known context windows for common OpenRouter models
CONTEXT_WINDOWS = {
    # Anthropic models
    "anthropic/claude-3.5-sonnet": 200000,
    "anthropic/claude-3-opus": 200000,
    "anthropic/claude-3-haiku": 200000,
    "anthropic/claude-2": 100000,

    # OpenAI models
    "openai/gpt-4-turbo": 128000,
    "openai/gpt-4": 8192,
    "openai/gpt-3.5-turbo": 16385,

    # Meta Llama models
    "meta-llama/llama-3.1-405b-instruct": 131072,
    "meta-llama/llama-3.1-70b-instruct": 131072,
    "meta-llama/llama-3.1-8b-instruct": 131072,

    # Mistral models
    "mistralai/mistral-large": 32768,
    "mistralai/mixtral-8x7b-instruct": 32768,

    # Google models
    "google/gemini-pro": 32768,
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_completion_response(
    model: str = "anthropic/claude-3.5-sonnet",
    content: str = "Test response",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    finish_reason: str = "stop",
    include_usage: bool = True
) -> Dict[str, Any]:
    """Create a custom completion response for testing.

    Args:
        model: Model identifier (provider/model-name format)
        content: Response text content
        prompt_tokens: Number of prompt tokens used
        completion_tokens: Number of completion tokens generated
        finish_reason: Reason for completion ("stop", "length", "content_filter")
        include_usage: Whether to include usage data

    Returns:
        Dictionary containing OpenRouter completion response

    Example:
        >>> response = create_completion_response(
        ...     model="openai/gpt-4-turbo",
        ...     content="Custom response",
        ...     completion_tokens=50
        ... )
    """
    response = {
        "id": f"gen-test-{hash(content) & 0xFFFFFFFF:08x}",
        "model": model,
        "object": "chat.completion",
        "created": 1728475200,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": finish_reason
            }
        ]
    }

    if include_usage:
        response["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }

    return response


def create_error_response(
    status_code: int,
    message: str,
    error_type: str = "invalid_request_error",
    code: str = "error"
) -> Dict[str, Any]:
    """Create a custom error response for testing.

    Args:
        status_code: HTTP status code
        message: Error message text
        error_type: Error type category
        code: Specific error code

    Returns:
        Dictionary containing OpenRouter error response

    Example:
        >>> error = create_error_response(
        ...     429,
        ...     "Rate limit exceeded",
        ...     "rate_limit_error",
        ...     "rate_limit_exceeded"
        ... )
    """
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": code
        }
    }


def create_model_info(
    model_id: str,
    model_name: str,
    context_length: int = 8192,
    prompt_price: str = "0.000001",
    completion_price: str = "0.000002",
    description: str = "Test model"
) -> Dict[str, Any]:
    """Create a custom model info entry for testing.

    Args:
        model_id: Model identifier (provider/model-name format)
        model_name: Human-readable model name
        context_length: Context window size in tokens
        prompt_price: Price per token for prompts (as string)
        completion_price: Price per token for completions (as string)
        description: Model description

    Returns:
        Dictionary containing OpenRouter model info

    Example:
        >>> model = create_model_info(
        ...     model_id="custom/test-model",
        ...     model_name="Test Model",
        ...     context_length=16384
        ... )
    """
    return {
        "id": model_id,
        "name": model_name,
        "created": 1728475200,
        "description": description,
        "context_length": context_length,
        "pricing": {
            "prompt": prompt_price,
            "completion": completion_price
        },
        "top_provider": {
            "context_length": context_length,
            "max_completion_tokens": min(context_length // 2, 8192),
            "is_moderated": False
        }
    }


def create_models_list(model_infos: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a custom models list response for testing.

    Args:
        model_infos: List of model info dictionaries

    Returns:
        Dictionary containing OpenRouter models list response

    Example:
        >>> models = create_models_list([
        ...     create_model_info("model1", "Model 1", 8192),
        ...     create_model_info("model2", "Model 2", 16384)
        ... ])
    """
    return {
        "data": model_infos
    }


def create_models_list_from_ids(
    model_ids: List[str],
    context_length: int = 8192
) -> Dict[str, Any]:
    """Create a custom models list response from model IDs.

    Args:
        model_ids: List of model identifiers
        context_length: Default context length for all models

    Returns:
        Dictionary containing OpenRouter models list response

    Example:
        >>> models = create_models_list_from_ids([
        ...     "openai/gpt-4-turbo",
        ...     "anthropic/claude-3.5-sonnet"
        ... ])
    """
    return {
        "data": [
            create_model_info(
                model_id=model_id,
                model_name=model_id.split("/")[-1].replace("-", " ").title(),
                context_length=context_length
            )
            for model_id in model_ids
        ]
    }
