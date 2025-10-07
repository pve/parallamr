"""Mock API response fixtures for OpenAI provider testing."""

from typing import Any, Dict

# Successful completion response
SUCCESSFUL_COMPLETION = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
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
        "completion_tokens": 20,
        "total_tokens": 30
    }
}

# Successful completion with streaming disabled
COMPLETION_NO_STREAM = {
    "id": "chatcmpl-456",
    "object": "chat.completion",
    "created": 1677652289,
    "model": "gpt-3.5-turbo",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "GPT-3.5-turbo response"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 10,
        "total_tokens": 15
    }
}

# Response with multiple choices (not typical for chat completions but possible)
COMPLETION_MULTIPLE_CHOICES = {
    "id": "chatcmpl-789",
    "object": "chat.completion",
    "created": 1677652290,
    "model": "gpt-4",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "First response"
            },
            "finish_reason": "stop"
        },
        {
            "index": 1,
            "message": {
                "role": "assistant",
                "content": "Second response"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 30,
        "total_tokens": 40
    }
}

# Response without usage data (fallback to token estimation)
COMPLETION_NO_USAGE = {
    "id": "chatcmpl-999",
    "object": "chat.completion",
    "created": 1677652291,
    "model": "gpt-4",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Response without usage data"
            },
            "finish_reason": "stop"
        }
    ]
}

# Models list response (OpenAI /v1/models format)
MODELS_LIST_RESPONSE = {
    "object": "list",
    "data": [
        {
            "id": "gpt-4",
            "object": "model",
            "created": 1677610602,
            "owned_by": "openai",
            "permission": [],
            "root": "gpt-4",
            "parent": None
        },
        {
            "id": "gpt-4-turbo-preview",
            "object": "model",
            "created": 1677610603,
            "owned_by": "openai",
            "permission": [],
            "root": "gpt-4-turbo-preview",
            "parent": None
        },
        {
            "id": "gpt-3.5-turbo",
            "object": "model",
            "created": 1677610604,
            "owned_by": "openai",
            "permission": [],
            "root": "gpt-3.5-turbo",
            "parent": None
        },
        {
            "id": "gpt-3.5-turbo-16k",
            "object": "model",
            "created": 1677610605,
            "owned_by": "openai",
            "permission": [],
            "root": "gpt-3.5-turbo-16k",
            "parent": None
        }
    ]
}

# Empty models list
MODELS_LIST_EMPTY = {
    "object": "list",
    "data": []
}

# Model information with context window
MODEL_INFO_GPT4 = {
    "id": "gpt-4",
    "object": "model",
    "created": 1677610602,
    "owned_by": "openai",
    "context_window": 8192,
    "max_tokens": 8192
}

MODEL_INFO_GPT35_TURBO = {
    "id": "gpt-3.5-turbo",
    "object": "model",
    "created": 1677610604,
    "owned_by": "openai",
    "context_window": 4096,
    "max_tokens": 4096
}

MODEL_INFO_GPT35_16K = {
    "id": "gpt-3.5-turbo-16k",
    "object": "model",
    "created": 1677610605,
    "owned_by": "openai",
    "context_window": 16384,
    "max_tokens": 16384
}

# Error responses
ERROR_401_UNAUTHORIZED = {
    "error": {
        "message": "Incorrect API key provided",
        "type": "invalid_request_error",
        "param": None,
        "code": "invalid_api_key"
    }
}

ERROR_403_FORBIDDEN = {
    "error": {
        "message": "You do not have access to this model",
        "type": "invalid_request_error",
        "param": "model",
        "code": "model_not_found"
    }
}

ERROR_404_NOT_FOUND = {
    "error": {
        "message": "The model 'invalid-model' does not exist",
        "type": "invalid_request_error",
        "param": "model",
        "code": "model_not_found"
    }
}

ERROR_429_RATE_LIMIT = {
    "error": {
        "message": "Rate limit exceeded. Please try again later.",
        "type": "rate_limit_error",
        "param": None,
        "code": "rate_limit_exceeded"
    }
}

ERROR_500_INTERNAL_SERVER = {
    "error": {
        "message": "The server had an error processing your request",
        "type": "server_error",
        "param": None,
        "code": "internal_server_error"
    }
}

ERROR_502_BAD_GATEWAY = {
    "error": {
        "message": "Bad gateway",
        "type": "server_error",
        "param": None,
        "code": "bad_gateway"
    }
}

ERROR_503_SERVICE_UNAVAILABLE = {
    "error": {
        "message": "Service temporarily unavailable",
        "type": "server_error",
        "param": None,
        "code": "service_unavailable"
    }
}

ERROR_413_PAYLOAD_TOO_LARGE = {
    "error": {
        "message": "This model's maximum context length is 4096 tokens. However, your messages resulted in 5000 tokens.",
        "type": "invalid_request_error",
        "param": "messages",
        "code": "context_length_exceeded"
    }
}

ERROR_400_BAD_REQUEST = {
    "error": {
        "message": "Invalid request: missing required parameter 'messages'",
        "type": "invalid_request_error",
        "param": "messages",
        "code": "missing_required_parameter"
    }
}

# Azure OpenAI specific responses
AZURE_COMPLETION_RESPONSE = {
    "id": "chatcmpl-azure-123",
    "object": "chat.completion",
    "created": 1677652300,
    "model": "gpt-4",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Response from Azure OpenAI"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 8,
        "completion_tokens": 15,
        "total_tokens": 23
    }
}

# LocalAI compatible response
LOCALAI_COMPLETION_RESPONSE = {
    "id": "chatcmpl-local-456",
    "object": "chat.completion",
    "created": 1677652301,
    "model": "llama-2-7b",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Response from LocalAI"
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

# Together AI compatible response
TOGETHER_COMPLETION_RESPONSE = {
    "id": "chatcmpl-together-789",
    "object": "chat.completion",
    "created": 1677652302,
    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Response from Together AI"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30
    }
}

# Context window data by model
CONTEXT_WINDOWS = {
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-turbo-preview": 128000,
    "gpt-4-1106-preview": 128000,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-3.5-turbo-1106": 16385,
    "text-davinci-003": 4096,
    "text-davinci-002": 4096,
}


def create_completion_response(
    model: str = "gpt-4",
    content: str = "Test response",
    prompt_tokens: int = 10,
    completion_tokens: int = 20
) -> Dict[str, Any]:
    """Create a custom completion response for testing."""
    return {
        "id": f"chatcmpl-test-{model}",
        "object": "chat.completion",
        "created": 1677652288,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }


def create_error_response(
    status_code: int,
    message: str,
    error_type: str = "invalid_request_error",
    code: str = "error"
) -> Dict[str, Any]:
    """Create a custom error response for testing."""
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code
        }
    }


def create_models_list_response(model_ids: list[str]) -> Dict[str, Any]:
    """Create a custom models list response for testing."""
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 1677610600 + i,
                "owned_by": "openai",
                "permission": [],
                "root": model_id,
                "parent": None
            }
            for i, model_id in enumerate(model_ids)
        ]
    }
