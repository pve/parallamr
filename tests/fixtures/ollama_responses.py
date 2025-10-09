"""Mock API response fixtures for Ollama provider testing.

This module provides comprehensive test fixtures for the Ollama API provider,
including successful responses, error cases, and edge cases based on the
actual Ollama API format.

Ollama API Documentation: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from typing import Any, Dict, List


# ============================================================================
# SUCCESSFUL COMPLETION RESPONSES
# ============================================================================

# Standard successful completion response (non-streaming)
SUCCESSFUL_COMPLETION = {
    "model": "llama3.1:latest",
    "created_at": "2024-10-09T12:00:00.000000Z",
    "response": "This is a test response from Llama 3.1.",
    "done": True,
    "context": [1, 2, 3, 4, 5],
    "total_duration": 5000000000,
    "load_duration": 1000000000,
    "prompt_eval_count": 10,
    "prompt_eval_duration": 500000000,
    "eval_count": 20,
    "eval_duration": 3500000000
}

# Completion for a smaller model (mistral)
COMPLETION_MISTRAL = {
    "model": "mistral:latest",
    "created_at": "2024-10-09T12:01:00.000000Z",
    "response": "Response from Mistral 7B model.",
    "done": True,
    "context": [10, 20, 30],
    "total_duration": 3000000000,
    "load_duration": 500000000,
    "prompt_eval_count": 8,
    "prompt_eval_duration": 400000000,
    "eval_count": 15,
    "eval_duration": 2100000000
}

# Completion for code generation model (codellama)
COMPLETION_CODELLAMA = {
    "model": "codellama:13b",
    "created_at": "2024-10-09T12:02:00.000000Z",
    "response": "def hello_world():\n    print('Hello, World!')",
    "done": True,
    "context": [5, 15, 25, 35],
    "total_duration": 7000000000,
    "load_duration": 2000000000,
    "prompt_eval_count": 12,
    "prompt_eval_duration": 600000000,
    "eval_count": 25,
    "eval_duration": 4400000000
}

# Completion with minimal response (empty context)
COMPLETION_MINIMAL = {
    "model": "llama3.1:latest",
    "created_at": "2024-10-09T12:03:00.000000Z",
    "response": "Short reply.",
    "done": True
}

# Completion without timing data (older Ollama versions)
COMPLETION_NO_TIMING = {
    "model": "llama2:7b",
    "created_at": "2024-10-09T12:04:00.000000Z",
    "response": "Response without detailed timing information.",
    "done": True,
    "context": [1, 2, 3]
}

# Empty response (edge case)
COMPLETION_EMPTY_RESPONSE = {
    "model": "llama3.1:latest",
    "created_at": "2024-10-09T12:05:00.000000Z",
    "response": "",
    "done": True,
    "context": []
}


# ============================================================================
# MODEL LIST RESPONSES (/api/tags)
# ============================================================================

# Standard models list with various model types
MODELS_LIST_RESPONSE = {
    "models": [
        {
            "name": "llama3.1:latest",
            "modified_at": "2024-10-01T10:00:00.000000Z",
            "size": 4661211648,
            "digest": "sha256:abc123def456",
            "details": {
                "format": "gguf",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "8B",
                "quantization_level": "Q4_0"
            }
        },
        {
            "name": "mistral:latest",
            "modified_at": "2024-09-15T14:30:00.000000Z",
            "size": 4109865216,
            "digest": "sha256:def789ghi012",
            "details": {
                "format": "gguf",
                "family": "mistral",
                "families": ["mistral"],
                "parameter_size": "7B",
                "quantization_level": "Q4_0"
            }
        },
        {
            "name": "codellama:13b",
            "modified_at": "2024-08-20T08:15:00.000000Z",
            "size": 7365960384,
            "digest": "sha256:ghi345jkl678",
            "details": {
                "format": "gguf",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "13B",
                "quantization_level": "Q4_0"
            }
        },
        {
            "name": "llama2:7b",
            "modified_at": "2024-07-10T16:45:00.000000Z",
            "size": 3826793728,
            "digest": "sha256:jkl901mno234",
            "details": {
                "format": "gguf",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "7B",
                "quantization_level": "Q4_0"
            }
        }
    ]
}

# Empty models list (no models installed)
MODELS_LIST_EMPTY = {
    "models": []
}

# Models list with single model
MODELS_LIST_SINGLE = {
    "models": [
        {
            "name": "llama3.1:latest",
            "modified_at": "2024-10-01T10:00:00.000000Z",
            "size": 4661211648,
            "digest": "sha256:abc123def456",
            "details": {
                "format": "gguf",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "8B",
                "quantization_level": "Q4_0"
            }
        }
    ]
}


# ============================================================================
# MODEL INFO RESPONSES (/api/show)
# ============================================================================

# Detailed model info for Llama 3.1 with context window
MODEL_INFO_LLAMA31 = {
    "modelfile": "# Modelfile for llama3.1:latest",
    "parameters": "num_ctx 4096\nstop \"<|end|>\"\nstop \"<|eot_id|>\"",
    "template": "{{ .System }}\n\n{{ .Prompt }}",
    "details": {
        "format": "gguf",
        "family": "llama",
        "families": ["llama"],
        "parameter_size": "8B",
        "quantization_level": "Q4_0"
    },
    "model_info": {
        "general.architecture": "llama",
        "general.file_type": 2,
        "general.parameter_count": 8030261248,
        "general.quantization_version": 2,
        "llama.attention.head_count": 32,
        "llama.attention.head_count_kv": 8,
        "llama.attention.layer_norm_rms_epsilon": 0.00001,
        "llama.block_count": 32,
        "llama.context_length": 131072,
        "llama.embedding_length": 4096,
        "llama.feed_forward_length": 14336,
        "llama.rope.dimension_count": 128,
        "llama.rope.freq_base": 500000,
        "llama.vocab_size": 128256
    }
}

# Model info for Mistral with smaller context window
MODEL_INFO_MISTRAL = {
    "modelfile": "# Modelfile for mistral:latest",
    "parameters": "num_ctx 8192",
    "template": "[INST] {{ .Prompt }} [/INST]",
    "details": {
        "format": "gguf",
        "family": "mistral",
        "families": ["mistral"],
        "parameter_size": "7B",
        "quantization_level": "Q4_0"
    },
    "model_info": {
        "general.architecture": "mistral",
        "general.file_type": 2,
        "general.parameter_count": 7241732096,
        "mistral.attention.head_count": 32,
        "mistral.attention.head_count_kv": 8,
        "mistral.block_count": 32,
        "mistral.context_length": 8192,
        "mistral.embedding_length": 4096,
        "mistral.feed_forward_length": 14336,
        "mistral.vocab_size": 32000
    }
}

# Model info for CodeLlama
MODEL_INFO_CODELLAMA = {
    "modelfile": "# Modelfile for codellama:13b",
    "parameters": "num_ctx 16384",
    "template": "{{ .System }}\n\n{{ .Prompt }}",
    "details": {
        "format": "gguf",
        "family": "llama",
        "families": ["llama"],
        "parameter_size": "13B",
        "quantization_level": "Q4_0"
    },
    "model_info": {
        "general.architecture": "llama",
        "general.file_type": 2,
        "general.parameter_count": 13015863808,
        "llama.attention.head_count": 40,
        "llama.attention.head_count_kv": 40,
        "llama.block_count": 40,
        "llama.context_length": 16384,
        "llama.embedding_length": 5120,
        "llama.feed_forward_length": 13824,
        "llama.vocab_size": 32016
    }
}

# Model info without context length (edge case)
MODEL_INFO_NO_CONTEXT = {
    "modelfile": "# Modelfile for custom:model",
    "parameters": "",
    "template": "{{ .Prompt }}",
    "details": {
        "format": "gguf",
        "family": "llama",
        "families": ["llama"],
        "parameter_size": "7B",
        "quantization_level": "Q4_0"
    },
    "model_info": {
        "general.architecture": "llama",
        "general.file_type": 2,
        "general.parameter_count": 7000000000
    }
}


# ============================================================================
# ERROR RESPONSES
# ============================================================================

# Model not found (404)
ERROR_404_MODEL_NOT_FOUND = {
    "error": "model 'invalid-model:latest' not found, try pulling it first"
}

# Invalid request format (400)
ERROR_400_INVALID_REQUEST = {
    "error": "invalid request: missing required field 'model'"
}

# Model not loaded / server error (500)
ERROR_500_MODEL_NOT_LOADED = {
    "error": "failed to load model: model file corrupted or incompatible"
}

# Service unavailable (503)
ERROR_503_SERVICE_UNAVAILABLE = {
    "error": "Ollama service is temporarily unavailable"
}

# Bad gateway (502)
ERROR_502_BAD_GATEWAY = {
    "error": "Bad gateway: unable to connect to model backend"
}

# Context length exceeded (400)
ERROR_400_CONTEXT_EXCEEDED = {
    "error": "prompt exceeds model context length of 4096 tokens"
}

# Out of memory error (500)
ERROR_500_OUT_OF_MEMORY = {
    "error": "insufficient memory to load model, try a smaller model or quantization"
}


# ============================================================================
# EDGE CASES & MALFORMED RESPONSES
# ============================================================================

# Response with missing 'done' field
COMPLETION_MISSING_DONE = {
    "model": "llama3.1:latest",
    "created_at": "2024-10-09T12:06:00.000000Z",
    "response": "Response without done flag"
}

# Response with malformed timestamp
COMPLETION_MALFORMED_TIMESTAMP = {
    "model": "llama3.1:latest",
    "created_at": "invalid-timestamp",
    "response": "Response with invalid timestamp",
    "done": True
}

# Response with negative timing values (corrupted)
COMPLETION_NEGATIVE_TIMING = {
    "model": "llama3.1:latest",
    "created_at": "2024-10-09T12:07:00.000000Z",
    "response": "Response with corrupted timing",
    "done": True,
    "total_duration": -1000000000,
    "eval_count": -5
}

# Models list with missing required fields
MODELS_LIST_MALFORMED = {
    "models": [
        {
            "name": "llama3.1:latest",
            # missing 'modified_at'
            "size": 4661211648,
            "digest": "sha256:abc123def456"
        },
        {
            # missing 'name'
            "modified_at": "2024-10-01T10:00:00.000000Z",
            "size": 4109865216,
            "digest": "sha256:def789ghi012"
        }
    ]
}

# Completion response with extra unexpected fields
COMPLETION_EXTRA_FIELDS = {
    "model": "llama3.1:latest",
    "created_at": "2024-10-09T12:08:00.000000Z",
    "response": "Response with extra fields",
    "done": True,
    "extra_field_1": "unexpected_value",
    "extra_field_2": 12345,
    "nested_extra": {
        "key": "value"
    }
}


# ============================================================================
# CONTEXT WINDOW MAPPING
# ============================================================================

# Known context windows for common Ollama models
CONTEXT_WINDOWS = {
    "llama3.1:latest": 131072,
    "llama3.1:8b": 131072,
    "llama3.1:70b": 131072,
    "llama2:7b": 4096,
    "llama2:13b": 4096,
    "llama2:70b": 4096,
    "mistral:latest": 8192,
    "mistral:7b": 8192,
    "codellama:7b": 16384,
    "codellama:13b": 16384,
    "codellama:34b": 16384,
    "phi:latest": 2048,
    "gemma:2b": 8192,
    "gemma:7b": 8192,
    "qwen:latest": 32768,
    "vicuna:7b": 2048,
    "vicuna:13b": 2048,
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_completion_response(
    model: str = "llama3.1:latest",
    response: str = "Test response",
    prompt_eval_count: int = 10,
    eval_count: int = 20,
    include_timing: bool = True,
    include_context: bool = True
) -> Dict[str, Any]:
    """Create a custom completion response for testing.

    Args:
        model: Model identifier
        response: Response text content
        prompt_eval_count: Number of prompt tokens evaluated
        eval_count: Number of response tokens generated
        include_timing: Whether to include timing data
        include_context: Whether to include context array

    Returns:
        Dictionary containing Ollama completion response

    Example:
        >>> response = create_completion_response(
        ...     model="llama3.1:latest",
        ...     response="Custom test response",
        ...     eval_count=50
        ... )
    """
    completion = {
        "model": model,
        "created_at": "2024-10-09T12:00:00.000000Z",
        "response": response,
        "done": True
    }

    if include_context:
        completion["context"] = list(range(1, prompt_eval_count + 1))

    if include_timing:
        completion.update({
            "total_duration": 5000000000,
            "load_duration": 1000000000,
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_duration": 500000000,
            "eval_count": eval_count,
            "eval_duration": 3500000000
        })

    return completion


def create_error_response(
    error_message: str,
    error_type: str = "general"
) -> Dict[str, str]:
    """Create a custom error response for testing.

    Args:
        error_message: Error message text
        error_type: Type of error (not used in Ollama API, for categorization)

    Returns:
        Dictionary containing Ollama error response

    Example:
        >>> error = create_error_response("Model not found")
    """
    return {
        "error": error_message
    }


def create_model_info(
    model_name: str,
    size: int = 4661211648,
    parameter_size: str = "8B",
    context_length: int = 4096,
    architecture: str = "llama"
) -> Dict[str, Any]:
    """Create a custom model info response for testing.

    Args:
        model_name: Model identifier
        size: Model file size in bytes
        parameter_size: Model parameter count (e.g., "7B", "13B")
        context_length: Context window size in tokens
        architecture: Model architecture (llama, mistral, etc.)

    Returns:
        Dictionary containing Ollama model info response

    Example:
        >>> model = create_model_info(
        ...     model_name="custom:model",
        ...     context_length=8192
        ... )
    """
    return {
        "name": model_name,
        "modified_at": "2024-10-01T10:00:00.000000Z",
        "size": size,
        "digest": f"sha256:{hash(model_name) & 0xFFFFFFFF:08x}",
        "details": {
            "format": "gguf",
            "family": architecture,
            "families": [architecture],
            "parameter_size": parameter_size,
            "quantization_level": "Q4_0"
        }
    }


def create_model_show_response(
    model_name: str,
    context_length: int = 4096,
    architecture: str = "llama"
) -> Dict[str, Any]:
    """Create a custom model show (/api/show) response for testing.

    Args:
        model_name: Model identifier
        context_length: Context window size in tokens
        architecture: Model architecture (llama, mistral, etc.)

    Returns:
        Dictionary containing Ollama model show response

    Example:
        >>> show = create_model_show_response(
        ...     model_name="llama3.1:latest",
        ...     context_length=131072
        ... )
    """
    model_info_key = f"{architecture}.context_length"

    return {
        "modelfile": f"# Modelfile for {model_name}",
        "parameters": f"num_ctx {context_length}",
        "template": "{{ .Prompt }}",
        "details": {
            "format": "gguf",
            "family": architecture,
            "families": [architecture],
            "parameter_size": "8B",
            "quantization_level": "Q4_0"
        },
        "model_info": {
            "general.architecture": architecture,
            "general.file_type": 2,
            model_info_key: context_length
        }
    }


def create_models_list(model_names: List[str]) -> Dict[str, Any]:
    """Create a custom models list response for testing.

    Args:
        model_names: List of model identifiers

    Returns:
        Dictionary containing Ollama models list response

    Example:
        >>> models_list = create_models_list([
        ...     "llama3.1:latest",
        ...     "mistral:latest",
        ...     "codellama:13b"
        ... ])
    """
    return {
        "models": [
            create_model_info(name)
            for name in model_names
        ]
    }
