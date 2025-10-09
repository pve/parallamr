"""Provider-agnostic common response fixtures.

This module provides shared fixtures for error responses and common patterns
that are consistent across all provider implementations.
"""

from typing import Dict, Any


# ============================================================================
# GENERIC ERROR RESPONSE FACTORY
# ============================================================================

def create_generic_error(
    status_code: int,
    message: str,
    error_type: str = "error",
    code: str = None
) -> Dict[str, Any]:
    """Create generic error response that works for any provider.

    Args:
        status_code: HTTP status code
        message: Error message text
        error_type: Type/category of error
        code: Specific error code (optional)

    Returns:
        Dictionary containing generic error structure

    Example:
        >>> error = create_generic_error(429, "Rate limit exceeded")
    """
    error_dict = {
        "error": {
            "message": message,
            "type": error_type,
            "status_code": status_code
        }
    }

    if code:
        error_dict["error"]["code"] = code

    return error_dict


# ============================================================================
# COMMON HTTP ERROR RESPONSES
# ============================================================================

# 4xx Client Errors
ERROR_400_BAD_REQUEST = create_generic_error(
    400,
    "Bad Request: Invalid request parameters",
    "invalid_request_error",
    "bad_request"
)

ERROR_401_UNAUTHORIZED = create_generic_error(
    401,
    "Unauthorized: Invalid or missing API key",
    "authentication_error",
    "invalid_api_key"
)

ERROR_403_FORBIDDEN = create_generic_error(
    403,
    "Forbidden: You do not have access to this resource",
    "permission_error",
    "forbidden"
)

ERROR_404_NOT_FOUND = create_generic_error(
    404,
    "Not Found: The requested resource does not exist",
    "not_found_error",
    "not_found"
)

ERROR_413_PAYLOAD_TOO_LARGE = create_generic_error(
    413,
    "Payload Too Large: Request exceeds maximum size limit",
    "request_too_large_error",
    "payload_too_large"
)

ERROR_429_RATE_LIMIT = create_generic_error(
    429,
    "Rate Limit Exceeded: Too many requests, please slow down",
    "rate_limit_error",
    "rate_limit_exceeded"
)

# 5xx Server Errors
ERROR_500_INTERNAL_SERVER = create_generic_error(
    500,
    "Internal Server Error: The server encountered an unexpected condition",
    "server_error",
    "internal_error"
)

ERROR_502_BAD_GATEWAY = create_generic_error(
    502,
    "Bad Gateway: Invalid response from upstream server",
    "server_error",
    "bad_gateway"
)

ERROR_503_SERVICE_UNAVAILABLE = create_generic_error(
    503,
    "Service Unavailable: The service is temporarily unavailable",
    "server_error",
    "service_unavailable"
)

ERROR_504_GATEWAY_TIMEOUT = create_generic_error(
    504,
    "Gateway Timeout: Upstream server did not respond in time",
    "server_error",
    "gateway_timeout"
)


# ============================================================================
# ERROR RESPONSE COLLECTIONS (for parameterized tests)
# ============================================================================

# All client errors (4xx)
CLIENT_ERRORS = [
    (400, ERROR_400_BAD_REQUEST, "bad request"),
    (401, ERROR_401_UNAUTHORIZED, "unauthorized"),
    (403, ERROR_403_FORBIDDEN, "forbidden"),
    (404, ERROR_404_NOT_FOUND, "not found"),
    (413, ERROR_413_PAYLOAD_TOO_LARGE, "payload too large"),
    (429, ERROR_429_RATE_LIMIT, "rate limit"),
]

# All server errors (5xx)
SERVER_ERRORS = [
    (500, ERROR_500_INTERNAL_SERVER, "internal server error"),
    (502, ERROR_502_BAD_GATEWAY, "bad gateway"),
    (503, ERROR_503_SERVICE_UNAVAILABLE, "service unavailable"),
    (504, ERROR_504_GATEWAY_TIMEOUT, "gateway timeout"),
]

# All HTTP errors combined
ALL_HTTP_ERRORS = CLIENT_ERRORS + SERVER_ERRORS


# ============================================================================
# ERROR DESCRIPTION MAPPINGS
# ============================================================================

ERROR_STATUS_TO_DESCRIPTION = {
    400: "Bad Request - Invalid request parameters",
    401: "Unauthorized - Invalid or missing credentials",
    403: "Forbidden - Access denied",
    404: "Not Found - Resource does not exist",
    413: "Payload Too Large - Request exceeds size limit",
    429: "Rate Limit - Too many requests",
    500: "Internal Server Error - Server encountered an error",
    502: "Bad Gateway - Invalid upstream response",
    503: "Service Unavailable - Service temporarily down",
    504: "Gateway Timeout - Upstream timeout",
}


# ============================================================================
# HELPER FUNCTIONS FOR TESTING
# ============================================================================

def is_client_error(status_code: int) -> bool:
    """Check if status code is a client error (4xx).

    Args:
        status_code: HTTP status code

    Returns:
        True if 400-499, False otherwise

    Example:
        >>> is_client_error(404)
        True
        >>> is_client_error(500)
        False
    """
    return 400 <= status_code < 500


def is_server_error(status_code: int) -> bool:
    """Check if status code is a server error (5xx).

    Args:
        status_code: HTTP status code

    Returns:
        True if 500-599, False otherwise

    Example:
        >>> is_server_error(500)
        True
        >>> is_server_error(404)
        False
    """
    return 500 <= status_code < 600


def get_error_description(status_code: int) -> str:
    """Get human-readable description for HTTP status code.

    Args:
        status_code: HTTP status code

    Returns:
        Description string

    Example:
        >>> get_error_description(404)
        'Not Found - Resource does not exist'
    """
    return ERROR_STATUS_TO_DESCRIPTION.get(
        status_code,
        f"HTTP Error {status_code}"
    )
