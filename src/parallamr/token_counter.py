"""Token counting utilities for estimating LLM input/output sizes."""

from typing import Dict, Optional


def estimate_tokens(text: str) -> int:
    """
    Estimate tokens using character count / 4 approximation.

    This is a simple, provider-agnostic approach that provides
    reasonable estimates for most text. More sophisticated
    tokenization could be added in future versions.

    Args:
        text: Input text to count tokens for

    Returns:
        Estimated number of tokens
    """
    if not text:
        return 0

    # Basic character-based estimation
    # This accounts for the fact that tokens are typically 3-4 characters on average
    return len(text) // 4


def estimate_tokens_detailed(text: str) -> Dict[str, int]:
    """
    Provide detailed token estimation breakdown.

    Args:
        text: Input text to analyze

    Returns:
        Dictionary with detailed token estimation metrics
    """
    if not text:
        return {
            "characters": 0,
            "words": 0,
            "lines": 0,
            "estimated_tokens": 0,
        }

    character_count = len(text)
    word_count = len(text.split())
    line_count = text.count('\n') + 1
    estimated_tokens = estimate_tokens(text)

    return {
        "characters": character_count,
        "words": word_count,
        "lines": line_count,
        "estimated_tokens": estimated_tokens,
    }


def validate_context_window(
    input_tokens: int,
    context_window: Optional[int],
    buffer_percentage: float = 0.1,
    model: Optional[str] = None,
    provider: Optional[str] = None
) -> tuple[bool, Optional[str]]:
    """
    Validate if input fits within model's context window.

    Args:
        input_tokens: Number of input tokens
        context_window: Model's context window size (None if unknown)
        buffer_percentage: Percentage of context window to reserve for output
        model: Model name (optional, for better error messages)
        provider: Provider name (optional, for better error messages)

    Returns:
        Tuple of (is_valid, warning_message)
    """
    if context_window is None:
        model_info = f"{provider}/{model}" if provider and model else "model"
        return True, f"Context window unknown for {model_info}"

    # Reserve buffer for output tokens
    available_tokens = int(context_window * (1 - buffer_percentage))

    if input_tokens > available_tokens:
        return False, f"Input tokens ({input_tokens}) exceed available context window ({available_tokens}/{context_window})"
    elif input_tokens > context_window * 0.8:  # Warning at 80% usage
        return True, f"Input tokens ({input_tokens}) approaching context window limit ({context_window})"

    return True, None


def format_token_info(tokens: int, context_window: Optional[int] = None) -> str:
    """
    Format token information for display.

    Args:
        tokens: Number of tokens
        context_window: Context window size if available

    Returns:
        Formatted string with token information
    """
    if context_window:
        percentage = (tokens / context_window) * 100
        return f"{tokens:,} tokens ({percentage:.1f}% of {context_window:,} context window)"
    else:
        return f"{tokens:,} tokens"