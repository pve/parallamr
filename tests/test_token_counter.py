"""Tests for token counting utilities."""

import pytest

from parallamr.token_counter import (
    estimate_tokens,
    estimate_tokens_detailed,
    format_token_info,
    validate_context_window,
)


class TestEstimateTokens:
    """Test token estimation."""

    def test_estimate_tokens_empty(self):
        """Test estimating tokens for empty text."""
        assert estimate_tokens("") == 0

    def test_estimate_tokens_simple(self):
        """Test estimating tokens for simple text."""
        text = "Hello world"  # 11 characters
        expected = 11 // 4  # 2 tokens
        assert estimate_tokens(text) == expected

    def test_estimate_tokens_longer_text(self):
        """Test estimating tokens for longer text."""
        text = "This is a longer text with multiple words and punctuation."  # 58 characters
        expected = 58 // 4  # 14 tokens
        assert estimate_tokens(text) == expected

    def test_estimate_tokens_unicode(self):
        """Test estimating tokens with Unicode characters."""
        text = "Hello 世界"  # 8 characters including Unicode
        expected = 8 // 4  # 2 tokens
        assert estimate_tokens(text) == expected

    def test_estimate_tokens_newlines(self):
        """Test estimating tokens with newlines."""
        text = "Line 1\nLine 2\nLine 3"  # 19 characters
        expected = 19 // 4  # 4 tokens
        assert estimate_tokens(text) == expected


class TestEstimateTokensDetailed:
    """Test detailed token estimation."""

    def test_detailed_empty(self):
        """Test detailed estimation for empty text."""
        result = estimate_tokens_detailed("")

        expected = {
            "characters": 0,
            "words": 0,
            "lines": 0,
            "estimated_tokens": 0,
        }

        assert result == expected

    def test_detailed_simple_text(self):
        """Test detailed estimation for simple text."""
        text = "Hello world"
        result = estimate_tokens_detailed(text)

        expected = {
            "characters": 11,
            "words": 2,
            "lines": 1,
            "estimated_tokens": 2,  # 11 // 4
        }

        assert result == expected

    def test_detailed_multiline_text(self):
        """Test detailed estimation for multiline text."""
        text = "Line 1\nLine 2\nLine 3"
        result = estimate_tokens_detailed(text)

        expected = {
            "characters": 19,
            "words": 6,
            "lines": 3,
            "estimated_tokens": 4,  # 19 // 4
        }

        assert result == expected

    def test_detailed_complex_text(self):
        """Test detailed estimation for complex text."""
        text = "This is a test.\n\nWith multiple lines,\npunctuation, and 123 numbers!"
        result = estimate_tokens_detailed(text)

        assert result["characters"] == len(text)
        assert result["words"] == len(text.split())
        assert result["lines"] == text.count('\n') + 1
        assert result["estimated_tokens"] == len(text) // 4


class TestValidateContextWindow:
    """Test context window validation."""

    def test_validate_unknown_context_window(self):
        """Test validation with unknown context window."""
        is_valid, warning = validate_context_window(100, None)

        assert is_valid is True
        assert warning == "Context window unknown for model"

    def test_validate_within_limits(self):
        """Test validation within context window limits."""
        is_valid, warning = validate_context_window(100, 1000)

        assert is_valid is True
        assert warning is None

    def test_validate_approaching_limit(self):
        """Test validation approaching context window limit."""
        # 850 tokens with 1000 context window (85% usage)
        is_valid, warning = validate_context_window(850, 1000)

        assert is_valid is True
        assert warning is not None
        assert "approaching context window limit" in warning

    def test_validate_exceeds_available(self):
        """Test validation exceeding available context window."""
        # 950 tokens with 1000 context window and 10% buffer = 900 available
        is_valid, warning = validate_context_window(950, 1000, buffer_percentage=0.1)

        assert is_valid is False
        assert warning is not None
        assert "exceed available context window" in warning

    def test_validate_custom_buffer(self):
        """Test validation with custom buffer percentage."""
        # 800 tokens with 1000 context window and 20% buffer = 800 available
        is_valid, warning = validate_context_window(800, 1000, buffer_percentage=0.2)

        assert is_valid is True
        assert warning is None

    def test_validate_zero_tokens(self):
        """Test validation with zero tokens."""
        is_valid, warning = validate_context_window(0, 1000)

        assert is_valid is True
        assert warning is None


class TestFormatTokenInfo:
    """Test token information formatting."""

    def test_format_without_context_window(self):
        """Test formatting without context window."""
        result = format_token_info(1000)
        assert result == "1,000 tokens"

    def test_format_with_context_window(self):
        """Test formatting with context window."""
        result = format_token_info(500, 2000)
        assert result == "500 tokens (25.0% of 2,000 context window)"

    def test_format_large_numbers(self):
        """Test formatting with large numbers."""
        result = format_token_info(123456, 1000000)
        assert result == "123,456 tokens (12.3% of 1,000,000 context window)"

    def test_format_zero_tokens(self):
        """Test formatting with zero tokens."""
        result = format_token_info(0, 1000)
        assert result == "0 tokens (0.0% of 1,000 context window)"

    def test_format_exact_percentage(self):
        """Test formatting with exact percentage."""
        result = format_token_info(250, 1000)
        assert result == "250 tokens (25.0% of 1,000 context window)"