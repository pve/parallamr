"""Tests for path template substitution and filename sanitization."""

import pytest
from pathlib import Path

from parallamr.path_template import (
    sanitize_filename,
    substitute_path_template,
    PathSubstitutionError,
)


class TestSanitizeFilename:
    """Tests for filename sanitization."""

    def test_clean_filename_unchanged(self):
        """Clean filenames should not be modified."""
        assert sanitize_filename("results.csv") == "results.csv"
        assert sanitize_filename("my-file_123.txt") == "my-file_123.txt"
        assert sanitize_filename("results-2024.csv") == "results-2024.csv"

    def test_forbidden_characters_replaced(self):
        """Forbidden characters should be replaced with underscores."""
        # Windows forbidden characters: < > : " / \ | ? *
        assert sanitize_filename("file<name>.csv") == "file_name_.csv"
        assert sanitize_filename("file>name.csv") == "file_name.csv"
        assert sanitize_filename("file:name.csv") == "file_name.csv"
        assert sanitize_filename('file"name.csv') == "file_name.csv"
        assert sanitize_filename("file/name.csv") == "file_name.csv"
        assert sanitize_filename("file\\name.csv") == "file_name.csv"
        assert sanitize_filename("file|name.csv") == "file_name.csv"
        assert sanitize_filename("file?name.csv") == "file_name.csv"
        assert sanitize_filename("file*name.csv") == "file_name.csv"

    def test_multiple_forbidden_characters(self):
        """Multiple forbidden characters should all be replaced."""
        assert sanitize_filename("a<b>c:d.csv") == "a_b_c_d.csv"
        assert sanitize_filename("file/with\\slashes.txt") == "file_with_slashes.txt"

    def test_trailing_dots_and_spaces_removed(self):
        """Trailing dots and spaces should be removed (Windows requirement)."""
        assert sanitize_filename("filename.") == "filename"
        assert sanitize_filename("filename ") == "filename"
        assert sanitize_filename("filename. ") == "filename"
        assert sanitize_filename("filename.csv.") == "filename.csv"

    def test_windows_reserved_names_modified(self):
        """Windows reserved names should be modified."""
        reserved = ["CON", "PRN", "AUX", "NUL", "COM1", "COM2", "LPT1", "LPT2"]
        for name in reserved:
            # Test uppercase
            assert sanitize_filename(name) == f"{name}_file"
            # Test lowercase
            assert sanitize_filename(name.lower()) == f"{name.lower()}_file"
            # Test with extension
            assert sanitize_filename(f"{name}.csv") == f"{name}_file.csv"

    def test_length_limit(self):
        """Filenames should be truncated to 255 characters."""
        long_name = "a" * 300
        result = sanitize_filename(long_name)
        assert len(result) == 255

    def test_length_limit_preserves_extension(self):
        """Long filenames should preserve extension if possible."""
        long_name = "a" * 300 + ".csv"
        result = sanitize_filename(long_name)
        assert len(result) <= 255
        # Should end with .csv if there's room
        if len(result) > 4:
            assert result.endswith(".csv")

    def test_empty_filename_raises_error(self):
        """Empty filename should raise error."""
        with pytest.raises(ValueError, match="Filename cannot be empty"):
            sanitize_filename("")
        with pytest.raises(ValueError, match="Filename cannot be empty"):
            sanitize_filename("   ")

    def test_unicode_characters_preserved(self):
        """Unicode characters should be preserved."""
        assert sanitize_filename("résumé.csv") == "résumé.csv"
        assert sanitize_filename("文件.txt") == "文件.txt"
        assert sanitize_filename("файл.csv") == "файл.csv"


class TestSubstitutePathTemplate:
    """Tests for path template substitution."""

    def test_no_template_returns_original(self):
        """Paths without templates should be returned unchanged."""
        result = substitute_path_template("results.csv", {})
        assert result == Path("results.csv")

    def test_simple_variable_substitution(self):
        """Simple variable substitution should work."""
        result = substitute_path_template(
            "results-{{topic}}.csv",
            {"topic": "AI"}
        )
        assert result == Path("results-AI.csv")

    def test_multiple_variables(self):
        """Multiple variables should be substituted."""
        result = substitute_path_template(
            "{{provider}}-{{model}}-output.csv",
            {"provider": "openrouter", "model": "claude"}
        )
        assert result == Path("openrouter-claude-output.csv")

    def test_directory_with_variable(self):
        """Variables in directory paths should work."""
        result = substitute_path_template(
            "{{provider}}/results.csv",
            {"provider": "openrouter"}
        )
        assert result == Path("openrouter/results.csv")

    def test_nested_directory_with_variables(self):
        """Multiple directory levels with variables should work."""
        result = substitute_path_template(
            "{{provider}}/{{model}}/output.csv",
            {"provider": "openrouter", "model": "claude"}
        )
        assert result == Path("openrouter/claude/output.csv")

    def test_filename_sanitization_applied(self):
        """Filename should be sanitized after substitution."""
        result = substitute_path_template(
            "{{model}}.csv",
            {"model": "anthropic/claude-sonnet-4"}
        )
        # Forward slash in model name should be replaced with underscore
        assert result == Path("anthropic_claude-sonnet-4.csv")

    def test_directory_separator_preserved(self):
        """Directory separators in path should be preserved."""
        result = substitute_path_template(
            "outputs/{{topic}}/results.csv",
            {"topic": "AI"}
        )
        assert result == Path("outputs/AI/results.csv")

    def test_missing_variable_raises_error(self):
        """Missing variables should raise clear error."""
        with pytest.raises(PathSubstitutionError, match="Template variable.*not found"):
            substitute_path_template(
                "results-{{topic}}.csv",
                {"provider": "openrouter"}
            )

    def test_path_traversal_sanitized_in_filename(self):
        """Path traversal in filename should be sanitized, not rejected."""
        # When ../../../etc/passwd is in filename position, slashes get sanitized
        result = substitute_path_template(
            "{{evil}}.csv",
            {"evil": "../../../etc/passwd"}
        )
        # Slashes become underscores, dots preserved (they're valid in filenames, even ..)
        assert result == Path(".._.._.._etc_passwd.csv")
        # The key point is slashes are replaced, making path traversal impossible
        assert "/" not in result.name

    def test_path_traversal_in_directory_rejected(self):
        """Path traversal in directory component should be rejected."""
        with pytest.raises(PathSubstitutionError, match="Path traversal detected"):
            substitute_path_template(
                "{{dir}}/results.csv",
                {"dir": "../../secrets"}
            )

    def test_absolute_path_sanitized_in_filename(self):
        """Absolute paths in filename should be sanitized."""
        result = substitute_path_template(
            "{{path}}.csv",
            {"path": "/etc/passwd"}
        )
        # Leading slash becomes underscore
        assert result == Path("_etc_passwd.csv")

    def test_windows_reserved_name_sanitized(self):
        """Windows reserved names should be sanitized."""
        result = substitute_path_template(
            "{{name}}.csv",
            {"name": "CON"}
        )
        assert result == Path("CON_file.csv")

    def test_complex_realistic_example(self):
        """Test realistic complex path template."""
        result = substitute_path_template(
            "experiments/{{date}}/{{provider}}/{{model}}-{{topic}}.csv",
            {
                "date": "2024-01-15",
                "provider": "openrouter",
                "model": "anthropic/claude-sonnet-4",
                "topic": "AI Ethics"
            }
        )
        # Model should have / replaced, topic should have space preserved
        assert result == Path("experiments/2024-01-15/openrouter/anthropic_claude-sonnet-4-AI Ethics.csv")

    def test_base_dir_parameter(self):
        """Base directory parameter should be prepended."""
        result = substitute_path_template(
            "{{topic}}/results.csv",
            {"topic": "AI"},
            base_dir=Path("/tmp/experiments")
        )
        assert str(result).startswith("/tmp/experiments")
        assert result.name == "results.csv"

    def test_path_object_input(self):
        """Should accept Path objects as input."""
        result = substitute_path_template(
            Path("results-{{topic}}.csv"),
            {"topic": "AI"}
        )
        assert result == Path("results-AI.csv")

    def test_empty_variable_value(self):
        """Empty string variable values should be handled."""
        result = substitute_path_template(
            "results-{{suffix}}.csv",
            {"suffix": ""}
        )
        assert result == Path("results-.csv")

    def test_none_variable_value_treated_as_empty(self):
        """None variable values should be treated as empty string."""
        result = substitute_path_template(
            "results-{{suffix}}.csv",
            {"suffix": None}
        )
        assert result == Path("results-.csv")

    def test_numeric_variable_converted_to_string(self):
        """Numeric variables should be converted to strings."""
        result = substitute_path_template(
            "experiment-{{number}}.csv",
            {"number": 42}
        )
        assert result == Path("experiment-42.csv")

    def test_dots_in_directory_names_allowed(self):
        """Dots in directory names should be allowed (not ../ though)."""
        result = substitute_path_template(
            "{{version}}/results.csv",
            {"version": "v1.2.3"}
        )
        assert result == Path("v1.2.3/results.csv")

    def test_hidden_files_allowed(self):
        """Files starting with dot should be allowed."""
        result = substitute_path_template(
            ".{{config}}.csv",
            {"config": "hidden"}
        )
        assert result == Path(".hidden.csv")


class TestPathTemplateEdgeCases:
    """Edge case tests for path template substitution."""

    def test_multiple_slashes_in_variable_value(self):
        """Multiple slashes in variable value should all be sanitized."""
        result = substitute_path_template(
            "{{path}}.csv",
            {"path": "a/b/c/d"}
        )
        assert result == Path("a_b_c_d.csv")

    def test_backslashes_in_variable_value(self):
        """Backslashes should be sanitized."""
        result = substitute_path_template(
            "{{path}}.csv",
            {"path": "a\\b\\c"}
        )
        assert result == Path("a_b_c.csv")

    def test_mixed_slashes_in_variable_value(self):
        """Mixed forward and backslashes should be sanitized."""
        result = substitute_path_template(
            "{{path}}.csv",
            {"path": "a/b\\c/d"}
        )
        assert result == Path("a_b_c_d.csv")

    def test_special_characters_in_directory_preserved(self):
        """Special characters in static directory names should be preserved."""
        result = substitute_path_template(
            "my-results_2024/{{topic}}.csv",
            {"topic": "AI"}
        )
        assert result == Path("my-results_2024/AI.csv")

    def test_very_long_path(self):
        """Very long paths should be handled."""
        long_topic = "a" * 300
        result = substitute_path_template(
            "{{topic}}.csv",
            {"topic": long_topic}
        )
        # Filename should be truncated
        assert len(result.name) <= 255

    def test_unicode_in_variables(self):
        """Unicode in variable values should be preserved."""
        result = substitute_path_template(
            "{{topic}}.csv",
            {"topic": "人工智能"}
        )
        assert result == Path("人工智能.csv")
