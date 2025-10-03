"""Tests for template engine."""

import pytest

from parallamr.template import (
    combine_files_with_variables,
    extract_variables,
    replace_variables,
    validate_template_syntax,
)


class TestReplaceVariables:
    """Test variable replacement functionality."""

    def test_replace_single_variable(self):
        """Test replacing a single variable."""
        text = "Hello {{name}}"
        variables = {"name": "World"}

        result, missing = replace_variables(text, variables)

        assert result == "Hello World"
        assert missing == []

    def test_replace_multiple_variables(self):
        """Test replacing multiple variables."""
        text = "{{greeting}} {{name}}, welcome to {{place}}"
        variables = {"greeting": "Hello", "name": "Alice", "place": "Paradise"}

        result, missing = replace_variables(text, variables)

        assert result == "Hello Alice, welcome to Paradise"
        assert missing == []

    def test_replace_with_missing_variables(self):
        """Test replacement with missing variables."""
        text = "{{greeting}} {{name}}, welcome to {{place}}"
        variables = {"greeting": "Hello", "name": "Alice"}

        result, missing = replace_variables(text, variables)

        assert result == "Hello Alice, welcome to {{place}}"
        assert missing == ["place"]

    def test_replace_with_none_value(self):
        """Test replacement with None value."""
        text = "Value: {{value}}"
        variables = {"value": None}

        result, missing = replace_variables(text, variables)

        assert result == "Value: "
        assert missing == []

    def test_replace_with_numeric_value(self):
        """Test replacement with numeric value."""
        text = "Count: {{count}}"
        variables = {"count": 42}

        result, missing = replace_variables(text, variables)

        assert result == "Count: 42"
        assert missing == []

    def test_replace_no_variables(self):
        """Test text with no variables."""
        text = "This is plain text"
        variables = {"unused": "value"}

        result, missing = replace_variables(text, variables)

        assert result == "This is plain text"
        assert missing == []

    def test_replace_empty_text(self):
        """Test replacement with empty text."""
        text = ""
        variables = {"name": "Alice"}

        result, missing = replace_variables(text, variables)

        assert result == ""
        assert missing == []

    def test_replace_duplicate_variables(self):
        """Test replacement with duplicate variables."""
        text = "{{name}} and {{name}} are friends"
        variables = {"name": "Alice"}

        result, missing = replace_variables(text, variables)

        assert result == "Alice and Alice are friends"
        assert missing == []


class TestValidateTemplateSyntax:
    """Test template syntax validation."""

    def test_valid_syntax(self):
        """Test valid template syntax."""
        text = "Hello {{name}}, welcome to {{place}}"
        errors = validate_template_syntax(text)
        assert errors == []

    def test_unmatched_opening_braces(self):
        """Test unmatched opening braces."""
        text = "Hello {{name, welcome to {{place}}"
        errors = validate_template_syntax(text)
        assert len(errors) == 1
        assert "Unmatched braces" in errors[0]

    def test_unmatched_closing_braces(self):
        """Test unmatched closing braces."""
        text = "Hello name}}, welcome to {{place}}"
        errors = validate_template_syntax(text)
        assert len(errors) == 1
        assert "Unmatched braces" in errors[0]

    def test_invalid_variable_name(self):
        """Test invalid variable names."""
        text = "Hello {{123invalid}}, welcome to {{valid_name}}"
        errors = validate_template_syntax(text)
        assert len(errors) == 1
        assert "Invalid variable name" in errors[0]
        assert "123invalid" in errors[0]

    def test_variable_with_spaces(self):
        """Test variable names with spaces."""
        text = "Hello {{name with spaces}}"
        errors = validate_template_syntax(text)
        assert len(errors) == 1
        assert "Invalid variable name" in errors[0]

    def test_empty_variable_name(self):
        """Test empty variable name."""
        text = "Hello {{}}"
        errors = validate_template_syntax(text)
        assert len(errors) == 1
        assert "Invalid variable name" in errors[0]


class TestExtractVariables:
    """Test variable extraction."""

    def test_extract_single_variable(self):
        """Test extracting single variable."""
        text = "Hello {{name}}"
        variables = extract_variables(text)
        assert variables == ["name"]

    def test_extract_multiple_variables(self):
        """Test extracting multiple variables."""
        text = "{{greeting}} {{name}}, welcome to {{place}}"
        variables = extract_variables(text)
        assert set(variables) == {"greeting", "name", "place"}

    def test_extract_duplicate_variables(self):
        """Test extracting duplicate variables (should be unique)."""
        text = "{{name}} and {{name}} are friends"
        variables = extract_variables(text)
        assert variables == ["name"]

    def test_extract_no_variables(self):
        """Test extracting from text with no variables."""
        text = "This is plain text"
        variables = extract_variables(text)
        assert variables == []

    def test_extract_with_underscores(self):
        """Test extracting variables with underscores."""
        text = "{{first_name}} {{last_name}}"
        variables = extract_variables(text)
        assert set(variables) == {"first_name", "last_name"}


class TestCombineFilesWithVariables:
    """Test file combination and variable replacement."""

    def test_combine_no_context_files(self):
        """Test combining with no context files."""
        primary_content = "Hello {{name}}"
        context_files = []
        variables = {"name": "World"}

        result, missing = combine_files_with_variables(
            primary_content, context_files, variables
        )

        assert result == "Hello World"
        assert missing == []

    def test_combine_with_context_files(self):
        """Test combining with context files."""
        primary_content = "Main content: {{topic}}"
        context_files = [
            ("doc1.txt", "Document 1 content"),
            ("doc2.txt", "Document 2 content")
        ]
        variables = {"topic": "AI"}

        result, missing = combine_files_with_variables(
            primary_content, context_files, variables
        )

        expected = (
            "Main content: AI\n\n"
            "## Document: doc1.txt\n\n"
            "Document 1 content\n\n"
            "---\n\n\n\n"
            "## Document: doc2.txt\n\n"
            "Document 2 content\n\n"
            "---\n\n"
        )

        assert result == expected
        assert missing == []

    def test_combine_with_variables_in_context(self):
        """Test combining with variables in context files."""
        primary_content = "Main: {{topic}}"
        context_files = [
            ("context.txt", "Context about {{topic}} and {{subtopic}}")
        ]
        variables = {"topic": "AI", "subtopic": "ML"}

        result, missing = combine_files_with_variables(
            primary_content, context_files, variables
        )

        assert "Main: AI" in result
        assert "Context about AI and ML" in result
        assert missing == []

    def test_combine_with_missing_variables(self):
        """Test combining with missing variables."""
        primary_content = "Main: {{topic}}"
        context_files = [
            ("context.txt", "About {{missing_var}}")
        ]
        variables = {"topic": "AI"}

        result, missing = combine_files_with_variables(
            primary_content, context_files, variables
        )

        assert "Main: AI" in result
        assert "About {{missing_var}}" in result
        assert missing == ["missing_var"]