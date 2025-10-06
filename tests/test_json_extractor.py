"""Tests for JSON extraction and flattening functionality."""

import pytest

from parallamr import json_extractor


class TestJSONExtraction:
    """Test JSON extraction from various output formats."""

    def test_extract_from_markdown_code_fence(self):
        """Test extracting JSON from markdown code fence."""
        output = """Here's the result:
```json
{"name": "Alice", "age": 30}
```
That's it!"""

        result = json_extractor.extract_json(output)

        assert result == {"name": "Alice", "age": 30}

    def test_extract_from_markdown_code_fence_no_language(self):
        """Test extracting JSON from code fence without language specifier."""
        output = """Result:
```
{"name": "Bob", "score": 95}
```"""

        result = json_extractor.extract_json(output)

        assert result == {"name": "Bob", "score": 95}

    def test_extract_from_direct_json(self):
        """Test extracting JSON when output is pure JSON."""
        output = '{"status": "success", "count": 42}'

        result = json_extractor.extract_json(output)

        assert result == {"status": "success", "count": 42}

    def test_extract_from_inline_json(self):
        """Test extracting JSON embedded in text."""
        output = 'The result is {"temperature": 72, "humidity": 65} for today.'

        result = json_extractor.extract_json(output)

        assert result == {"temperature": 72, "humidity": 65}

    def test_extract_no_json_returns_none(self):
        """Test that missing JSON returns None without errors."""
        output = "This is just plain text with no JSON at all."

        result = json_extractor.extract_json(output)

        assert result is None

    def test_extract_empty_string_returns_none(self):
        """Test that empty string returns None."""
        result = json_extractor.extract_json("")

        assert result is None

    def test_extract_invalid_json_returns_none(self):
        """Test that invalid JSON returns None without raising exception."""
        output = '{"name": "Alice", invalid}'

        result = json_extractor.extract_json(output)

        assert result is None

    def test_extract_json_array_wrapped(self):
        """Test that JSON arrays are wrapped in {"items": [...]}."""
        output = '[1, 2, 3, 4, 5]'

        result = json_extractor.extract_json(output)

        assert result == {"items": [1, 2, 3, 4, 5]}

    def test_extract_multiple_code_fences_uses_first(self):
        """Test that multiple code fences extracts the first one."""
        output = """First:
```json
{"first": 1}
```
Second:
```json
{"second": 2}
```"""

        result = json_extractor.extract_json(output)

        assert result == {"first": 1}


class TestJSONFlattening:
    """Test JSON flattening with various structures."""

    def test_flatten_simple_object(self):
        """Test flattening a simple flat object."""
        data = {"name": "Alice", "age": 30, "city": "NYC"}

        result = json_extractor.flatten_json(data)

        assert result == {"name": "Alice", "age": 30, "city": "NYC"}

    def test_flatten_nested_object(self):
        """Test flattening nested objects with underscore separator."""
        data = {
            "user": {
                "name": "Bob",
                "address": {
                    "city": "Boston",
                    "zip": "02101"
                }
            }
        }

        result = json_extractor.flatten_json(data)

        assert result == {
            "user_name": "Bob",
            "user_address_city": "Boston",
            "user_address_zip": "02101"
        }

    def test_flatten_array_to_indexed_columns(self):
        """Test flattening arrays with index suffixes."""
        data = {"scores": [85, 92, 78]}

        result = json_extractor.flatten_json(data)

        assert result == {
            "scores_0": 85,
            "scores_1": 92,
            "scores_2": 78
        }

    def test_flatten_array_respects_max_limit(self):
        """Test that arrays are truncated at max_array limit."""
        data = {"values": list(range(15))}

        result = json_extractor.flatten_json(data, max_array=10)

        # Should only have indices 0-9
        assert len(result) == 10
        assert "values_9" in result
        assert "values_10" not in result

    def test_flatten_deep_nesting_respects_max_depth(self):
        """Test that deep nesting is limited to max_depth."""
        data = {
            "l1": {
                "l2": {
                    "l3": {
                        "l4": {
                            "l5": {
                                "l6": {
                                    "l7": "too deep"
                                }
                            }
                        }
                    }
                }
            }
        }

        result = json_extractor.flatten_json(data, max_depth=5)

        # Should flatten nested objects, but stop recursing at max_depth
        # This creates keys like l1_l2_l3_l4_l5_l6 where l6 is the stringified remainder
        assert any("l1_l2_l3_l4_l5" in key for key in result.keys())
        # Should have stopped flattening deeper levels
        assert "l1_l2_l3_l4_l5_l6_l7" not in result

    def test_flatten_mixed_nested_and_arrays(self):
        """Test flattening with both nested objects and arrays."""
        data = {
            "user": "Alice",
            "scores": [10, 20, 30],
            "profile": {
                "tags": ["python", "ai"]
            }
        }

        result = json_extractor.flatten_json(data)

        assert result["user"] == "Alice"
        assert result["scores_0"] == 10
        assert result["scores_1"] == 20
        assert result["scores_2"] == 30
        assert result["profile_tags_0"] == "python"
        assert result["profile_tags_1"] == "ai"

    def test_flatten_empty_object(self):
        """Test flattening an empty object."""
        data = {}

        result = json_extractor.flatten_json(data)

        assert result == {}

    def test_flatten_none_values_converted_to_empty_string(self):
        """Test that None values are converted to empty strings."""
        data = {"name": "Alice", "middle": None, "age": 30}

        result = json_extractor.flatten_json(data)

        assert result == {"name": "Alice", "middle": "", "age": 30}


class TestColumnNameResolution:
    """Test column name conflict resolution."""

    def test_resolve_no_conflicts(self):
        """Test that non-conflicting names remain unchanged."""
        flat_data = {"score": 95, "rank": 1}
        reserved = {"provider", "model", "status", "output"}
        experiment_vars = {"topic", "source"}

        result = json_extractor.resolve_column_names(flat_data, reserved, experiment_vars)

        assert result == {"score": 95, "rank": 1}

    def test_resolve_conflict_with_reserved(self):
        """Test that conflicts with reserved columns get json_ prefix."""
        flat_data = {"status": "completed", "output": "result", "score": 95}
        reserved = {"provider", "model", "status", "output"}
        experiment_vars = set()

        result = json_extractor.resolve_column_names(flat_data, reserved, experiment_vars)

        assert result == {"json_status": "completed", "json_output": "result", "score": 95}

    def test_resolve_conflict_with_experiment_vars(self):
        """Test that conflicts with experiment variables get json_ prefix."""
        flat_data = {"topic": "AI", "score": 95}
        reserved = {"provider", "model"}
        experiment_vars = {"topic", "source"}

        result = json_extractor.resolve_column_names(flat_data, reserved, experiment_vars)

        assert result == {"json_topic": "AI", "score": 95}


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_flatten_with_boolean_values(self):
        """Test flattening with boolean values."""
        data = {"active": True, "verified": False}

        result = json_extractor.flatten_json(data)

        assert result == {"active": True, "verified": False}

    def test_flatten_with_numeric_values(self):
        """Test flattening with various numeric types."""
        data = {"integer": 42, "float": 3.14, "negative": -10}

        result = json_extractor.flatten_json(data)

        assert result == {"integer": 42, "float": 3.14, "negative": -10}

    def test_flatten_with_empty_array(self):
        """Test flattening with empty array."""
        data = {"items": []}

        result = json_extractor.flatten_json(data)

        # Empty arrays should result in no columns
        assert result == {}

    def test_flatten_with_empty_nested_object(self):
        """Test flattening with empty nested object."""
        data = {"user": {}, "age": 30}

        result = json_extractor.flatten_json(data)

        # Empty objects should result in no columns for that key
        assert result == {"age": 30}

    def test_extract_json_with_unicode(self):
        """Test extracting JSON with unicode characters."""
        output = '{"message": "Hello 世界", "emoji": "🎉"}'

        result = json_extractor.extract_json(output)

        assert result == {"message": "Hello 世界", "emoji": "🎉"}

    def test_flatten_preserves_string_numbers(self):
        """Test that string numbers are not converted."""
        data = {"id": "12345", "code": "ABC-123"}

        result = json_extractor.flatten_json(data)

        assert result == {"id": "12345", "code": "ABC-123"}
        assert isinstance(result["id"], str)
