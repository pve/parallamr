"""JSON extraction and flattening utilities for LLM outputs."""

import json
import re
from typing import Any, Dict, List, Optional, Set


def extract_json(output: str) -> Optional[dict]:
    """
    Extract JSON from LLM output with multiple fallback strategies.

    Tries extraction in the following order:
    1. JSON from markdown code blocks (```json ... ```)
    2. Direct JSON parsing of the entire output
    3. Finding inline JSON objects or arrays

    Args:
        output: Raw LLM output string

    Returns:
        Extracted JSON as dictionary, or None if no valid JSON found

    Note:
        Returns None silently if no JSON is found - this is not considered an error.
    """
    if not output or not isinstance(output, str):
        return None

    output = output.strip()
    if not output:
        return None

    # Strategy 1: Try markdown code block extraction
    json_obj = _extract_from_markdown(output)
    if json_obj is not None:
        return json_obj

    # Strategy 2: Try direct JSON parsing
    json_obj = _try_parse_json(output)
    if json_obj is not None:
        return json_obj

    # Strategy 3: Try finding inline JSON
    json_obj = _extract_inline_json(output)
    if json_obj is not None:
        return json_obj

    return None


def flatten_json(
    data: dict,
    max_depth: int = 5,
    max_array: int = 10,
    parent_key: str = "",
    current_depth: int = 0
) -> Dict[str, Any]:
    """
    Flatten nested JSON structure into single-level dictionary.

    Flattening rules:
    - Nested objects: {"a": {"b": "c"}} -> {"a_b": "c"}
    - Arrays: {"x": [1, 2]} -> {"x_0": 1, "x_1": 2}
    - Arrays are limited to max_array elements
    - Nesting is limited to max_depth levels

    Args:
        data: Dictionary to flatten
        max_depth: Maximum nesting depth to process (default: 5)
        max_array: Maximum array elements to extract (default: 10)
        parent_key: Internal - current parent key for recursion
        current_depth: Internal - current recursion depth

    Returns:
        Flattened dictionary with string keys and primitive values

    Examples:
        >>> flatten_json({"a": {"b": "c"}})
        {"a_b": "c"}

        >>> flatten_json({"x": [1, 2, 3]})
        {"x_0": 1, "x_1": 2, "x_2": 3}
    """
    if not isinstance(data, dict):
        return {}

    result = {}

    for key, value in data.items():
        # Sanitize the key name
        sanitized_key = _sanitize_key(key)

        # Build the full key path
        new_key = f"{parent_key}_{sanitized_key}" if parent_key else sanitized_key

        # Handle different value types
        if value is None:
            result[new_key] = ""
        elif isinstance(value, (str, int, float, bool)):
            result[new_key] = value
        elif isinstance(value, dict) and current_depth < max_depth:
            # Recursively flatten nested objects
            nested = flatten_json(
                value,
                max_depth=max_depth,
                max_array=max_array,
                parent_key=new_key,
                current_depth=current_depth + 1
            )
            result.update(nested)
        elif isinstance(value, list) and current_depth < max_depth:
            # Flatten arrays with element indexing
            flattened_array = _flatten_array(
                value,
                new_key,
                max_array=max_array,
                max_depth=max_depth,
                current_depth=current_depth
            )
            result.update(flattened_array)
        else:
            # Depth limit reached or unsupported type - convert to string
            result[new_key] = str(value)

    return result


def resolve_column_names(
    flat_data: Dict[str, Any],
    reserved_columns: Set[str],
    experiment_vars: Set[str]
) -> Dict[str, Any]:
    """
    Resolve column name conflicts by prefixing with 'json_'.

    Args:
        flat_data: Flattened JSON data
        reserved_columns: Set of reserved column names (status, output, etc.)
        experiment_vars: Set of experiment variable names from CSV

    Returns:
        Dictionary with resolved column names

    Examples:
        >>> resolve_column_names(
        ...     {"status": "ok", "value": 42},
        ...     {"status", "output"},
        ...     set()
        ... )
        {"json_status": "ok", "value": 42}
    """
    if not flat_data:
        return {}

    all_reserved = reserved_columns | experiment_vars
    resolved = {}

    for key, value in flat_data.items():
        # Check if key conflicts with reserved or experiment columns
        if key in all_reserved:
            resolved_key = f"json_{key}"
        else:
            resolved_key = key

        resolved[resolved_key] = value

    return resolved


def _extract_from_markdown(output: str) -> Optional[dict]:
    """
    Extract JSON from markdown code blocks.

    Looks for patterns like:
    ```json
    {"key": "value"}
    ```

    or

    ```
    {"key": "value"}
    ```
    """
    # Try with explicit json language specifier
    patterns = [
        r'```json\s*\n(.*?)\n```',  # ```json ... ```
        r'```\s*\n(\{.*?\})\s*\n```',  # ``` {...} ```
        r'```\s*\n(\[.*?\])\s*\n```',  # ``` [...] ```
    ]

    for pattern in patterns:
        match = re.search(pattern, output, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            parsed = _try_parse_json(json_str)
            if parsed is not None:
                return parsed

    return None


def _extract_inline_json(output: str) -> Optional[dict]:
    """
    Find and extract JSON objects or arrays embedded in text.

    Looks for the first valid JSON object {...} or array [...] in the text.
    """
    # Try to find JSON object
    obj_match = re.search(r'\{[^}]*\}', output, re.DOTALL)
    if obj_match:
        # Expand match to handle nested braces
        start = obj_match.start()
        json_str = _extract_balanced_json(output, start)
        if json_str:
            parsed = _try_parse_json(json_str)
            if parsed is not None:
                return parsed

    # Try to find JSON array
    arr_match = re.search(r'\[[^\]]*\]', output, re.DOTALL)
    if arr_match:
        start = arr_match.start()
        json_str = _extract_balanced_json(output, start)
        if json_str:
            parsed = _try_parse_json(json_str)
            if parsed is not None and isinstance(parsed, list):
                # Wrap array in object to maintain dict return type
                return {"items": parsed}

    return None


def _extract_balanced_json(text: str, start_pos: int) -> Optional[str]:
    """
    Extract a balanced JSON structure starting from start_pos.

    Handles nested braces and brackets properly.
    """
    if start_pos >= len(text):
        return None

    open_char = text[start_pos]
    if open_char == '{':
        close_char = '}'
    elif open_char == '[':
        close_char = ']'
    else:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start_pos, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start_pos:i+1]

    return None


def _try_parse_json(text: str) -> Optional[dict]:
    """
    Attempt to parse JSON string, returning None on failure.
    """
    try:
        parsed = json.loads(text)
        # Only return if it's a dictionary
        if isinstance(parsed, dict):
            return parsed
        # If it's a list, wrap it in a dict
        elif isinstance(parsed, list):
            return {"items": parsed}
        return None
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _flatten_array(
    array: List[Any],
    key: str,
    max_array: int,
    max_depth: int,
    current_depth: int
) -> Dict[str, Any]:
    """
    Flatten an array into indexed keys.

    Args:
        array: List to flatten
        key: Parent key name
        max_array: Maximum elements to process
        max_depth: Maximum nesting depth
        current_depth: Current recursion depth

    Returns:
        Dictionary with indexed keys (key_0, key_1, etc.)
    """
    result = {}

    # Limit array size
    items_to_process = min(len(array), max_array)

    for i in range(items_to_process):
        item = array[i]
        item_key = f"{key}_{i}"

        if item is None:
            result[item_key] = ""
        elif isinstance(item, (str, int, float, bool)):
            result[item_key] = item
        elif isinstance(item, dict) and current_depth < max_depth:
            # Recursively flatten nested objects in arrays
            nested = flatten_json(
                item,
                max_depth=max_depth,
                max_array=max_array,
                parent_key=item_key,
                current_depth=current_depth + 1
            )
            result.update(nested)
        elif isinstance(item, list) and current_depth < max_depth:
            # Recursively flatten nested arrays
            nested_array = _flatten_array(
                item,
                item_key,
                max_array=max_array,
                max_depth=max_depth,
                current_depth=current_depth + 1
            )
            result.update(nested_array)
        else:
            # Convert complex types to string
            result[item_key] = str(item)

    return result


def _sanitize_key(key: str) -> str:
    """
    Sanitize JSON key for use as CSV column name.

    Rules:
    - Replace hyphens with underscores
    - Replace spaces with underscores
    - Remove or replace other special characters
    - Ensure key starts with letter or underscore

    Args:
        key: Original key name

    Returns:
        Sanitized key name safe for CSV columns
    """
    # Replace common separators with underscores
    sanitized = key.replace('-', '_').replace(' ', '_')

    # Remove or replace other special characters
    sanitized = re.sub(r'[^\w]', '_', sanitized)

    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)

    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')

    # Ensure it starts with letter or underscore (not a digit)
    if sanitized and sanitized[0].isdigit():
        sanitized = f"_{sanitized}"

    # Handle empty key
    if not sanitized:
        sanitized = "field"

    return sanitized
