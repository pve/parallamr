"""Template engine for variable replacement in experiment prompts."""

import re
from typing import Any, Dict, List, Tuple


def replace_variables(text: str, variables: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Replace {{variable}} placeholders with values from the variables dictionary.

    Args:
        text: Text containing {{variable}} placeholders
        variables: Dictionary mapping variable names to their values

    Returns:
        Tuple of (replaced_text, list_of_missing_variables)
    """
    # Pattern to match {{variable_name}} allowing alphanumeric and underscore
    pattern = r'\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}'

    missing_variables = []

    def replacer(match: re.Match[str]) -> str:
        variable_name = match.group(1)
        if variable_name in variables:
            # Convert value to string, handling None gracefully
            value = variables[variable_name]
            return str(value) if value is not None else ""
        else:
            missing_variables.append(variable_name)
            # Leave the placeholder unchanged for missing variables
            return match.group(0)

    replaced_text = re.sub(pattern, replacer, text)

    return replaced_text, missing_variables


def validate_template_syntax(text: str) -> List[str]:
    """
    Validate template syntax and return any syntax errors.

    Args:
        text: Text to validate

    Returns:
        List of error messages (empty if no errors)
    """
    errors = []

    # Check for unmatched braces
    open_braces = text.count("{{")
    close_braces = text.count("}}")

    if open_braces != close_braces:
        errors.append(f"Unmatched braces: {open_braces} opening '{{{{' vs {close_braces} closing '}}}}'")

    # Check for malformed variable names
    malformed_pattern = r'\{\{([^}]*)\}\}'
    variable_pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'

    for match in re.finditer(malformed_pattern, text):
        variable_name = match.group(1)
        if not re.match(variable_pattern, variable_name):
            errors.append(f"Invalid variable name: '{variable_name}' (must start with letter/underscore, contain only alphanumeric/underscore)")

    return errors


def extract_variables(text: str) -> List[str]:
    """
    Extract all variable names from template text.

    Args:
        text: Text containing {{variable}} placeholders

    Returns:
        List of unique variable names found in the text
    """
    pattern = r'\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}'
    variables = re.findall(pattern, text)
    return list(set(variables))  # Remove duplicates


def combine_files_with_variables(
    primary_file_content: str,
    context_files: List[Tuple[str, str]],
    variables: Dict[str, Any]
) -> Tuple[str, List[str]]:
    """
    Combine primary file and context files, then replace variables.

    Args:
        primary_file_content: Content of the primary prompt file
        context_files: List of (filename, content) tuples for context files
        variables: Dictionary of variables to replace

    Returns:
        Tuple of (combined_and_replaced_text, list_of_missing_variables)
    """
    # Start with primary content
    combined_content = primary_file_content

    # Add context files with separators
    for filename, content in context_files:
        combined_content += f"\n\n## Document: {filename}\n\n{content}\n\n---\n\n"

    # Replace variables in the combined content
    return replace_variables(combined_content, variables)