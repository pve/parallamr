"""Path template substitution and filename sanitization."""

import re
from pathlib import Path
from typing import Any, Dict, Optional, Union


class PathSubstitutionError(Exception):
    """Raised when path template substitution fails."""
    pass


# Windows forbidden characters in filenames
FORBIDDEN_CHARS = r'<>:"/\|?*'

# Windows reserved names
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
}

# Maximum filename length (cross-platform safe)
MAX_FILENAME_LENGTH = 255


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename for filesystem safety.

    Removes or replaces forbidden characters, handles Windows reserved names,
    and enforces length limits.

    Args:
        filename: The filename to sanitize

    Returns:
        Sanitized filename safe for all filesystems

    Raises:
        ValueError: If filename is empty or becomes empty after sanitization
    """
    if not filename or not filename.strip():
        raise ValueError("Filename cannot be empty")

    # Replace forbidden characters with underscore
    for char in FORBIDDEN_CHARS:
        filename = filename.replace(char, '_')

    # Remove trailing dots and spaces (Windows requirement)
    filename = filename.rstrip('. ')

    # Check for empty filename after sanitization
    if not filename:
        raise ValueError("Filename cannot be empty after sanitization")

    # Handle Windows reserved names
    # Split filename into name and extension
    name_parts = filename.rsplit('.', 1)
    base_name = name_parts[0]
    extension = '.' + name_parts[1] if len(name_parts) > 1 else ''

    # Check if base name (without extension) is a reserved name
    if base_name.upper() in WINDOWS_RESERVED_NAMES:
        base_name = base_name + '_file'

    # Reconstruct filename
    filename = base_name + extension

    # Enforce length limit (preserve extension if possible)
    if len(filename) > MAX_FILENAME_LENGTH:
        if extension:
            # Try to preserve extension
            max_base_length = MAX_FILENAME_LENGTH - len(extension)
            if max_base_length > 0:
                filename = base_name[:max_base_length] + extension
            else:
                # Extension too long, just truncate everything
                filename = filename[:MAX_FILENAME_LENGTH]
        else:
            filename = filename[:MAX_FILENAME_LENGTH]

    return filename


def substitute_path_template(
    path_template: Union[str, Path],
    variables: Dict[str, Any],
    base_dir: Optional[Path] = None
) -> Path:
    """
    Substitute template variables in a path and sanitize the result.

    Template variables use {{variable_name}} syntax. Before substitution, we
    split the path into directory and filename parts based on the static structure.
    Variable values in the filename portion are sanitized to remove forbidden
    characters like slashes.

    Args:
        path_template: Path template with {{variable}} placeholders
        variables: Dictionary mapping variable names to values
        base_dir: Optional base directory to prepend to result

    Returns:
        Sanitized Path object with variables substituted

    Raises:
        PathSubstitutionError: If template variables are missing or path is unsafe

    Examples:
        >>> substitute_path_template("results-{{topic}}.csv", {"topic": "AI"})
        Path("results-AI.csv")

        >>> substitute_path_template("{{provider}}/{{model}}.csv",
        ...                          {"provider": "openrouter", "model": "claude"})
        Path("openrouter/claude.csv")
    """
    # Convert to string if Path object
    template_str = str(path_template)

    # First, split template into static directory and filename template parts
    # This must be done BEFORE substitution to identify the filename component
    template_path = Path(template_str)
    directory_template = str(template_path.parent) if template_path.parent != Path('.') else None
    filename_template = template_path.name

    # Pattern to match {{variable_name}}
    pattern = r'\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}'

    # Track missing variables
    missing_variables = []

    def make_replacer(sanitize: bool):
        """Create a replacer function that optionally sanitizes the value."""
        def replacer(match: re.Match[str]) -> str:
            variable_name = match.group(1)
            if variable_name in variables:
                # Convert value to string, handling None gracefully
                value = variables[variable_name]
                value_str = str(value) if value is not None else ""

                # If this is for the filename, sanitize slashes and forbidden chars
                if sanitize:
                    # Replace filesystem forbidden characters
                    for char in FORBIDDEN_CHARS:
                        value_str = value_str.replace(char, '_')

                return value_str
            else:
                if variable_name not in missing_variables:
                    missing_variables.append(variable_name)
                return match.group(0)  # Leave placeholder unchanged
        return replacer

    # Substitute variables in filename (with sanitization)
    filename = re.sub(pattern, make_replacer(sanitize=True), filename_template)

    # Substitute variables in directory (without sanitizing slashes - they're valid there)
    # But we still need to check for path traversal
    if directory_template:
        directory = re.sub(pattern, make_replacer(sanitize=False), directory_template)
    else:
        directory = None

    # Check for missing variables
    if missing_variables:
        raise PathSubstitutionError(
            f"Template variable(s) not found in experiment: {', '.join(missing_variables)}"
        )

    # Additional sanitization for filename (reserved names, length limits, trailing chars)
    filename = filename.rstrip('. ')
    if not filename:
        raise PathSubstitutionError("Filename is empty after substitution and sanitization")

    # Check Windows reserved names
    name_parts = filename.rsplit('.', 1)
    base_name = name_parts[0]
    extension = '.' + name_parts[1] if len(name_parts) > 1 else ''
    if base_name.upper() in WINDOWS_RESERVED_NAMES:
        base_name = base_name + '_file'
    filename = base_name + extension

    # Enforce length limit
    if len(filename) > MAX_FILENAME_LENGTH:
        if extension:
            max_base_length = MAX_FILENAME_LENGTH - len(extension)
            if max_base_length > 0:
                filename = base_name[:max_base_length] + extension
            else:
                filename = filename[:MAX_FILENAME_LENGTH]
        else:
            filename = filename[:MAX_FILENAME_LENGTH]

    # Reconstruct path
    if directory:
        safe_path = Path(directory) / filename
    else:
        safe_path = Path(filename)

    # Prepend base directory if provided
    if base_dir is not None:
        safe_path = base_dir / safe_path

    # Validate no path traversal attempts
    try:
        # Get the directory we should be within
        allowed_base = base_dir.resolve() if base_dir else Path.cwd().resolve()

        # For validation, resolve the path relative to allowed_base
        if base_dir is not None:
            test_path = safe_path.resolve()
        else:
            test_path = (Path.cwd() / safe_path).resolve()

        # Check if resolved path is within allowed base
        test_path.relative_to(allowed_base)

    except ValueError:
        # relative_to() raises ValueError if path is not relative to base
        raise PathSubstitutionError(
            f"Path traversal detected: resolved path escapes base directory"
        )

    return safe_path
