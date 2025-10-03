"""Utility functions for parallamr."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .models import Experiment


def load_experiments_from_csv(csv_path: str | Path) -> List[Experiment]:
    """
    Load experiments from a CSV file.

    Args:
        csv_path: Path to the experiments CSV file

    Returns:
        List of Experiment objects

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the CSV is malformed or missing required columns
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Experiments CSV file not found: {csv_path}")

    experiments = []

    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            if not reader.fieldnames:
                raise ValueError("CSV file appears to be empty or malformed")

            # Check for required columns
            required_columns = {"provider", "model"}
            missing_columns = required_columns - set(reader.fieldnames)
            if missing_columns:
                raise ValueError(f"Missing required columns in CSV: {missing_columns}")

            for row_number, row in enumerate(reader, start=1):
                # Remove empty string values and None values
                cleaned_row = {k: v for k, v in row.items() if v is not None and v != ""}

                try:
                    experiment = Experiment.from_csv_row(cleaned_row, row_number)
                    experiments.append(experiment)
                except Exception as e:
                    raise ValueError(f"Error parsing row {row_number}: {e}")

    except csv.Error as e:
        raise ValueError(f"CSV parsing error: {e}")

    if not experiments:
        raise ValueError("No valid experiments found in CSV file")

    return experiments


def load_file_content(file_path: str | Path) -> str:
    """
    Load content from a text file.

    Args:
        file_path: Path to the file

    Returns:
        File content as string

    Raises:
        FileNotFoundError: If the file doesn't exist
        UnicodeDecodeError: If the file can't be decoded as UTF-8
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        return file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            e.encoding, e.object, e.start, e.end,
            f"Could not decode file {file_path} as UTF-8: {e.reason}"
        )


def load_context_files(file_paths: List[str | Path]) -> List[Tuple[str, str]]:
    """
    Load multiple context files.

    Args:
        file_paths: List of file paths to load

    Returns:
        List of (filename, content) tuples

    Raises:
        FileNotFoundError: If any file doesn't exist
    """
    context_files = []

    for file_path in file_paths:
        file_path = Path(file_path)
        content = load_file_content(file_path)
        context_files.append((file_path.name, content))

    return context_files


def validate_output_path(output_path: str | Path) -> Path:
    """
    Validate and prepare output path.

    Args:
        output_path: Path for the output file

    Returns:
        Validated Path object

    Raises:
        ValueError: If the output path is invalid
    """
    output_path = Path(output_path)

    # Ensure parent directory exists
    parent_dir = output_path.parent
    if not parent_dir.exists():
        try:
            parent_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise ValueError(f"Cannot create output directory {parent_dir}: {e}")

    # Check if we can write to the directory
    if parent_dir.exists() and not os.access(parent_dir, os.W_OK):
        raise ValueError(f"No write permission for directory {parent_dir}")

    return output_path


def format_experiment_summary(experiments: List[Experiment]) -> str:
    """
    Format a summary of experiments for logging.

    Args:
        experiments: List of experiments

    Returns:
        Formatted summary string
    """
    if not experiments:
        return "No experiments to run"

    provider_counts = {}
    for exp in experiments:
        provider_counts[exp.provider] = provider_counts.get(exp.provider, 0) + 1

    summary_lines = [
        f"Loaded {len(experiments)} experiments:",
    ]

    for provider, count in sorted(provider_counts.items()):
        summary_lines.append(f"  - {provider}: {count} experiment(s)")

    return "\n".join(summary_lines)


import os  # Add this import that was missing