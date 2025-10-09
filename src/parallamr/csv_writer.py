"""Incremental CSV writer for experiment results."""

import asyncio
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

from .models import ExperimentResult


class IncrementalCSVWriter:
    """
    Handles incremental writing to CSV with proper escaping.
    Writes headers on first call, appends data subsequently.
    Supports writing to stdout by passing None as output_path.
    Async-safe with persistent file handle for better parallel performance.
    """

    def __init__(self, output_path: Optional[str | Path]):
        """
        Initialize the CSV writer.

        Args:
            output_path: Path to the output CSV file, or None for stdout
        """
        self.output_path = Path(output_path) if output_path else None
        self._headers_written = False
        self._fieldnames: Optional[List[str]] = None
        self._is_stdout = output_path is None
        self._file_handle: Optional[TextIO] = None
        self._lock = asyncio.Lock()  # Async lock for concurrent coroutine safety
        self._closed = False

    async def write_result(self, result: ExperimentResult) -> None:
        """
        Append a single result row to the CSV file.
        Async-safe with locking for concurrent coroutines.

        Args:
            result: ExperimentResult to write to CSV

        Raises:
            ValueError: If writer is closed
        """
        async with self._lock:
            if self._closed and not self._is_stdout:
                raise ValueError("Cannot write to closed CSV writer")

            row_data = result.to_csv_row()

            # Determine fieldnames from first result if not set
            if self._fieldnames is None:
                self._fieldnames = self._determine_fieldnames(row_data)

            # Write headers if this is the first write
            if not self._headers_written:
                self._write_headers()
                self._headers_written = True

            # Append the result row
            self._write_row(row_data)

    async def write_results(self, results: List[ExperimentResult]) -> None:
        """
        Write multiple results to the CSV file.

        Args:
            results: List of ExperimentResult objects to write
        """
        for result in results:
            await self.write_result(result)

    def _determine_fieldnames(self, row_data: Dict[str, Any]) -> List[str]:
        """
        Determine the CSV fieldnames based on the row data.
        Orders them logically with core fields first, then variables, then result fields.

        Args:
            row_data: Dictionary representing a CSV row

        Returns:
            Ordered list of fieldnames
        """
        # Core experiment fields (from CSV)
        core_fields = ["provider", "model"]

        # Result fields (added by parallamr)
        result_fields = [
            "status",
            "input_tokens",
            "context_window",
            "output_tokens",
            "output",
            "error_message"
        ]

        # Variable fields (everything else)
        variable_fields = [
            key for key in row_data.keys()
            if key not in core_fields + result_fields
        ]

        return core_fields + variable_fields + result_fields

    def _write_headers(self) -> None:
        """Write CSV headers to the file or stdout."""
        if self._fieldnames is None:
            raise ValueError("Cannot write headers without fieldnames")

        if self._is_stdout:
            writer = csv.DictWriter(sys.stdout, fieldnames=self._fieldnames, dialect='excel', lineterminator='\n')
            writer.writeheader()
            sys.stdout.flush()
        else:
            # Open file handle for persistent use
            if self._file_handle is None:
                self._file_handle = open(self.output_path, 'w', newline='')
            writer = csv.DictWriter(self._file_handle, fieldnames=self._fieldnames, dialect='excel')
            writer.writeheader()
            self._file_handle.flush()

    def _write_row(self, row_data: Dict[str, Any]) -> None:
        """
        Write a single row to the CSV file or stdout.

        Args:
            row_data: Dictionary representing the row to write
        """
        if self._fieldnames is None:
            raise ValueError("Cannot write row without fieldnames")

        # Ensure all expected fields are present (fill missing with empty strings)
        complete_row = {field: row_data.get(field, "") for field in self._fieldnames}

        if self._is_stdout:
            writer = csv.DictWriter(sys.stdout, fieldnames=self._fieldnames, dialect='excel', lineterminator='\n')
            writer.writerow(complete_row)
            sys.stdout.flush()
        else:
            # Use persistent file handle
            if self._file_handle is None:
                raise ValueError("File handle not initialized")
            writer = csv.DictWriter(self._file_handle, fieldnames=self._fieldnames, dialect='excel')
            writer.writerow(complete_row)
            self._file_handle.flush()

    @property
    def exists(self) -> bool:
        """Check if the output file already exists."""
        if self._is_stdout:
            return False
        return self.output_path.exists()

    @property
    def headers_written(self) -> bool:
        """Check if headers have been written to the file."""
        return self._headers_written

    async def close(self) -> None:
        """
        Close the file handle if open.
        Idempotent - can be called multiple times safely.
        """
        async with self._lock:
            if self._file_handle is not None and not self._file_handle.closed:
                self._file_handle.close()
            self._closed = True

    async def reset(self) -> None:
        """Reset the writer state (useful for testing)."""
        async with self._lock:
            # Close file handle before resetting
            await self.close()
            self._headers_written = False
            self._fieldnames = None
            self._file_handle = None
            self._closed = False

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and close file handle."""
        self.close()
        return False

    def get_existing_fieldnames(self) -> Optional[List[str]]:
        """
        Get fieldnames from existing CSV file.

        Returns:
            List of fieldnames if file exists and has headers, None otherwise
        """
        if not self.exists:
            return None

        try:
            with open(self.output_path, 'r', newline='') as file:
                reader = csv.reader(file)
                headers = next(reader, None)
                return headers if headers else None
        except (IOError, StopIteration):
            return None

    def validate_compatibility(self, result: ExperimentResult) -> tuple[bool, Optional[str]]:
        """
        Validate if a result is compatible with existing CSV structure.

        Args:
            result: ExperimentResult to validate

        Returns:
            Tuple of (is_compatible, error_message)
        """
        if not self.exists:
            return True, None

        existing_fieldnames = self.get_existing_fieldnames()
        if existing_fieldnames is None:
            return True, None

        new_fieldnames = self._determine_fieldnames(result.to_csv_row())

        # Check if new fieldnames are compatible (subset or superset)
        existing_set = set(existing_fieldnames)
        new_set = set(new_fieldnames)

        if existing_set != new_set:
            missing_in_new = existing_set - new_set
            extra_in_new = new_set - existing_set

            messages = []
            if missing_in_new:
                messages.append(f"missing fields: {sorted(missing_in_new)}")
            if extra_in_new:
                messages.append(f"extra fields: {sorted(extra_in_new)}")

            return False, f"CSV structure mismatch - {', '.join(messages)}"

        return True, None