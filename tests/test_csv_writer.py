"""Tests for CSV writer functionality."""

import csv
import tempfile
from pathlib import Path

import pytest

from parallamr.csv_writer import IncrementalCSVWriter
from parallamr.models import ExperimentResult, ExperimentStatus


class TestIncrementalCSVWriter:
    """Test incremental CSV writer."""

    def test_write_single_result(self, tmp_path):
        """Test writing a single result."""
        output_file = tmp_path / "test_output.csv"
        writer = IncrementalCSVWriter(output_file)

        result = ExperimentResult(
            provider="mock",
            model="test-model",
            variables={"source": "Wikipedia", "topic": "AI"},
            row_number=1,
            status=ExperimentStatus.OK,
            input_tokens=50,
            context_window=8192,
            output_tokens=10,
            output="Test output",
            error_message=None
        )

        writer.write_result(result)

        # Verify file was created and contains correct data
        assert output_file.exists()

        with open(output_file, 'r', newline='') as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["provider"] == "mock"
        assert rows[0]["model"] == "test-model"
        assert rows[0]["source"] == "Wikipedia"
        assert rows[0]["topic"] == "AI"
        assert rows[0]["status"] == "ok"
        assert rows[0]["input_tokens"] == "50"
        assert rows[0]["output"] == "Test output"

    def test_write_multiple_results(self, tmp_path):
        """Test writing multiple results."""
        output_file = tmp_path / "test_output.csv"
        writer = IncrementalCSVWriter(output_file)

        results = [
            ExperimentResult(
                provider="mock",
                model="model1",
                variables={"topic": "AI"},
                row_number=1,
                status=ExperimentStatus.OK,
                input_tokens=50,
                context_window=None,
                output_tokens=10,
                output="Output 1",
                error_message=None
            ),
            ExperimentResult(
                provider="mock",
                model="model2",
                variables={"topic": "ML"},
                row_number=2,
                status=ExperimentStatus.WARNING,
                input_tokens=75,
                context_window=8192,
                output_tokens=15,
                output="Output 2",
                error_message="Warning message"
            )
        ]

        for result in results:
            writer.write_result(result)

        # Verify file contains both results
        with open(output_file, 'r', newline='') as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["topic"] == "AI"
        assert rows[1]["topic"] == "ML"
        assert rows[1]["error_message"] == "Warning message"

    def test_csv_escaping(self, tmp_path):
        """Test CSV escaping of special characters."""
        output_file = tmp_path / "test_output.csv"
        writer = IncrementalCSVWriter(output_file)

        result = ExperimentResult(
            provider="mock",
            model="test-model",
            variables={"description": "Text with, commas and \"quotes\""},
            row_number=1,
            status=ExperimentStatus.OK,
            input_tokens=50,
            context_window=None,
            output_tokens=10,
            output="Output with\nnewlines and, commas",
            error_message=None
        )

        writer.write_result(result)

        # Verify proper CSV escaping
        with open(output_file, 'r', newline='') as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["description"] == "Text with, commas and \"quotes\""
        assert rows[0]["output"] == "Output with\nnewlines and, commas"

    def test_fieldname_ordering(self, tmp_path):
        """Test that fieldnames are ordered correctly."""
        output_file = tmp_path / "test_output.csv"
        writer = IncrementalCSVWriter(output_file)

        result = ExperimentResult(
            provider="mock",
            model="test-model",
            variables={"var1": "value1", "var2": "value2"},
            row_number=1,
            status=ExperimentStatus.OK,
            input_tokens=50,
            context_window=None,
            output_tokens=10,
            output="Test output",
            error_message=None
        )

        writer.write_result(result)

        # Check fieldname order
        with open(output_file, 'r', newline='') as file:
            reader = csv.reader(file)
            headers = next(reader)

        # Core fields should come first
        assert headers[0] == "provider"
        assert headers[1] == "model"

        # Result fields should come last
        assert headers[-1] == "error_message"
        assert headers[-2] == "output"

        # Variable fields should be in between
        assert "var1" in headers
        assert "var2" in headers

    def test_write_results_batch(self, tmp_path):
        """Test writing multiple results in a batch."""
        output_file = tmp_path / "test_output.csv"
        writer = IncrementalCSVWriter(output_file)

        results = [
            ExperimentResult(
                provider="mock",
                model="model1",
                variables={"topic": "AI"},
                row_number=1,
                status=ExperimentStatus.OK,
                input_tokens=50,
                context_window=None,
                output_tokens=10,
                output="Output 1",
                error_message=None
            ),
            ExperimentResult(
                provider="mock",
                model="model2",
                variables={"topic": "ML"},
                row_number=2,
                status=ExperimentStatus.OK,
                input_tokens=75,
                context_window=None,
                output_tokens=15,
                output="Output 2",
                error_message=None
            )
        ]

        writer.write_results(results)

        # Verify both results were written
        with open(output_file, 'r', newline='') as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        assert len(rows) == 2

    def test_get_existing_fieldnames(self, tmp_path):
        """Test getting fieldnames from existing CSV file."""
        output_file = tmp_path / "test_output.csv"

        # Create CSV with headers
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["provider", "model", "topic", "status", "output"])

        csv_writer = IncrementalCSVWriter(output_file)
        fieldnames = csv_writer.get_existing_fieldnames()

        assert fieldnames == ["provider", "model", "topic", "status", "output"]

    def test_get_existing_fieldnames_nonexistent(self, tmp_path):
        """Test getting fieldnames from nonexistent file."""
        output_file = tmp_path / "nonexistent.csv"
        writer = IncrementalCSVWriter(output_file)

        fieldnames = writer.get_existing_fieldnames()
        assert fieldnames is None

    def test_validate_compatibility_same_structure(self, tmp_path):
        """Test compatibility validation with same structure."""
        output_file = tmp_path / "test_output.csv"

        # Create initial result
        result1 = ExperimentResult(
            provider="mock",
            model="model1",
            variables={"topic": "AI"},
            row_number=1,
            status=ExperimentStatus.OK,
            input_tokens=50,
            context_window=None,
            output_tokens=10,
            output="Output 1",
            error_message=None
        )

        writer = IncrementalCSVWriter(output_file)
        writer.write_result(result1)

        # Test compatibility with same structure
        result2 = ExperimentResult(
            provider="mock",
            model="model2",
            variables={"topic": "ML"},
            row_number=2,
            status=ExperimentStatus.OK,
            input_tokens=75,
            context_window=None,
            output_tokens=15,
            output="Output 2",
            error_message=None
        )

        is_compatible, error = writer.validate_compatibility(result2)
        assert is_compatible is True
        assert error is None

    def test_validate_compatibility_different_structure(self, tmp_path):
        """Test compatibility validation with different structure."""
        output_file = tmp_path / "test_output.csv"

        # Create initial result
        result1 = ExperimentResult(
            provider="mock",
            model="model1",
            variables={"topic": "AI"},
            row_number=1,
            status=ExperimentStatus.OK,
            input_tokens=50,
            context_window=None,
            output_tokens=10,
            output="Output 1",
            error_message=None
        )

        writer = IncrementalCSVWriter(output_file)
        writer.write_result(result1)

        # Test compatibility with different structure
        result2 = ExperimentResult(
            provider="mock",
            model="model2",
            variables={"topic": "ML", "source": "Wikipedia"},  # Extra variable
            row_number=2,
            status=ExperimentStatus.OK,
            input_tokens=75,
            context_window=None,
            output_tokens=15,
            output="Output 2",
            error_message=None
        )

        is_compatible, error = writer.validate_compatibility(result2)
        assert is_compatible is False
        assert "extra fields" in error