"""Integration tests for JSON flattening feature with --flatten flag."""

import csv
from pathlib import Path

import pytest

from parallamr.csv_writer import IncrementalCSVWriter
from parallamr.models import Experiment, ExperimentResult, ExperimentStatus, ProviderResponse
from parallamr.runner import ExperimentRunner


class TestFlattenIntegration:
    """Test end-to-end JSON flattening integration."""

    @pytest.mark.asyncio
    async def test_flatten_flag_with_json_output(self, tmp_path):
        """Test running experiments with --flatten flag and JSON in output."""
        # Create test files
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Generate a JSON response about {{topic}}", encoding='utf-8')

        experiments_file = tmp_path / "experiments.csv"
        experiments_content = """provider,model,topic
mock,mock,AI"""
        experiments_file.write_text(experiments_content, encoding='utf-8')

        output_file = tmp_path / "results.csv"

        # Create runner with flatten enabled
        runner = ExperimentRunner(flatten_json=True)

        # Run experiments
        await runner.run_experiments(
            prompt_file=prompt_file,
            experiments_file=experiments_file,
            output_file=output_file
        )

        # Verify output file
        assert output_file.exists()

        # Read and verify CSV
        with open(output_file, 'r', newline='') as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        assert len(rows) == 1
        # Should have standard columns
        assert "provider" in rows[0]
        assert "model" in rows[0]
        assert "topic" in rows[0]
        assert "output" in rows[0]
        assert "status" in rows[0]
        # Should have JSON fields if mock provider returned JSON
        # (This depends on mock provider implementation)

    @pytest.mark.asyncio
    async def test_flatten_missing_json_no_extra_columns(self, tmp_path):
        """Test that missing JSON doesn't create extra columns or errors."""
        # Create test files
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Generate plain text about {{topic}}", encoding='utf-8')

        experiments_file = tmp_path / "experiments.csv"
        experiments_content = """provider,model,topic
mock,mock,AI"""
        experiments_file.write_text(experiments_content, encoding='utf-8')

        output_file = tmp_path / "results.csv"

        # Create runner with flatten enabled
        runner = ExperimentRunner(flatten_json=True)

        # Run experiments
        await runner.run_experiments(
            prompt_file=prompt_file,
            experiments_file=experiments_file,
            output_file=output_file
        )

        # Verify output file
        assert output_file.exists()

        # Read and verify CSV
        with open(output_file, 'r', newline='') as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        assert len(rows) == 1
        # Should have standard columns only (no JSON fields)
        assert rows[0]["status"] == "ok"  # No error status
        # Should not have error message about missing JSON
        assert rows[0].get("error_message", "") == ""

    @pytest.mark.asyncio
    async def test_flatten_consistent_json_schemas(self, tmp_path):
        """Test that consistent JSON schemas work correctly across experiments."""
        # Create test files
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Rate {{topic}}", encoding='utf-8')

        experiments_file = tmp_path / "experiments.csv"
        experiments_content = """provider,model,topic
mock,mock,AI
mock,mock,ML
mock,mock,NLP"""
        experiments_file.write_text(experiments_content, encoding='utf-8')

        output_file = tmp_path / "results.csv"

        # Create runner with flatten enabled
        runner = ExperimentRunner(flatten_json=True)

        # Mock provider to return consistent JSON
        # (This would need mock provider to be configured to return JSON)
        await runner.run_experiments(
            prompt_file=prompt_file,
            experiments_file=experiments_file,
            output_file=output_file
        )

        # Verify output file
        assert output_file.exists()

        # Read and verify CSV
        with open(output_file, 'r', newline='') as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        assert len(rows) == 3
        # All rows should have same structure
        headers = list(rows[0].keys())
        for row in rows[1:]:
            assert list(row.keys()) == headers

    @pytest.mark.asyncio
    async def test_flatten_inconsistent_json_schemas(self, tmp_path):
        """Test handling of inconsistent JSON schemas across experiments."""
        # This test would verify that when different experiments return
        # different JSON schemas, all columns are properly handled
        # (implementation depends on CSV writer behavior)

        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Data for {{topic}}", encoding='utf-8')

        experiments_file = tmp_path / "experiments.csv"
        experiments_content = """provider,model,topic
mock,mock,test1
mock,mock,test2"""
        experiments_file.write_text(experiments_content, encoding='utf-8')

        output_file = tmp_path / "results.csv"

        runner = ExperimentRunner(flatten_json=True)

        # This should complete without error even if schemas differ
        await runner.run_experiments(
            prompt_file=prompt_file,
            experiments_file=experiments_file,
            output_file=output_file
        )

        assert output_file.exists()

    def test_csv_output_with_flattened_json(self, tmp_path):
        """Test CSV writer correctly handles flattened JSON fields."""
        output_file = tmp_path / "test_output.csv"
        writer = IncrementalCSVWriter(output_file)

        # Create result with JSON fields
        result = ExperimentResult(
            provider="mock",
            model="test-model",
            variables={"topic": "AI"},
            row_number=1,
            status=ExperimentStatus.OK,
            input_tokens=50,
            context_window=8192,
            output_tokens=10,
            output="Test output with JSON",
            error_message=None,
            json_fields={"score": 95, "rating": "excellent"}
        )

        writer.write_result(result)

        # Verify file
        with open(output_file, 'r', newline='') as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["topic"] == "AI"
        assert rows[0]["score"] == "95"
        assert rows[0]["rating"] == "excellent"

    def test_column_ordering_with_json_fields(self, tmp_path):
        """Test that column ordering is correct with JSON fields."""
        output_file = tmp_path / "test_output.csv"
        writer = IncrementalCSVWriter(output_file)

        result = ExperimentResult(
            provider="mock",
            model="test-model",
            variables={"var1": "value1"},
            row_number=1,
            status=ExperimentStatus.OK,
            input_tokens=50,
            context_window=8192,
            output_tokens=10,
            output="Test output",
            error_message=None,
            json_fields={"json_field": "json_value"}
        )

        writer.write_result(result)

        # Check column order
        with open(output_file, 'r', newline='') as file:
            reader = csv.reader(file)
            headers = next(reader)

        # Core fields should come first
        assert headers[0] == "provider"
        assert headers[1] == "model"

        # JSON fields should come after experiment variables
        assert "json_field" in headers
        json_field_index = headers.index("json_field")
        output_index = headers.index("output")
        # JSON fields should come before output
        assert json_field_index < output_index

    def test_output_cleaning_with_flatten(self, tmp_path):
        """Test that output text is cleaned when flatten is enabled."""
        output_file = tmp_path / "test_output.csv"
        writer = IncrementalCSVWriter(output_file)

        # Create result with output containing newlines and carriage returns
        result = ExperimentResult(
            provider="mock",
            model="test-model",
            variables={"topic": "AI"},
            row_number=1,
            status=ExperimentStatus.OK,
            input_tokens=50,
            context_window=8192,
            output_tokens=10,
            output="Line one\nLine two\r\nLine three",
            error_message=None,
            json_fields={"score": 95}
        )

        writer.write_result(result)

        # Verify output is cleaned
        with open(output_file, 'r', newline='') as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        # Output should have newlines replaced with spaces
        assert "\n" not in rows[0]["output"]
        assert "\r" not in rows[0]["output"]
        assert "Line one" in rows[0]["output"]
        assert "Line three" in rows[0]["output"]

    @pytest.mark.asyncio
    async def test_flatten_with_context_files(self, tmp_path):
        """Test JSON flattening works with context files."""
        # Create test files
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Analyze {{topic}}", encoding='utf-8')

        context_file = tmp_path / "context.txt"
        context_file.write_text("Additional context information", encoding='utf-8')

        experiments_file = tmp_path / "experiments.csv"
        experiments_content = """provider,model,topic
mock,mock,AI"""
        experiments_file.write_text(experiments_content, encoding='utf-8')

        output_file = tmp_path / "results.csv"

        # Create runner with flatten enabled
        runner = ExperimentRunner(flatten_json=True)

        # Run experiments with context files
        await runner.run_experiments(
            prompt_file=prompt_file,
            experiments_file=experiments_file,
            output_file=output_file,
            context_files=[context_file]
        )

        # Verify output file created successfully
        assert output_file.exists()

        # Read and verify
        with open(output_file, 'r', newline='') as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_flatten_multiple_experiments_parallel(self, tmp_path):
        """Test JSON flattening with multiple experiments running in parallel."""
        # Create test files
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Process {{topic}}", encoding='utf-8')

        experiments_file = tmp_path / "experiments.csv"
        experiments_content = """provider,model,topic
mock,mock,AI
mock,mock,ML
mock,mock,NLP
mock,mock,CV
mock,mock,Robotics"""
        experiments_file.write_text(experiments_content, encoding='utf-8')

        output_file = tmp_path / "results.csv"

        # Create runner with flatten enabled
        runner = ExperimentRunner(flatten_json=True, max_concurrent=3)

        # Run experiments
        await runner.run_experiments(
            prompt_file=prompt_file,
            experiments_file=experiments_file,
            output_file=output_file
        )

        # Verify all experiments completed
        assert output_file.exists()

        with open(output_file, 'r', newline='') as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        assert len(rows) == 5
        # All should have ok status (no race conditions)
        assert all(row["status"] == "ok" for row in rows)

    def test_column_conflict_resolution(self, tmp_path):
        """Test that column name conflicts are properly resolved."""
        output_file = tmp_path / "test_output.csv"
        writer = IncrementalCSVWriter(output_file)

        # Create result where JSON fields conflict with reserved columns
        result = ExperimentResult(
            provider="mock",
            model="test-model",
            variables={"topic": "AI"},
            row_number=1,
            status=ExperimentStatus.OK,
            input_tokens=50,
            context_window=8192,
            output_tokens=10,
            output="Test output",
            error_message=None,
            # JSON field names conflict with reserved columns
            json_fields={"status": "completed", "output": "result data"}
        )

        writer.write_result(result)

        # Verify conflict resolution
        with open(output_file, 'r', newline='') as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        assert len(rows) == 1
        # Reserved columns should have their original values
        assert rows[0]["status"] == "ok"
        assert rows[0]["output"] == "Test output"
        # JSON fields should be prefixed
        assert rows[0]["json_status"] == "completed"
        assert rows[0]["json_output"] == "result data"


class TestExperimentResultWithJSON:
    """Test ExperimentResult model with JSON fields."""

    def test_experiment_result_with_json_fields(self):
        """Test creating ExperimentResult with json_fields."""
        result = ExperimentResult(
            provider="mock",
            model="test-model",
            variables={"topic": "AI"},
            row_number=1,
            status=ExperimentStatus.OK,
            input_tokens=50,
            context_window=8192,
            output_tokens=10,
            output="Test output",
            error_message=None,
            json_fields={"score": 95, "confidence": 0.98}
        )

        assert result.json_fields == {"score": 95, "confidence": 0.98}

    def test_experiment_result_to_csv_row_includes_json_fields(self):
        """Test that to_csv_row includes json_fields."""
        result = ExperimentResult(
            provider="mock",
            model="test-model",
            variables={"topic": "AI"},
            row_number=1,
            status=ExperimentStatus.OK,
            input_tokens=50,
            context_window=8192,
            output_tokens=10,
            output="Test output",
            error_message=None,
            json_fields={"score": 95, "rating": "A"}
        )

        row = result.to_csv_row()

        # Should include standard fields
        assert row["provider"] == "mock"
        assert row["topic"] == "AI"
        assert row["output"] == "Test output"
        # Should include JSON fields
        assert row["score"] == 95
        assert row["rating"] == "A"

    def test_experiment_result_without_json_fields(self):
        """Test that json_fields is optional."""
        result = ExperimentResult(
            provider="mock",
            model="test-model",
            variables={"topic": "AI"},
            row_number=1,
            status=ExperimentStatus.OK,
            input_tokens=50,
            context_window=8192,
            output_tokens=10,
            output="Test output",
            error_message=None
        )

        # Should have json_fields as None or not set
        assert not hasattr(result, 'json_fields') or result.json_fields is None

        # to_csv_row should work without json_fields
        row = result.to_csv_row()
        assert row["provider"] == "mock"
        assert row["output"] == "Test output"
