"""Tests for utility functions."""

import csv
import tempfile
from pathlib import Path

import pytest

from parallamr.models import Experiment
from parallamr.utils import (
    format_experiment_summary,
    load_context_files,
    load_experiments_from_csv,
    load_file_content,
    validate_output_path,
)


class TestLoadExperimentsFromCSV:
    """Test loading experiments from CSV files."""

    def test_load_valid_experiments(self, tmp_path):
        """Test loading valid experiments."""
        csv_file = tmp_path / "experiments.csv"
        csv_content = """provider,model,source,topic
mock,mock,Wikipedia,AI
mock,mock,Encyclopedia,ML"""

        csv_file.write_text(csv_content, encoding='utf-8')

        experiments = load_experiments_from_csv(csv_file)

        assert len(experiments) == 2
        assert experiments[0].provider == "mock"
        assert experiments[0].model == "mock"
        assert experiments[0].variables == {"source": "Wikipedia", "topic": "AI"}
        assert experiments[0].row_number == 1

        assert experiments[1].variables == {"source": "Encyclopedia", "topic": "ML"}
        assert experiments[1].row_number == 2

    def test_load_missing_provider_column(self, tmp_path):
        """Test loading CSV missing provider column."""
        csv_file = tmp_path / "experiments.csv"
        csv_content = """model,source,topic
mock,Wikipedia,AI"""

        csv_file.write_text(csv_content, encoding='utf-8')

        with pytest.raises(ValueError, match="Missing required columns"):
            load_experiments_from_csv(csv_file)

    def test_load_missing_model_column(self, tmp_path):
        """Test loading CSV missing model column."""
        csv_file = tmp_path / "experiments.csv"
        csv_content = """provider,source,topic
mock,Wikipedia,AI"""

        csv_file.write_text(csv_content, encoding='utf-8')

        with pytest.raises(ValueError, match="Missing required columns"):
            load_experiments_from_csv(csv_file)

    def test_load_empty_csv(self, tmp_path):
        """Test loading empty CSV file."""
        csv_file = tmp_path / "experiments.csv"
        csv_file.write_text("", encoding='utf-8')

        with pytest.raises(ValueError, match="empty or malformed"):
            load_experiments_from_csv(csv_file)

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading nonexistent CSV file."""
        csv_file = tmp_path / "nonexistent.csv"

        with pytest.raises(FileNotFoundError):
            load_experiments_from_csv(csv_file)

    def test_load_csv_with_empty_values(self, tmp_path):
        """Test loading CSV with empty values."""
        csv_file = tmp_path / "experiments.csv"
        csv_content = """provider,model,source,topic
mock,mock,Wikipedia,AI
mock,mock,,ML
mock,mock,Database,"""

        csv_file.write_text(csv_content, encoding='utf-8')

        experiments = load_experiments_from_csv(csv_file)

        assert len(experiments) == 3
        # Empty values should be filtered out
        assert experiments[1].variables == {"topic": "ML"}
        assert experiments[2].variables == {"source": "Database"}


class TestLoadFileContent:
    """Test loading file content."""

    def test_load_existing_file(self, tmp_path):
        """Test loading existing file."""
        test_file = tmp_path / "test.txt"
        content = "This is test content\nwith multiple lines"
        test_file.write_text(content, encoding='utf-8')

        loaded_content = load_file_content(test_file)

        assert loaded_content == content

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading nonexistent file."""
        test_file = tmp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundError):
            load_file_content(test_file)

    def test_load_unicode_file(self, tmp_path):
        """Test loading file with Unicode content."""
        test_file = tmp_path / "unicode.txt"
        content = "Hello 世界! 🌍"
        test_file.write_text(content, encoding='utf-8')

        loaded_content = load_file_content(test_file)

        assert loaded_content == content


class TestLoadContextFiles:
    """Test loading context files."""

    def test_load_single_context_file(self, tmp_path):
        """Test loading single context file."""
        context_file = tmp_path / "context.txt"
        content = "Context content"
        context_file.write_text(content, encoding='utf-8')

        result = load_context_files([context_file])

        assert len(result) == 1
        assert result[0] == ("context.txt", content)

    def test_load_multiple_context_files(self, tmp_path):
        """Test loading multiple context files."""
        file1 = tmp_path / "context1.txt"
        file2 = tmp_path / "context2.txt"

        content1 = "Content 1"
        content2 = "Content 2"

        file1.write_text(content1, encoding='utf-8')
        file2.write_text(content2, encoding='utf-8')

        result = load_context_files([file1, file2])

        assert len(result) == 2
        assert result[0] == ("context1.txt", content1)
        assert result[1] == ("context2.txt", content2)

    def test_load_empty_context_files_list(self):
        """Test loading empty list of context files."""
        result = load_context_files([])

        assert result == []

    def test_load_nonexistent_context_file(self, tmp_path):
        """Test loading nonexistent context file."""
        nonexistent_file = tmp_path / "nonexistent.txt"

        with pytest.raises(FileNotFoundError):
            load_context_files([nonexistent_file])


class TestValidateOutputPath:
    """Test output path validation."""

    def test_validate_existing_directory(self, tmp_path):
        """Test validating path in existing directory."""
        output_file = tmp_path / "output.csv"

        result = validate_output_path(output_file)

        assert result == output_file
        assert result.parent.exists()

    def test_validate_nonexistent_directory(self, tmp_path):
        """Test validating path in nonexistent directory."""
        nested_dir = tmp_path / "nested" / "dir"
        output_file = nested_dir / "output.csv"

        result = validate_output_path(output_file)

        assert result == output_file
        assert result.parent.exists()

    def test_validate_path_as_string(self, tmp_path):
        """Test validating path provided as string."""
        output_file = str(tmp_path / "output.csv")

        result = validate_output_path(output_file)

        assert isinstance(result, Path)
        assert result.name == "output.csv"


class TestFormatExperimentSummary:
    """Test experiment summary formatting."""

    def test_format_empty_experiments(self):
        """Test formatting empty experiments list."""
        result = format_experiment_summary([])

        assert result == "No experiments to run"

    def test_format_single_provider(self):
        """Test formatting experiments with single provider."""
        experiments = [
            Experiment("mock", "model1", {"topic": "AI"}, 1),
            Experiment("mock", "model2", {"topic": "ML"}, 2),
        ]

        result = format_experiment_summary(experiments)

        assert "Loaded 2 experiments:" in result
        assert "mock: 2 experiment(s)" in result

    def test_format_multiple_providers(self):
        """Test formatting experiments with multiple providers."""
        experiments = [
            Experiment("mock", "model1", {"topic": "AI"}, 1),
            Experiment("openrouter", "model2", {"topic": "ML"}, 2),
            Experiment("ollama", "model3", {"topic": "NLP"}, 3),
            Experiment("mock", "model4", {"topic": "DL"}, 4),
        ]

        result = format_experiment_summary(experiments)

        assert "Loaded 4 experiments:" in result
        assert "mock: 2 experiment(s)" in result
        assert "ollama: 1 experiment(s)" in result
        assert "openrouter: 1 experiment(s)" in result