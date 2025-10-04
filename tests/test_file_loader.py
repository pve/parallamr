"""Tests for file loading abstractions."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from parallamr.file_loader import FileLoader
from parallamr.models import Experiment


class TestFileLoader:
    """Test FileLoader class."""

    def test_load_prompt_from_file(self, tmp_path):
        """Test loading prompt from file."""
        # Create test file
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Test prompt content", encoding='utf-8')

        loader = FileLoader()
        content = loader.load_prompt(prompt_file, use_stdin=False)

        assert content == "Test prompt content"

    def test_load_prompt_from_stdin(self):
        """Test loading prompt from stdin."""
        loader = FileLoader()

        # Mock stdin
        loader._read_stdin = MagicMock(return_value="Stdin prompt content")

        content = loader.load_prompt(None, use_stdin=True)

        assert content == "Stdin prompt content"
        loader._read_stdin.assert_called_once()

    def test_load_experiments_from_file(self, tmp_path):
        """Test loading experiments from CSV file."""
        # Create test CSV
        csv_file = tmp_path / "experiments.csv"
        csv_content = """provider,model,topic
mock,mock,AI
mock,mock,ML"""
        csv_file.write_text(csv_content, encoding='utf-8')

        loader = FileLoader()
        experiments = loader.load_experiments(csv_file, use_stdin=False)

        assert len(experiments) == 2
        assert isinstance(experiments[0], Experiment)
        assert experiments[0].provider == "mock"
        assert experiments[0].model == "mock"
        assert experiments[0].variables == {"topic": "AI"}

    def test_load_experiments_from_stdin(self):
        """Test loading experiments from stdin."""
        loader = FileLoader()

        csv_content = """provider,model,topic
mock,mock,Test"""

        # Mock stdin
        loader._read_stdin = MagicMock(return_value=csv_content)

        experiments = loader.load_experiments(None, use_stdin=True)

        assert len(experiments) == 1
        assert experiments[0].provider == "mock"
        assert experiments[0].variables == {"topic": "Test"}
        loader._read_stdin.assert_called_once()

    def test_load_context_files(self, tmp_path):
        """Test loading context files."""
        # Create test files
        file1 = tmp_path / "context1.txt"
        file1.write_text("Context 1 content", encoding='utf-8')

        file2 = tmp_path / "context2.txt"
        file2.write_text("Context 2 content", encoding='utf-8')

        loader = FileLoader()
        context = loader.load_context([file1, file2])

        assert len(context) == 2
        assert context[0] == ("context1.txt", "Context 1 content")
        assert context[1] == ("context2.txt", "Context 2 content")

    def test_load_context_empty_list(self):
        """Test loading context with empty list."""
        loader = FileLoader()
        context = loader.load_context([])

        assert context == []

    def test_load_context_none(self):
        """Test loading context with None."""
        loader = FileLoader()
        context = loader.load_context(None)

        assert context == []


class TestFileLoaderIntegration:
    """Integration tests for FileLoader with ExperimentRunner."""

    def test_runner_uses_injected_file_loader(self, tmp_path):
        """Test that ExperimentRunner uses injected FileLoader."""
        from parallamr.runner import ExperimentRunner

        # Create mock FileLoader
        mock_loader = MagicMock(spec=FileLoader)
        mock_loader.load_prompt.return_value = "Mock prompt"
        mock_loader.load_experiments.return_value = []
        mock_loader.load_context.return_value = []

        # Inject mock loader
        runner = ExperimentRunner(file_loader=mock_loader)

        # Verify runner has the mock loader
        assert runner.file_loader == mock_loader

    def test_runner_creates_default_file_loader(self):
        """Test that ExperimentRunner creates default FileLoader."""
        from parallamr.runner import ExperimentRunner

        runner = ExperimentRunner()

        # Verify runner has a FileLoader instance
        assert isinstance(runner.file_loader, FileLoader)
