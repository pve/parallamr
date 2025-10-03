"""Tests for CLI interface."""

import pytest
from click.testing import CliRunner

from parallamr.cli import cli


class TestCLI:
    """Test CLI commands."""

    def test_cli_version(self):
        """Test CLI version command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])

        assert result.exit_code == 0
        assert "parallamr" in result.output

    def test_cli_help(self):
        """Test CLI help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])

        assert result.exit_code == 0
        assert "Parallamr" in result.output
        assert "command-line tool" in result.output

    def test_run_command_help(self):
        """Test run command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['run', '--help'])

        assert result.exit_code == 0
        assert "--prompt" in result.output
        assert "--experiments" in result.output
        assert "--output" in result.output

    def test_run_command_missing_required_args(self):
        """Test run command with missing required arguments."""
        runner = CliRunner()
        result = runner.invoke(cli, ['run'])

        assert result.exit_code != 0
        assert "Missing option" in result.output

    def test_run_command_nonexistent_files(self):
        """Test run command with nonexistent files."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            'run',
            '--prompt', 'nonexistent_prompt.txt',
            '--experiments', 'nonexistent_experiments.csv',
            '--output', 'output.csv'
        ])

        assert result.exit_code != 0

    def test_providers_command(self):
        """Test providers command."""
        runner = CliRunner()
        result = runner.invoke(cli, ['providers'])

        assert result.exit_code == 0
        assert "Available providers:" in result.output
        assert "mock" in result.output
        assert "openrouter" in result.output
        assert "ollama" in result.output

    def test_models_command_mock(self):
        """Test models command with mock provider."""
        runner = CliRunner()
        result = runner.invoke(cli, ['models', 'mock'])

        # Mock provider doesn't have a models command, should fail
        assert result.exit_code == 2  # Click error for invalid choice

    def test_models_command_ollama(self):
        """Test models command with ollama provider (async handling)."""
        runner = CliRunner()
        result = runner.invoke(cli, ['models', 'ollama'])

        # Should not have RuntimeWarning about unawaited coroutine
        assert "RuntimeWarning" not in result.output
        assert "coroutine" not in result.output

        # Should either succeed or fail gracefully (if Ollama not available)
        # Exit code 0 = success, 1 = error (e.g., can't connect to Ollama)
        assert result.exit_code in [0, 1]

        if result.exit_code == 0:
            assert "Fetching models" in result.output
        else:
            # If Ollama is not available, should show error message
            assert "Cannot connect" in result.output or "Provider" in result.output

    def test_init_command(self, tmp_path):
        """Test init command creates example files."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ['init'])

            assert result.exit_code == 0
            assert "Created example experiments file" in result.output
            assert "Created example prompt file" in result.output
            assert "Created example context file" in result.output

            # Check files were created
            import os
            assert os.path.exists("experiments.csv")
            assert os.path.exists("prompt.txt")
            assert os.path.exists("context.txt")

    def test_init_command_custom_output(self, tmp_path):
        """Test init command with custom output file."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ['init', '--output', 'custom_experiments.csv'])

            assert result.exit_code == 0
            assert "custom_experiments.csv" in result.output

            # Check custom file was created
            import os
            assert os.path.exists("custom_experiments.csv")

    def test_run_validate_only(self, tmp_path):
        """Test run command with validate-only flag."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create test files using init
            result = runner.invoke(cli, ['init'])
            assert result.exit_code == 0

            # Run validation
            result = runner.invoke(cli, [
                'run',
                '--prompt', 'prompt.txt',
                '--experiments', 'experiments.csv',
                '--output', 'results.csv',
                '--validate-only'
            ])

            assert result.exit_code == 0
            assert "Validating experiments" in result.output
            assert "Validation successful" in result.output

    def test_run_with_context_files(self, tmp_path):
        """Test run command with context files."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create test files
            result = runner.invoke(cli, ['init'])
            assert result.exit_code == 0

            # Run with context files
            result = runner.invoke(cli, [
                'run',
                '--prompt', 'prompt.txt',
                '--context', 'context.txt',
                '--experiments', 'experiments.csv',
                '--output', 'results.csv',
                '--validate-only'
            ])

            assert result.exit_code == 0

    def test_run_with_verbose(self, tmp_path):
        """Test run command with verbose flag."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create test files
            result = runner.invoke(cli, ['init'])
            assert result.exit_code == 0

            # Run with verbose
            result = runner.invoke(cli, [
                'run',
                '--prompt', 'prompt.txt',
                '--experiments', 'experiments.csv',
                '--output', 'results.csv',
                '--verbose',
                '--validate-only'
            ])

            assert result.exit_code == 0

    def test_run_invalid_timeout(self):
        """Test run command with invalid timeout."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            'run',
            '--prompt', 'prompt.txt',
            '--experiments', 'experiments.csv',
            '--output', 'results.csv',
            '--timeout', '0'
        ])

        assert result.exit_code == 1
        assert "Timeout must be positive" in result.output

    @pytest.mark.asyncio
    async def test_full_run_integration(self, tmp_path):
        """Test full run integration with mock provider."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create test files with mock provider only
            prompt_content = "Test prompt about {{topic}}"
            with open("prompt.txt", "w") as f:
                f.write(prompt_content)

            experiments_content = """provider,model,topic
mock,mock,AI
mock,mock,ML"""
            with open("experiments.csv", "w") as f:
                f.write(experiments_content)

            # Run experiments
            result = runner.invoke(cli, [
                'run',
                '--prompt', 'prompt.txt',
                '--experiments', 'experiments.csv',
                '--output', 'results.csv',
                '--verbose'
            ])

            # Should complete successfully with mock provider
            assert result.exit_code == 0

            # Check output file was created
            import os
            assert os.path.exists("results.csv")

            # Verify content
            import csv
            with open("results.csv", "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            assert all(row["status"] == "ok" for row in rows)
            assert all("MOCK RESPONSE" in row["output"] for row in rows)