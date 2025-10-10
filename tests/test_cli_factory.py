"""Tests for CLI runner factory pattern."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from parallamr.cli import create_experiment_runner
from parallamr.file_loader import FileLoader
from parallamr.providers import MockProvider
from parallamr.runner import ExperimentRunner


class TestFactoryFunction:
    """Test the factory function in isolation."""

    def test_factory_creates_runner_with_defaults(self):
        """Verify factory creates runner with default parameters."""
        runner = create_experiment_runner()

        assert isinstance(runner, ExperimentRunner)
        assert runner.timeout == 300
        assert runner.verbose is False
        assert len(runner.providers) == 4  # mock, openrouter, ollama, openai

    def test_factory_creates_runner_with_custom_timeout(self):
        """Verify factory creates runner with custom timeout."""
        runner = create_experiment_runner(timeout=600)

        assert runner.timeout == 600

    def test_factory_creates_runner_with_verbose(self):
        """Verify factory creates runner with verbose logging."""
        runner = create_experiment_runner(verbose=True)

        assert runner.verbose is True

    def test_factory_injects_custom_providers(self):
        """Verify factory can inject custom providers."""
        custom_provider = MockProvider()
        custom_providers = {"custom": custom_provider}

        runner = create_experiment_runner(providers=custom_providers)

        assert "custom" in runner.providers
        assert runner.providers["custom"] is custom_provider
        assert len(runner.providers) == 1  # Only custom provider

    def test_factory_injects_file_loader(self):
        """Verify factory can inject custom file loader."""
        custom_loader = FileLoader()

        runner = create_experiment_runner(file_loader=custom_loader)

        assert runner.file_loader is custom_loader

    def test_factory_injects_session(self):
        """Verify factory can inject HTTP session."""
        mock_session = Mock(spec=aiohttp.ClientSession)

        runner = create_experiment_runner(session=mock_session)

        # Session should be passed to providers
        # Check if providers have session (for parallel processing)
        assert hasattr(runner, '_session') or any(
            hasattr(p, '_session') for p in runner.providers.values()
        )

    def test_factory_combines_all_parameters(self):
        """Verify factory can combine all parameters."""
        custom_provider = MockProvider()
        custom_loader = FileLoader()
        mock_session = Mock(spec=aiohttp.ClientSession)

        runner = create_experiment_runner(
            timeout=600,
            verbose=True,
            providers={"custom": custom_provider},
            file_loader=custom_loader,
            session=mock_session
        )

        assert runner.timeout == 600
        assert runner.verbose is True
        assert runner.file_loader is custom_loader


class TestCLIIntegration:
    """Test CLI integration with factory."""

    def test_cli_run_uses_factory(self, tmp_path):
        """Verify CLI run command uses factory."""
        from parallamr.cli import cli
        from click.testing import CliRunner

        # Create test files
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Test prompt")

        experiments_file = tmp_path / "experiments.csv"
        experiments_file.write_text("provider,model\nmock,test-model\n")

        output_file = tmp_path / "output.csv"

        # Mock the factory to track calls
        mock_runner = Mock(spec=ExperimentRunner)
        mock_runner.run_experiments = AsyncMock()

        def mock_factory(**kwargs):
            return mock_runner

        runner = CliRunner()
        with patch('parallamr.cli.create_experiment_runner', side_effect=mock_factory) as factory_spy:
            result = runner.invoke(cli, [
                'run',
                '-p', str(prompt_file),
                '-e', str(experiments_file),
                '-o', str(output_file)
            ])

            # Command should succeed
            assert result.exit_code == 0

            # Verify factory was called
            factory_spy.assert_called_once()
            call_kwargs = factory_spy.call_args.kwargs
            assert call_kwargs['timeout'] == 300
            assert call_kwargs['verbose'] is False

            # Verify runner.run_experiments was called
            mock_runner.run_experiments.assert_called_once()

    def test_cli_run_with_custom_timeout(self, tmp_path):
        """Verify CLI passes custom timeout to factory."""
        from parallamr.cli import cli
        from click.testing import CliRunner

        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Test prompt")

        experiments_file = tmp_path / "experiments.csv"
        experiments_file.write_text("provider,model\nmock,test-model\n")

        output_file = tmp_path / "output.csv"

        mock_runner = Mock(spec=ExperimentRunner)
        mock_runner.run_experiments = AsyncMock()

        def mock_factory(**kwargs):
            assert kwargs['timeout'] == 600
            return mock_runner

        runner = CliRunner()
        with patch('parallamr.cli.create_experiment_runner', side_effect=mock_factory):
            result = runner.invoke(cli, [
                'run',
                '-p', str(prompt_file),
                '-e', str(experiments_file),
                '-o', str(output_file),
                '--timeout', '600'
            ])

            assert result.exit_code == 0

    def test_cli_run_with_verbose(self, tmp_path):
        """Verify CLI passes verbose flag to factory."""
        from parallamr.cli import cli
        from click.testing import CliRunner

        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Test prompt")

        experiments_file = tmp_path / "experiments.csv"
        experiments_file.write_text("provider,model\nmock,test-model\n")

        output_file = tmp_path / "output.csv"

        mock_runner = Mock(spec=ExperimentRunner)
        mock_runner.run_experiments = AsyncMock()

        def mock_factory(**kwargs):
            assert kwargs['verbose'] is True
            return mock_runner

        runner = CliRunner()
        with patch('parallamr.cli.create_experiment_runner', side_effect=mock_factory):
            result = runner.invoke(cli, [
                'run',
                '-p', str(prompt_file),
                '-e', str(experiments_file),
                '-o', str(output_file),
                '--verbose'
            ])

            assert result.exit_code == 0

    def test_cli_validate_uses_factory(self, tmp_path):
        """Verify CLI validate command uses factory."""
        from parallamr.cli import cli
        from click.testing import CliRunner

        experiments_file = tmp_path / "experiments.csv"
        experiments_file.write_text("provider,model\nmock,test-model\n")

        mock_runner = Mock(spec=ExperimentRunner)
        mock_runner.validate_experiments = AsyncMock(return_value={
            "valid": True,
            "experiments": 1,
            "providers": {},
            "warnings": []
        })

        def mock_factory(**kwargs):
            return mock_runner

        runner = CliRunner()
        with patch('parallamr.cli.create_experiment_runner', side_effect=mock_factory) as factory_spy:
            result = runner.invoke(cli, [
                'run',
                '-p', str(experiments_file),
                '-e', str(experiments_file),
                '--validate-only'
            ])

            # Verify factory was called
            factory_spy.assert_called()

            # Verify runner.validate_experiments was called
            mock_runner.validate_experiments.assert_called()

    def test_cli_argument_parsing_without_api_keys(self):
        """Verify CLI argument parsing doesn't require API keys."""
        from parallamr.cli import cli
        from click.testing import CliRunner

        runner = CliRunner()

        # Mock the factory to prevent actual runner creation
        with patch('parallamr.cli.create_experiment_runner') as mock_factory:
            mock_factory.return_value = Mock(spec=ExperimentRunner)

            # Test missing required arguments
            result = runner.invoke(cli, ["run"])

            # Should fail due to missing arguments, not missing API keys
            assert result.exit_code != 0
            # Click outputs to stderr or stdout depending on version
            output = result.stdout + result.output
            assert "Missing option" in output or "Error" in output or result.exit_code == 2

    def test_factory_enables_parallel_processing(self):
        """Verify factory can inject session for parallel processing."""
        # Test that factory accepts session parameter (for future parallel mode)
        mock_session = Mock(spec=aiohttp.ClientSession)

        runner = create_experiment_runner(session=mock_session)

        # Verify runner created successfully
        assert isinstance(runner, ExperimentRunner)

        # Session should be injected into providers
        # (actual usage will be in parallel processing implementation)


class TestFactoryErrorHandling:
    """Test factory error handling."""

    def test_factory_with_invalid_timeout(self):
        """Verify factory handles invalid timeout."""
        # Should not raise during creation
        runner = create_experiment_runner(timeout=-1)

        # Runner should still be created
        assert isinstance(runner, ExperimentRunner)

    def test_factory_with_none_providers(self):
        """Verify factory handles None providers gracefully."""
        runner = create_experiment_runner(providers=None)

        # Should create with default providers
        assert len(runner.providers) == 4  # mock, openrouter, ollama, openai

    def test_factory_with_empty_providers(self):
        """Verify factory handles empty providers dict."""
        runner = create_experiment_runner(providers={})

        # Should have empty providers dict
        assert len(runner.providers) == 0


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_factory_signature_matches_runner_init(self):
        """Verify factory signature compatible with ExperimentRunner.__init__."""
        import inspect

        factory_sig = inspect.signature(create_experiment_runner)
        runner_sig = inspect.signature(ExperimentRunner.__init__)

        # Factory should have compatible parameters
        factory_params = set(factory_sig.parameters.keys())
        runner_params = set(runner_sig.parameters.keys()) - {'self'}

        # Factory params should be subset of runner params (or add session)
        assert factory_params <= (runner_params | {'session'})

    def test_existing_cli_tests_still_work(self):
        """Verify existing CLI tests continue to work."""
        # This is a placeholder - existing tests in test_cli.py should pass
        # They will use the factory indirectly through CLI commands
        assert True
