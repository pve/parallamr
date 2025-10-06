# Factory Pattern Test Templates

Complete test function signatures and implementations for TDD.

## test_factory.py

```python
"""Tests for ExperimentRunner factory function."""

import pytest
from unittest.mock import Mock
import aiohttp

from parallamr.cli import create_experiment_runner
from parallamr.providers import MockProvider
from parallamr.file_loader import FileLoader


class TestFactoryFunction:
    """Test create_experiment_runner factory function."""

    def test_factory_creates_runner_with_default_providers(self):
        """Factory creates runner with default providers when none specified."""
        runner = create_experiment_runner(timeout=60, verbose=True)

        assert runner.timeout == 60
        assert runner.verbose is True
        assert "mock" in runner.providers
        assert "openrouter" in runner.providers
        assert "ollama" in runner.providers

    def test_factory_creates_runner_with_custom_providers(self):
        """Factory accepts custom providers dict."""
        custom_providers = {"mock": MockProvider(timeout=30)}

        runner = create_experiment_runner(timeout=60, providers=custom_providers)

        assert runner.timeout == 60
        assert runner.providers == custom_providers
        assert "openrouter" not in runner.providers

    def test_factory_creates_runner_with_custom_file_loader(self):
        """Factory accepts custom file loader."""
        mock_loader = Mock(spec=FileLoader)

        runner = create_experiment_runner(timeout=60, file_loader=mock_loader)

        assert runner.file_loader is mock_loader

    def test_factory_creates_runner_with_session_injection(self):
        """Factory creates providers with shared session when provided."""
        mock_session = Mock(spec=aiohttp.ClientSession)

        runner = create_experiment_runner(timeout=60, session=mock_session)

        # Verify session injected into HTTP-based providers
        assert runner.providers["openrouter"]._session is mock_session
        assert runner.providers["ollama"]._session is mock_session

    def test_factory_custom_providers_override_session(self):
        """When custom providers given, session parameter ignored."""
        mock_session = Mock(spec=aiohttp.ClientSession)
        custom_providers = {"mock": MockProvider(timeout=30)}

        runner = create_experiment_runner(
            timeout=60,
            session=mock_session,
            providers=custom_providers
        )

        assert runner.providers == custom_providers

    def test_factory_default_parameters(self):
        """Factory uses sensible defaults."""
        runner = create_experiment_runner()

        assert runner.timeout == 300
        assert runner.verbose is False
        assert runner.providers is not None
        assert runner.file_loader is not None

    def test_factory_accepts_zero_timeout(self):
        """Factory accepts any timeout value (validation in CLI)."""
        runner = create_experiment_runner(timeout=0)
        assert runner.timeout == 0

    def test_factory_accepts_negative_timeout(self):
        """Factory accepts negative timeout (CLI validates)."""
        runner = create_experiment_runner(timeout=-1)
        assert runner.timeout == -1

    def test_factory_none_providers_creates_defaults(self):
        """Explicitly passing None providers creates defaults."""
        runner = create_experiment_runner(providers=None)

        assert "mock" in runner.providers
        assert "openrouter" in runner.providers
        assert "ollama" in runner.providers

    def test_factory_empty_providers_dict(self):
        """Factory accepts empty providers dict."""
        runner = create_experiment_runner(providers={})

        assert runner.providers == {}
        assert len(runner.providers) == 0

    def test_factory_preserves_provider_timeout(self):
        """Factory passes timeout to created providers."""
        runner = create_experiment_runner(timeout=120)

        # Default providers should have the timeout
        assert runner.providers["mock"].timeout == 120
        assert runner.providers["openrouter"].timeout == 120
        assert runner.providers["ollama"].timeout == 120

    def test_factory_session_creates_providers_with_session(self):
        """Session parameter triggers provider creation with session."""
        mock_session = Mock(spec=aiohttp.ClientSession)

        runner = create_experiment_runner(timeout=60, session=mock_session)

        # All HTTP providers share the same session
        assert (
            runner.providers["openrouter"]._session
            is runner.providers["ollama"]._session
        )
        assert runner.providers["openrouter"]._session is mock_session
```

## test_cli_factory.py

```python
"""Tests for CLI factory pattern integration."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from click.testing import CliRunner
from pathlib import Path

from parallamr.cli import cli, create_experiment_runner
from parallamr.runner import ExperimentRunner
from parallamr.providers import MockProvider


class TestCLIFactoryInjection:
    """Test factory injection into CLI commands."""

    def test_run_command_uses_default_factory(self, tmp_path):
        """Run command uses DEFAULT_RUNNER_FACTORY by default."""
        runner = CliRunner()

        factory_calls = []

        def tracking_factory(**kwargs):
            factory_calls.append(kwargs)
            return create_experiment_runner(**kwargs)

        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("prompt.txt").write_text("Test prompt")
            Path("experiments.csv").write_text("provider,model\nmock,mock")

            with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', tracking_factory):
                result = runner.invoke(cli, [
                    'run',
                    '--prompt', 'prompt.txt',
                    '--experiments', 'experiments.csv',
                    '--output', 'results.csv',
                    '--timeout', '60',
                    '--verbose'
                ])

            assert result.exit_code == 0
            assert len(factory_calls) == 1
            assert factory_calls[0]['timeout'] == 60
            assert factory_calls[0]['verbose'] is True

    def test_run_command_with_mock_runner(self, tmp_path):
        """CLI argument parsing tested without running experiments."""
        runner = CliRunner()

        mock_runner = Mock(spec=ExperimentRunner)
        mock_runner.run_experiments = AsyncMock()

        def mock_factory(**kwargs):
            return mock_runner

        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("prompt.txt").write_text("Test prompt")
            Path("experiments.csv").write_text("provider,model\nmock,mock")

            with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', mock_factory):
                result = runner.invoke(cli, [
                    'run',
                    '--prompt', 'prompt.txt',
                    '--experiments', 'experiments.csv',
                    '--output', 'results.csv'
                ])

            # Should not raise errors from validation
            assert result.exit_code == 0
            # Verify runner.run_experiments was called
            mock_runner.run_experiments.assert_called_once()

    def test_validate_command_uses_factory(self, tmp_path):
        """Validate command creates runner via factory."""
        runner = CliRunner()

        mock_runner = Mock(spec=ExperimentRunner)
        mock_runner.validate_experiments = AsyncMock(return_value={
            "valid": True,
            "experiments": 2,
            "providers": {},
            "warnings": []
        })

        def mock_factory(**kwargs):
            return mock_runner

        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("experiments.csv").write_text("provider,model\nmock,mock")

            with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', mock_factory):
                result = runner.invoke(cli, [
                    'run',
                    '--prompt', '-',
                    '--experiments', 'experiments.csv',
                    '--validate-only'
                ], input="Test prompt")

            assert result.exit_code == 0
            mock_runner.validate_experiments.assert_called_once()

    def test_models_command_uses_factory(self):
        """Models command creates runner via factory."""
        runner = CliRunner()

        mock_provider = Mock()
        mock_provider.list_models = AsyncMock(return_value=["model1", "model2"])

        mock_runner = Mock(spec=ExperimentRunner)
        mock_runner.providers = {"ollama": mock_provider}

        def mock_factory(**kwargs):
            return mock_runner

        with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', mock_factory):
            result = runner.invoke(cli, ['models', 'ollama'])

        assert result.exit_code == 0
        assert "model1" in result.output
        assert "model2" in result.output
        mock_provider.list_models.assert_called_once()


class TestCLIArgumentParsingWithoutAPIKeys:
    """Test CLI can parse arguments without requiring API keys."""

    def test_run_missing_required_args_no_api_key_needed(self):
        """Argument validation happens before runner creation."""
        runner = CliRunner()

        result = runner.invoke(cli, ['run'], env={})

        assert result.exit_code != 0
        assert "Missing option" in result.output
        # Should NOT mention API key
        assert "API key" not in result.output.lower()

    def test_run_nonexistent_file_no_api_key_needed(self):
        """File validation happens before runner creation."""
        runner = CliRunner()

        result = runner.invoke(cli, [
            'run',
            '--prompt', 'nonexistent.txt',
            '--experiments', 'nonexistent.csv'
        ], env={})

        assert result.exit_code != 0
        assert "not found" in result.output
        # Should NOT reach runner creation
        assert "API key" not in result.output.lower()

    def test_run_invalid_timeout_no_api_key_needed(self):
        """Timeout validation happens before runner creation."""
        runner = CliRunner()

        result = runner.invoke(cli, [
            'run',
            '--prompt', 'prompt.txt',
            '--experiments', 'experiments.csv',
            '--timeout', '0'
        ], env={})

        assert result.exit_code != 0
        assert "Timeout must be positive" in result.output
        # Should NOT reach runner creation
        assert "API key" not in result.output.lower()

    def test_run_both_stdin_error_no_api_key_needed(self):
        """Stdin validation happens before runner creation."""
        runner = CliRunner()

        result = runner.invoke(cli, [
            'run',
            '--prompt', '-',
            '--experiments', '-',
            '--output', 'results.csv'
        ], env={})

        assert result.exit_code != 0
        assert "Cannot read both" in result.output or "stdin" in result.output.lower()
        # Should NOT reach runner creation
        assert "API key" not in result.output.lower()

    def test_validation_before_factory_call(self, tmp_path):
        """Factory not called when validation fails."""
        runner = CliRunner()

        factory_called = False

        def tracking_factory(**kwargs):
            nonlocal factory_called
            factory_called = True
            return create_experiment_runner(**kwargs)

        with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', tracking_factory):
            # Invalid timeout
            result = runner.invoke(cli, [
                'run',
                '--prompt', 'prompt.txt',
                '--experiments', 'experiments.csv',
                '--timeout', '-1'
            ])

        assert result.exit_code != 0
        # Factory should NOT have been called
        assert factory_called is False


class TestCLIErrorHandlingWithFactory:
    """Test error handling paths using factory pattern."""

    def test_run_handles_runner_creation_error(self, tmp_path):
        """CLI handles errors from factory gracefully."""
        runner = CliRunner()

        def failing_factory(**kwargs):
            raise RuntimeError("Factory initialization failed")

        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("prompt.txt").write_text("Test")
            Path("experiments.csv").write_text("provider,model\nmock,mock")

            with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', failing_factory):
                result = runner.invoke(cli, [
                    'run',
                    '--prompt', 'prompt.txt',
                    '--experiments', 'experiments.csv'
                ])

            # Should fail gracefully (might be caught or propagate)
            assert result.exit_code != 0

    def test_run_handles_experiment_error(self, tmp_path):
        """CLI handles errors from runner.run_experiments."""
        runner = CliRunner()

        mock_runner = Mock(spec=ExperimentRunner)
        mock_runner.run_experiments = AsyncMock(
            side_effect=RuntimeError("Experiment failed")
        )

        def mock_factory(**kwargs):
            return mock_runner

        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("prompt.txt").write_text("Test")
            Path("experiments.csv").write_text("provider,model\nmock,mock")

            with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', mock_factory):
                result = runner.invoke(cli, [
                    'run',
                    '--prompt', 'prompt.txt',
                    '--experiments', 'experiments.csv'
                ])

            assert result.exit_code == 1
            assert "Error running experiments" in result.output

    def test_validate_handles_validation_error(self, tmp_path):
        """CLI handles errors from runner.validate_experiments."""
        runner = CliRunner()

        mock_runner = Mock(spec=ExperimentRunner)
        mock_runner.validate_experiments = AsyncMock(
            side_effect=ValueError("Invalid experiment format")
        )

        def mock_factory(**kwargs):
            return mock_runner

        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("experiments.csv").write_text("invalid,csv")

            with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', mock_factory):
                result = runner.invoke(cli, [
                    'run',
                    '--prompt', '-',
                    '--experiments', 'experiments.csv',
                    '--validate-only'
                ], input="test")

            assert result.exit_code == 1
            assert "Error validating experiments" in result.output


class TestFactoryWithProviderInjection:
    """Test factory enables provider injection for CLI testing."""

    def test_run_with_custom_mock_provider(self, tmp_path):
        """CLI can use custom provider via factory."""
        runner = CliRunner()

        custom_mock = MockProvider(timeout=30)

        def custom_factory(**kwargs):
            return create_experiment_runner(
                **kwargs,
                providers={"mock": custom_mock}
            )

        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("prompt.txt").write_text("Test prompt")
            Path("experiments.csv").write_text("provider,model\nmock,mock")

            with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', custom_factory):
                result = runner.invoke(cli, [
                    'run',
                    '--prompt', 'prompt.txt',
                    '--experiments', 'experiments.csv',
                    '--output', 'results.csv'
                ])

            assert result.exit_code == 0
            assert Path("results.csv").exists()

    def test_run_with_spy_provider_tracks_calls(self, tmp_path):
        """Factory enables spying on provider calls."""
        runner = CliRunner()

        call_log = []

        class SpyMockProvider(MockProvider):
            async def get_completion(self, prompt, model, **kwargs):
                call_log.append({
                    "prompt": prompt,
                    "model": model,
                    "kwargs": kwargs
                })
                return await super().get_completion(prompt, model, **kwargs)

        def spy_factory(**kwargs):
            return create_experiment_runner(
                **kwargs,
                providers={"mock": SpyMockProvider(timeout=30)}
            )

        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("prompt.txt").write_text("Test {{var}}")
            Path("experiments.csv").write_text(
                "provider,model,var\nmock,mock,value1\nmock,mock,value2"
            )

            with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', spy_factory):
                result = runner.invoke(cli, [
                    'run',
                    '--prompt', 'prompt.txt',
                    '--experiments', 'experiments.csv',
                    '--output', 'results.csv'
                ])

            # Verify provider was called twice
            assert len(call_log) == 2
            assert "value1" in call_log[0]["prompt"]
            assert "value2" in call_log[1]["prompt"]


class TestFactoryWithFileLoaderInjection:
    """Test factory enables file loader injection."""

    def test_run_with_custom_file_loader(self, tmp_path):
        """CLI can use custom file loader via factory."""
        runner = CliRunner()

        mock_loader = Mock(spec=FileLoader)
        mock_loader.load_file = Mock(return_value="Mocked prompt content")

        def custom_factory(**kwargs):
            return create_experiment_runner(
                **kwargs,
                providers={"mock": MockProvider(timeout=30)},
                file_loader=mock_loader
            )

        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("prompt.txt").write_text("Real prompt")
            Path("experiments.csv").write_text("provider,model\nmock,mock")

            with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', custom_factory):
                result = runner.invoke(cli, [
                    'run',
                    '--prompt', 'prompt.txt',
                    '--experiments', 'experiments.csv',
                    '--output', 'results.csv'
                ])

            # Verify mock file loader was used
            mock_loader.load_file.assert_called()


class TestFactoryWithSessionInjection:
    """Test factory enables session injection for parallel processing."""

    def test_factory_creates_providers_with_shared_session(self):
        """Factory can inject shared session into providers."""
        mock_session = Mock(spec=aiohttp.ClientSession)

        runner = create_experiment_runner(timeout=60, session=mock_session)

        # Verify providers share session
        assert runner.providers["openrouter"]._session is mock_session
        assert runner.providers["ollama"]._session is mock_session

        # Both providers use same session instance
        assert (
            runner.providers["openrouter"]._session
            is runner.providers["ollama"]._session
        )

    def test_cli_can_use_factory_with_session(self, tmp_path):
        """CLI can pass session through factory for parallel mode."""
        runner = CliRunner()

        mock_session = Mock(spec=aiohttp.ClientSession)

        def session_factory(**kwargs):
            return create_experiment_runner(**kwargs, session=mock_session)

        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("prompt.txt").write_text("Test")
            Path("experiments.csv").write_text("provider,model\nmock,mock")

            with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', session_factory):
                result = runner.invoke(cli, [
                    'run',
                    '--prompt', 'prompt.txt',
                    '--experiments', 'experiments.csv',
                    '--output', 'results.csv'
                ])

            assert result.exit_code == 0


class TestCompleteEndToEnd:
    """Complete end-to-end tests demonstrating all patterns."""

    def test_complete_cli_flow_with_tracking(self, tmp_path):
        """End-to-end CLI test with all components tracked."""
        runner = CliRunner()

        interactions = {
            "factory_calls": [],
            "provider_calls": [],
        }

        class TrackingMockProvider(MockProvider):
            async def get_completion(self, prompt, model, **kwargs):
                interactions["provider_calls"].append({
                    "prompt": prompt,
                    "model": model
                })
                return await super().get_completion(prompt, model, **kwargs)

        def tracking_factory(**kwargs):
            interactions["factory_calls"].append(kwargs)
            return create_experiment_runner(
                **kwargs,
                providers={"mock": TrackingMockProvider()}
            )

        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("prompt.txt").write_text("Test {{var}}")
            Path("experiments.csv").write_text(
                "provider,model,var\nmock,mock,value1\nmock,mock,value2"
            )

            with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', tracking_factory):
                result = runner.invoke(cli, [
                    'run',
                    '--prompt', 'prompt.txt',
                    '--experiments', 'experiments.csv',
                    '--output', 'results.csv',
                    '--timeout', '60',
                    '--verbose'
                ])

        assert result.exit_code == 0

        # Factory called once with correct params
        assert len(interactions["factory_calls"]) == 1
        assert interactions["factory_calls"][0]["timeout"] == 60
        assert interactions["factory_calls"][0]["verbose"] is True

        # Provider called twice (two experiments)
        assert len(interactions["provider_calls"]) == 2
        assert "value1" in interactions["provider_calls"][0]["prompt"]
        assert "value2" in interactions["provider_calls"][1]["prompt"]

        # Output file created
        assert Path("results.csv").exists()
```

## test_cli.py additions

```python
# Add to existing test_cli.py

class TestCLIWithFactoryPattern:
    """Tests verifying factory pattern integration."""

    def test_run_creates_runner_with_correct_timeout(self, tmp_path):
        """Run command passes timeout to factory."""
        runner = CliRunner()

        created_runners = []

        def tracking_factory(**kwargs):
            runner_instance = create_experiment_runner(**kwargs)
            created_runners.append(runner_instance)
            return runner_instance

        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("prompt.txt").write_text("Test")
            Path("experiments.csv").write_text("provider,model\nmock,mock")

            with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', tracking_factory):
                result = runner.invoke(cli, [
                    'run',
                    '--prompt', 'prompt.txt',
                    '--experiments', 'experiments.csv',
                    '--timeout', '120'
                ])

            assert len(created_runners) == 1
            assert created_runners[0].timeout == 120

    def test_run_creates_runner_with_verbose_flag(self, tmp_path):
        """Run command passes verbose flag to factory."""
        runner = CliRunner()

        created_runners = []

        def tracking_factory(**kwargs):
            runner_instance = create_experiment_runner(**kwargs)
            created_runners.append(runner_instance)
            return runner_instance

        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("prompt.txt").write_text("Test")
            Path("experiments.csv").write_text("provider,model\nmock,mock")

            with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', tracking_factory):
                result = runner.invoke(cli, [
                    'run',
                    '--prompt', 'prompt.txt',
                    '--experiments', 'experiments.csv',
                    '--verbose'
                ])

            assert len(created_runners) == 1
            assert created_runners[0].verbose is True

    def test_models_command_creates_runner_via_factory(self):
        """Models command uses factory to create runner."""
        runner = CliRunner()

        mock_provider = Mock()
        mock_provider.list_models = AsyncMock(return_value=["model1"])

        def mock_factory(**kwargs):
            mock_runner = Mock(spec=ExperimentRunner)
            mock_runner.providers = {"ollama": mock_provider}
            return mock_runner

        with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', mock_factory):
            result = runner.invoke(cli, ['models', 'ollama'])

        assert result.exit_code == 0
        assert "model1" in result.output
```

## Implementation Order

### Step 1: Write failing tests
```bash
# Create test files with all tests above
touch tests/test_factory.py
touch tests/test_cli_factory.py

# Tests will fail because factory doesn't exist yet
pytest tests/test_factory.py tests/test_cli_factory.py -v
```

### Step 2: Implement factory
```python
# In cli.py, add after imports:

def create_experiment_runner(
    timeout: int = 300,
    verbose: bool = False,
    providers: Optional[Dict[str, Provider]] = None,
    file_loader: Optional[FileLoader] = None,
    session: Optional[aiohttp.ClientSession] = None
) -> ExperimentRunner:
    """
    Factory function to create ExperimentRunner with proper dependency injection.

    Args:
        timeout: Request timeout in seconds
        verbose: Enable verbose logging
        providers: Optional custom providers dict (for testing)
        file_loader: Optional custom file loader (for testing)
        session: Optional shared HTTP session for parallel processing

    Returns:
        Configured ExperimentRunner instance
    """
    # If session provided, inject it into providers
    if session and providers is None:
        from .providers import MockProvider, OpenRouterProvider, OllamaProvider
        providers = {
            "mock": MockProvider(timeout=timeout),
            "openrouter": OpenRouterProvider(timeout=timeout, session=session),
            "ollama": OllamaProvider(timeout=timeout, session=session),
        }

    return ExperimentRunner(
        timeout=timeout,
        verbose=verbose,
        providers=providers,
        file_loader=file_loader
    )


DEFAULT_RUNNER_FACTORY = create_experiment_runner
```

### Step 3: Update CLI commands
```python
# Line 129 - in run() command:
# Before:
runner = ExperimentRunner(timeout=timeout, verbose=verbose)

# After:
runner = DEFAULT_RUNNER_FACTORY(timeout=timeout, verbose=verbose)


# Line 244 - in _list_models() function:
# Before:
runner = ExperimentRunner()

# After:
runner = DEFAULT_RUNNER_FACTORY()
```

### Step 4: Run tests
```bash
# Run factory tests
pytest tests/test_factory.py -v

# Run CLI factory integration tests
pytest tests/test_cli_factory.py -v

# Run all CLI tests
pytest tests/test_cli.py -v

# Run all tests
pytest tests/ -v
```

### Step 5: Verify coverage
```bash
# Check coverage
pytest tests/ --cov=src/parallamr --cov-report=term-missing

# Verify cli.py lines 129 and 244 are covered
pytest tests/test_cli_factory.py --cov=src/parallamr/cli --cov-report=term-missing
```

## Quick Start Commands

```bash
# 1. Create test files
cat > tests/test_factory.py << 'EOF'
# Paste test_factory.py content here
EOF

cat > tests/test_cli_factory.py << 'EOF'
# Paste test_cli_factory.py content here
EOF

# 2. Run tests (should fail)
pytest tests/test_factory.py tests/test_cli_factory.py -v

# 3. Implement factory in cli.py
# (Add create_experiment_runner function and DEFAULT_RUNNER_FACTORY)

# 4. Update lines 129 and 244 in cli.py
# (Use DEFAULT_RUNNER_FACTORY instead of ExperimentRunner)

# 5. Run tests (should pass)
pytest tests/test_factory.py tests/test_cli_factory.py -v

# 6. Run all tests
pytest tests/ -v

# 7. Check coverage
pytest tests/ --cov=src/parallamr/cli --cov-report=term-missing
```
