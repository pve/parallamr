# TDD Design: CLI Runner Factory Pattern

## Problem Analysis

### Current Issues (Lines 129, 244 in cli.py)
1. **Direct instantiation**: `runner = ExperimentRunner(timeout=timeout, verbose=verbose)`
2. **Testing challenges**:
   - CLI tests require actual API keys even for argument parsing tests
   - Cannot test CLI logic without executing experiments
   - Tight coupling prevents mocking runner behavior
   - Hard to test error handling paths in CLI

### Current Architecture
- **cli.py**: Contains Click commands (`run`, `validate`, `models`)
- **runner.py**: ExperimentRunner already supports dependency injection for:
  - `providers`: Dict[str, Provider]
  - `file_loader`: FileLoader
  - Session injection is implemented in providers (see test_session_injection.py)

## Factory Pattern Design

### Option 1: Factory Function (RECOMMENDED)

**Rationale**: Simple, functional, matches existing codebase patterns, easy to test.

```python
# In cli.py or new factory.py module

from typing import Callable, Dict, Optional
from pathlib import Path
import aiohttp
from .runner import ExperimentRunner
from .providers import Provider
from .file_loader import FileLoader

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

    Example:
        # Production usage
        runner = create_experiment_runner(timeout=60, verbose=True)

        # Test usage with mocks
        runner = create_experiment_runner(
            timeout=30,
            providers={'mock': MockProvider()},
            file_loader=FakeFileLoader()
        )
    """
    # If session provided, inject it into providers
    if session and providers is None:
        # Create default providers with shared session
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


# Type alias for factory function (enables injection)
RunnerFactory = Callable[..., ExperimentRunner]
```

### Option 2: Factory Class (Alternative)

```python
class ExperimentRunnerFactory:
    """
    Factory class for creating ExperimentRunner instances.

    Useful if you need to maintain factory state or configuration.
    """

    def __init__(
        self,
        default_timeout: int = 300,
        session: Optional[aiohttp.ClientSession] = None
    ):
        """
        Initialize factory with default configuration.

        Args:
            default_timeout: Default timeout for created runners
            session: Shared session for all runners (parallel mode)
        """
        self.default_timeout = default_timeout
        self.session = session

    def create(
        self,
        timeout: Optional[int] = None,
        verbose: bool = False,
        providers: Optional[Dict[str, Provider]] = None,
        file_loader: Optional[FileLoader] = None
    ) -> ExperimentRunner:
        """Create a runner instance."""
        actual_timeout = timeout or self.default_timeout

        # Inject session if available
        if self.session and providers is None:
            providers = self._create_providers_with_session(actual_timeout)

        return ExperimentRunner(
            timeout=actual_timeout,
            verbose=verbose,
            providers=providers,
            file_loader=file_loader
        )

    def _create_providers_with_session(
        self,
        timeout: int
    ) -> Dict[str, Provider]:
        """Create providers with shared session."""
        from .providers import MockProvider, OpenRouterProvider, OllamaProvider
        return {
            "mock": MockProvider(timeout=timeout),
            "openrouter": OpenRouterProvider(timeout=timeout, session=self.session),
            "ollama": OllamaProvider(timeout=timeout, session=self.session),
        }
```

**Recommendation**: Use **Option 1 (Factory Function)** because:
- Simpler and more Pythonic
- Matches existing functional style in codebase
- Easier to mock in tests
- No unnecessary state management
- Stateless factories are easier to reason about

## CLI Integration

### Updated cli.py Structure

```python
# cli.py

from typing import Optional, Callable
from .runner import ExperimentRunner

# Default factory (can be overridden in tests)
DEFAULT_RUNNER_FACTORY = create_experiment_runner

@cli.command()
@click.option("--timeout", type=int, default=300)
@click.option("--verbose", "-v", is_flag=True)
# ... other options ...
def run(
    prompt: str,
    experiments: str,
    output: Optional[Path],
    context: tuple[Path, ...],
    verbose: bool,
    timeout: int,
    validate_only: bool,
    _runner_factory: Callable = None  # Hidden parameter for testing
) -> None:
    """Run experiments across multiple LLM providers and models."""

    # Validation logic (unchanged)
    if timeout <= 0:
        click.echo("Error: Timeout must be positive", err=True)
        sys.exit(1)

    # ... file validation ...

    # Use injected factory or default
    factory = _runner_factory or DEFAULT_RUNNER_FACTORY

    # Create runner via factory
    runner = factory(timeout=timeout, verbose=verbose)

    # Rest of logic unchanged
    if validate_only:
        asyncio.run(_validate_experiments(runner, experiments_path, experiments == "-"))
    else:
        asyncio.run(_run_experiments(
            runner=runner,
            prompt_file=prompt_path,
            experiments_file=experiments_path,
            output_file=output,
            context_files=list(context) if context else None,
            read_prompt_stdin=prompt == "-",
            read_experiments_stdin=experiments == "-"
        ))


async def _list_models(
    provider: str,
    _runner_factory: Callable = None  # For testing
) -> None:
    """List models for a provider asynchronously."""
    factory = _runner_factory or DEFAULT_RUNNER_FACTORY
    runner = factory()

    # Rest of logic unchanged
    if provider not in runner.providers:
        click.echo(f"Provider '{provider}' not available", err=True)
        sys.exit(1)

    # ... rest of implementation ...


@cli.command()
@click.argument("provider", type=click.Choice(["openrouter", "ollama"]))
def models(
    provider: str,
    _runner_factory: Callable = None  # Hidden for testing
) -> None:
    """List available models for a specific provider."""
    asyncio.run(_list_models(provider, _runner_factory=_runner_factory))
```

### Alternative: Global Factory Override for Tests

```python
# cli.py

# Module-level factory (can be monkey-patched in tests)
_runner_factory: Callable = create_experiment_runner

def set_runner_factory(factory: Callable) -> None:
    """Set custom runner factory (for testing)."""
    global _runner_factory
    _runner_factory = factory

def get_runner_factory() -> Callable:
    """Get current runner factory."""
    return _runner_factory


@cli.command()
def run(...):
    """Run experiments."""
    # Use module-level factory
    runner = _runner_factory(timeout=timeout, verbose=verbose)
    # ... rest of implementation ...
```

**Recommendation**: Use **hidden parameter approach** (`_runner_factory`) because:
- Click will ignore parameters starting with `_`
- More explicit in tests (no global state mutation)
- Each test can use different factory without side effects
- No monkey-patching required

## Test Design

### Test File Structure

```
tests/
  test_cli_factory.py          # New: Factory pattern tests
  test_cli.py                   # Updated: CLI with factory injection
  test_factory.py               # New: Factory function tests
```

### Test Scenarios

#### 1. Factory Function Tests (test_factory.py)

```python
"""Tests for ExperimentRunner factory function."""

import pytest
from unittest.mock import Mock, AsyncMock
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
        custom_providers = {
            "mock": MockProvider(timeout=30)
        }

        runner = create_experiment_runner(
            timeout=60,
            providers=custom_providers
        )

        assert runner.timeout == 60
        assert runner.providers == custom_providers
        assert "openrouter" not in runner.providers  # Only custom providers

    def test_factory_creates_runner_with_custom_file_loader(self):
        """Factory accepts custom file loader."""
        mock_loader = Mock(spec=FileLoader)

        runner = create_experiment_runner(
            timeout=60,
            file_loader=mock_loader
        )

        assert runner.file_loader is mock_loader

    def test_factory_creates_runner_with_session_injection(self):
        """Factory creates providers with shared session when provided."""
        mock_session = Mock(spec=aiohttp.ClientSession)

        runner = create_experiment_runner(
            timeout=60,
            session=mock_session
        )

        # Verify session injected into HTTP-based providers
        assert runner.providers["openrouter"]._session is mock_session
        assert runner.providers["ollama"]._session is mock_session
        # Mock provider doesn't use session
        assert not hasattr(runner.providers["mock"], "_session")

    def test_factory_custom_providers_override_session(self):
        """When custom providers given, session parameter ignored."""
        mock_session = Mock(spec=aiohttp.ClientSession)
        custom_providers = {"mock": MockProvider(timeout=30)}

        runner = create_experiment_runner(
            timeout=60,
            session=mock_session,
            providers=custom_providers
        )

        # Custom providers used, session not injected
        assert runner.providers == custom_providers

    def test_factory_default_parameters(self):
        """Factory uses sensible defaults."""
        runner = create_experiment_runner()

        assert runner.timeout == 300  # Default timeout
        assert runner.verbose is False  # Default not verbose
        assert runner.providers is not None  # Default providers created
        assert runner.file_loader is not None  # Default file loader


class TestFactoryParameterValidation:
    """Test factory parameter validation and edge cases."""

    def test_factory_accepts_zero_timeout(self):
        """Factory accepts any timeout value (validation in CLI)."""
        # Factory doesn't validate, CLI layer does
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
```

#### 2. CLI Factory Integration Tests (test_cli_factory.py)

```python
"""Tests for CLI factory pattern integration."""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from click.testing import CliRunner
from pathlib import Path

from parallamr.cli import cli, create_experiment_runner
from parallamr.runner import ExperimentRunner
from parallamr.providers import MockProvider


class TestCLIFactoryInjection:
    """Test factory injection into CLI commands."""

    def test_run_command_uses_factory_to_create_runner(self, tmp_path):
        """Run command creates runner via factory."""
        runner = CliRunner()

        # Track factory calls
        factory_called = []

        def mock_factory(**kwargs):
            factory_called.append(kwargs)
            # Return real runner for mock provider
            return create_experiment_runner(**kwargs)

        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create test files
            Path("prompt.txt").write_text("Test prompt")
            Path("experiments.csv").write_text("provider,model\nmock,mock")

            # Inject factory via hidden parameter
            result = runner.invoke(cli, [
                'run',
                '--prompt', 'prompt.txt',
                '--experiments', 'experiments.csv',
                '--output', 'results.csv',
                '--timeout', '60',
                '--verbose'
            ], obj={'_runner_factory': mock_factory})

            # Verify factory was called with correct parameters
            assert len(factory_called) == 1
            assert factory_called[0]['timeout'] == 60
            assert factory_called[0]['verbose'] is True

    def test_run_command_with_mock_runner(self, tmp_path):
        """CLI argument parsing tested without running experiments."""
        runner = CliRunner()

        # Create mock runner
        mock_runner = Mock(spec=ExperimentRunner)
        mock_runner.run_experiments = AsyncMock()

        def mock_factory(**kwargs):
            return mock_runner

        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("prompt.txt").write_text("Test prompt")
            Path("experiments.csv").write_text("provider,model\nmock,mock")

            # This approach requires Click context obj support
            # Alternative: use monkeypatch on module level
            with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', mock_factory):
                result = runner.invoke(cli, [
                    'run',
                    '--prompt', 'prompt.txt',
                    '--experiments', 'experiments.csv',
                    '--output', 'results.csv'
                ])

            # Verify runner.run_experiments was called
            mock_runner.run_experiments.assert_called_once()

    def test_validate_command_uses_factory(self, tmp_path):
        """Validate command creates runner via factory."""
        runner = CliRunner()

        # Mock runner with validate capability
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
                    '--prompt', '-',  # stdin
                    '--experiments', 'experiments.csv',
                    '--validate-only'
                ], input="Test prompt")

            # Verify validate_experiments was called
            mock_runner.validate_experiments.assert_called_once()

    def test_models_command_uses_factory(self):
        """Models command creates runner via factory."""
        runner = CliRunner()

        # Mock runner with models capability
        mock_provider = Mock()
        mock_provider.list_models = AsyncMock(return_value=["model1", "model2"])

        mock_runner = Mock(spec=ExperimentRunner)
        mock_runner.providers = {"ollama": mock_provider}

        def mock_factory(**kwargs):
            return mock_runner

        with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', mock_factory):
            result = runner.invoke(cli, ['models', 'ollama'])

        assert result.exit_code == 0
        mock_provider.list_models.assert_called_once()


class TestCLIArgumentParsingWithoutAPIKeys:
    """Test CLI can parse arguments without requiring API keys."""

    def test_run_missing_required_args_no_api_key_needed(self):
        """Argument validation happens before runner creation."""
        runner = CliRunner()

        # Don't provide required arguments
        result = runner.invoke(cli, ['run'], env={})

        # Should fail on missing arguments, not API key
        assert result.exit_code != 0
        assert "Missing option" in result.output
        # Should NOT mention API key
        assert "API key" not in result.output

    def test_run_nonexistent_file_no_api_key_needed(self):
        """File validation happens before runner creation."""
        runner = CliRunner()

        result = runner.invoke(cli, [
            'run',
            '--prompt', 'nonexistent.txt',
            '--experiments', 'nonexistent.csv'
        ], env={})

        # Should fail on file not found, not API key
        assert result.exit_code != 0
        assert "not found" in result.output
        # Should NOT reach runner creation
        assert "API key" not in result.output

    def test_run_invalid_timeout_no_api_key_needed(self):
        """Timeout validation happens before runner creation."""
        runner = CliRunner()

        result = runner.invoke(cli, [
            'run',
            '--prompt', 'prompt.txt',
            '--experiments', 'experiments.csv',
            '--timeout', '-1'
        ], env={})

        assert result.exit_code != 0
        assert "Timeout must be positive" in result.output
        # Should NOT reach runner creation
        assert "API key" not in result.output

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
        assert "Cannot read both" in result.output or "stdin" in result.output
        # Should NOT reach runner creation
        assert "API key" not in result.output


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

            # Should fail gracefully
            assert result.exit_code != 0
            # Error should be shown
            assert "Factory initialization failed" in result.output or "Error" in result.output

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

        # Custom mock provider that always succeeds
        custom_mock = MockProvider(timeout=30)

        def custom_factory(**kwargs):
            return create_experiment_runner(
                **kwargs,
                providers={"custom_mock": custom_mock}
            )

        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("prompt.txt").write_text("Test prompt")
            Path("experiments.csv").write_text("provider,model\ncustom_mock,mock")

            with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', custom_factory):
                result = runner.invoke(cli, [
                    'run',
                    '--prompt', 'prompt.txt',
                    '--experiments', 'experiments.csv',
                    '--output', 'results.csv'
                ])

            assert result.exit_code == 0
            # Verify results file created
            assert Path("results.csv").exists()

    def test_run_with_spy_provider_tracks_calls(self, tmp_path):
        """Factory enables spying on provider calls."""
        runner = CliRunner()

        # Spy on provider calls
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
            Path("experiments.csv").write_text("provider,model,var\nmock,mock,value1\nmock,mock,value2")

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

        # Mock file loader
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

    @pytest.mark.asyncio
    async def test_factory_creates_providers_with_shared_session(self):
        """Factory can inject shared session into providers."""
        mock_session = Mock(spec=aiohttp.ClientSession)

        runner = create_experiment_runner(
            timeout=60,
            session=mock_session
        )

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
            return create_experiment_runner(
                **kwargs,
                session=mock_session
            )

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

            # Should succeed
            assert result.exit_code == 0
```

#### 3. Updated CLI Tests (test_cli.py additions)

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

## Implementation Strategy

### Phase 1: Create Factory Function
1. Add `create_experiment_runner()` function to `cli.py`
2. Write tests in `test_factory.py`
3. Verify factory creates runners correctly

### Phase 2: Integrate Factory into CLI
1. Update `run()` command to use factory
2. Update `_list_models()` to use factory
3. Add module-level `DEFAULT_RUNNER_FACTORY` variable
4. Write tests in `test_cli_factory.py`

### Phase 3: Update Existing CLI Tests
1. Add factory-specific test cases to `test_cli.py`
2. Verify all existing tests still pass
3. Add new tests for error handling

### Phase 4: Documentation
1. Update docstrings
2. Add usage examples
3. Document testing patterns

## Benefits Summary

### Testability Improvements
1. **CLI argument parsing** can be tested without API keys
2. **Error handling paths** can be tested with mock failures
3. **Provider behavior** can be verified with spies/mocks
4. **File loading** can be tested with fake file systems
5. **Parallel processing** can be tested with mock sessions

### Code Quality
1. **Separation of concerns**: CLI handles arguments, factory handles creation
2. **Dependency injection**: All dependencies explicit and testable
3. **Single Responsibility**: Factory only creates runners
4. **Open/Closed**: Easy to add new configuration options

### Maintainability
1. **Centralized creation logic**: One place to configure runners
2. **Easy to extend**: Add new parameters to factory signature
3. **Backward compatible**: Existing code continues working
4. **Clear testing patterns**: Consistent approach for all CLI tests

## Mock Strategy Summary

### 1. Mock Runner for CLI Logic Testing
```python
mock_runner = Mock(spec=ExperimentRunner)
mock_runner.run_experiments = AsyncMock()
```
**Use case**: Test CLI argument parsing and flow control

### 2. Mock Providers for Provider Testing
```python
custom_providers = {"mock": MockProvider()}
runner = create_experiment_runner(providers=custom_providers)
```
**Use case**: Test experiment execution without real API calls

### 3. Mock File Loader for File Operations
```python
mock_loader = Mock(spec=FileLoader)
runner = create_experiment_runner(file_loader=mock_loader)
```
**Use case**: Test file handling without actual files

### 4. Mock Session for Parallel Processing
```python
mock_session = Mock(spec=aiohttp.ClientSession)
runner = create_experiment_runner(session=mock_session)
```
**Use case**: Test parallel execution without network calls

### 5. Spy Pattern for Call Verification
```python
class SpyProvider(MockProvider):
    async def get_completion(self, prompt, model, **kwargs):
        # Log call
        return await super().get_completion(prompt, model, **kwargs)
```
**Use case**: Verify provider is called with correct arguments

## Integration Points

### 1. cli.py Lines 129
**Before**:
```python
runner = ExperimentRunner(timeout=timeout, verbose=verbose)
```

**After**:
```python
runner = DEFAULT_RUNNER_FACTORY(timeout=timeout, verbose=verbose)
```

### 2. cli.py Line 244
**Before**:
```python
runner = ExperimentRunner()
```

**After**:
```python
runner = DEFAULT_RUNNER_FACTORY()
```

### 3. Import Addition
```python
# At top of cli.py
from .factory import create_experiment_runner

DEFAULT_RUNNER_FACTORY = create_experiment_runner
```

### 4. Test Pattern
```python
# In test files
with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', mock_factory):
    result = runner.invoke(cli, [...])
```

## File Structure After Implementation

```
src/parallamr/
  cli.py                 # Updated: uses factory
  factory.py             # New: factory function (or add to cli.py)
  runner.py             # Unchanged: already supports DI

tests/
  test_cli.py           # Updated: new factory tests added
  test_cli_factory.py   # New: factory integration tests
  test_factory.py       # New: factory function tests
```

## Alternative: Keep Factory in cli.py

Instead of creating `factory.py`, add `create_experiment_runner()` directly to `cli.py` since it's only used there.

**Pros**:
- Simpler file structure
- Factory close to usage
- No new module needed

**Cons**:
- cli.py gets slightly longer
- Factory harder to test in isolation

**Recommendation**: Keep in `cli.py` initially, extract if it grows complex.

## Testing Checklist

- [ ] Factory creates runner with default providers
- [ ] Factory accepts custom providers
- [ ] Factory accepts custom file loader
- [ ] Factory accepts session for parallel mode
- [ ] Factory handles None vs empty dict providers
- [ ] CLI uses factory for run command
- [ ] CLI uses factory for models command
- [ ] CLI argument parsing works without API keys
- [ ] CLI file validation happens before runner creation
- [ ] CLI timeout validation happens before runner creation
- [ ] Mock runner enables CLI flow testing
- [ ] Mock providers enable execution testing
- [ ] Mock file loader enables file operation testing
- [ ] Mock session enables parallel testing
- [ ] Error handling tested with factory failures
- [ ] Error handling tested with runner failures
- [ ] Spy pattern verifies provider calls
- [ ] Existing tests still pass after refactoring
