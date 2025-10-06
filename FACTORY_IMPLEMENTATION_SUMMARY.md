# CLI Runner Factory Pattern - Implementation Summary

## Quick Reference

### Problem
- Lines 129, 244 in `cli.py` create `ExperimentRunner` directly
- CLI tests require API keys even for argument parsing
- Cannot mock runner to test CLI logic independently
- Tight coupling prevents testability

### Solution
Factory function pattern with dependency injection

## Design Decisions

### 1. Factory Function (Not Class)
**Chosen**: Simple function-based factory

```python
def create_experiment_runner(
    timeout: int = 300,
    verbose: bool = False,
    providers: Optional[Dict[str, Provider]] = None,
    file_loader: Optional[FileLoader] = None,
    session: Optional[aiohttp.ClientSession] = None
) -> ExperimentRunner:
    """Factory function to create ExperimentRunner with DI."""
    # Inject session into providers if provided
    if session and providers is None:
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
```

**Why**: Simpler, matches codebase style, easier to test, no unnecessary state

### 2. Module-Level Factory Variable
**Chosen**: Patchable module variable with default factory

```python
# cli.py
DEFAULT_RUNNER_FACTORY = create_experiment_runner

@cli.command()
def run(...):
    runner = DEFAULT_RUNNER_FACTORY(timeout=timeout, verbose=verbose)
```

**Why**: Easy to patch in tests, no hidden Click parameters, clean test isolation

### 3. Factory Location
**Chosen**: Add directly to `cli.py` (not separate module)

**Why**:
- Only used by CLI
- Keeps related code together
- Simpler file structure
- Can extract later if needed

## Integration Points

### Change 1: cli.py line 129
```python
# Before
runner = ExperimentRunner(timeout=timeout, verbose=verbose)

# After
runner = DEFAULT_RUNNER_FACTORY(timeout=timeout, verbose=verbose)
```

### Change 2: cli.py line 244
```python
# Before
runner = ExperimentRunner()

# After
runner = DEFAULT_RUNNER_FACTORY()
```

### Change 3: Add factory function (top of cli.py after imports)
```python
def create_experiment_runner(...) -> ExperimentRunner:
    """Factory function..."""
    # Implementation

DEFAULT_RUNNER_FACTORY = create_experiment_runner
```

## Test Structure

### New Test Files
1. **test_factory.py**: Test factory function in isolation
   - Default provider creation
   - Custom provider injection
   - File loader injection
   - Session injection
   - Parameter validation

2. **test_cli_factory.py**: Test CLI integration with factory
   - Factory called with correct parameters
   - Mock runner for CLI flow testing
   - Provider injection for integration tests
   - Session injection for parallel tests
   - Error handling with mock failures

### Updated Test File
3. **test_cli.py**: Add factory-specific tests
   - Verify timeout passed to factory
   - Verify verbose flag passed to factory
   - Verify models command uses factory

## Key Test Scenarios

### 1. CLI Argument Parsing Without API Keys
```python
def test_run_missing_args_no_api_key_needed():
    """Argument validation before runner creation."""
    result = runner.invoke(cli, ['run'], env={})
    assert "Missing option" in result.output
    assert "API key" not in result.output
```

### 2. Mock Runner for CLI Logic
```python
def test_run_with_mock_runner():
    """Test CLI flow without running experiments."""
    mock_runner = Mock(spec=ExperimentRunner)
    mock_runner.run_experiments = AsyncMock()

    def mock_factory(**kwargs):
        return mock_runner

    with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', mock_factory):
        result = runner.invoke(cli, ['run', ...])

    mock_runner.run_experiments.assert_called_once()
```

### 3. Provider Injection
```python
def test_run_with_custom_provider():
    """Inject custom provider for testing."""
    custom_provider = MockProvider()

    def custom_factory(**kwargs):
        return create_experiment_runner(
            **kwargs,
            providers={"custom": custom_provider}
        )

    with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', custom_factory):
        result = runner.invoke(cli, ['run', ...])
```

### 4. Session Injection (Parallel Mode)
```python
def test_factory_with_session():
    """Factory injects session into providers."""
    mock_session = Mock(spec=aiohttp.ClientSession)

    runner = create_experiment_runner(session=mock_session)

    assert runner.providers["openrouter"]._session is mock_session
    assert runner.providers["ollama"]._session is mock_session
```

### 5. Spy Pattern for Call Verification
```python
class SpyMockProvider(MockProvider):
    def __init__(self):
        super().__init__()
        self.calls = []

    async def get_completion(self, prompt, model, **kwargs):
        self.calls.append({"prompt": prompt, "model": model})
        return await super().get_completion(prompt, model, **kwargs)

def test_verify_provider_calls():
    """Verify provider called with correct arguments."""
    spy = SpyMockProvider()

    def spy_factory(**kwargs):
        return create_experiment_runner(providers={"mock": spy})

    with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', spy_factory):
        runner.invoke(cli, ['run', ...])

    assert len(spy.calls) == 2
    assert "expected prompt" in spy.calls[0]["prompt"]
```

## Implementation Phases

### Phase 1: Factory Function (1-2 hours)
1. Add `create_experiment_runner()` to `cli.py`
2. Add `DEFAULT_RUNNER_FACTORY` variable
3. Write `test_factory.py` with 10+ test cases
4. Run tests: `pytest tests/test_factory.py -v`

### Phase 2: CLI Integration (1-2 hours)
1. Update line 129: `run` command
2. Update line 244: `_list_models` function
3. Write `test_cli_factory.py` with 15+ test cases
4. Run tests: `pytest tests/test_cli_factory.py -v`

### Phase 3: Existing Tests (30 min)
1. Add 3 new tests to `test_cli.py`
2. Verify all existing tests pass
3. Run full test suite: `pytest tests/ -v`

### Phase 4: Documentation (30 min)
1. Update CLI docstrings
2. Add examples to factory docstring
3. Update README if needed

**Total Estimated Time**: 4-5 hours

## Testing Best Practices

### Do's
- Use `patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', mock_factory)` for factory mocking
- Create factory functions inline in tests for clarity
- Test argument parsing separately from experiment execution
- Use `AsyncMock()` for async runner methods
- Verify factory called with expected parameters

### Don'ts
- Don't test actual API calls in factory tests (use mock providers)
- Don't skip validation tests (timeout, file exists, stdin)
- Don't forget to test error handling paths
- Don't use real API keys in any factory tests
- Don't mix unit tests with integration tests

## Expected Benefits

### Immediate
1. CLI argument parsing tests don't need API keys
2. Can test CLI error handling with mock failures
3. Can verify CLI calls runner with correct parameters
4. Parallel mode testable with mock sessions

### Long-term
1. Easier to add new CLI options
2. Better separation of concerns
3. More comprehensive test coverage
4. Faster test execution (less mocking needed)

## Success Metrics

- [ ] All existing CLI tests pass
- [ ] New tests in `test_factory.py` (10+ tests)
- [ ] New tests in `test_cli_factory.py` (15+ tests)
- [ ] CLI argument tests don't require API keys
- [ ] Mock runner enables CLI flow testing
- [ ] Provider injection enables integration testing
- [ ] Session injection enables parallel testing
- [ ] Code coverage increases for cli.py
- [ ] Test execution time acceptable (< 5s for CLI tests)

## Example: Complete Test Flow

```python
# test_cli_factory.py

def test_complete_cli_flow_with_mocks(tmp_path):
    """End-to-end CLI test with all components mocked."""
    runner = CliRunner()

    # Track all interactions
    interactions = {
        "factory_calls": [],
        "provider_calls": [],
        "file_reads": []
    }

    # Mock provider with tracking
    class TrackingMockProvider(MockProvider):
        async def get_completion(self, prompt, model, **kwargs):
            interactions["provider_calls"].append({
                "prompt": prompt,
                "model": model
            })
            return await super().get_completion(prompt, model, **kwargs)

    # Mock file loader with tracking
    class TrackingFileLoader(FileLoader):
        def load_file(self, path):
            interactions["file_reads"].append(str(path))
            return super().load_file(path)

    # Factory with tracking
    def tracking_factory(**kwargs):
        interactions["factory_calls"].append(kwargs)
        return create_experiment_runner(
            **kwargs,
            providers={"mock": TrackingMockProvider()},
            file_loader=TrackingFileLoader()
        )

    # Run CLI command
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("prompt.txt").write_text("Test {{var}}")
        Path("experiments.csv").write_text(
            "provider,model,var\n"
            "mock,mock,value1\n"
            "mock,mock,value2"
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

    # Verify complete flow
    assert result.exit_code == 0

    # Factory called once with correct params
    assert len(interactions["factory_calls"]) == 1
    assert interactions["factory_calls"][0]["timeout"] == 60
    assert interactions["factory_calls"][0]["verbose"] is True

    # Provider called twice (two experiments)
    assert len(interactions["provider_calls"]) == 2
    assert "value1" in interactions["provider_calls"][0]["prompt"]
    assert "value2" in interactions["provider_calls"][1]["prompt"]

    # Files read
    assert "prompt.txt" in interactions["file_reads"]

    # Output file created
    assert Path("results.csv").exists()
```

## Next Steps

1. Review this design document
2. Start with Phase 1 (factory function)
3. Write tests first (TDD approach)
4. Implement factory function
5. Verify tests pass
6. Continue to Phase 2 (CLI integration)
7. Update documentation

## Questions to Consider

1. Should factory be in separate `factory.py` module or in `cli.py`?
   - **Recommendation**: Start in `cli.py`, extract if it grows

2. Should we add a `set_runner_factory()` function for tests?
   - **Recommendation**: No, use `patch()` for cleaner test isolation

3. How to handle backwards compatibility?
   - **Answer**: No breaking changes - existing code works as-is

4. Should factory validate parameters?
   - **Recommendation**: No, let CLI handle validation (already does)

5. How to test async factory behavior?
   - **Answer**: Factory is sync, returns runner which has async methods
