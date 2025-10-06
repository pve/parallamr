# CLI Factory Pattern Architecture

## Current Architecture (Before Factory)

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI Layer                           │
│                        (cli.py)                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  run() command (line 129)                                  │
│    ├─ Validate arguments                                   │
│    ├─ runner = ExperimentRunner(timeout, verbose) ◄────────┼─── PROBLEM: Direct instantiation
│    └─ asyncio.run(_run_experiments(runner, ...))           │    Cannot inject mocks
│                                                             │
│  models() command (line 244)                               │
│    ├─ Validate provider choice                             │
│    ├─ runner = ExperimentRunner() ◄─────────────────────────┼─── PROBLEM: Direct instantiation
│    └─ asyncio.run(_list_models(runner, provider))          │    Requires API keys for all tests
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ExperimentRunner                         │
│                      (runner.py)                            │
├─────────────────────────────────────────────────────────────┤
│  __init__(timeout, verbose, providers, file_loader)        │
│    ├─ self.timeout = timeout                               │
│    ├─ self.verbose = verbose                               │
│    ├─ self.providers = providers or _create_defaults()     │
│    └─ self.file_loader = file_loader or FileLoader()       │
│                                                             │
│  Already supports DI! ✓                                    │
│    - Can inject providers                                   │
│    - Can inject file_loader                                 │
│    - Providers support session injection                    │
└─────────────────────────────────────────────────────────────┘
```

## New Architecture (With Factory)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLI Layer                                      │
│                             (cli.py)                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────┐        │
│  │ Factory Function (NEW)                                        │        │
│  │                                                               │        │
│  │ def create_experiment_runner(                                 │        │
│  │     timeout=300,                                              │        │
│  │     verbose=False,                                            │        │
│  │     providers=None,      ◄──── Dependency Injection         │        │
│  │     file_loader=None,    ◄──── Dependency Injection         │        │
│  │     session=None         ◄──── For parallel processing       │        │
│  │ ) -> ExperimentRunner:                                        │        │
│  │     if session and providers is None:                         │        │
│  │         providers = create_providers_with_session(session)    │        │
│  │     return ExperimentRunner(...)                              │        │
│  │                                                               │        │
│  │ DEFAULT_RUNNER_FACTORY = create_experiment_runner             │        │
│  └───────────────────────────────────────────────────────────────┘        │
│                              ▲                                              │
│                              │ Testable!                                    │
│  run() command              │ Can patch in tests                           │
│    ├─ Validate arguments    │                                              │
│    ├─ runner = DEFAULT_RUNNER_FACTORY(timeout, verbose) ◄───────────────┐ │
│    └─ asyncio.run(_run_experiments(runner, ...))                        │ │
│                                                                          │ │
│  models() command                                                        │ │
│    ├─ Validate provider                                                 │ │
│    ├─ runner = DEFAULT_RUNNER_FACTORY() ◄───────────────────────────────┘ │
│    └─ asyncio.run(_list_models(runner, provider))                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
        ┌─────────────────────────────────────────────────────┐
        │         ExperimentRunner (unchanged)                │
        │              (runner.py)                            │
        ├─────────────────────────────────────────────────────┤
        │  __init__(timeout, verbose, providers, file_loader) │
        │    ├─ self.timeout = timeout                        │
        │    ├─ self.verbose = verbose                        │
        │    ├─ self.providers = providers or _create_...()   │
        │    └─ self.file_loader = file_loader or ...()       │
        │                                                      │
        │  run_experiments(...)                               │
        │  validate_experiments(...)                          │
        └─────────────────────────────────────────────────────┘
```

## Testing Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          Test Layer                                        │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Test Strategy 1: Mock Factory Function                                   │
│  ┌──────────────────────────────────────────────────────────────┐        │
│  │ def test_cli_with_mock_runner():                              │        │
│  │     mock_runner = Mock(spec=ExperimentRunner)                 │        │
│  │     mock_runner.run_experiments = AsyncMock()                 │        │
│  │                                                                │        │
│  │     def mock_factory(**kwargs):                               │        │
│  │         return mock_runner  ◄── Return mock instead of real  │        │
│  │                                                                │        │
│  │     with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY',        │        │
│  │                mock_factory):                                  │        │
│  │         result = runner.invoke(cli, ['run', ...])             │        │
│  │                                                                │        │
│  │     # Verify CLI logic without running experiments            │        │
│  │     mock_runner.run_experiments.assert_called_once()          │        │
│  └──────────────────────────────────────────────────────────────┘        │
│                                                                            │
│  Test Strategy 2: Inject Custom Providers                                 │
│  ┌──────────────────────────────────────────────────────────────┐        │
│  │ def test_cli_with_custom_provider():                          │        │
│  │     spy_provider = SpyMockProvider()  ◄── Track calls        │        │
│  │                                                                │        │
│  │     def custom_factory(**kwargs):                             │        │
│  │         return create_experiment_runner(                      │        │
│  │             **kwargs,                                          │        │
│  │             providers={"spy": spy_provider}                   │        │
│  │         )                                                      │        │
│  │                                                                │        │
│  │     with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY',        │        │
│  │                custom_factory):                                │        │
│  │         result = runner.invoke(cli, ['run', ...])             │        │
│  │                                                                │        │
│  │     # Verify provider was called correctly                    │        │
│  │     assert len(spy_provider.calls) == 2                       │        │
│  └──────────────────────────────────────────────────────────────┘        │
│                                                                            │
│  Test Strategy 3: Inject Mock File Loader                                 │
│  ┌──────────────────────────────────────────────────────────────┐        │
│  │ def test_cli_with_mock_file_loader():                         │        │
│  │     mock_loader = Mock(spec=FileLoader)                       │        │
│  │     mock_loader.load_file = Mock(return_value="content")      │        │
│  │                                                                │        │
│  │     def custom_factory(**kwargs):                             │        │
│  │         return create_experiment_runner(                      │        │
│  │             **kwargs,                                          │        │
│  │             file_loader=mock_loader                           │        │
│  │         )                                                      │        │
│  │                                                                │        │
│  │     with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY',        │        │
│  │                custom_factory):                                │        │
│  │         result = runner.invoke(cli, ['run', ...])             │        │
│  │                                                                │        │
│  │     # Verify file loader was used                             │        │
│  │     mock_loader.load_file.assert_called()                     │        │
│  └──────────────────────────────────────────────────────────────┘        │
│                                                                            │
│  Test Strategy 4: Inject Session for Parallel Mode                        │
│  ┌──────────────────────────────────────────────────────────────┐        │
│  │ def test_factory_with_session():                              │        │
│  │     mock_session = Mock(spec=aiohttp.ClientSession)           │        │
│  │                                                                │        │
│  │     runner = create_experiment_runner(                        │        │
│  │         timeout=60,                                            │        │
│  │         session=mock_session  ◄── Shared session             │        │
│  │     )                                                          │        │
│  │                                                                │        │
│  │     # Verify session injected into providers                  │        │
│  │     assert runner.providers["openrouter"]._session is ...     │        │
│  │     assert runner.providers["ollama"]._session is ...         │        │
│  └──────────────────────────────────────────────────────────────┘        │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

## Dependency Flow

```
Production Mode:
─────────────

CLI run() command
    │
    ├─ Validates arguments (timeout, file paths, stdin)
    │
    ├─ Calls: DEFAULT_RUNNER_FACTORY(timeout=timeout, verbose=verbose)
    │     │
    │     └─ create_experiment_runner()
    │           │
    │           ├─ Creates default providers (mock, openrouter, ollama)
    │           ├─ Creates default FileLoader()
    │           └─ Returns: ExperimentRunner(...)
    │
    └─ Calls: runner.run_experiments(...)
          │
          └─ Executes experiments with real providers


Test Mode (Mock Runner):
────────────────────────

CLI test
    │
    ├─ Patches: DEFAULT_RUNNER_FACTORY → mock_factory
    │
    ├─ Invokes: CLI run command
    │     │
    │     └─ Calls: mock_factory(timeout=timeout, verbose=verbose)
    │           │
    │           └─ Returns: mock_runner (Mock object)
    │
    └─ Verifies: mock_runner.run_experiments.assert_called_once()


Test Mode (Custom Providers):
──────────────────────────────

CLI test
    │
    ├─ Creates: spy_provider = SpyMockProvider()
    │
    ├─ Patches: DEFAULT_RUNNER_FACTORY → custom_factory
    │     │
    │     └─ custom_factory returns:
    │           create_experiment_runner(providers={"spy": spy_provider})
    │
    ├─ Invokes: CLI run command
    │     │
    │     └─ Runner uses spy_provider
    │
    └─ Verifies: spy_provider.calls == [...]


Test Mode (Session Injection):
───────────────────────────────

Test
    │
    ├─ Creates: mock_session = Mock(spec=aiohttp.ClientSession)
    │
    ├─ Calls: runner = create_experiment_runner(session=mock_session)
    │     │
    │     └─ Factory creates providers with session:
    │           providers = {
    │               "openrouter": OpenRouterProvider(session=mock_session),
    │               "ollama": OllamaProvider(session=mock_session),
    │           }
    │
    └─ Verifies: runner.providers["openrouter"]._session is mock_session
```

## Validation Flow (Shows Separation of Concerns)

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Layer Validates                      │
│               (Before Factory is Called)                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ✓ Timeout > 0                                             │
│  ✓ File paths exist                                        │
│  ✓ stdin usage valid (not both prompt and experiments)    │
│  ✓ Required arguments provided                             │
│                                                             │
│  If validation fails:                                       │
│    → Print error message                                    │
│    → sys.exit(1)                                           │
│    → Factory never called                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                         │ All valid
                         ▼
┌─────────────────────────────────────────────────────────────┐
│             Factory Creates Runner                          │
│         (Only called after validation)                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  runner = DEFAULT_RUNNER_FACTORY(                          │
│      timeout=timeout,     ◄── Already validated           │
│      verbose=verbose      ◄── Safe to use                 │
│  )                                                          │
│                                                             │
│  Factory doesn't validate - trusts CLI layer               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│             Runner Executes                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  await runner.run_experiments(...)                         │
│    ├─ Load files via file_loader                           │
│    ├─ Parse CSV                                             │
│    ├─ Execute experiments via providers                     │
│    └─ Write results                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘

This separation means:
  - CLI tests can verify validation without creating runner
  - Factory tests don't need to handle invalid inputs
  - Runner tests assume valid configuration
```

## Benefits Visualization

```
Before Factory:
──────────────

CLI Tests                    Runner Tests
┌────────────┐              ┌────────────┐
│ Need API   │──────────────│ Need API   │
│ keys for   │              │ keys for   │
│ argument   │              │ provider   │
│ parsing!   │              │ tests      │
└────────────┘              └────────────┘
      │                            │
      └────────────┬───────────────┘
                   ▼
          High coupling,
          hard to test,
          slow test suite


After Factory:
─────────────

CLI Tests                    Factory Tests              Runner Tests
┌────────────┐              ┌────────────┐             ┌────────────┐
│ Mock       │              │ Test DI    │             │ Test with  │
│ factory    │              │ without    │             │ mock       │
│ = fast,    │              │ API keys   │             │ providers  │
│ no API key │              │            │             │            │
└────────────┘              └────────────┘             └────────────┘
      │                            │                          │
      └────────────┬───────────────┴──────────────────────────┘
                   ▼
          Clean separation,
          easy to test,
          fast test suite,
          no API keys needed
```

## File Organization

```
src/parallamr/
│
├── cli.py                          (MODIFIED)
│   ├── create_experiment_runner()  ◄── NEW: Factory function
│   ├── DEFAULT_RUNNER_FACTORY      ◄── NEW: Module variable
│   ├── run()                        ◄── MODIFIED: Uses factory
│   └── models()                     ◄── MODIFIED: Uses factory
│
├── runner.py                        (UNCHANGED)
│   └── ExperimentRunner             ✓ Already supports DI
│
├── providers/
│   ├── openrouter.py                (UNCHANGED)
│   │   └── OpenRouterProvider       ✓ Already supports session injection
│   └── ollama.py                    (UNCHANGED)
│       └── OllamaProvider           ✓ Already supports session injection
│
└── file_loader.py                   (UNCHANGED)
    └── FileLoader                   ✓ Already injectable

tests/
│
├── test_cli.py                      (MODIFIED)
│   └── + 3 new tests for factory integration
│
├── test_cli_factory.py              (NEW)
│   ├── TestCLIFactoryInjection
│   ├── TestCLIArgumentParsingWithoutAPIKeys
│   ├── TestCLIErrorHandlingWithFactory
│   ├── TestFactoryWithProviderInjection
│   └── TestFactoryWithFileLoaderInjection
│
└── test_factory.py                  (NEW)
    ├── TestFactoryFunction
    └── TestFactoryParameterValidation
```

## Call Graph (Production vs Test)

```
Production:
──────────

main()
  └─ cli()
      └─ run()
          ├─ Validate arguments
          ├─ DEFAULT_RUNNER_FACTORY(timeout, verbose)
          │   └─ create_experiment_runner()
          │       └─ ExperimentRunner(timeout, verbose, providers=None, file_loader=None)
          │           ├─ _create_default_providers()
          │           │   ├─ MockProvider(timeout)
          │           │   ├─ OpenRouterProvider(timeout)  ◄── Reads OPENROUTER_API_KEY
          │           │   └─ OllamaProvider(timeout)
          │           └─ FileLoader()
          └─ runner.run_experiments(...)
              ├─ file_loader.load_file()
              ├─ csv.DictReader()
              └─ For each experiment:
                  └─ providers[name].get_completion()


Test (Mock Runner):
───────────────────

test_cli_with_mock_runner()
  └─ with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', mock_factory):
      └─ runner.invoke(cli, ['run', ...])
          └─ run()
              ├─ Validate arguments
              ├─ mock_factory(timeout, verbose)  ◄── Returns mock_runner
              │   └─ return mock_runner
              └─ mock_runner.run_experiments(...)  ◄── AsyncMock, doesn't execute


Test (Custom Provider):
───────────────────────

test_cli_with_custom_provider()
  └─ with patch('parallamr.cli.DEFAULT_RUNNER_FACTORY', custom_factory):
      └─ runner.invoke(cli, ['run', ...])
          └─ run()
              ├─ Validate arguments
              ├─ custom_factory(timeout, verbose)
              │   └─ create_experiment_runner(providers={"spy": spy_provider})
              │       └─ ExperimentRunner(providers={"spy": spy_provider})
              └─ runner.run_experiments(...)
                  └─ providers["spy"].get_completion()  ◄── Spy tracks calls
```

## Summary

### Key Changes
1. Add `create_experiment_runner()` factory function
2. Add `DEFAULT_RUNNER_FACTORY` module variable
3. Change 2 lines: 129 and 244 to use factory
4. Write ~30 new tests

### Key Benefits
1. CLI tests don't need API keys
2. Can mock runner for CLI flow testing
3. Can inject providers for integration testing
4. Can inject session for parallel testing
5. Better separation of concerns
6. Easier to test error handling

### Key Principles
1. Factory is simple function, not class
2. CLI validates before factory call
3. Factory trusts CLI validation
4. Factory doesn't modify inputs
5. Tests use `patch()` for clean isolation
