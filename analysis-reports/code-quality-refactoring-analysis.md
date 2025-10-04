# Code Quality & Testability Refactoring Analysis
**Project:** Parallamr
**Date:** 2025-10-04
**Analyzed by:** CODER Agent (tdrefactor swarm)

## Executive Summary

This analysis examined the Parallamr codebase for testability issues, tight coupling, code duplication, and refactoring opportunities. The codebase is generally well-structured with good separation of concerns, but there are several concrete refactoring opportunities that would significantly improve testability and maintainability.

**Key Findings:**
- 8 high-priority refactoring opportunities identified
- 12 medium-priority improvements recommended
- 6 low-priority enhancements suggested
- Estimated total refactoring effort: 3-5 days

---

## HIGH PRIORITY REFACTORING NEEDS

### 1. Hard-coded Provider Instantiation in ExperimentRunner

**File:** `/workspaces/parallamr/src/parallamr/runner.py` (Lines 24-40)

**Issue:** Direct provider instantiation prevents dependency injection and mocking in tests.

**Current Code:**
```python
def __init__(self, timeout: int = 300, verbose: bool = False):
    """Initialize the experiment runner."""
    self.timeout = timeout
    self.verbose = verbose

    # Initialize providers
    self.providers: Dict[str, Provider] = {
        "mock": MockProvider(timeout=timeout),
        "openrouter": OpenRouterProvider(timeout=timeout),
        "ollama": OllamaProvider(timeout=timeout),
    }

    self._setup_logging()
```

**Problems:**
- Cannot inject mock providers for testing
- Hard to test different provider configurations
- Requires API keys even when testing unrelated functionality
- Violates dependency injection principle

**Proposed Refactoring:**
```python
def __init__(
    self,
    timeout: int = 300,
    verbose: bool = False,
    providers: Optional[Dict[str, Provider]] = None
):
    """Initialize the experiment runner.

    Args:
        timeout: Request timeout in seconds
        verbose: Enable verbose logging
        providers: Optional provider dictionary (defaults to standard providers)
    """
    self.timeout = timeout
    self.verbose = verbose

    # Use injected providers or create defaults
    if providers is not None:
        self.providers = providers
    else:
        self.providers = self._create_default_providers(timeout)

    self._setup_logging()

def _create_default_providers(self, timeout: int) -> Dict[str, Provider]:
    """Create default provider instances.

    Separated for easier testing and configuration.
    """
    return {
        "mock": MockProvider(timeout=timeout),
        "openrouter": OpenRouterProvider(timeout=timeout),
        "ollama": OllamaProvider(timeout=timeout),
    }
```

**Benefits:**
- Tests can inject mock providers without network calls
- Easier to test runner logic in isolation
- Supports custom provider configurations
- No breaking changes for existing code

**Estimated Effort:** 2 hours (including test updates)

---

### 2. File I/O Operations Scattered Throughout ExperimentRunner

**File:** `/workspaces/parallamr/src/parallamr/runner.py` (Lines 53-97)

**Issue:** Direct file I/O and stdin reading in `run_experiments()` makes the method hard to test.

**Current Code:**
```python
async def run_experiments(
    self,
    prompt_file: Optional[str | Path],
    experiments_file: Optional[str | Path],
    output_file: Optional[str | Path],
    context_files: Optional[List[str | Path]] = None,
    read_prompt_stdin: bool = False,
    read_experiments_stdin: bool = False,
) -> None:
    # Load and validate inputs
    if read_prompt_stdin:
        logger.info("Reading prompt from stdin")
        primary_content = sys.stdin.read()
    else:
        logger.info(f"Loading prompt from {prompt_file}")
        primary_content = load_file_content(prompt_file)

    if read_experiments_stdin:
        logger.info("Reading experiments from stdin")
        experiments_content = sys.stdin.read()
        experiments = load_experiments_from_csv(csv_content=experiments_content)
    else:
        logger.info(f"Loading experiments from {experiments_file}")
        experiments = load_experiments_from_csv(csv_path=experiments_file)
```

**Problems:**
- Direct `sys.stdin.read()` calls impossible to mock properly
- File loading logic mixed with experiment execution logic
- High cyclomatic complexity (multiple conditional branches)
- Difficult to test stdin/file scenarios independently

**Proposed Refactoring:**

Create a `FileLoader` abstraction:
```python
# New file: src/parallamr/file_loader.py
class FileLoader:
    """Abstraction for file/stdin loading operations."""

    def load_prompt(
        self,
        file_path: Optional[Path],
        use_stdin: bool
    ) -> str:
        """Load prompt from file or stdin."""
        if use_stdin:
            return self._read_stdin()
        return load_file_content(file_path)

    def load_experiments(
        self,
        file_path: Optional[Path],
        use_stdin: bool
    ) -> List[Experiment]:
        """Load experiments from file or stdin."""
        if use_stdin:
            content = self._read_stdin()
            return load_experiments_from_csv(csv_content=content)
        return load_experiments_from_csv(csv_path=file_path)

    def _read_stdin(self) -> str:
        """Read from stdin (mockable)."""
        return sys.stdin.read()


# Updated ExperimentRunner:
def __init__(
    self,
    timeout: int = 300,
    verbose: bool = False,
    providers: Optional[Dict[str, Provider]] = None,
    file_loader: Optional[FileLoader] = None  # NEW
):
    self.timeout = timeout
    self.verbose = verbose
    self.providers = providers or self._create_default_providers(timeout)
    self.file_loader = file_loader or FileLoader()  # NEW
    self._setup_logging()

async def run_experiments(
    self,
    prompt_file: Optional[str | Path],
    experiments_file: Optional[str | Path],
    output_file: Optional[str | Path],
    context_files: Optional[List[str | Path]] = None,
    read_prompt_stdin: bool = False,
    read_experiments_stdin: bool = False,
) -> None:
    # Load inputs using injected loader
    primary_content = self.file_loader.load_prompt(
        Path(prompt_file) if prompt_file else None,
        read_prompt_stdin
    )

    experiments = self.file_loader.load_experiments(
        Path(experiments_file) if experiments_file else None,
        read_experiments_stdin
    )
    # ... rest of method
```

**Benefits:**
- FileLoader can be mocked in tests
- Clean separation of I/O from business logic
- Easier to test different input sources
- Reduces cyclomatic complexity of run_experiments()

**Estimated Effort:** 4 hours (new class + test updates)

---

### 3. Logging Configuration Side Effects in Constructor

**File:** `/workspaces/parallamr/src/parallamr/runner.py` (Lines 44-51)

**Issue:** `_setup_logging()` modifies global logging state, causing test pollution.

**Current Code:**
```python
def _setup_logging(self) -> None:
    """Setup logging configuration."""
    level = logging.INFO if self.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='[%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )
```

**Problems:**
- `logging.basicConfig()` affects global logging state
- Tests that create ExperimentRunner instances interfere with each other
- Cannot easily capture or suppress logs in tests
- Changes persist across test runs

**Proposed Refactoring:**
```python
def _setup_logging(self) -> None:
    """Setup logging configuration for this runner instance."""
    level = logging.INFO if self.verbose else logging.WARNING

    # Use instance-specific logger instead of root logger
    self.logger = logging.getLogger(f"{__name__}.{id(self)}")
    self.logger.setLevel(level)

    # Only add handler if not already present
    if not self.logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter('[%(levelname)s] %(message)s')
        )
        self.logger.addHandler(handler)

    # Prevent propagation to root logger
    self.logger.propagate = False

# Update all logger.info/warning/error calls to use self.logger
```

**Benefits:**
- No global state pollution
- Tests can configure logging independently
- Easy to capture logs per-instance
- Follows logging best practices

**Estimated Effort:** 2 hours (update all logging calls)

---

### 4. OpenRouterProvider with Hard-coded Environment Variable Access

**File:** `/workspaces/parallamr/src/parallamr/providers/openrouter.py` (Lines 25-36)

**Issue:** Direct `os.getenv()` calls make testing API interactions difficult.

**Current Code:**
```python
def __init__(self, api_key: Optional[str] = None, timeout: int = 300):
    """Initialize the OpenRouter provider."""
    super().__init__(timeout)
    self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not self.api_key:
        raise AuthenticationError("OpenRouter API key not provided")

    self.base_url = "https://openrouter.ai/api/v1"
    self._model_cache: Optional[Dict[str, Any]] = None
```

**Problems:**
- Cannot easily test with different API key scenarios
- Test environment variables can leak between tests
- Hard to test error handling for missing keys
- Base URL is hard-coded (can't test with mock server)

**Proposed Refactoring:**
```python
def __init__(
    self,
    api_key: Optional[str] = None,
    timeout: int = 300,
    base_url: Optional[str] = None,
    env_getter: Optional[Callable[[str, str], Optional[str]]] = None
):
    """Initialize the OpenRouter provider.

    Args:
        api_key: OpenRouter API key (if None, reads from env)
        timeout: Request timeout in seconds
        base_url: API base URL (for testing with mock servers)
        env_getter: Function to get env vars (defaults to os.getenv)
    """
    super().__init__(timeout)

    # Use injected env_getter for testability
    _env_getter = env_getter or os.getenv
    self.api_key = api_key or _env_getter("OPENROUTER_API_KEY")

    if not self.api_key:
        raise AuthenticationError("OpenRouter API key not provided")

    self.base_url = base_url or "https://openrouter.ai/api/v1"
    self._model_cache: Optional[Dict[str, Any]] = None
```

**Benefits:**
- Tests can inject mock env_getter
- Can test against local mock servers
- No environment pollution in tests
- Easier to test authentication errors

**Estimated Effort:** 2 hours (including test updates)

**Same pattern applies to OllamaProvider** (Lines 23-33 in `/workspaces/parallamr/src/parallamr/providers/ollama.py`)

---

### 5. HTTP Client Session Management in Provider Classes

**File:** `/workspaces/parallamr/src/parallamr/providers/openrouter.py` (Lines 82-147)
**File:** `/workspaces/parallamr/src/parallamr/providers/ollama.py` (Lines 70-134)

**Issue:** Each method creates its own `aiohttp.ClientSession`, leading to resource waste and testing difficulties.

**Current Code (OpenRouter example):**
```python
async def get_completion(self, prompt: str, model: str, **kwargs) -> ProviderResponse:
    # ... validation ...

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                # ... handle response ...
```

**Problems:**
- Creates new connection pool for each request
- Cannot inject mock HTTP client for testing
- Resource inefficient (should reuse sessions)
- Hard to test network errors and retries
- No connection pooling benefits

**Proposed Refactoring:**
```python
# Option 1: Injected session (better for testing)
def __init__(
    self,
    api_key: Optional[str] = None,
    timeout: int = 300,
    session: Optional[aiohttp.ClientSession] = None
):
    super().__init__(timeout)
    self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not self.api_key:
        raise AuthenticationError("OpenRouter API key not provided")

    self.base_url = "https://openrouter.ai/api/v1"
    self._model_cache: Optional[Dict[str, Any]] = None
    self._session = session  # Injected session
    self._owns_session = session is None

async def _get_session(self) -> aiohttp.ClientSession:
    """Get or create HTTP session."""
    if self._session is None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
    return self._session

async def close(self) -> None:
    """Close HTTP session if owned by this instance."""
    if self._owns_session and self._session is not None:
        await self._session.close()
        self._session = None

async def get_completion(self, prompt: str, model: str, **kwargs) -> ProviderResponse:
    session = await self._get_session()

    try:
        async with session.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload
        ) as response:
            # ... handle response ...
```

**Benefits:**
- Can inject mock HTTP session for testing
- Reuses connection pools (performance)
- Easier to test network failures
- Better resource management
- Supports async context manager pattern

**Estimated Effort:** 4 hours per provider (8 hours total for OpenRouter + Ollama)

---

### 6. Long Method with High Cyclomatic Complexity: `_run_single_experiment()`

**File:** `/workspaces/parallamr/src/parallamr/runner.py` (Lines 137-222)

**Issue:** 85-line method with multiple responsibilities and complex error handling.

**Current Structure:**
- Provider lookup (4 lines)
- Template variable replacement (6 lines)
- Token estimation (2 lines)
- Context window validation (10 lines)
- Warning aggregation (9 lines)
- Provider completion call (5 lines)
- Result creation (9 lines)
- Exception handling (6 lines)

**Problems:**
- Hard to test individual steps in isolation
- High cyclomatic complexity (8+ branches)
- Multiple responsibilities in one method
- Error handling mixed with business logic

**Proposed Refactoring:**

Break into smaller, testable methods:

```python
async def _run_single_experiment(
    self,
    experiment: Experiment,
    primary_content: str,
    context_files: List[Tuple[str, str]],
) -> ExperimentResult:
    """Run a single experiment (orchestrator)."""
    try:
        # Step 1: Validate provider
        provider = self._get_provider(experiment)
        if provider is None:
            return self._create_error_result(
                experiment,
                f"Unknown provider: {experiment.provider}"
            )

        # Step 2: Prepare prompt with variables
        combined_content, warnings = self._prepare_prompt(
            primary_content, context_files, experiment.variables
        )

        # Step 3: Validate token limits
        validation = await self._validate_token_limits(
            combined_content, provider, experiment
        )
        if not validation.valid:
            return self._create_error_result(
                experiment, validation.error_message, validation.input_tokens
            )
        warnings.extend(validation.warnings)

        # Step 4: Get completion from provider
        provider_response = await provider.get_completion(
            prompt=combined_content,
            model=experiment.model,
            variables=experiment.variables
        )

        # Step 5: Create result
        return ExperimentResult.from_experiment_and_response(
            experiment=experiment,
            response=provider_response,
            input_tokens=validation.input_tokens,
            template_warnings=warnings if warnings else None
        )

    except Exception as e:
        logger.exception(f"Unexpected error in experiment {experiment.row_number}")
        return self._create_error_result(experiment, f"Unexpected error: {str(e)}")


def _get_provider(self, experiment: Experiment) -> Optional[Provider]:
    """Get provider for experiment (testable)."""
    return self.providers.get(experiment.provider)


def _prepare_prompt(
    self,
    primary_content: str,
    context_files: List[Tuple[str, str]],
    variables: Dict[str, Any]
) -> Tuple[str, List[str]]:
    """Prepare prompt with variable substitution (testable)."""
    combined_content, missing_variables = combine_files_with_variables(
        primary_content, context_files, variables
    )

    warnings = []
    for var in missing_variables:
        warnings.append(f"Variable '{{{{{var}}}}}' in template has no value")

    return combined_content, warnings


@dataclass
class TokenValidationResult:
    """Result of token validation."""
    valid: bool
    input_tokens: int
    warnings: List[str]
    error_message: Optional[str] = None


async def _validate_token_limits(
    self,
    combined_content: str,
    provider: Provider,
    experiment: Experiment
) -> TokenValidationResult:
    """Validate token limits for experiment (testable)."""
    input_tokens = estimate_tokens(combined_content)
    context_window = await provider.get_context_window(experiment.model)

    context_valid, context_warning = validate_context_window(
        input_tokens,
        context_window,
        model=experiment.model,
        provider=experiment.provider
    )

    warnings = []
    if context_warning:
        warnings.append(context_warning)

    error_message = None
    if not context_valid:
        error_message = f"Input tokens ({input_tokens}) exceed model context window ({context_window})"

    return TokenValidationResult(
        valid=context_valid,
        input_tokens=input_tokens,
        warnings=warnings,
        error_message=error_message
    )
```

**Benefits:**
- Each method has single responsibility
- Easy to unit test each step
- Reduced cyclomatic complexity (2-3 per method)
- Clear separation of concerns
- Better error isolation

**Estimated Effort:** 6 hours (including comprehensive test coverage)

---

### 7. CLI Functions That Directly Instantiate ExperimentRunner

**File:** `/workspaces/parallamr/src/parallamr/cli.py` (Lines 129, 244)

**Issue:** Direct instantiation prevents testing CLI logic without running experiments.

**Current Code:**
```python
@cli.command()
def run(...) -> None:
    # ... validation ...

    # Create runner
    runner = ExperimentRunner(timeout=timeout, verbose=verbose)

    if validate_only:
        asyncio.run(_validate_experiments(runner, experiments_path, experiments == "-"))
    else:
        asyncio.run(_run_experiments(runner, ...))


async def _list_models(provider: str) -> None:
    """List models for a provider asynchronously."""
    runner = ExperimentRunner()  # Direct instantiation

    if provider not in runner.providers:
        click.echo(f"Provider '{provider}' not available", err=True)
        sys.exit(1)
    # ...
```

**Problems:**
- Cannot test CLI logic without creating real providers
- CLI tests require API keys even when testing argument parsing
- Hard to test error handling paths
- Tight coupling between CLI and runner

**Proposed Refactoring:**

Create runner factory function:

```python
# New file: src/parallamr/factory.py
def create_experiment_runner(
    timeout: int = 300,
    verbose: bool = False,
    providers: Optional[Dict[str, Provider]] = None
) -> ExperimentRunner:
    """Factory function for creating ExperimentRunner instances.

    This allows tests to inject the factory and return mock runners.
    """
    return ExperimentRunner(
        timeout=timeout,
        verbose=verbose,
        providers=providers
    )


# Update CLI:
# Add runner_factory parameter to commands (for testing)
def run(
    prompt: str,
    experiments: str,
    output: Optional[Path],
    context: tuple[Path, ...],
    verbose: bool,
    timeout: int,
    validate_only: bool,
    runner_factory: Callable = create_experiment_runner  # Injectable
) -> None:
    # ... validation ...

    # Create runner using factory
    runner = runner_factory(timeout=timeout, verbose=verbose)

    if validate_only:
        asyncio.run(_validate_experiments(runner, experiments_path, experiments == "-"))
    else:
        asyncio.run(_run_experiments(runner, ...))
```

**Benefits:**
- Tests can inject factory that returns mocks
- CLI logic testable without real providers
- No API keys needed for CLI tests
- Maintains backward compatibility

**Estimated Effort:** 3 hours

---

### 8. IncrementalCSVWriter Creates File Handles on Every Write

**File:** `/workspaces/parallamr/src/parallamr/csv_writer.py` (Lines 107-127)

**Issue:** Opens and closes file for every row write, inefficient and hard to test.

**Current Code:**
```python
def _write_row(self, row_data: Dict[str, Any]) -> None:
    """Write a single row to the CSV file or stdout."""
    if self._fieldnames is None:
        raise ValueError("Cannot write row without fieldnames")

    complete_row = {field: row_data.get(field, "") for field in self._fieldnames}

    if self._is_stdout:
        writer = csv.DictWriter(sys.stdout, fieldnames=self._fieldnames, lineterminator='\n')
        writer.writerow(complete_row)
        sys.stdout.flush()
    else:
        with open(self.output_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=self._fieldnames)
            writer.writerow(complete_row)
```

**Problems:**
- Opens file for every write (I/O inefficient)
- Direct `sys.stdout` access (hard to test)
- No buffering control
- Cannot inject output stream for testing

**Proposed Refactoring:**

```python
from typing import TextIO

class IncrementalCSVWriter:
    """Handles incremental writing to CSV with proper escaping."""

    def __init__(
        self,
        output_path: Optional[str | Path],
        output_stream: Optional[TextIO] = None  # NEW: Injectable stream
    ):
        """Initialize the CSV writer.

        Args:
            output_path: Path to the output CSV file, or None for stdout
            output_stream: Output stream (for testing), defaults to sys.stdout
        """
        self.output_path = Path(output_path) if output_path else None
        self._headers_written = False
        self._fieldnames: Optional[List[str]] = None
        self._is_stdout = output_path is None
        self._output_stream = output_stream or sys.stdout  # NEW
        self._file_handle: Optional[TextIO] = None  # NEW

    def _get_writer_and_stream(self) -> Tuple[csv.DictWriter, TextIO]:
        """Get CSV writer and output stream (cached for file mode)."""
        if self._is_stdout:
            stream = self._output_stream
        else:
            # Open file once and reuse (append mode)
            if self._file_handle is None or self._file_handle.closed:
                mode = 'a' if self._headers_written else 'w'
                self._file_handle = open(
                    self.output_path, mode, newline='', encoding='utf-8'
                )
            stream = self._file_handle

        writer = csv.DictWriter(
            stream,
            fieldnames=self._fieldnames,
            lineterminator='\n'
        )
        return writer, stream

    def _write_row(self, row_data: Dict[str, Any]) -> None:
        """Write a single row to the CSV file or stdout."""
        if self._fieldnames is None:
            raise ValueError("Cannot write row without fieldnames")

        complete_row = {field: row_data.get(field, "") for field in self._fieldnames}

        writer, stream = self._get_writer_and_stream()
        writer.writerow(complete_row)
        stream.flush()

    def close(self) -> None:
        """Close file handle if open."""
        if self._file_handle is not None and not self._file_handle.closed:
            self._file_handle.close()
            self._file_handle = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
```

**Benefits:**
- File opened once and reused (performance)
- Output stream injectable for testing
- Support for context manager pattern
- No direct stdout dependency
- Buffering control

**Estimated Effort:** 3 hours (including test updates)

---

## MEDIUM PRIORITY IMPROVEMENTS

### 9. Duplicate Error Handling Pattern Across Providers

**Files:**
- `/workspaces/parallamr/src/parallamr/providers/openrouter.py` (Lines 127-147)
- `/workspaces/parallamr/src/parallamr/providers/ollama.py` (Lines 107-134)

**Issue:** Same try-except-error-response pattern duplicated in both providers.

**Current Pattern (repeated):**
```python
except asyncio.TimeoutError:
    return ProviderResponse(
        output="",
        output_tokens=0,
        success=False,
        error_message=f"Request timeout after {self.timeout} seconds"
    )
except aiohttp.ClientError as e:
    return ProviderResponse(
        output="",
        output_tokens=0,
        success=False,
        error_message=f"Network error: {str(e)}"
    )
except Exception as e:
    return ProviderResponse(
        output="",
        output_tokens=0,
        success=False,
        error_message=f"Unexpected error: {str(e)}"
    )
```

**Proposed Solution:**

Create shared error handler in base class:

```python
# In base.py
class Provider(ABC):
    # ... existing methods ...

    def _create_error_response(
        self,
        error: Exception,
        context_window: Optional[int] = None
    ) -> ProviderResponse:
        """Create error response from exception (DRY)."""
        if isinstance(error, asyncio.TimeoutError):
            message = f"Request timeout after {self.timeout} seconds"
        elif isinstance(error, aiohttp.ClientConnectorError):
            message = f"Cannot connect to provider service: {str(error)}"
        elif isinstance(error, aiohttp.ClientError):
            message = f"Network error: {str(error)}"
        else:
            message = f"Unexpected error: {str(error)}"

        return ProviderResponse(
            output="",
            output_tokens=0,
            success=False,
            error_message=message,
            context_window=context_window
        )

# In provider implementations:
try:
    # ... actual request logic ...
except Exception as e:
    return self._create_error_response(e, context_window)
```

**Benefits:**
- DRY principle (Don't Repeat Yourself)
- Consistent error messages
- Easier to add new error types
- Single place to update error handling

**Estimated Effort:** 2 hours

---

### 10. Template Variable Missing Warning Generation

**File:** `/workspaces/parallamr/src/parallamr/runner.py` (Lines 186-188)

**Issue:** Warning message construction with complex string formatting.

**Current Code:**
```python
if missing_variables:
    for var in missing_variables:
        warnings.append(f"Variable '{{{{{var}}}}}' in template has no value in experiment row {experiment.row_number}")
```

**Proposed Refactoring:**
```python
# Move to models.py or template.py
def format_missing_variable_warning(
    variable_name: str,
    row_number: int
) -> str:
    """Format warning for missing template variable (testable)."""
    return f"Variable '{{{{{variable_name}}}}}' in template has no value in experiment row {row_number}"

# In runner.py:
if missing_variables:
    warnings.extend([
        format_missing_variable_warning(var, experiment.row_number)
        for var in missing_variables
    ])
```

**Benefits:**
- Testable string formatting
- Consistent warning format
- Easier to internationalize later
- Single source of truth

**Estimated Effort:** 30 minutes

---

### 11. Provider Model Cache Pattern Duplication

**Files:**
- `/workspaces/parallamr/src/parallamr/providers/openrouter.py` (Lines 191-226)
- `/workspaces/parallamr/src/parallamr/providers/ollama.py` (Lines 176-204)

**Issue:** Similar caching logic for model lists in both providers.

**Proposed Solution:**

Add caching mixin or helper in base class:

```python
# In base.py
class CachedModelProvider(Provider):
    """Provider base class with model list caching."""

    def __init__(self, timeout: int = 300):
        super().__init__(timeout)
        self._model_cache: Optional[List[str]] = None

    async def list_models(self) -> List[str]:
        """List models with caching."""
        if self._model_cache is not None:
            return self._model_cache

        self._model_cache = await self._fetch_models()
        return self._model_cache

    @abstractmethod
    async def _fetch_models(self) -> List[str]:
        """Fetch models from provider (implement in subclass)."""
        pass

    def clear_cache(self) -> None:
        """Clear model cache (useful for testing)."""
        self._model_cache = None
```

**Benefits:**
- Shared caching logic
- Easier to test caching behavior
- Consistent cache invalidation
- Cache clearing for tests

**Estimated Effort:** 2 hours

---

### 12. Environment Variable Loading in CLI

**File:** `/workspaces/parallamr/src/parallamr/cli.py` (Lines 16-18, 28)

**Issue:** `load_environment()` called as side effect in CLI decorator.

**Current Code:**
```python
def load_environment() -> None:
    """Load environment variables from .env file crawling up to the top."""
    load_dotenv(find_dotenv())


@click.group()
@click.version_option(version=__version__, prog_name="parallamr")
def cli() -> None:
    """CLI entry point."""
    load_environment()  # Side effect
```

**Problems:**
- Side effect in decorator makes testing harder
- Cannot test CLI without loading .env
- Global state modification

**Proposed Refactoring:**
```python
@click.group()
@click.version_option(version=__version__, prog_name="parallamr")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """CLI entry point."""
    # Only load env if not already loaded (testing control)
    if ctx.obj is None:
        ctx.obj = {}

    if not ctx.obj.get('env_loaded', False):
        load_environment()
        ctx.obj['env_loaded'] = True
```

**Benefits:**
- Tests can set env_loaded=True to skip
- Better control over environment
- Follows click best practices

**Estimated Effort:** 1 hour

---

### 13. CSV Field Name Determination Logic

**File:** `/workspaces/parallamr/src/parallamr/csv_writer.py` (Lines 61-91)

**Issue:** Complex field ordering logic that would benefit from being testable independently.

**Proposed Refactoring:**

Extract to pure function:
```python
def determine_csv_field_order(row_data: Dict[str, Any]) -> List[str]:
    """Determine CSV field order (pure function, testable).

    Orders fields logically: core -> variables -> results
    """
    CORE_FIELDS = ["provider", "model"]
    RESULT_FIELDS = [
        "status", "input_tokens", "context_window",
        "output_tokens", "output", "error_message"
    ]

    variable_fields = [
        key for key in row_data.keys()
        if key not in CORE_FIELDS + RESULT_FIELDS
    ]

    return CORE_FIELDS + variable_fields + RESULT_FIELDS


# In IncrementalCSVWriter:
def _determine_fieldnames(self, row_data: Dict[str, Any]) -> List[str]:
    """Determine fieldnames (delegates to testable function)."""
    return determine_csv_field_order(row_data)
```

**Benefits:**
- Pure function (easy to test)
- No dependencies on instance state
- Clear input/output contract
- Can test edge cases easily

**Estimated Effort:** 1 hour

---

### 14. Token Estimation Functions Need Better Testing Hooks

**File:** `/workspaces/parallamr/src/parallamr/token_counter.py`

**Issue:** Simple functions but used throughout; would benefit from strategy pattern for testing.

**Current Code:**
```python
def estimate_tokens(text: str) -> int:
    """Estimate tokens using character count / 4 approximation."""
    if not text:
        return 0
    return len(text) // 4
```

**Proposed Refactoring:**
```python
class TokenEstimator:
    """Strategy for token estimation (mockable)."""

    def estimate(self, text: str) -> int:
        """Estimate tokens in text."""
        if not text:
            return 0
        return len(text) // 4


class ExactTokenEstimator(TokenEstimator):
    """Exact token counting using tiktoken (for testing)."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        import tiktoken
        self.encoding = tiktoken.get_encoding(encoding_name)

    def estimate(self, text: str) -> int:
        """Get exact token count."""
        if not text:
            return 0
        return len(self.encoding.encode(text))


# Default instance
_default_estimator = TokenEstimator()

def estimate_tokens(text: str, estimator: TokenEstimator = None) -> int:
    """Estimate tokens (with injectable estimator)."""
    estimator = estimator or _default_estimator
    return estimator.estimate(text)
```

**Benefits:**
- Can inject exact counter for tests
- Strategy pattern for different models
- Backward compatible
- Better test accuracy

**Estimated Effort:** 2 hours

---

### 15. Context Window Validation Return Type

**File:** `/workspaces/parallamr/src/parallamr/token_counter.py` (Lines 59-91)

**Issue:** Returns tuple instead of structured object.

**Current Code:**
```python
def validate_context_window(...) -> tuple[bool, Optional[str]]:
    """Returns: Tuple of (is_valid, warning_message)"""
    # ...
```

**Proposed Refactoring:**
```python
@dataclass
class ContextValidation:
    """Result of context window validation."""
    is_valid: bool
    warning: Optional[str] = None
    percentage_used: Optional[float] = None

    def __bool__(self) -> bool:
        """Allow boolean checks."""
        return self.is_valid


def validate_context_window(...) -> ContextValidation:
    """Validate if input fits within model's context window."""
    if context_window is None:
        model_info = f"{provider}/{model}" if provider and model else "model"
        return ContextValidation(
            is_valid=True,
            warning=f"Context window unknown for {model_info}"
        )

    available_tokens = int(context_window * (1 - buffer_percentage))
    percentage = (input_tokens / context_window) * 100

    if input_tokens > available_tokens:
        return ContextValidation(
            is_valid=False,
            warning=f"Input tokens ({input_tokens}) exceed available context window ({available_tokens}/{context_window})",
            percentage_used=percentage
        )
    elif input_tokens > context_window * 0.8:
        return ContextValidation(
            is_valid=True,
            warning=f"Input tokens ({input_tokens}) approaching context window limit ({context_window})",
            percentage_used=percentage
        )

    return ContextValidation(is_valid=True, percentage_used=percentage)
```

**Benefits:**
- Named fields instead of tuple unpacking
- Type safety
- Can add more fields without breaking API
- Self-documenting code

**Estimated Effort:** 2 hours

---

### 16. Experiment Status Determination Logic

**File:** `/workspaces/parallamr/src/parallamr/models.py` (Lines 52-60)

**Issue:** Status logic in property, hard to test edge cases.

**Current Code:**
```python
@property
def status(self) -> ExperimentStatus:
    """Determine status based on response state."""
    if not self.success:
        return ExperimentStatus.ERROR
    elif self.error_message:
        return ExperimentStatus.WARNING
    else:
        return ExperimentStatus.OK
```

**Proposed Refactoring:**
```python
def determine_experiment_status(
    success: bool,
    error_message: Optional[str]
) -> ExperimentStatus:
    """Determine experiment status (pure function, testable)."""
    if not success:
        return ExperimentStatus.ERROR
    elif error_message:
        return ExperimentStatus.WARNING
    else:
        return ExperimentStatus.OK


@property
def status(self) -> ExperimentStatus:
    """Determine status based on response state."""
    return determine_experiment_status(self.success, self.error_message)
```

**Benefits:**
- Pure function easy to test
- Can test all combinations
- Reusable logic
- Clear contract

**Estimated Effort:** 30 minutes

---

### 17. File Content Loading Error Messages

**File:** `/workspaces/parallamr/src/parallamr/utils.py` (Lines 80-104)

**Issue:** Error message construction mixed with I/O logic.

**Proposed Refactoring:**

Separate error handling:
```python
class FileLoadError(Exception):
    """Base class for file loading errors."""
    pass


class FileNotFoundError(FileLoadError):
    """File not found error with helpful message."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        super().__init__(f"File not found: {file_path}")


class FileEncodingError(FileLoadError):
    """File encoding error with helpful message."""

    def __init__(self, file_path: Path, encoding: str, reason: str):
        self.file_path = file_path
        self.encoding = encoding
        self.reason = reason
        super().__init__(
            f"Could not decode file {file_path} as {encoding}: {reason}"
        )


def load_file_content(file_path: str | Path) -> str:
    """Load content from a text file."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    try:
        return file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError as e:
        raise FileEncodingError(file_path, 'utf-8', e.reason) from e
```

**Benefits:**
- Structured exceptions
- Easier to test error cases
- Better error messages
- Can catch specific errors in tests

**Estimated Effort:** 1 hour

---

### 18. CSV Validation Logic Extraction

**File:** `/workspaces/parallamr/src/parallamr/utils.py` (Lines 13-77)

**Issue:** CSV loading with inline validation, hard to test validation separately.

**Proposed Refactoring:**

```python
def validate_csv_structure(
    fieldnames: List[str],
    required_columns: Set[str] = {"provider", "model"}
) -> Optional[str]:
    """Validate CSV structure (pure function, testable).

    Returns:
        Error message if invalid, None if valid
    """
    if not fieldnames:
        return "CSV file appears to be empty or malformed"

    missing_columns = required_columns - set(fieldnames)
    if missing_columns:
        return f"Missing required columns in CSV: {missing_columns}"

    return None


def load_experiments_from_csv(...) -> List[Experiment]:
    """Load experiments from CSV."""
    # ... setup reader ...

    # Validate structure using testable function
    validation_error = validate_csv_structure(reader.fieldnames)
    if validation_error:
        raise ValueError(validation_error)

    # ... rest of loading ...
```

**Benefits:**
- Validation testable independently
- Pure function
- Can test all validation cases
- Reusable validation logic

**Estimated Effort:** 1 hour

---

### 19. Async Method Testing Support

**Files:** Multiple provider files

**Issue:** Many async methods make testing verbose with `@pytest.mark.asyncio`.

**Proposed Improvement:**

Add synchronous wrappers for testing:

```python
# In base.py or test_utils.py
def sync_get_completion(
    provider: Provider,
    prompt: str,
    model: str,
    **kwargs
) -> ProviderResponse:
    """Synchronous wrapper for testing (no asyncio needed)."""
    import asyncio
    return asyncio.run(provider.get_completion(prompt, model, **kwargs))


# Or use pytest-asyncio fixtures more effectively
@pytest.fixture
def event_loop():
    """Create event loop for all async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
```

**Benefits:**
- Simpler test code
- Less boilerplate
- Shared test utilities
- Consistent async handling

**Estimated Effort:** 2 hours

---

### 20. Hard-coded Provider Names in CLI

**File:** `/workspaces/parallamr/src/parallamr/cli.py` (Lines 212-232)

**Issue:** Provider names and configuration hard-coded in `providers` command.

**Current Code:**
```python
@cli.command()
def providers() -> None:
    """List available LLM providers and their configuration."""
    # List available provider types (don't instantiate to avoid API key errors)
    click.echo("Available providers:")
    click.echo("  - mock")
    click.echo("  - openrouter")
    click.echo("  - ollama")
```

**Proposed Refactoring:**

```python
# In runner.py or new registry.py
class ProviderRegistry:
    """Registry of available provider types."""

    @staticmethod
    def get_provider_info() -> Dict[str, Dict[str, Any]]:
        """Get information about available providers."""
        return {
            "mock": {
                "name": "Mock",
                "description": "Mock provider for testing",
                "requires_api_key": False,
                "config_keys": []
            },
            "openrouter": {
                "name": "OpenRouter",
                "description": "Access multiple LLMs via OpenRouter API",
                "requires_api_key": True,
                "config_keys": ["OPENROUTER_API_KEY"]
            },
            "ollama": {
                "name": "Ollama",
                "description": "Local LLM inference with Ollama",
                "requires_api_key": False,
                "config_keys": ["OLLAMA_BASE_URL"]
            }
        }

    @staticmethod
    def check_provider_config(provider_name: str) -> Dict[str, Any]:
        """Check if provider is configured."""
        info = ProviderRegistry.get_provider_info().get(provider_name, {})
        config_status = {}

        for key in info.get("config_keys", []):
            config_status[key] = os.getenv(key) is not None

        return {
            "provider": provider_name,
            "configured": all(config_status.values()) or not info.get("requires_api_key"),
            "config_keys": config_status
        }


# In CLI:
@cli.command()
def providers() -> None:
    """List available LLM providers and their configuration."""
    click.echo("Available providers:")

    for provider_name, info in ProviderRegistry.get_provider_info().items():
        click.echo(f"  - {provider_name}: {info['description']}")

    click.echo("\nConfiguration:")

    for provider_name in ProviderRegistry.get_provider_info():
        status = ProviderRegistry.check_provider_config(provider_name)
        # ... format output ...
```

**Benefits:**
- Single source of truth for providers
- Easier to add new providers
- Testable configuration checking
- Metadata-driven CLI output

**Estimated Effort:** 3 hours

---

## LOW PRIORITY ENHANCEMENTS

### 21. Add Type Hints for kwargs in Provider Methods

**Files:** Provider implementations

**Issue:** `**kwargs` not type-hinted, unclear what parameters are accepted.

**Proposed:** Use TypedDict or Protocols to define expected kwargs.

**Estimated Effort:** 2 hours

---

### 22. Extract Format Functions to Utilities

**File:** `/workspaces/parallamr/src/parallamr/token_counter.py` (Lines 94-109)

**Issue:** Format functions could be more reusable.

**Proposed:** Move to shared formatting module.

**Estimated Effort:** 1 hour

---

### 23. Add Retry Logic to Provider Calls

**Files:** Provider implementations

**Issue:** No retry for transient failures.

**Proposed:** Add configurable retry decorator with exponential backoff.

**Estimated Effort:** 4 hours

---

### 24. Implement Provider Health Checks

**Files:** Provider classes

**Issue:** No way to check if provider is healthy before running experiments.

**Proposed:** Add `async def health_check()` method to Provider base class.

**Estimated Effort:** 3 hours

---

### 25. Add Request/Response Logging for Debugging

**Files:** Provider implementations

**Issue:** Hard to debug API issues without request/response logging.

**Proposed:** Add debug-level logging of all HTTP requests/responses.

**Estimated Effort:** 2 hours

---

### 26. Create Provider Factory Pattern

**Files:** Multiple

**Issue:** Provider instantiation scattered across codebase.

**Proposed:** Centralized factory with configuration.

**Estimated Effort:** 4 hours

---

## SUMMARY TABLE

| Priority | Item | File(s) | Effort | Impact |
|----------|------|---------|--------|--------|
| HIGH | 1. Provider DI in ExperimentRunner | runner.py | 2h | High |
| HIGH | 2. FileLoader Abstraction | runner.py | 4h | High |
| HIGH | 3. Logging Isolation | runner.py | 2h | Medium |
| HIGH | 4. Environment Variable Injection | openrouter.py, ollama.py | 2h | High |
| HIGH | 5. HTTP Session Management | openrouter.py, ollama.py | 8h | Medium |
| HIGH | 6. Method Decomposition | runner.py | 6h | High |
| HIGH | 7. CLI Runner Factory | cli.py | 3h | Medium |
| HIGH | 8. CSV Writer Optimization | csv_writer.py | 3h | Medium |
| MED | 9. Error Handling DRY | providers/*.py | 2h | Medium |
| MED | 10. Warning Format Extraction | runner.py | 0.5h | Low |
| MED | 11. Model Cache Pattern | providers/*.py | 2h | Low |
| MED | 12. Environment Loading | cli.py | 1h | Low |
| MED | 13. CSV Field Order Function | csv_writer.py | 1h | Low |
| MED | 14. Token Estimator Strategy | token_counter.py | 2h | Medium |
| MED | 15. Validation Return Type | token_counter.py | 2h | Medium |
| MED | 16. Status Determination | models.py | 0.5h | Low |
| MED | 17. File Load Errors | utils.py | 1h | Low |
| MED | 18. CSV Validation Extraction | utils.py | 1h | Low |
| MED | 19. Async Test Helpers | test files | 2h | Medium |
| MED | 20. Provider Registry | cli.py | 3h | Medium |
| LOW | 21. Type Hints for kwargs | providers/*.py | 2h | Low |
| LOW | 22. Format Utilities | token_counter.py | 1h | Low |
| LOW | 23. Retry Logic | providers/*.py | 4h | High |
| LOW | 24. Health Checks | providers/*.py | 3h | Medium |
| LOW | 25. Request Logging | providers/*.py | 2h | Low |
| LOW | 26. Provider Factory | multiple | 4h | Medium |

**Total Estimated Effort:**
- High Priority: 30 hours (3-4 days)
- Medium Priority: 20 hours (2-3 days)
- Low Priority: 18 hours (2-3 days)
- **Grand Total: 68 hours (8-9 days)**

---

## PRIORITY RANKING METHODOLOGY

Items were prioritized based on:

1. **Impact on Testability** - How much does this improve test coverage and reliability?
2. **Code Coupling** - Does this reduce tight coupling between components?
3. **Maintainability** - How much easier will the code be to maintain?
4. **Performance** - Does this improve runtime performance?
5. **Effort/Benefit Ratio** - Quick wins ranked higher

**High Priority** = Blocking testability issues or major coupling problems
**Medium Priority** = DRY violations, moderate testability improvements
**Low Priority** = Nice-to-have enhancements, polish

---

## RECOMMENDED REFACTORING ORDER

For maximum impact with minimum disruption, implement in this order:

### Phase 1 (Week 1): Foundation - High Priority Items 1-4
1. Provider DI in ExperimentRunner (#1)
2. Environment Variable Injection (#4)
3. Logging Isolation (#3)
4. FileLoader Abstraction (#2)

**Rationale:** These establish dependency injection patterns that make everything else easier to test.

### Phase 2 (Week 2): Complexity Reduction - High Priority Items 5-8
5. Method Decomposition (#6)
6. HTTP Session Management (#5)
7. CSV Writer Optimization (#8)
8. CLI Runner Factory (#7)

**Rationale:** Reduces complexity and improves performance while building on DI foundation.

### Phase 3 (Week 3): Polish - Selected Medium Priority
9. Error Handling DRY (#9)
10. Validation Return Types (#15)
11. Token Estimator Strategy (#14)
12. Provider Registry (#20)

**Rationale:** Improves code quality and makes system more extensible.

### Phase 4 (Optional): Low Priority Enhancements
- Implement based on specific needs and time availability

---

## TESTING IMPACT

After implementing high priority refactorings:

**Before:**
- Tests require environment variables
- Tests hit real file system
- Tests modify global logging state
- Cannot test providers in isolation
- CLI tests require API keys

**After:**
- All dependencies injectable
- File I/O mockable
- Logging isolated per instance
- Providers testable with mock HTTP
- CLI testable without real runners
- 90%+ test coverage achievable

---

## MIGRATION STRATEGY

All refactorings designed to be backward compatible:

1. Add new parameters with defaults
2. Keep existing signatures working
3. Deprecate old patterns gradually
4. Update tests incrementally
5. Document migration path

**Example:**
```python
# Old code still works:
runner = ExperimentRunner(timeout=300)

# New code uses DI:
runner = ExperimentRunner(
    timeout=300,
    providers={"mock": MockProvider()}
)
```

---

## CONCLUSION

The Parallamr codebase is well-structured but has several testability challenges stemming from:
- Direct instantiation of dependencies
- Global state modification (logging)
- Hard-coded environment access
- Mixed I/O and business logic

Implementing the high-priority refactorings will significantly improve testability while maintaining backward compatibility. The recommended phased approach allows for incremental improvement without disrupting development.

**Next Steps:**
1. Review this analysis with team
2. Prioritize based on current sprint goals
3. Create tickets for Phase 1 items
4. Begin implementation with provider DI (#1)

---

**Generated by:** CODER Agent (tdrefactor swarm)
**Contact:** swarm-1759590488674-ssi4bfby8
