# Comprehensive Testing Strategy for Parallamr
## TESTER Agent - HiveMind Swarm Analysis

**Date**: 2025-10-04
**Agent**: TESTER (swarm-1759590488674-ssi4bfby8)
**Project**: parallamr v0.2.0
**Codebase Size**: 2,080 LOC (source), 121 existing tests

---

## Executive Summary

This testing strategy provides a multi-layered approach to ensure quality and reliability for parallamr, a CLI tool for systematic LLM experimentation. The strategy is informed by the existing codebase analysis, current test coverage (~113 passing tests), and identified gaps.

**Current State**:
- Test count: 121 tests (113 passing, 6 failing, 2 skipped)
- Test organization: Flat structure in `/tests` directory
- Coverage areas: Unit tests for core components, integration tests for CLI
- Gaps: Missing E2E tests, mock strategy needs enhancement, performance testing incomplete

**Recommended Approach**: 5-level testing pyramid with emphasis on unit testing and intelligent mocking.

---

## 1. Testing Pyramid Architecture

### Level 1: Unit Tests (Target: 95%+ coverage)
**Scope**: Individual functions and classes
**Speed**: < 100ms per test
**Quantity**: ~200-300 tests
**Isolation**: Complete (all external dependencies mocked)

### Level 2: Integration Tests (Target: 90% feature coverage)
**Scope**: Component interactions
**Speed**: < 1s per test
**Quantity**: ~50-80 tests
**Isolation**: Partial (external APIs mocked, internal components real)

### Level 3: Contract Tests (Target: 100% provider coverage)
**Scope**: LLM provider interfaces
**Speed**: < 500ms per test
**Quantity**: ~30-40 tests
**Isolation**: Mock provider responses with realistic data

### Level 4: API/CLI Tests (Target: 100% command coverage)
**Scope**: CLI interface and Unix I/O
**Speed**: < 2s per test
**Quantity**: ~40-50 tests
**Isolation**: Subprocess testing with mock providers

### Level 5: E2E Tests (Target: Critical paths only)
**Scope**: Complete workflows
**Speed**: < 10s per test
**Quantity**: ~10-15 tests
**Isolation**: None (uses real mock provider, real files)

---

## 2. Unit Testing Strategy

### 2.1 Template Engine (`template.py`)

**Test Categories**:
1. **Variable Replacement** (15 tests)
   - Single/multiple variable replacement
   - Missing variables (warning generation)
   - None values, numeric values, empty strings
   - Duplicate variables in template
   - Case sensitivity handling

2. **Syntax Validation** (8 tests)
   - Valid template syntax
   - Unmatched braces (opening/closing)
   - Invalid variable names (spaces, numbers, special chars)
   - Empty variable names
   - Nested braces (edge case)

3. **Variable Extraction** (6 tests)
   - Extract single/multiple variables
   - Deduplicate variable names
   - Variables with underscores, numbers
   - Unicode in variable names

4. **File Combination** (10 tests)
   - Combine with/without context files
   - Variables in context files
   - Missing variables across files
   - Multiple context files
   - Large file handling (10MB+)

**Mocking Strategy**: No external dependencies - pure functions

**Current Status**: ✅ Well covered (39 tests)
**Gaps**:
- Nested template syntax edge cases
- Performance testing with large templates
- Unicode edge cases (emoji variable names)

---

### 2.2 Token Counter (`token_counter.py`)

**Test Categories**:
1. **Basic Estimation** (12 tests)
   - Empty/simple/complex text
   - Unicode characters (emoji, accents)
   - Whitespace handling
   - Newlines and special characters
   - Very large texts (100KB, 1MB, 10MB)

2. **Detailed Estimation** (8 tests)
   - Character/word/line counting
   - Multiline text statistics
   - Mixed content (code, prose, data)
   - Edge cases (empty lines, tabs)

3. **Context Window Validation** (10 tests)
   - Unknown context window handling
   - Within limits, approaching limits
   - Exceeding limits
   - Custom buffer percentages
   - Zero tokens, negative edge cases

4. **Token Formatting** (5 tests)
   - Number formatting (thousands separator)
   - Percentage calculation
   - Large numbers (millions of tokens)

**Mocking Strategy**: No dependencies - pure calculation

**Current Status**: ⚠️ 3 failing tests (newline handling bug)
**Gaps**:
- Performance benchmarks for large texts
- Accuracy validation against real tokenizers
- Edge cases: RTL text, control characters

---

### 2.3 CSV Writer (`csv_writer.py`)

**Test Categories**:
1. **Basic Writing** (10 tests)
   - Single result write
   - Multiple results (incremental)
   - Header written once
   - File creation/append
   - Empty results handling

2. **CSV Escaping** (12 tests)
   - Multiline content (`\n`, `\r\n`)
   - Quotes and double quotes
   - Commas in content
   - Special characters (tabs, null bytes)
   - Unicode content
   - CSV injection prevention

3. **Field Management** (8 tests)
   - Variable columns handling
   - Field ordering consistency
   - Missing fields
   - Extra fields
   - Type coercion (int, float, None)

4. **File Operations** (6 tests)
   - Permission errors
   - Disk space errors
   - Concurrent access
   - File locking
   - Stdout handling (new feature)

**Mocking Strategy**: Mock file system for error cases

**Current Status**: ✅ Well covered (9 tests)
**Gaps**:
- Stdout output testing
- Large batch performance (10K+ rows)
- Concurrent write safety
- CSV injection security tests

---

### 2.4 Data Models (`models.py`)

**Test Categories**:
1. **Experiment Model** (8 tests)
   - Creation from CSV row
   - Missing required fields (provider, model)
   - Variable extraction
   - Row number tracking
   - Type validation

2. **ProviderResponse Model** (6 tests)
   - Status determination (ok/warning/error)
   - Success flag handling
   - Context window optional
   - Token counts validation

3. **ExperimentResult Model** (10 tests)
   - Creation from experiment + response
   - Template warning integration
   - Status propagation
   - CSV row conversion
   - Error message combination

**Mocking Strategy**: No dependencies - data classes

**Current Status**: ✅ Adequate (8 tests)
**Gaps**:
- Validation edge cases
- Deep copy behavior
- Serialization/deserialization

---

### 2.5 Provider Classes

#### 2.5.1 Base Provider (`providers/base.py`)

**Test Categories**:
1. **Abstract Interface** (5 tests)
   - Cannot instantiate abstract class
   - All abstract methods defined
   - Timeout configuration
   - Provider name generation

2. **Exception Hierarchy** (6 tests)
   - ProviderError base class
   - ModelNotAvailableError
   - AuthenticationError
   - RateLimitError
   - TimeoutError
   - ContextWindowExceededError

**Mocking Strategy**: Not applicable (abstract class)

**Current Status**: ⚠️ No dedicated tests
**Gaps**: Complete coverage needed

---

#### 2.5.2 Mock Provider (`providers/mock.py`)

**Test Categories**:
1. **Completion Generation** (8 tests)
   - Basic completion
   - With variables
   - Token counting
   - Response structure
   - Deterministic output

2. **Model Information** (4 tests)
   - Context window (None)
   - Model listing
   - Model availability
   - Provider name

**Mocking Strategy**: None needed (already a mock)

**Current Status**: ✅ Well covered (9 tests)
**Gaps**: None significant

---

#### 2.5.3 OpenRouter Provider (`providers/openrouter.py`)

**Test Categories**:
1. **Authentication** (8 tests)
   - Valid API key
   - Missing API key
   - Invalid API key (401)
   - Environment variable loading
   - API key in constructor vs env

2. **Request Handling** (15 tests)
   - Successful completion
   - Rate limiting (429)
   - Context exceeded (413)
   - Network errors
   - Timeout errors
   - Malformed responses
   - Empty responses

3. **Model Management** (10 tests)
   - List models (API call)
   - Model availability check
   - Context window retrieval
   - Model caching behavior
   - Cache invalidation

4. **Error Handling** (8 tests)
   - Connection refused
   - DNS failure
   - SSL errors
   - Invalid JSON response
   - Partial response

**Mocking Strategy**: Mock `aiohttp.ClientSession`

**Current Status**: ⚠️ Minimal (skipped integration tests)
**Gaps**: Complete mock-based unit testing needed

**Recommended Mock Pattern**:
```python
@pytest.fixture
def mock_openrouter_session():
    """Mock aiohttp session for OpenRouter"""
    async def mock_post(*args, **kwargs):
        response = AsyncMock()
        response.status = 200
        response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"completion_tokens": 50}
        })
        return response

    session = AsyncMock()
    session.post = mock_post
    return session
```

---

#### 2.5.4 Ollama Provider (`providers/ollama.py`)

**Test Categories**:
1. **Connection** (6 tests)
   - Default URL (localhost:11434)
   - Custom URL from env
   - Connection refused
   - Invalid URL format
   - Timeout handling

2. **Model Operations** (12 tests)
   - List models API call
   - Model name preservation (with tags)
   - Model availability check
   - Empty model list
   - API response parsing

3. **Context Window Retrieval** (8 tests)
   - Context window from model info
   - Missing context window
   - Different architectures (llama, mistral)
   - Context length field variations
   - **BUG FIX**: Currently looking in wrong fields

4. **Completion Requests** (10 tests)
   - Successful completion
   - Model not found
   - Network errors
   - Streaming response handling
   - Token counting

**Mocking Strategy**: Mock `aiohttp.ClientSession`

**Current Status**: ⚠️ Has regression tests but needs expansion
**Known Issues**:
- Context window parsing looks in wrong fields (line 157-171)
- Should use `model_info["llama.context_length"]` pattern

**Gaps**:
- Comprehensive error handling tests
- Different Ollama versions compatibility
- Streaming response tests

---

### 2.6 Experiment Runner (`runner.py`)

**Test Categories**:
1. **Initialization** (5 tests)
   - Default configuration
   - Custom timeout
   - Verbose logging
   - Provider registration

2. **Single Experiment Execution** (15 tests)
   - Successful execution
   - Missing variables warning
   - Unknown provider error
   - Context window validation
   - Template processing
   - Token counting integration

3. **Batch Execution** (10 tests)
   - Multiple experiments
   - Sequential processing
   - Error recovery (continue on error)
   - Progress tracking
   - CSV writing integration

4. **Validation** (8 tests)
   - Valid experiments
   - Invalid CSV structure
   - Unknown providers
   - Missing required columns
   - Empty experiments file

**Mocking Strategy**: Mock providers for controlled testing

**Current Status**: ⚠️ 3 failing tests
**Gaps**:
- Parallel execution testing (future feature)
- Memory management for large batches
- Graceful interruption handling

---

### 2.7 Utilities (`utils.py`)

**Test Categories**:
1. **File Operations** (8 tests)
   - Load file content
   - Load context files
   - Load experiments CSV
   - CSV parsing (string vs file)
   - File not found errors
   - Invalid CSV format

2. **Validation** (6 tests)
   - Output path validation
   - Stdout handling (None path)
   - Invalid paths
   - Permission checks

3. **Formatting** (5 tests)
   - Experiment summary formatting
   - Provider grouping
   - Counts and statistics

**Mocking Strategy**: Mock file system for errors

**Current Status**: ✅ Well covered (8 tests)
**Gaps**: Edge cases for malformed CSVs

---

## 3. Integration Testing Strategy

### 3.1 Component Integration Tests

**Test Scenarios** (25 tests):

1. **Template → Token Counter** (5 tests)
   - Process template, count resulting tokens
   - Variable replacement impact on tokens
   - Multiple files concatenation

2. **Runner → Provider** (8 tests)
   - Runner calls correct provider
   - Provider response handling
   - Error propagation
   - Timeout handling

3. **Runner → CSV Writer** (7 tests)
   - Incremental writing during execution
   - Field ordering consistency
   - Error handling in CSV writing

4. **Complete Flow (no CLI)** (5 tests)
   - Load files → Process → Execute → Write
   - With/without context files
   - Error recovery
   - Warning accumulation

**Mocking Strategy**: Mock external APIs only, use real internal components

---

### 3.2 CLI Integration Tests

**Test Scenarios** (30+ tests):

1. **Command Parsing** (8 tests)
   - All required arguments
   - Optional arguments
   - Flag combinations
   - Invalid argument combinations
   - Help text verification

2. **File I/O** (12 tests)
   - Read from files
   - Write to files
   - **NEW**: Stdin support (`-`)
   - **NEW**: Stdout support (no `-o`)
   - Error messages for missing files
   - Path resolution (relative/absolute)

3. **Subcommands** (10 tests)
   - `init` command
   - `providers` command
   - `models` command (openrouter/ollama)
   - `run` command variants
   - `--validate-only` flag

**Current Status**: ✅ Excellent (19 tests in test_cli.py)
**New Features Tested**:
- stdin/stdout support (issue #6)
- Validation preventing both stdin sources

---

## 4. Contract Testing Strategy

### 4.1 Provider Contract Tests

**Purpose**: Ensure all providers implement consistent interface

**Test Template** (applies to all providers):

```python
class ProviderContractTests:
    """Base contract tests for all providers"""

    @abstractmethod
    def get_provider(self):
        """Return provider instance for testing"""
        pass

    def test_get_completion_returns_response(self):
        """All providers must return ProviderResponse"""
        provider = self.get_provider()
        response = await provider.get_completion("test", "model")
        assert isinstance(response, ProviderResponse)

    def test_response_has_required_fields(self):
        """Response must have output, tokens, success"""
        # ... implementation

    def test_handles_timeout_gracefully(self):
        """Must handle timeout without crashing"""
        # ... implementation

    # ... 15 more contract tests
```

**Implementation** (3 provider classes × 20 contract tests = 60 tests):
- MockProvider contract tests
- OpenRouterProvider contract tests
- OllamaProvider contract tests

**Benefits**:
- Ensures interface consistency
- Catches breaking changes
- Validates error handling standards

---

### 4.2 API Response Contract Tests

**Purpose**: Validate external API response handling

**Test Data**: Realistic mock responses from:
- OpenRouter API (`/v1/chat/completions`, `/v1/models`)
- Ollama API (`/api/generate`, `/api/tags`, `/api/show`)

**Test Cases** (20 tests):
1. Valid response parsing
2. Missing fields handling
3. Extra fields tolerance
4. Type coercion
5. Error response formats
6. Rate limit headers
7. Retry-after headers
8. API version compatibility

**Implementation**:
```python
@pytest.fixture
def openrouter_valid_response():
    return {
        "id": "gen-123",
        "choices": [{
            "message": {"content": "Response text"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 50,
            "total_tokens": 60
        }
    }

def test_parse_openrouter_response(openrouter_valid_response):
    response = parse_openrouter_response(openrouter_valid_response)
    assert response.output == "Response text"
    assert response.output_tokens == 50
```

---

## 5. Mocking Strategy

### 5.1 Mock Categories

#### Category 1: External HTTP APIs
**What to Mock**:
- OpenRouter API calls
- Ollama API calls

**How to Mock**:
```python
# Using aioresponses library
from aioresponses import aioresponses

@pytest.fixture
def mock_openrouter_api():
    with aioresponses() as m:
        m.post(
            'https://openrouter.ai/api/v1/chat/completions',
            payload={
                'choices': [{'message': {'content': 'Test'}}],
                'usage': {'completion_tokens': 10}
            }
        )
        yield m
```

**Benefits**:
- No API keys needed
- Fast execution
- Deterministic results
- Offline testing

---

#### Category 2: File System Operations
**What to Mock**:
- File reads/writes for error cases
- Permission errors
- Disk space errors

**How to Mock**:
```python
def test_file_permission_error(mocker):
    mocker.patch('pathlib.Path.read_text',
                 side_effect=PermissionError("Access denied"))

    with pytest.raises(PermissionError):
        load_file_content("test.txt")
```

**When NOT to Mock**:
- Use real temporary files for happy path tests
- Only mock for error conditions

---

#### Category 3: Time-based Operations
**What to Mock**:
- Timeouts
- Rate limiting
- Retry delays

**How to Mock**:
```python
@pytest.mark.asyncio
async def test_timeout_handling(mocker):
    mocker.patch('asyncio.sleep')  # Skip actual delays

    provider = OpenRouterProvider(timeout=1)
    # Simulate timeout
    with pytest.raises(TimeoutError):
        await provider.get_completion("test", "model")
```

---

#### Category 4: Environment Variables
**What to Mock**:
- API keys
- Configuration URLs
- Feature flags

**How to Mock**:
```python
@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://test:11434")
    yield
    # Automatic cleanup
```

---

### 5.2 Mock Data Fixtures

**Location**: `tests/fixtures/`

**Structure**:
```
fixtures/
├── api_responses/
│   ├── openrouter_success.json
│   ├── openrouter_rate_limit.json
│   ├── ollama_models.json
│   ├── ollama_show_llama.json
│   └── ollama_error.json
├── prompts/
│   ├── basic.txt
│   ├── with_variables.txt
│   ├── multiline.txt
│   ├── unicode.txt
│   └── large_10kb.txt
├── contexts/
│   ├── simple.txt
│   ├── code_context.txt
│   └── structured_data.json
├── experiments/
│   ├── valid_basic.csv
│   ├── missing_columns.csv
│   ├── invalid_provider.csv
│   ├── large_1000_rows.csv
│   └── unicode_variables.csv
└── expected_outputs/
    ├── basic_run.csv
    ├── warning_case.csv
    └── error_case.csv
```

**Fixture Loading Utility**:
```python
# tests/conftest.py
import json
from pathlib import Path

@pytest.fixture(scope="session")
def fixtures_dir():
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def load_fixture(fixtures_dir):
    def _load(category, filename):
        path = fixtures_dir / category / filename
        if path.suffix == '.json':
            return json.loads(path.read_text())
        return path.read_text()
    return _load
```

---

## 6. E2E Testing Strategy

### 6.1 Critical User Flows

**Flow 1: First-time User Setup** (1 test)
```bash
# Steps:
parallamr init
# Edit .env (mocked)
parallamr run -p prompt.txt -e experiments.csv -o results.csv
# Verify results.csv exists and valid
```

**Flow 2: Multi-provider Comparison** (1 test)
```csv
provider,model,task
mock,mock,test
# Would include openrouter/ollama with mocks
```

**Flow 3: Large Batch Processing** (1 test)
- 100+ experiments
- Verify incremental writing
- Test interruption handling

**Flow 4: Error Recovery** (2 tests)
- Some experiments fail
- Successful experiments still written
- Error messages captured

**Flow 5: Stdin/Stdout Pipeline** (2 tests)
```bash
cat experiments.csv | parallamr run -p prompt.txt -e - > results.csv
cat prompt.txt | parallamr run -p - -e experiments.csv -o results.csv
```

**Flow 6: Validation Workflow** (1 test)
```bash
parallamr run -p prompt.txt -e experiments.csv --validate-only
# Fix errors
parallamr run -p prompt.txt -e experiments.csv -o results.csv
```

### 6.2 E2E Test Implementation

**Framework**: Subprocess testing with Click's CliRunner

```python
def test_complete_user_flow(tmp_path):
    """Test complete workflow from init to results"""
    runner = CliRunner()

    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Step 1: Initialize
        result = runner.invoke(cli, ['init'])
        assert result.exit_code == 0

        # Step 2: Modify experiments to use mock only
        # ... modify files ...

        # Step 3: Run experiments
        result = runner.invoke(cli, [
            'run',
            '-p', 'prompt.txt',
            '-e', 'experiments.csv',
            '-o', 'results.csv',
            '--verbose'
        ])
        assert result.exit_code == 0

        # Step 4: Verify results
        results_path = Path('results.csv')
        assert results_path.exists()

        # Verify CSV structure
        with open(results_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) > 0
            assert all('status' in row for row in rows)
```

---

## 7. Test Data Management

### 7.1 Realistic Test Data

**Prompt Templates**:
- Basic: "Summarize {{topic}}"
- Complex: Multi-paragraph with multiple variables
- Edge case: Unicode, emojis, code blocks
- Large: 5KB+ prompts

**Context Files**:
- Simple text
- Structured data (JSON, CSV)
- Code snippets
- Large documents (100KB+)

**Experiment CSVs**:
- Minimal (2 columns: provider, model)
- Standard (5-10 columns with variables)
- Large (1000+ rows)
- Malformed (missing columns, wrong types)

### 7.2 Sensitive Data Handling

**Rules**:
1. Never commit real API keys
2. Use obvious fake keys: `"test-key-12345"`
3. Mock all external API responses
4. Sanitize any real data examples

**Environment Variable Management**:
```python
# Good: Explicit test values
@pytest.fixture
def test_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake-key")

# Bad: Using real keys from environment
def test_with_real_key():
    api_key = os.getenv("OPENROUTER_API_KEY")  # DON'T DO THIS
```

---

## 8. Testing Tools & Frameworks

### 8.1 Core Testing Stack

**Primary Framework**: pytest
- Async support: `pytest-asyncio`
- Coverage: `pytest-cov`
- Fixtures: Built-in pytest fixtures
- Parametrization: `@pytest.mark.parametrize`

**Mocking Libraries**:
- HTTP mocking: `aioresponses` (for aiohttp)
- Function mocking: `pytest-mock` (wrapper for unittest.mock)
- Time mocking: `freezegun` (for time-sensitive tests)

**Assertion Libraries**:
- Standard: pytest built-in assertions
- Advanced: `pytest-clarity` (better diffs)

### 8.2 CI/CD Integration

**GitHub Actions Workflow**:
```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12]

    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run pytest --cov=src/parallamr --cov-report=xml
      - uses: codecov/codecov-action@v3
```

**Test Commands**:
```bash
# Quick smoke test (unit tests only)
pytest tests/ -k "not integration and not slow" -v

# Full test suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/parallamr --cov-report=html --cov-report=term

# Only integration tests
pytest tests/ -m integration -v

# Skip slow tests
pytest tests/ -m "not slow"

# Parallel execution (future)
pytest tests/ -n auto
```

### 8.3 Code Quality Tools

**Type Checking**: mypy
```bash
mypy src/parallamr --strict
```

**Linting**: ruff
```bash
ruff check src/ tests/
```

**Formatting**: black
```bash
black src/ tests/ --check
```

**Security**: bandit
```bash
bandit -r src/parallamr
```

---

## 9. Test Organization & Structure

### 9.1 Recommended Directory Structure

```
tests/
├── __init__.py
├── conftest.py                      # Shared fixtures
├── unit/                            # Unit tests (fast, isolated)
│   ├── __init__.py
│   ├── test_template.py            ✅ EXISTS (39 tests)
│   ├── test_token_counter.py       ✅ EXISTS (35 tests) ⚠️ 3 failing
│   ├── test_csv_writer.py          ✅ EXISTS (9 tests)
│   ├── test_models.py              ✅ EXISTS (8 tests)
│   ├── test_utils.py               ✅ EXISTS (8 tests)
│   ├── test_runner.py              ✅ EXISTS (7 tests) ⚠️ 3 failing
│   └── providers/
│       ├── __init__.py
│       ├── test_base.py            ❌ NEW
│       ├── test_mock.py            ✅ EXISTS (partial)
│       ├── test_openrouter.py      ❌ NEW (currently skipped)
│       └── test_ollama.py          ✅ EXISTS (regression tests)
├── integration/                     # Integration tests
│   ├── __init__.py
│   ├── test_cli.py                 ✅ EXISTS (19 tests)
│   ├── test_runner_integration.py  ❌ NEW
│   └── test_end_to_end.py          ❌ NEW
├── contract/                        # Contract tests
│   ├── __init__.py
│   ├── test_provider_contracts.py  ❌ NEW
│   └── test_api_contracts.py       ❌ NEW
├── performance/                     # Performance tests
│   ├── __init__.py
│   ├── test_token_counting.py      ❌ NEW
│   ├── test_csv_writing.py         ❌ NEW
│   └── test_large_batches.py       ❌ NEW
└── fixtures/                        # Test data
    ├── api_responses/               ❌ NEW
    ├── prompts/                     ✅ EXISTS (3 files)
    ├── contexts/                    ✅ EXISTS (1 file)
    ├── experiments/                 ✅ EXISTS (1 file)
    └── expected_outputs/            ❌ NEW
```

### 9.2 Test Naming Conventions

**Pattern**: `test_<component>_<scenario>_<expected_result>`

**Examples**:
```python
# Good
def test_replace_variables_with_missing_variables_returns_warnings():
def test_openrouter_provider_with_invalid_key_raises_auth_error():
def test_csv_writer_with_multiline_content_escapes_correctly():

# Bad
def test_template():
def test_error():
def test_it_works():
```

**Class Organization**:
```python
class TestTemplateEngine:
    """Tests for template.py"""

    class TestReplaceVariables:
        """Tests for replace_variables function"""

        def test_with_valid_variables():
            pass

        def test_with_missing_variables():
            pass

    class TestValidateSyntax:
        """Tests for validate_template_syntax function"""
        pass
```

---

## 10. Success Criteria & KPIs

### 10.1 Coverage Metrics

**Code Coverage Targets**:
- Overall: ≥ 90%
- Unit tests: ≥ 95%
- Critical paths (template, runner, providers): 100%
- CLI: ≥ 85% (some error paths hard to reach)

**Branch Coverage**: ≥ 85%

**Mutation Testing** (future): ≥ 70% mutation score

### 10.2 Test Quality Metrics

**Test Count by Level**:
- Unit: 250-300 tests
- Integration: 50-80 tests
- Contract: 60-80 tests
- E2E: 10-15 tests
- **Total**: 370-475 tests

**Test Speed**:
- Unit tests: < 10s total (< 50ms average)
- Integration: < 60s total (< 1s average)
- Full suite: < 2 minutes

**Flakiness**: < 0.1% (1 in 1000 test runs)

### 10.3 Quality Gates

**Pre-commit**:
- All tests pass
- No linting errors
- Type checking passes

**PR Merge Requirements**:
- All tests pass
- Coverage doesn't decrease
- New code has tests (enforced by coverage)
- No security vulnerabilities

**Release Requirements**:
- Full test suite passes
- Manual E2E testing completed
- Performance benchmarks met
- Documentation updated

---

## 11. Implementation Roadmap

### Phase 1: Foundation (Week 1)
**Priority**: Critical bugs and infrastructure

Tasks:
1. Fix failing tests (6 tests)
   - Token counter newline handling
   - Runner integration tests
   - Template syntax validation
2. Set up test fixtures directory structure
3. Add missing conftest.py fixtures
4. Configure CI/CD with coverage reporting

**Deliverables**:
- All existing tests passing
- Coverage baseline established
- CI/CD pipeline working

---

### Phase 2: Unit Test Expansion (Week 2-3)
**Priority**: Achieve 95% unit test coverage

Tasks:
1. Complete provider unit tests (60 tests)
   - OpenRouter with aiohttp mocking
   - Ollama with aiohttp mocking
   - Base provider tests
2. Expand edge case coverage (40 tests)
   - Template edge cases
   - Token counter edge cases
   - CSV writer edge cases
3. Add performance tests (20 tests)

**Deliverables**:
- 250+ unit tests
- 95%+ unit test coverage
- Performance benchmarks established

---

### Phase 3: Integration & Contract Tests (Week 4)
**Priority**: Component interaction validation

Tasks:
1. Component integration tests (25 tests)
2. Provider contract tests (60 tests)
3. API contract tests (20 tests)
4. Enhanced CLI tests (15 tests)

**Deliverables**:
- 120+ integration/contract tests
- Provider interface consistency validated
- CLI fully tested

---

### Phase 4: E2E & Polish (Week 5)
**Priority**: Critical user flows and documentation

Tasks:
1. E2E test scenarios (10-15 tests)
2. Test documentation
3. Developer testing guide
4. Performance optimization

**Deliverables**:
- Complete test suite
- Testing documentation
- Performance benchmarks met
- Ready for production

---

## 12. Risk Assessment & Mitigation

### 12.1 High-Risk Areas

**Risk 1: API Key Leakage**
- **Impact**: Security breach, cost
- **Likelihood**: Medium
- **Mitigation**:
  - Never use real keys in tests
  - Mock all external APIs
  - Git pre-commit hooks to scan for keys
  - Environment variable validation

**Risk 2: CSV Injection**
- **Impact**: Security vulnerability
- **Likelihood**: Low
- **Mitigation**:
  - Comprehensive CSV escaping tests
  - Security-focused test scenarios
  - Fuzzing with malicious inputs

**Risk 3: Provider API Changes**
- **Impact**: Breaking changes
- **Likelihood**: Medium
- **Mitigation**:
  - Contract tests validate API structure
  - Version pinning
  - Regular integration test runs with real APIs (manual)

**Risk 4: Race Conditions**
- **Impact**: Flaky tests, data corruption
- **Likelihood**: Low (currently sequential)
- **Mitigation**:
  - File locking tests
  - Concurrent write tests
  - Prepare for future parallel execution

### 12.2 Known Issues

**Issue 1**: Ollama context window retrieval bug
- **Location**: `providers/ollama.py` lines 157-171
- **Fix**: Use `model_info["llama.context_length"]` pattern
- **Test**: Add regression test before fixing

**Issue 2**: Token counter newline handling
- **Location**: `token_counter.py`
- **Symptom**: Counts vary with different newline styles
- **Fix**: Normalize newlines before counting
- **Test**: Already failing tests exist

**Issue 3**: Template syntax validation over-reporting
- **Location**: `template.py`
- **Symptom**: Reports multiple errors for single issue
- **Fix**: Improve error deduplication
- **Test**: Adjust expectations in failing test

---

## 13. Testing Best Practices

### 13.1 Test Writing Guidelines

**AAA Pattern**: Arrange, Act, Assert
```python
def test_replace_variables_with_valid_data():
    # Arrange
    text = "Hello {{name}}"
    variables = {"name": "World"}

    # Act
    result, missing = replace_variables(text, variables)

    # Assert
    assert result == "Hello World"
    assert missing == []
```

**Single Responsibility**: One test, one concept
```python
# Good
def test_csv_writer_escapes_newlines():
    # Test only newline escaping

def test_csv_writer_escapes_quotes():
    # Test only quote escaping

# Bad
def test_csv_writer_escapes_everything():
    # Tests newlines, quotes, commas, tabs all in one
```

**Descriptive Names**: Self-documenting tests
```python
# Good
def test_openrouter_provider_raises_auth_error_with_invalid_key():

# Bad
def test_auth():
```

### 13.2 Common Anti-Patterns to Avoid

**Anti-Pattern 1**: Testing implementation details
```python
# Bad - tests internal method
def test_runner_calls_private_method():
    runner = ExperimentRunner()
    runner._internal_method()  # Don't test private methods

# Good - tests public interface
def test_runner_processes_experiment():
    runner = ExperimentRunner()
    result = runner.run_experiment(...)
    assert result.status == "ok"
```

**Anti-Pattern 2**: Over-mocking
```python
# Bad - mocks everything
def test_runner_with_all_mocks(mocker):
    mocker.patch('parallamr.template.replace_variables')
    mocker.patch('parallamr.token_counter.estimate_tokens')
    mocker.patch('parallamr.providers.get_provider')
    # ... test becomes meaningless

# Good - mock only external dependencies
def test_runner_integration(mocker):
    mocker.patch('aiohttp.ClientSession')  # Only external API
    # Use real internal components
```

**Anti-Pattern 3**: Fragile tests
```python
# Bad - relies on exact string matching
assert "Error: something went wrong" in output

# Good - tests essential content
assert "Error" in output
assert "wrong" in output.lower()
```

---

## 14. Continuous Improvement

### 14.1 Test Metrics Monitoring

**Dashboard Metrics**:
- Test count trend (should increase)
- Coverage trend (should increase then stabilize)
- Test duration trend (should stay stable or decrease)
- Flaky test rate (should approach 0%)
- PR test failure rate (good indicator of code quality)

**Tools**:
- Codecov: Coverage tracking
- pytest-benchmark: Performance tracking
- Custom scripts: Flakiness detection

### 14.2 Regular Test Maintenance

**Monthly**:
- Review skipped tests (bring back or remove)
- Review slow tests (optimize or mark as slow)
- Update test fixtures for new features
- Check for obsolete tests

**Quarterly**:
- Refactor test code (reduce duplication)
- Update testing documentation
- Review mocking strategy (are mocks still accurate?)
- Performance benchmark review

**Annually**:
- Major test framework updates
- Test architecture review
- Developer testing survey

---

## 15. Developer Guidelines

### 15.1 Writing Tests for New Features

**Checklist**:
- [ ] Unit tests for new functions/classes
- [ ] Integration tests if interacting with existing components
- [ ] Update contract tests if changing provider interface
- [ ] E2E test if new user-facing feature
- [ ] Update test fixtures if new data formats
- [ ] Add performance test if performance-critical

**Example Workflow**:
```bash
# 1. Write failing test first (TDD)
def test_new_feature():
    result = new_function()
    assert result == expected

# 2. Run test (should fail)
pytest tests/test_new.py::test_new_feature -v

# 3. Implement feature

# 4. Run test (should pass)
pytest tests/test_new.py::test_new_feature -v

# 5. Run full suite
pytest tests/
```

### 15.2 Debugging Failing Tests

**Strategies**:

1. **Verbose Output**:
```bash
pytest tests/test_failing.py -vv  # Very verbose
pytest tests/test_failing.py -s   # Show print statements
```

2. **Specific Test**:
```bash
pytest tests/test_file.py::TestClass::test_method
```

3. **Drop into Debugger**:
```python
def test_something():
    result = function()
    import pdb; pdb.set_trace()  # Debugger here
    assert result == expected
```

4. **Capture Logs**:
```bash
pytest tests/test_failing.py --log-cli-level=DEBUG
```

### 15.3 Test Review Checklist

When reviewing PRs with tests:
- [ ] Tests are in correct directory (unit/integration/etc.)
- [ ] Test names are descriptive
- [ ] Tests follow AAA pattern
- [ ] No hard-coded values (use fixtures/parametrize)
- [ ] Mocking is appropriate (not over/under-mocked)
- [ ] Tests are deterministic (not flaky)
- [ ] Edge cases are covered
- [ ] Performance considerations addressed
- [ ] Documentation updated if needed

---

## Appendix A: Test Count Summary

| Category | Current | Target | Gap |
|----------|---------|--------|-----|
| **Unit Tests** | | | |
| Template | 39 | 40 | 1 |
| Token Counter | 35 | 40 | 5 |
| CSV Writer | 9 | 30 | 21 |
| Models | 8 | 25 | 17 |
| Utils | 8 | 20 | 12 |
| Providers (base) | 0 | 10 | 10 |
| Providers (mock) | 9 | 15 | 6 |
| Providers (openrouter) | 2 | 40 | 38 |
| Providers (ollama) | 3 | 35 | 32 |
| Runner | 7 | 40 | 33 |
| **Integration Tests** | | | |
| CLI | 19 | 30 | 11 |
| Component Integration | 0 | 25 | 25 |
| **Contract Tests** | 0 | 80 | 80 |
| **Performance Tests** | 0 | 20 | 20 |
| **E2E Tests** | 0 | 15 | 15 |
| **TOTAL** | 139* | 465 | 326 |

*Note: 121 collected, but some are duplicates or skipped

---

## Appendix B: Mock Response Examples

### OpenRouter Success Response
```json
{
  "id": "gen-1234567890",
  "model": "anthropic/claude-sonnet-4",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "This is the model's response to your prompt."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 89,
    "total_tokens": 134
  }
}
```

### Ollama Model List Response
```json
{
  "models": [
    {
      "name": "llama3.1:latest",
      "modified_at": "2025-10-01T10:00:00Z",
      "size": 4661211648
    },
    {
      "name": "llama3.1:8b",
      "modified_at": "2025-10-01T10:00:00Z",
      "size": 4661211648
    }
  ]
}
```

### Ollama Model Info Response
```json
{
  "model_info": {
    "general.architecture": "llama",
    "general.file_type": 2,
    "llama.context_length": 131072,
    "llama.embedding_length": 4096,
    "llama.block_count": 32,
    "llama.attention.head_count": 32,
    "llama.attention.head_count_kv": 8
  }
}
```

---

## Appendix C: Useful Pytest Plugins

| Plugin | Purpose | Installation |
|--------|---------|--------------|
| pytest-asyncio | Async test support | `uv add pytest-asyncio --dev` |
| pytest-cov | Coverage reporting | `uv add pytest-cov --dev` |
| pytest-mock | Enhanced mocking | `uv add pytest-mock --dev` |
| pytest-benchmark | Performance testing | `uv add pytest-benchmark --dev` |
| pytest-xdist | Parallel execution | `uv add pytest-xdist --dev` |
| pytest-clarity | Better assertions | `uv add pytest-clarity --dev` |
| aioresponses | Mock aiohttp | `uv add aioresponses --dev` |
| freezegun | Time mocking | `uv add freezegun --dev` |
| pytest-timeout | Test timeout | `uv add pytest-timeout --dev` |

---

## Conclusion

This comprehensive testing strategy provides:

1. **Multi-level testing pyramid** ensuring quality at every layer
2. **Intelligent mocking strategy** for fast, reliable tests
3. **Clear implementation roadmap** with phased approach
4. **Comprehensive coverage targets** with defined success criteria
5. **Practical guidelines** for developers writing tests
6. **Risk mitigation** for high-risk areas
7. **Continuous improvement** framework

**Next Steps**:
1. Fix 6 failing tests (Phase 1)
2. Implement provider unit tests with mocking (Phase 2)
3. Add contract tests (Phase 3)
4. Complete E2E tests (Phase 4)
5. Achieve 95%+ coverage

**Estimated Effort**: 5 weeks for complete implementation
**Expected Outcome**: 465+ tests, 95%+ coverage, production-ready quality assurance

---

**Document Version**: 1.0
**Last Updated**: 2025-10-04
**Author**: TESTER Agent (HiveMind Swarm)
**Review Status**: Ready for Queen review
