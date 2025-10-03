# Parallamr Testing Strategy
## HiveMind-Tester-Delta Quality Assurance Framework

### Overview
This document outlines a comprehensive testing strategy for the Parallamr project, a command-line tool for running systematic experiments across multiple LLM providers. The testing framework ensures robust, reliable software through systematic validation of all components.

## Testing Objectives
1. **Reliability**: Ensure consistent behavior across different providers and configurations
2. **Robustness**: Handle edge cases, errors, and unexpected inputs gracefully
3. **Performance**: Validate token counting accuracy and efficient CSV processing
4. **Security**: Prevent API key leakage and validate input sanitization
5. **Usability**: Ensure clear error messages and proper CLI behavior

## Test Categories

### 1. Unit Tests (Target: >95% coverage)

#### A. Template Engine (`template.py`)
**Core Functionality:**
- Variable replacement with valid variables
- Handling missing variables (warning generation)
- Malformed variable syntax handling
- Empty template handling
- Special characters in variable names

**Test Cases:**
```python
# Test valid replacement
assert replace_variables("Hello {{name}}", {"name": "World"}) == ("Hello World", [])

# Test missing variables
result, missing = replace_variables("Hello {{name}} {{age}}", {"name": "Alice"})
assert result == "Hello Alice {{age}}"
assert missing == ["age"]

# Test malformed syntax
assert replace_variables("Hello {name}", {"name": "World"}) == ("Hello {name}", [])
```

#### B. Token Counter (`token_counter.py`)
**Core Functionality:**
- Character count / 4 estimation accuracy
- Unicode character handling
- Empty string handling
- Very large text handling

**Test Cases:**
```python
# Test basic estimation
assert estimate_tokens("Hello world") == 2  # 11 chars / 4 = 2.75 -> 2

# Test Unicode
assert estimate_tokens("🚀✨") == 0  # 2 chars / 4 = 0.5 -> 0

# Test large text
large_text = "a" * 10000
assert estimate_tokens(large_text) == 2500
```

#### C. CSV Writer (`csv_writer.py`)
**Core Functionality:**
- Header writing on first call only
- Proper CSV escaping for multiline content
- Incremental append behavior
- File handle management

**Test Cases:**
```python
# Test CSV escaping
result = ExperimentResult(output='Line 1\nLine 2,"quoted"')
# Should properly escape newlines and quotes

# Test incremental writing
writer = IncrementalCSVWriter("test.csv")
writer.write_result(result1)
writer.write_result(result2)
# Verify both results in file with single header
```

#### D. Provider Classes
**Base Provider Interface:**
- Abstract method implementation validation
- Response format consistency
- Error handling standardization

**Mock Provider:**
- Deterministic response generation
- Proper metadata inclusion
- Variable display in output

**OpenRouter Provider:**
- API key validation
- Rate limit handling
- Model availability checking
- Context window retrieval

**Ollama Provider:**
- Base URL connectivity
- Local model availability
- Error response handling

### 2. Integration Tests

#### A. End-to-End Experiment Execution
**Scenarios:**
- Complete experiment run with mock provider
- Multi-file input processing
- Variable replacement across files
- CSV output validation

**Test Setup:**
```
fixtures/
├── basic_prompt.txt
├── context_file.txt
├── experiments_valid.csv
├── experiments_missing_vars.csv
├── experiments_invalid_models.csv
└── expected_outputs/
    ├── basic_run.csv
    ├── missing_vars_run.csv
    └── error_handling_run.csv
```

#### B. Provider Integration
**Real Provider Tests** (Optional, with API keys):
- OpenRouter API integration
- Ollama local instance integration
- Rate limit handling validation

**Mock Integration Tests** (Always run):
- Provider factory behavior
- Error propagation
- Response format validation

### 3. Error Handling Tests

#### A. Input Validation
- Missing required files
- Malformed CSV files
- Invalid provider names
- Authentication failures

#### B. Runtime Error Handling
- Network connectivity issues
- API rate limits
- Invalid model names
- Context window overflow
- Timeout handling

#### C. File System Errors
- Permission denied for output file
- Disk space exhaustion
- Invalid file paths

### 4. Edge Case Tests

#### A. Boundary Conditions
- Empty prompt files
- Maximum CSV row limits
- Very large context files
- Unicode handling across all components

#### B. Performance Edge Cases
- Large experiment sets (1000+ rows)
- High token count inputs
- Rapid successive API calls

#### C. Configuration Edge Cases
- Missing environment variables
- Invalid API keys
- Unreachable Ollama instances

### 5. CLI Interface Tests

#### A. Argument Parsing
- Required argument validation
- Optional argument handling
- Help text generation
- Error message clarity

#### B. File Path Handling
- Relative vs absolute paths
- Non-existent files
- Permission issues
- Special characters in filenames

### 6. Security Tests

#### A. Input Sanitization
- Injection prevention in templates
- Safe file path handling
- API key protection in logs

#### B. Output Security
- Proper CSV escaping prevents injection
- Sensitive data handling in error messages

## Test Fixtures Design

### Directory Structure
```
tests/
├── __init__.py
├── conftest.py                 # Pytest configuration
├── unit/
│   ├── test_template.py
│   ├── test_token_counter.py
│   ├── test_csv_writer.py
│   └── test_providers.py
├── integration/
│   ├── test_end_to_end.py
│   ├── test_provider_integration.py
│   └── test_cli_integration.py
├── edge_cases/
│   ├── test_error_handling.py
│   ├── test_boundary_conditions.py
│   └── test_performance.py
└── fixtures/
    ├── prompts/
    │   ├── basic.txt
    │   ├── with_variables.txt
    │   ├── multiline.txt
    │   └── unicode.txt
    ├── contexts/
    │   ├── simple.txt
    │   ├── large.txt
    │   └── special_chars.txt
    ├── experiments/
    │   ├── valid_basic.csv
    │   ├── missing_variables.csv
    │   ├── invalid_providers.csv
    │   ├── large_dataset.csv
    │   └── malformed.csv
    └── expected_outputs/
        ├── basic_run.csv
        ├── error_cases.csv
        └── warning_cases.csv
```

### Key Test Fixtures

#### Basic Test Data
```csv
# experiments/valid_basic.csv
provider,model,topic,source
mock,mock,AI,Wikipedia
mock,mock,ML,Encyclopedia
```

```txt
# prompts/with_variables.txt
Please summarize information about {{topic}} from {{source}}.
The response should be concise and accurate.
```

#### Error Test Data
```csv
# experiments/invalid_providers.csv
provider,model,topic
invalid_provider,some_model,AI
openrouter,non_existent_model,ML
```

#### Edge Case Data
```txt
# prompts/unicode.txt
Explain {{concept}} using examples: 🚀 ✨ 🎯
Include émojis and spëcial characters: àáâãäå
```

## Test Automation Framework

### Pytest Configuration
```python
# conftest.py
import pytest
import tempfile
import os
from pathlib import Path

@pytest.fixture
def temp_output_file():
    """Provide temporary CSV output file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

@pytest.fixture
def sample_experiments():
    """Provide sample experiment data."""
    return [
        {"provider": "mock", "model": "mock", "topic": "AI"},
        {"provider": "mock", "model": "mock", "topic": "ML"},
    ]
```

### Test Execution Strategy

#### Local Development
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/parallamr --cov-report=html

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/edge_cases/ -v

# Run performance tests
pytest tests/edge_cases/test_performance.py -v --benchmark
```

#### Continuous Integration
```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12]
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Set up Python
        run: uv python install ${{ matrix.python-version }}
      - name: Install dependencies
        run: uv sync
      - name: Run tests
        run: uv run pytest tests/ --cov=src/parallamr --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Quality Metrics

### Coverage Requirements
- **Unit Tests**: >95% line coverage
- **Integration Tests**: >90% feature coverage
- **Edge Cases**: 100% error path coverage

### Performance Benchmarks
- **CSV Writing**: <1ms per row for standard results
- **Token Counting**: <1ms for 10K character strings
- **Template Processing**: <5ms for complex templates

### Quality Gates
1. All tests must pass before merge
2. Coverage requirements must be met
3. No security vulnerabilities in dependencies
4. Performance benchmarks must be maintained

## Test Data Management

### Sensitive Data Handling
- No real API keys in test fixtures
- Mock responses for all external APIs
- Sanitized example data only

### Test Data Versioning
- Version test fixtures with code changes
- Maintain backward compatibility where possible
- Document breaking changes in test data

## Monitoring and Reporting

### Test Result Reporting
- JUnit XML format for CI integration
- HTML coverage reports for developers
- Performance regression detection

### Quality Dashboard
- Test execution trends
- Coverage progression
- Performance metrics over time
- Error pattern analysis

## Risk Assessment

### High-Risk Areas
1. **API Key Handling**: Leakage in logs or errors
2. **CSV Injection**: Malicious content in outputs
3. **Rate Limiting**: API quota exhaustion
4. **File Handling**: Permission and path traversal issues

### Mitigation Strategies
1. Comprehensive input validation tests
2. Security-focused test scenarios
3. Mock-first testing approach
4. Systematic error handling validation

## Maintenance Strategy

### Test Maintenance
- Regular fixture updates with spec changes
- Deprecation warnings for outdated test patterns
- Refactoring tests alongside code changes

### Documentation Updates
- Keep testing docs synchronized with implementation
- Document new test patterns and utilities
- Maintain troubleshooting guides

This comprehensive testing strategy ensures the Parallamr project will be robust, reliable, and maintainable while providing excellent user experience and developer confidence.