# Testing Strategy Executive Summary
## Parallamr Project - Quick Reference Guide

**Project**: parallamr v0.2.0
**Current State**: 121 tests (113 passing, 6 failing, 2 skipped)
**Target**: 465 tests, 95%+ coverage
**Timeline**: 5 weeks to completion

---

## Current State Analysis

### ✅ Strengths
- Excellent CLI test coverage (19 tests)
- Good template engine tests (39 tests)
- Comprehensive token counter tests (35 tests)
- Well-structured unit test foundation
- Recent stdin/stdout support added (issue #6)

### ⚠️ Critical Issues
1. **6 failing tests** need immediate fixes:
   - Token counter newline handling (3 tests)
   - Runner integration tests (3 tests)
2. **Ollama provider bug**: Context window retrieval looks in wrong fields
3. **Missing provider tests**: OpenRouter/Ollama lack comprehensive unit tests
4. **No E2E tests**: Critical user flows not validated

### ❌ Gaps
- Provider unit tests: 2/75 implemented (97% gap)
- Contract tests: 0/80 needed
- Integration tests: 19/55 needed
- Performance tests: 0/20 needed
- E2E tests: 0/15 needed

---

## Testing Pyramid (5 Levels)

```
           E2E (10-15 tests)
         /                  \
    API/CLI (40-50 tests)
      /                    \
  Contract (60-80 tests)
    /                      \
Integration (50-80 tests)
  /                        \
Unit Tests (250-300 tests) ← FOUNDATION
```

### Level 1: Unit Tests (95%+ coverage target)
- **Current**: 113 tests
- **Target**: 250-300 tests
- **Gap**: 137-187 tests
- **Priority**: HIGH

### Level 2: Integration Tests (90% feature coverage)
- **Current**: 19 tests
- **Target**: 50-80 tests
- **Gap**: 31-61 tests
- **Priority**: MEDIUM

### Level 3: Contract Tests (100% provider coverage)
- **Current**: 0 tests
- **Target**: 60-80 tests
- **Gap**: 60-80 tests
- **Priority**: HIGH

### Level 4: API/CLI Tests (100% command coverage)
- **Current**: 19 tests (good!)
- **Target**: 40-50 tests
- **Gap**: 21-31 tests
- **Priority**: MEDIUM

### Level 5: E2E Tests (critical paths only)
- **Current**: 0 tests
- **Target**: 10-15 tests
- **Gap**: 10-15 tests
- **Priority**: MEDIUM

---

## Quick Wins (Week 1)

### 1. Fix Failing Tests (2 hours)
```bash
# Token counter newline bug
# Location: src/parallamr/token_counter.py
# Fix: Normalize newlines before counting
text = text.replace('\r\n', '\n')  # CRLF → LF

# Template syntax validation
# Location: src/parallamr/template.py
# Fix: Deduplicate error messages

# Runner tests
# Location: tests/test_runner.py
# Fix: Update expected mock response format
```

### 2. Add Provider Mock Tests (4 hours)
```python
# tests/unit/providers/test_openrouter.py
@pytest.mark.asyncio
async def test_openrouter_success(mock_aiohttp):
    mock_aiohttp.post('https://openrouter.ai/api/v1/chat/completions',
                      payload={'choices': [...]})
    provider = OpenRouterProvider(api_key="test-key")
    response = await provider.get_completion("test", "model")
    assert response.success is True
```

### 3. Setup Test Fixtures Directory (1 hour)
```bash
mkdir -p tests/fixtures/{api_responses,prompts,contexts,experiments,expected_outputs}
# Populate with realistic test data
```

### 4. Configure Coverage Reporting (1 hour)
```bash
# Already in pyproject.toml, just verify
pytest --cov=src/parallamr --cov-report=html
# Open htmlcov/index.html to see gaps
```

**Total Time**: ~8 hours for immediate improvements

---

## Critical Path: Provider Testing

### Why It's Critical
- Providers are external dependencies (high risk)
- Currently almost no unit tests (2/75)
- Integration tests skipped (require real APIs)
- Bug already found in Ollama provider

### Mock Strategy for Providers

#### OpenRouter Provider Tests (40 tests needed)
```python
# Mock aiohttp for all HTTP calls
from aioresponses import aioresponses

@pytest.fixture
def mock_openrouter():
    with aioresponses() as m:
        # Mock /chat/completions endpoint
        m.post('https://openrouter.ai/api/v1/chat/completions',
               payload={...})
        # Mock /models endpoint
        m.get('https://openrouter.ai/api/v1/models',
              payload={'data': [...]})
        yield m

# Test categories:
1. Authentication (8 tests)
   - Valid/invalid API key
   - Missing API key
   - Environment variable loading

2. Request handling (15 tests)
   - Successful completion
   - Rate limiting (429)
   - Context exceeded (413)
   - Network errors
   - Timeout errors

3. Model management (10 tests)
   - List models
   - Model availability
   - Context window retrieval
   - Caching behavior

4. Error handling (7 tests)
   - Connection errors
   - Malformed responses
   - Partial responses
```

#### Ollama Provider Tests (35 tests needed)
```python
@pytest.fixture
def mock_ollama():
    with aioresponses() as m:
        # Mock /api/generate endpoint
        m.post('http://localhost:11434/api/generate',
               payload={...})
        # Mock /api/tags endpoint
        m.get('http://localhost:11434/api/tags',
              payload={'models': [...]})
        # Mock /api/show endpoint
        m.post('http://localhost:11434/api/show',
               payload={'model_info': {...}})
        yield m

# Test categories:
1. Connection (6 tests)
2. Model operations (12 tests)
3. Context window (8 tests) ← BUG HERE
4. Completion requests (9 tests)
```

**Priority**: Fix Ollama context window bug first, add regression test

---

## Mocking Strategy Summary

### What to Mock
✅ **Always Mock**:
- External HTTP APIs (OpenRouter, Ollama)
- File system errors (permission, disk space)
- Network failures
- Time-sensitive operations (timeouts)
- Environment variables

❌ **Never Mock** (use real):
- Internal business logic
- Data models (Pydantic classes)
- Template processing
- Token counting
- CSV writing (use temp files)

### How to Mock
```python
# HTTP mocking (aiohttp)
from aioresponses import aioresponses

@pytest.fixture
def mock_http():
    with aioresponses() as m:
        m.post('https://api.example.com/endpoint',
               payload={'result': 'success'})
        yield m

# Function mocking (pytest-mock)
def test_with_mock(mocker):
    mocker.patch('module.function', return_value='mocked')

# Environment variables
def test_with_env(monkeypatch):
    monkeypatch.setenv('API_KEY', 'test-key')

# Time mocking (freezegun)
from freezegun import freeze_time

@freeze_time("2025-10-04 12:00:00")
def test_with_time():
    # Time is frozen
```

---

## Contract Testing

### Purpose
Ensure all providers implement consistent interface defined in `Provider` base class.

### Implementation Template
```python
# tests/contract/test_provider_contracts.py

class ProviderContractTests:
    """Base contract tests - apply to ALL providers"""

    @abstractmethod
    def get_provider(self):
        """Return provider instance"""
        pass

    @pytest.mark.asyncio
    async def test_get_completion_returns_provider_response(self):
        provider = self.get_provider()
        response = await provider.get_completion("test", "model")
        assert isinstance(response, ProviderResponse)

    @pytest.mark.asyncio
    async def test_response_has_required_fields(self):
        provider = self.get_provider()
        response = await provider.get_completion("test", "model")
        assert hasattr(response, 'output')
        assert hasattr(response, 'output_tokens')
        assert hasattr(response, 'success')

    # ... 18 more contract tests

class TestMockProviderContract(ProviderContractTests):
    def get_provider(self):
        return MockProvider()

class TestOpenRouterProviderContract(ProviderContractTests):
    def get_provider(self):
        # Returns mocked OpenRouter provider
        return OpenRouterProvider(api_key="test")

class TestOllamaProviderContract(ProviderContractTests):
    def get_provider(self):
        # Returns mocked Ollama provider
        return OllamaProvider()
```

**Result**: 20 contract tests × 3 providers = 60 tests with minimal code duplication

---

## E2E Testing Strategy

### Critical User Flows (10-15 tests)

#### Flow 1: First-time Setup
```bash
parallamr init
# Edit files
parallamr run -p prompt.txt -e experiments.csv -o results.csv
# Verify results.csv valid
```

#### Flow 2: Stdin/Stdout Pipeline
```bash
cat experiments.csv | parallamr run -p prompt.txt -e - > results.csv
cat prompt.txt | parallamr run -p - -e experiments.csv -o results.csv
```

#### Flow 3: Error Recovery
```csv
# Some experiments fail, some succeed
provider,model
invalid_provider,model  # This fails
mock,mock              # This succeeds
```
Verify: Failed experiments recorded with errors, successful ones have results

#### Flow 4: Large Batch Processing
```csv
# 100+ experiments
# Verify: Incremental writing works, no memory issues
```

### Implementation
```python
def test_complete_user_flow(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # init → modify → run → verify
        result = runner.invoke(cli, ['init'])
        # ... modify files to use mock provider
        result = runner.invoke(cli, ['run', '-p', 'prompt.txt', ...])
        assert result.exit_code == 0
        # Verify results.csv structure and content
```

---

## Test Organization Recommendations

### Current Structure (Flat)
```
tests/
├── test_cli.py           ✅ Good
├── test_csv_writer.py    ✅ Good
├── test_models.py        ✅ Good
├── test_providers.py     ⚠️ Minimal
├── test_runner.py        ✅ Good
├── test_template.py      ✅ Good
├── test_token_counter.py ✅ Good
├── test_utils.py         ✅ Good
└── fixtures/             ⚠️ Minimal
```

### Recommended Structure (Hierarchical)
```
tests/
├── conftest.py           # Shared fixtures
├── unit/                 # Fast, isolated tests
│   ├── test_template.py
│   ├── test_token_counter.py
│   ├── test_csv_writer.py
│   ├── test_models.py
│   ├── test_utils.py
│   ├── test_runner.py
│   └── providers/
│       ├── test_base.py
│       ├── test_mock.py
│       ├── test_openrouter.py  ← NEW
│       └── test_ollama.py      ← NEW
├── integration/          # Component interaction
│   ├── test_cli.py
│   ├── test_runner_integration.py
│   └── test_end_to_end.py
├── contract/             # Interface consistency
│   └── test_provider_contracts.py
├── performance/          # Speed benchmarks
│   ├── test_token_counting.py
│   ├── test_csv_writing.py
│   └── test_large_batches.py
└── fixtures/             # Test data
    ├── api_responses/    # Mock API responses
    ├── prompts/          # Test prompts
    ├── contexts/         # Test contexts
    ├── experiments/      # Test CSVs
    └── expected_outputs/ # Expected results
```

**Migration**: Can be done incrementally, no need to move all at once

---

## Performance Testing

### Benchmarks to Establish

#### Token Counting Performance
```python
def test_token_counting_large_text(benchmark):
    large_text = "This is a test. " * 10000  # ~160KB
    result = benchmark(estimate_tokens, large_text)
    assert result > 0
    # Target: < 10ms
```

#### CSV Writing Performance
```python
def test_csv_write_1000_results(benchmark):
    writer = IncrementalCSVWriter("test.csv")
    results = [create_test_result() for _ in range(1000)]

    def write_all():
        for result in results:
            writer.write_result(result)

    benchmark(write_all)
    # Target: < 1ms per write average
```

#### Template Processing Performance
```python
def test_template_large_file(benchmark):
    template = "{{var}}" * 1000
    variables = {"var": "value"}
    result = benchmark(replace_variables, template, variables)
    # Target: < 5ms
```

### Performance Regression Detection
- Run benchmarks on every PR
- Store historical results
- Alert if performance degrades > 10%

---

## Test Maintenance

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: tests
        name: Run tests
        entry: pytest tests/unit -x  # Fail fast
        language: system
        pass_filenames: false

      - id: type-check
        name: Type checking
        entry: mypy src/parallamr
        language: system
        pass_filenames: false
```

### CI/CD Pipeline
```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11', '3.12']

    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - name: Unit tests
        run: uv run pytest tests/unit -v --cov=src/parallamr
      - name: Integration tests
        run: uv run pytest tests/integration -v
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Monthly Checklist
- [ ] Review skipped tests (re-enable or remove)
- [ ] Review slow tests (optimize or mark)
- [ ] Update mock API responses (check for API changes)
- [ ] Review coverage gaps
- [ ] Update test documentation

---

## Success Metrics

### Coverage Targets
| Component | Current | Target | Priority |
|-----------|---------|--------|----------|
| Overall | ~75% | 90% | HIGH |
| Template | 95% | 95% | ✅ |
| Token Counter | 90% | 95% | MEDIUM |
| CSV Writer | 80% | 95% | HIGH |
| Models | 85% | 95% | MEDIUM |
| Providers | 20% | 95% | **CRITICAL** |
| Runner | 75% | 95% | HIGH |
| CLI | 85% | 85% | ✅ |
| Utils | 80% | 90% | MEDIUM |

### Test Quality Metrics
- **Test count**: 121 → 465 (3.8× increase)
- **Unit tests**: 113 → 250-300
- **Integration tests**: 19 → 50-80
- **Contract tests**: 0 → 60-80
- **E2E tests**: 0 → 10-15

### Speed Targets
- Unit tests: < 10 seconds total
- Integration tests: < 60 seconds total
- Full suite: < 2 minutes total
- Per-test average: < 50ms

### Quality Gates
- ✅ All tests must pass before merge
- ✅ Coverage cannot decrease
- ✅ No security vulnerabilities
- ✅ Type checking passes
- ✅ Linting passes

---

## Implementation Timeline

### Week 1: Foundation & Fixes
**Goal**: All tests passing, infrastructure ready

Tasks:
- [ ] Fix 6 failing tests
- [ ] Set up test directory structure
- [ ] Add fixture loading utilities
- [ ] Configure coverage reporting
- [ ] Set up CI/CD with coverage

**Deliverable**: Green test suite, baseline established

### Week 2: Provider Testing
**Goal**: 95% provider coverage

Tasks:
- [ ] Write OpenRouter unit tests (40 tests)
- [ ] Write Ollama unit tests (35 tests)
- [ ] Fix Ollama context window bug
- [ ] Add provider contract tests (60 tests)
- [ ] Mock API response fixtures

**Deliverable**: Providers fully tested with mocks

### Week 3: Component Integration
**Goal**: 90% integration coverage

Tasks:
- [ ] Component integration tests (25 tests)
- [ ] Enhanced CLI tests (15 tests)
- [ ] Runner integration tests (10 tests)
- [ ] API contract tests (20 tests)

**Deliverable**: All components working together

### Week 4: E2E & Performance
**Goal**: Critical flows validated

Tasks:
- [ ] E2E test scenarios (15 tests)
- [ ] Performance benchmarks (20 tests)
- [ ] Edge case coverage (30 tests)
- [ ] Security tests (10 tests)

**Deliverable**: Production-ready test suite

### Week 5: Documentation & Polish
**Goal**: Maintainable testing infrastructure

Tasks:
- [ ] Test documentation
- [ ] Developer testing guide
- [ ] CI/CD optimization
- [ ] Test maintenance procedures
- [ ] Performance optimization

**Deliverable**: Complete, documented test suite

---

## Key Recommendations

### Top 5 Priorities

1. **Fix failing tests immediately** (blocking all other work)
   - 6 tests failing = red test suite
   - Undermines confidence in testing

2. **Add comprehensive provider tests** (highest risk area)
   - External dependencies = highest failure risk
   - Currently almost no coverage
   - Use aioresponses for mocking

3. **Implement contract tests** (prevent regression)
   - Ensure interface consistency
   - Catch breaking changes early
   - Minimal code for maximum value

4. **Add E2E tests for critical flows** (user confidence)
   - First-time setup
   - Stdin/stdout pipelines
   - Error recovery
   - Large batches

5. **Establish performance benchmarks** (prevent regression)
   - Token counting speed
   - CSV writing speed
   - Template processing speed
   - Track over time

### Anti-Patterns to Avoid

❌ **Don't**:
- Use real API keys in tests
- Mock internal business logic
- Write tests that depend on execution order
- Hard-code expected strings (brittle)
- Skip tests indefinitely
- Test implementation details

✅ **Do**:
- Mock external dependencies only
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)
- Parametrize similar tests
- Use fixtures for common setup
- Keep tests isolated and deterministic

---

## Quick Reference Commands

```bash
# Run all tests
pytest tests/

# Run specific test level
pytest tests/unit/
pytest tests/integration/
pytest tests/contract/

# Run with coverage
pytest tests/ --cov=src/parallamr --cov-report=html

# Run specific test
pytest tests/test_file.py::TestClass::test_method

# Run fast tests only (skip slow)
pytest tests/ -m "not slow"

# Run in parallel (future)
pytest tests/ -n auto

# Run with verbose output
pytest tests/ -vv

# Run with debugging
pytest tests/ -s --pdb

# Check coverage gaps
pytest tests/ --cov=src/parallamr --cov-report=term-missing

# Run performance benchmarks
pytest tests/performance/ --benchmark-only
```

---

## Resources

### Required Libraries
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-mock>=3.11.0",      # NEW
    "aioresponses>=0.7.0",      # NEW
    "freezegun>=1.2.0",         # NEW (optional)
    "pytest-benchmark>=4.0.0",  # NEW (optional)
]
```

### Documentation
- Pytest: https://docs.pytest.org/
- pytest-asyncio: https://pytest-asyncio.readthedocs.io/
- aioresponses: https://github.com/pnuckowski/aioresponses
- Coverage.py: https://coverage.readthedocs.io/

### Internal Docs
- Full testing strategy: `analysis-reports/testing-strategy-comprehensive.md`
- Existing testing guide: `TEST_IMPLEMENTATION_GUIDE.md`
- Test fixtures spec: `TEST_FIXTURES_SPEC.md`

---

## Conclusion

**Current State**: Good foundation, critical gaps
**Target State**: Production-ready, comprehensive testing
**Effort Required**: 5 weeks, ~200 person-hours
**Risk Level**: Medium (failing tests, missing provider coverage)
**Recommendation**: Proceed with phased implementation

**Key Success Factors**:
1. Fix failing tests immediately (Week 1)
2. Focus on provider testing (Week 2)
3. Add contract tests for consistency (Week 3)
4. Validate critical E2E flows (Week 4)
5. Document and optimize (Week 5)

**Expected Outcome**:
- 465+ tests (from 121)
- 95%+ coverage (from ~75%)
- All providers thoroughly tested
- Critical user flows validated
- Performance benchmarks established
- Production-ready quality assurance

---

**Version**: 1.0
**Date**: 2025-10-04
**Author**: TESTER Agent (HiveMind Swarm)
**Status**: Ready for implementation
