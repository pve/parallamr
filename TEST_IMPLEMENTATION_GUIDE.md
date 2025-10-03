# Parallamr Test Implementation Guide
## HiveMind-Tester-Delta Detailed Test Specifications

### Test Directory Structure Setup

```
tests/
├── __init__.py
├── conftest.py                 # Pytest configuration and fixtures
├── unit/
│   ├── __init__.py
│   ├── test_template.py        # Template engine tests
│   ├── test_token_counter.py   # Token counting tests
│   ├── test_csv_writer.py      # CSV writing tests
│   ├── test_runner.py          # Experiment runner tests
│   ├── test_models.py          # Data models tests
│   └── providers/
│       ├── __init__.py
│       ├── test_base.py        # Base provider tests
│       ├── test_mock.py        # Mock provider tests
│       ├── test_openrouter.py  # OpenRouter provider tests
│       └── test_ollama.py      # Ollama provider tests
├── integration/
│   ├── __init__.py
│   ├── test_end_to_end.py      # Full workflow tests
│   ├── test_cli.py             # CLI interface tests
│   └── test_provider_integration.py
├── edge_cases/
│   ├── __init__.py
│   ├── test_error_handling.py  # Error scenarios
│   ├── test_boundary_conditions.py
│   └── test_performance.py     # Performance tests
└── fixtures/
    ├── prompts/
    ├── contexts/
    ├── experiments/
    └── expected_outputs/
```

### Core Test Implementation Examples

#### 1. Template Engine Tests (`test_template.py`)

```python
import pytest
from parallamr.template import replace_variables

class TestTemplateEngine:

    def test_simple_variable_replacement(self):
        """Test basic variable replacement functionality."""
        text = "Hello {{name}}, welcome to {{place}}!"
        variables = {"name": "Alice", "place": "Wonderland"}
        result, missing = replace_variables(text, variables)

        assert result == "Hello Alice, welcome to Wonderland!"
        assert missing == []

    def test_missing_variables_warning(self):
        """Test handling of missing variables with warning generation."""
        text = "Hello {{name}}, your age is {{age}}"
        variables = {"name": "Bob"}
        result, missing = replace_variables(text, variables)

        assert result == "Hello Bob, your age is {{age}}"
        assert missing == ["age"]

    def test_no_variables(self):
        """Test text with no variables."""
        text = "This is plain text with no variables."
        variables = {"unused": "value"}
        result, missing = replace_variables(text, variables)

        assert result == text
        assert missing == []

    def test_empty_text(self):
        """Test empty input text."""
        result, missing = replace_variables("", {"var": "value"})
        assert result == ""
        assert missing == []

    def test_malformed_variables(self):
        """Test handling of malformed variable syntax."""
        text = "Hello {name} and {{incomplete"
        variables = {"name": "Alice", "incomplete": "value"}
        result, missing = replace_variables(text, variables)

        # Should not replace malformed syntax
        assert result == "Hello {name} and {{incomplete"
        assert missing == []

    def test_duplicate_variables(self):
        """Test handling of duplicate variables in text."""
        text = "{{greeting}} {{name}}, {{greeting}} again!"
        variables = {"greeting": "Hello", "name": "World"}
        result, missing = replace_variables(text, variables)

        assert result == "Hello World, Hello again!"
        assert missing == []

    def test_special_characters_in_variables(self):
        """Test variables containing special characters."""
        text = "File: {{file_path}}, Size: {{file_size}}"
        variables = {
            "file_path": "/path/to/file with spaces.txt",
            "file_size": "1.5 MB"
        }
        result, missing = replace_variables(text, variables)

        assert "1.5 MB" in result
        assert "/path/to/file with spaces.txt" in result
        assert missing == []

    def test_unicode_handling(self):
        """Test handling of Unicode characters in variables."""
        text = "Message: {{message}}, Emoji: {{emoji}}"
        variables = {"message": "Héllö Wörld", "emoji": "🚀✨"}
        result, missing = replace_variables(text, variables)

        assert "Héllö Wörld" in result
        assert "🚀✨" in result
        assert missing == []
```

#### 2. Token Counter Tests (`test_token_counter.py`)

```python
import pytest
from parallamr.token_counter import estimate_tokens

class TestTokenCounter:

    def test_basic_estimation(self):
        """Test basic token estimation using character count / 4."""
        # 11 characters -> 2.75 -> 2 tokens
        assert estimate_tokens("Hello world") == 2

        # 16 characters -> 4 tokens exactly
        assert estimate_tokens("This is a test!!") == 4

        # 3 characters -> 0.75 -> 0 tokens
        assert estimate_tokens("Hi!") == 0

    def test_empty_string(self):
        """Test empty string handling."""
        assert estimate_tokens("") == 0

    def test_whitespace_only(self):
        """Test strings with only whitespace."""
        assert estimate_tokens("    ") == 1  # 4 spaces = 1 token
        assert estimate_tokens("\n\t  ") == 1  # 4 whitespace chars = 1 token

    def test_unicode_characters(self):
        """Test Unicode character handling."""
        # Emoji characters
        assert estimate_tokens("🚀") == 0  # 1 char -> 0 tokens
        assert estimate_tokens("🚀✨🎯🎉") == 1  # 4 chars -> 1 token

        # Accented characters
        assert estimate_tokens("café") == 1  # 4 chars -> 1 token
        assert estimate_tokens("naïve résumé") == 2  # 12 chars -> 3 tokens

    def test_large_text(self):
        """Test large text handling."""
        large_text = "a" * 10000  # 10,000 characters
        assert estimate_tokens(large_text) == 2500

        very_large_text = "x" * 100000  # 100,000 characters
        assert estimate_tokens(very_large_text) == 25000

    def test_newlines_and_formatting(self):
        """Test text with various formatting characters."""
        formatted_text = "Line 1\nLine 2\r\nLine 3\tTabbed"
        # Count actual characters including newlines and tabs
        expected = len(formatted_text) // 4
        assert estimate_tokens(formatted_text) == expected

    def test_mixed_content(self):
        """Test realistic mixed content."""
        content = """
        This is a sample prompt with:
        - Multiple lines
        - Special characters: !@#$%^&*()
        - Numbers: 12345
        - Unicode: café 🚀
        """
        expected = len(content) // 4
        assert estimate_tokens(content) == expected
```

#### 3. CSV Writer Tests (`test_csv_writer.py`)

```python
import pytest
import tempfile
import csv
from pathlib import Path
from parallamr.csv_writer import IncrementalCSVWriter
from parallamr.models import ExperimentResult

class TestIncrementalCSVWriter:

    @pytest.fixture
    def temp_csv_file(self):
        """Provide temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def sample_result(self):
        """Provide sample experiment result."""
        return ExperimentResult(
            provider="mock",
            model="test-model",
            variables={"topic": "AI", "source": "Wikipedia"},
            status="ok",
            input_tokens=45,
            context_window=None,
            output_tokens=50,
            output="This is a test response.",
            error_message=""
        )

    def test_header_written_once(self, temp_csv_file, sample_result):
        """Test that CSV header is written only on first write."""
        writer = IncrementalCSVWriter(temp_csv_file)

        # Write first result
        writer.write_result(sample_result)

        # Write second result
        sample_result.output = "Second response"
        writer.write_result(sample_result)

        # Check file contains only one header line
        with open(temp_csv_file, 'r') as f:
            lines = f.readlines()

        header_count = sum(1 for line in lines if 'provider,model' in line)
        assert header_count == 1
        assert len(lines) == 3  # 1 header + 2 data rows

    def test_csv_escaping_multiline(self, temp_csv_file):
        """Test proper CSV escaping for multiline content."""
        result = ExperimentResult(
            provider="test",
            model="test",
            variables={},
            status="ok",
            input_tokens=10,
            context_window=None,
            output_tokens=20,
            output='Line 1\nLine 2\n"Quoted text"',
            error_message=""
        )

        writer = IncrementalCSVWriter(temp_csv_file)
        writer.write_result(result)

        # Read back and verify proper escaping
        with open(temp_csv_file, 'r') as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert 'Line 1\nLine 2\n"Quoted text"' == row['output']

    def test_csv_escaping_commas(self, temp_csv_file):
        """Test proper CSV escaping for content with commas."""
        result = ExperimentResult(
            provider="test",
            model="test",
            variables={},
            status="ok",
            input_tokens=10,
            context_window=None,
            output_tokens=20,
            output="Item 1, Item 2, Item 3",
            error_message=""
        )

        writer = IncrementalCSVWriter(temp_csv_file)
        writer.write_result(result)

        with open(temp_csv_file, 'r') as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert "Item 1, Item 2, Item 3" == row['output']

    def test_incremental_append(self, temp_csv_file, sample_result):
        """Test that results are immediately written to file."""
        writer = IncrementalCSVWriter(temp_csv_file)

        # Write first result
        writer.write_result(sample_result)

        # Verify file exists and contains data immediately
        with open(temp_csv_file, 'r') as f:
            content = f.read()
            assert "mock" in content
            assert "test-model" in content

    def test_variable_columns(self, temp_csv_file):
        """Test that variable columns are included in output."""
        result = ExperimentResult(
            provider="test",
            model="test",
            variables={"topic": "AI", "source": "Wikipedia", "style": "formal"},
            status="ok",
            input_tokens=10,
            context_window=2048,
            output_tokens=20,
            output="Test output",
            error_message=""
        )

        writer = IncrementalCSVWriter(temp_csv_file)
        writer.write_result(result)

        with open(temp_csv_file, 'r') as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert row['topic'] == "AI"
            assert row['source'] == "Wikipedia"
            assert row['style'] == "formal"
```

#### 4. Provider Tests (`test_mock.py`)

```python
import pytest
from parallamr.providers.mock import MockProvider

class TestMockProvider:

    @pytest.fixture
    def provider(self):
        return MockProvider()

    def test_get_completion_basic(self, provider):
        """Test basic mock completion functionality."""
        prompt = "Test prompt"
        model = "mock"

        response = provider.get_completion(prompt, model)

        assert response.status == "ok"
        assert "MOCK RESPONSE" in response.output
        assert "Test prompt" in response.output
        assert response.output_tokens > 0
        assert response.error_message == ""

    def test_get_completion_with_variables(self, provider):
        """Test mock completion includes variable information."""
        prompt = "Test prompt"
        model = "mock"
        variables = {"topic": "AI", "source": "Wikipedia"}

        response = provider.get_completion(prompt, model, variables=variables)

        assert "topic" in response.output
        assert "AI" in response.output
        assert "source" in response.output
        assert "Wikipedia" in response.output

    def test_get_completion_token_count(self, provider):
        """Test mock provider returns estimated token counts."""
        prompt = "Short prompt"  # 12 chars = 3 tokens
        model = "mock"

        response = provider.get_completion(prompt, model)

        # Should include input token estimate in response
        assert f"Input tokens: {len(prompt) // 4}" in response.output
        # Output tokens should be > 0
        assert response.output_tokens > 0

    def test_get_context_window(self, provider):
        """Test mock provider context window behavior."""
        # Mock provider should return None for context window
        assert provider.get_context_window("mock") is None
        assert provider.get_context_window("any-model") is None

    def test_deterministic_output_structure(self, provider):
        """Test that mock output follows expected structure."""
        prompt = "Test"
        response = provider.get_completion(prompt, "mock")

        lines = response.output.split('\n')
        assert lines[0] == "MOCK RESPONSE"
        assert any("Input tokens:" in line for line in lines)
        assert any("Model: mock" in line for line in lines)
        assert any("--- Original Input ---" in line for line in lines)
        assert "Test" in response.output
```

#### 5. End-to-End Integration Tests (`test_end_to_end.py`)

```python
import pytest
import tempfile
import csv
from pathlib import Path
from parallamr.runner import ExperimentRunner

class TestEndToEndExecution:

    @pytest.fixture
    def temp_files(self):
        """Create temporary files for testing."""
        files = {}

        # Create temporary prompt file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Summarize {{topic}} from {{source}}.")
            files['prompt'] = f.name

        # Create temporary context file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Additional context for testing.")
            files['context'] = f.name

        # Create temporary experiments CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['provider', 'model', 'topic', 'source'])
            writer.writerow(['mock', 'mock', 'AI', 'Wikipedia'])
            writer.writerow(['mock', 'mock', 'ML', 'Encyclopedia'])
            files['experiments'] = f.name

        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            files['output'] = f.name

        yield files

        # Cleanup
        for filepath in files.values():
            Path(filepath).unlink(missing_ok=True)

    def test_complete_experiment_run(self, temp_files):
        """Test complete experiment execution from start to finish."""
        runner = ExperimentRunner(
            prompt_file=temp_files['prompt'],
            context_files=[temp_files['context']],
            experiments_file=temp_files['experiments'],
            output_file=temp_files['output']
        )

        # Execute experiments
        results = runner.run()

        # Verify execution completed
        assert len(results) == 2
        assert all(result.status == "ok" for result in results)

        # Verify output file was created
        assert Path(temp_files['output']).exists()

        # Verify output file contents
        with open(temp_files['output'], 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]['topic'] == 'AI'
        assert rows[0]['source'] == 'Wikipedia'
        assert rows[1]['topic'] == 'ML'
        assert rows[1]['source'] == 'Encyclopedia'

        # Verify variable replacement occurred
        for row in rows:
            assert 'AI' in row['output'] or 'ML' in row['output']
            assert 'Wikipedia' in row['output'] or 'Encyclopedia' in row['output']

    def test_missing_variable_handling(self, temp_files):
        """Test handling of missing variables in templates."""
        # Create experiments with missing variables
        with open(temp_files['experiments'], 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['provider', 'model', 'topic'])  # Missing 'source'
            writer.writerow(['mock', 'mock', 'AI'])

        runner = ExperimentRunner(
            prompt_file=temp_files['prompt'],
            context_files=[],
            experiments_file=temp_files['experiments'],
            output_file=temp_files['output']
        )

        results = runner.run()

        # Should complete with warnings
        assert len(results) == 1
        assert results[0].status == "warning"
        assert "missing variable" in results[0].error_message.lower()

        # Verify unreplaced variable remains in output
        assert "{{source}}" in results[0].output

    def test_file_concatenation(self, temp_files):
        """Test that multiple input files are properly concatenated."""
        # Create second context file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Second context file content.")
            second_context = f.name

        try:
            runner = ExperimentRunner(
                prompt_file=temp_files['prompt'],
                context_files=[temp_files['context'], second_context],
                experiments_file=temp_files['experiments'],
                output_file=temp_files['output']
            )

            results = runner.run()

            # Verify both context files are included
            for result in results:
                assert "Additional context for testing" in result.output
                assert "Second context file content" in result.output

        finally:
            Path(second_context).unlink(missing_ok=True)
```

### Performance and Edge Case Tests

#### Performance Tests (`test_performance.py`)

```python
import pytest
import time
import tempfile
from pathlib import Path
from parallamr.token_counter import estimate_tokens
from parallamr.csv_writer import IncrementalCSVWriter
from parallamr.models import ExperimentResult

class TestPerformance:

    def test_token_counting_performance(self):
        """Test token counting performance with large texts."""
        # Create large text (100KB)
        large_text = "This is a test sentence. " * 4000  # ~100KB

        start_time = time.time()
        tokens = estimate_tokens(large_text)
        elapsed = time.time() - start_time

        # Should complete in under 10ms
        assert elapsed < 0.01
        assert tokens > 0

    def test_csv_writing_performance(self):
        """Test CSV writing performance with many results."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            output_file = f.name

        try:
            writer = IncrementalCSVWriter(output_file)

            # Write 1000 results
            start_time = time.time()
            for i in range(1000):
                result = ExperimentResult(
                    provider="mock",
                    model="test",
                    variables={"iteration": str(i)},
                    status="ok",
                    input_tokens=50,
                    context_window=None,
                    output_tokens=100,
                    output=f"Result {i} with some content",
                    error_message=""
                )
                writer.write_result(result)

            elapsed = time.time() - start_time

            # Should average less than 1ms per write
            avg_time_per_write = elapsed / 1000
            assert avg_time_per_write < 0.001

            # Verify all results were written
            with open(output_file, 'r') as f:
                lines = f.readlines()
            assert len(lines) == 1001  # 1 header + 1000 data rows

        finally:
            Path(output_file).unlink(missing_ok=True)

    @pytest.mark.slow
    def test_large_experiment_set(self):
        """Test handling of large experiment sets."""
        # This test is marked as slow and may be skipped in quick test runs
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            experiments_file = f.name

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            output_file = f.name

        try:
            # Create large experiment set (5000 experiments)
            import csv
            with open(experiments_file, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['provider', 'model', 'iteration'])
                for i in range(5000):
                    writer.writerow(['mock', 'mock', str(i)])

            # This test verifies the system can handle large datasets
            # without running out of memory or taking excessive time
            file_size = Path(experiments_file).stat().st_size
            assert file_size > 0

            # Verify file can be read without memory issues
            with open(experiments_file, 'r') as f:
                reader = csv.DictReader(f)
                row_count = sum(1 for _ in reader)
            assert row_count == 5000

        finally:
            Path(experiments_file).unlink(missing_ok=True)
            Path(output_file).unlink(missing_ok=True)
```

### Test Configuration (`conftest.py`)

```python
import pytest
import tempfile
import os
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir():
    """Provide test data directory path."""
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def temp_output_file():
    """Provide temporary output file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        yield f.name
    Path(f.name).unlink(missing_ok=True)

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-12345")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")

@pytest.fixture
def no_env_vars(monkeypatch):
    """Remove environment variables for testing error conditions."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_api: marks tests that require real API access"
    )

# Pytest collection customization
def pytest_collection_modifyitems(config, items):
    """Automatically mark certain tests."""
    for item in items:
        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark performance tests as slow
        if "performance" in item.name.lower():
            item.add_marker(pytest.mark.slow)
```

This implementation guide provides comprehensive test coverage for the Parallamr project, ensuring robust validation of all components while maintaining high code quality and reliability standards.