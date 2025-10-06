# Parallamr

A command-line tool for running systematic experiments across multiple LLM providers and models, enabling prompt engineering and model comparison through parameterized testing.

## Features

- **Multi-Provider Support**: OpenRouter, Ollama, and Mock providers
- **Template Variables**: Dynamic prompt customization with `{{variable}}` syntax
- **Incremental CSV Output**: Real-time results writing for monitoring progress
- **Context Window Validation**: Automatic token counting and validation
- **Comprehensive Error Handling**: Graceful handling of API errors and rate limits
- **Flexible Configuration**: Environment-based configuration with .env support

## Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/parallamr.git
cd parallamr

# Install with uv
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Unix/macOS
# OR
.venv\Scripts\activate  # On Windows

# Install in development mode
uv pip install -e .
```

### Using pip

```bash
pip install -e .
```

## Quick Start

1. **Initialize example files**:
   ```bash
   parallamr init
   ```

2. **Configure API keys** (copy `.env.example` to `.env`):
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run experiments**:
   ```bash
   parallamr run -p prompt.txt -e experiments.csv -o results.csv
   ```

## Usage

### Basic Command

```bash
parallamr run --prompt prompt.txt --experiments experiments.csv --output results.csv
```

### With Context Files

```bash
parallamr run -p prompt.txt -c context1.txt -c context2.txt -e experiments.csv -o results.csv
```

### Validation Only

```bash
parallamr run -p prompt.txt -e experiments.csv -o results.csv --validate-only
```

### Verbose Output

```bash
parallamr run -p prompt.txt -e experiments.csv -o results.csv --verbose
```

### Templated Output Paths

Organize experiment results automatically by using template variables in output filenames:

```bash
# Separate files by topic
parallamr run -p prompt.txt -e experiments.csv -o "results-{{topic}}.csv"

# Organize by provider and model
parallamr run -p prompt.txt -e experiments.csv -o "{{provider}}/{{model}}-output.csv"

# Complex organization with subdirectories
parallamr run -p prompt.txt -e experiments.csv -o "experiments/{{date}}/{{provider}}/{{model}}-{{topic}}.csv"
```

**How it works:**
- Variables in output paths use `{{variable}}` syntax from your experiments CSV
- Experiments are automatically grouped by their resolved output path
- Directories are created automatically as needed
- Forbidden characters (like `/` in model names) are sanitized to `_`
- Each unique output path gets its own CSV file with corresponding experiments

**Example:**
```csv
# experiments.csv
provider,model,topic,date
openrouter,anthropic/claude-sonnet-4,AI,2024-01-15
openrouter,anthropic/claude-sonnet-4,Blockchain,2024-01-15
ollama,llama3.2,AI,2024-01-15
```

Running with `-o "{{provider}}/{{topic}}-results.csv"` creates:
```
openrouter/AI-results.csv        (1 experiment)
openrouter/Blockchain-results.csv (1 experiment)
ollama/AI-results.csv             (1 experiment)
```

**Security:**
- Path traversal attempts (`../`) are detected and sanitized
- Windows reserved names (CON, PRN, etc.) are handled safely
- Filenames are limited to 255 characters
- Cross-platform compatible (Windows, macOS, Linux)

## File Formats

### Prompt File (prompt.txt)

```
Summarize the following information from {{source}}:

The topic is {{topic}}.

Please provide a concise summary that covers the key concepts and applications.
```

### Experiments CSV (experiments.csv)

```csv
provider,model,source,topic
mock,mock,Wikipedia,AI
openrouter,anthropic/claude-sonnet-4,Encyclopedia,Machine Learning
ollama,llama3.2,Database,Natural Language Processing
```

**Required columns:**
- `provider`: LLM provider (mock, openrouter, ollama)
- `model`: Model identifier

**Optional columns:**
- Any other column becomes a template variable for `{{variable}}` replacement

### Output CSV (results.csv)

The output CSV contains all input columns plus:

| Column | Description |
|--------|-------------|
| `status` | `ok`, `warning`, or `error` |
| `input_tokens` | Estimated input token count |
| `context_window` | Model's context window (if available) |
| `output_tokens` | Response token count |
| `output` | Model response text |
| `error_message` | Error details (if any) |

## Configuration

### Environment Variables (.env)

```bash
# OpenRouter
OPENROUTER_API_KEY=sk-or-v1-your-api-key-here

# Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Optional: Request timeout
DEFAULT_TIMEOUT=300
```

### Provider Configuration

- **openrouter**: Requires `OPENROUTER_API_KEY`
- **ollama**: Requires `OLLAMA_BASE_URL` (default: http://localhost:11434)
- **mock**: No configuration required (for testing)

## Available Commands

### List Providers

```bash
parallamr providers
```

### List Models

```bash
parallamr models openrouter
parallamr models ollama
```

### Create Example Files

```bash
parallamr init [--output experiments.csv]
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/parallamr --cov-report=html

# Run specific test file
pytest tests/test_models.py
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff src/ tests/

# Type checking
mypy src/
```

### Project Structure

```
parallamr/
├── src/
│   └── parallamr/
│       ├── __init__.py
│       ├── cli.py              # CLI interface
│       ├── runner.py           # Experiment orchestration
│       ├── template.py         # Variable replacement
│       ├── token_counter.py    # Token estimation
│       ├── csv_writer.py       # Incremental CSV output
│       ├── models.py           # Data models
│       ├── utils.py            # Utilities
│       └── providers/
│           ├── __init__.py
│           ├── base.py         # Provider interface
│           ├── openrouter.py   # OpenRouter implementation
│           ├── ollama.py       # Ollama implementation
│           └── mock.py         # Mock provider
└── tests/
    ├── fixtures/               # Test data
    ├── test_*.py              # Test modules
    └── ...
```

## Features in Detail

### Template Variables

Use `{{variable}}` syntax in prompt and context files:

```
Hello {{name}}, please analyze {{topic}} using data from {{source}}.
```

Variables are replaced with values from the experiments CSV columns.

### Token Counting

- Uses character count ÷ 4 approximation
- Provider-agnostic estimation
- Context window validation with warnings

### Error Handling

- **OK**: Successful execution
- **WARNING**: Successful with issues (missing variables, context warnings)
- **ERROR**: Failed execution (invalid model, API error, timeout)

### Incremental Output

Results are written to CSV after each experiment completes:
- Real-time monitoring possible
- No data loss if interrupted
- Immediate feedback on errors

## Examples

### Example 1: Model Comparison

```csv
provider,model,task,domain
openrouter,anthropic/claude-sonnet-4,summarization,medical
openrouter,meta-llama/llama-3.1-70b-instruct,summarization,medical
ollama,llama3.2,summarization,medical
```

### Example 2: Prompt Variations

```csv
provider,model,style,tone
mock,mock,formal,professional
mock,mock,casual,friendly
mock,mock,technical,detailed
```

### Example 3: Multi-Provider Testing

```csv
provider,model,temperature,max_tokens
openrouter,anthropic/claude-sonnet-4,0.7,1000
ollama,llama3.2,0.7,1000
mock,mock,0.7,1000
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- Report issues on GitHub
- Check documentation for common questions
- Use `--verbose` flag for detailed logging

## Version History

### v0.1.0
- Initial implementation
- Multi-provider support (OpenRouter, Ollama, Mock)
- Template variable system
- Incremental CSV output
- Comprehensive test suite
- CLI interface with validation