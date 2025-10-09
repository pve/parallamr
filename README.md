# Parallamr

A command-line tool for running systematic experiments across multiple LLM providers and models, enabling prompt engineering and model comparison through parameterized testing.

## Features

- **Multi-Provider Support**: OpenAI, OpenRouter, Ollama, and Mock providers
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

### Templated File Paths

Organize experiments automatically by using template variables in **any file path** (prompt, context, or output):

#### Templated Outputs

```bash
# Separate output files by topic
parallamr run -p prompt.txt -e experiments.csv -o "results-{{topic}}.csv"

# Organize by provider and model
parallamr run -p prompt.txt -e experiments.csv -o "{{provider}}/{{model}}-output.csv"
```

#### Templated Prompts

```bash
# Different prompts per topic
parallamr run -p "prompts/{{topic}}-prompt.txt" -e experiments.csv -o results.csv"

# Model-specific prompts
parallamr run -p "prompts/{{model}}-instructions.txt" -e experiments.csv -o results.csv
```

#### Templated Contexts

```bash
# Topic-specific context files
parallamr run -p prompt.txt -c "context/{{topic}}-background.txt" -e experiments.csv -o results.csv

# Multiple templated contexts
parallamr run -p prompt.txt \
              -c "context/{{topic}}-info.txt" \
              -c "guides/{{provider}}-guide.txt" \
              -e experiments.csv \
              -o results.csv
```

#### Combined Templating

```bash
# Everything templated for maximum organization
parallamr run -p "prompts/{{topic}}/prompt.txt" \
              -c "context/{{topic}}-background.txt" \
              -c "guides/{{provider}}-guide.txt" \
              -e experiments.csv \
              -o "outputs/{{date}}/{{provider}}/{{model}}-{{topic}}.csv"
```

**How it works:**
- Variables use `{{variable}}` syntax from your experiments CSV columns
- For outputs: Experiments are grouped by resolved path; each group gets its own file
- For inputs: Each experiment loads its own files based on its variable values
- Directories are created automatically as needed
- Forbidden characters (like `/` in model names) are sanitized to `_`

**Example:**
```csv
# experiments.csv
provider,model,topic,date
openrouter,anthropic/claude-sonnet-4,AI,2024-01-15
openrouter,anthropic/claude-sonnet-4,Blockchain,2024-01-15
ollama,llama3.2,AI,2024-01-15
```

Running with templated paths creates organized structure:
```bash
parallamr run \
  -p "prompts/{{topic}}-prompt.txt" \
  -e experiments.csv \
  -o "{{provider}}/{{topic}}-results.csv"
```

Results in:
```
prompts/AI-prompt.txt               → Used for experiments with topic=AI
prompts/Blockchain-prompt.txt       → Used for experiments with topic=Blockchain

openrouter/AI-results.csv           → 1 experiment output
openrouter/Blockchain-results.csv   → 1 experiment output
ollama/AI-results.csv               → 1 experiment output
```

**Error Handling:**
- Missing input files create error results (experiments continue)
- Missing variables in templates raise clear errors
- Path traversal attempts are detected and blocked

**Security:**
- Path traversal attempts (`../`) are detected and sanitized in filenames
- Directory traversal in template structure is validated
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
- `provider`: LLM provider (openai, openrouter, ollama, mock)
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
# OpenAI
OPENAI_API_KEY=sk-proj-your-api-key-here

# OpenRouter
OPENROUTER_API_KEY=sk-or-v1-your-api-key-here

# Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Optional: Request timeout
DEFAULT_TIMEOUT=300
```

### Provider Configuration

#### OpenAI

Supports official OpenAI API and OpenAI-compatible providers (Azure OpenAI, LocalAI, Together AI, Groq, etc.).

**Basic Configuration:**
```bash
OPENAI_API_KEY=sk-proj-your-key-here
```

**Azure OpenAI Configuration:**
```bash
OPENAI_API_KEY=your-azure-key
OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment
```

**LocalAI / Together AI / Other Compatible Providers:**
```bash
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=http://localhost:8080/v1  # LocalAI
# OR
OPENAI_BASE_URL=https://api.together.xyz/v1  # Together AI
```

**Supported Models:**
- GPT-4o: `gpt-4o`, `gpt-4o-mini` (128K context)
- GPT-4 Turbo: `gpt-4-turbo`, `gpt-4-turbo-preview` (128K context)
- GPT-4: `gpt-4`, `gpt-4-32k` (8K / 32K context)
- GPT-3.5: `gpt-3.5-turbo`, `gpt-3.5-turbo-16k` (16K context)

**Example Usage:**
```csv
provider,model,topic
openai,gpt-4o-mini,Machine Learning
openai,gpt-4-turbo,Neural Networks
openai,gpt-3.5-turbo,Deep Learning
```

#### OpenRouter

- **openrouter**: Requires `OPENROUTER_API_KEY`

#### Ollama

- **ollama**: Requires `OLLAMA_BASE_URL` (default: http://localhost:11434)

#### Mock

- **mock**: No configuration required (for testing)

## Available Commands

### List Providers

```bash
parallamr providers
```

### List Models

```bash
parallamr models openai
parallamr models openrouter
parallamr models ollama
```

### Create Example Files

```bash
parallamr init [--output experiments.csv]
```

## Troubleshooting

### OpenAI Provider

**"Authentication failed - invalid API key"**
- Verify your API key starts with `sk-proj-` or `sk-`
- Check that `OPENAI_API_KEY` is set in your `.env` file
- Ensure the API key has not expired

**"Rate limit exceeded - please wait and retry"**
- You've hit OpenAI's rate limits for your tier
- Wait a few minutes and try again
- Consider upgrading your OpenAI tier for higher limits
- Reduce the number of parallel experiments

**"Request too large - input exceeds model context window"**
- Your prompt + context files exceed the model's context window
- Use a model with a larger context window (e.g., `gpt-4-turbo` with 128K)
- Reduce the size of your prompt or context files
- Check token counts with `--validate-only`

**"Model or endpoint not found"**
- Verify the model name is correct (e.g., `gpt-4o-mini`, not `gpt-4-mini`)
- Check that the model is available for your API key
- For Azure OpenAI, ensure your deployment name is correct

**Azure OpenAI Connection Issues**
- Verify your `OPENAI_BASE_URL` includes the full deployment path
- Ensure you're using your Azure API key, not an OpenAI key
- Check that the `api-version` parameter is in the URL if required

**LocalAI / Together AI Connection Issues**
- Verify the base URL is accessible (`curl http://localhost:8080/v1/models`)
- Check that the service is running and accepting connections
- Ensure the API key format matches the provider's requirements

### General Issues

**"No module named 'parallamr'"**
- Install the package: `pip install -e .`
- Activate your virtual environment

**CSV Formatting Errors**
- Ensure your experiments.csv has `provider` and `model` columns
- Check for proper CSV quoting if cells contain commas
- Verify the file encoding is UTF-8

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
provider,model
openrouter,anthropic/claude-sonnet-4
ollama,llama3.3:latest
mock,mock
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## using claude-flow

What seems to work quite nicely is, for example:

```
npx claude-flow@alpha hive-mind spawn "implement git issue 21" --claude
```

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