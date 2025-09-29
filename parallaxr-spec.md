# Parallaxr Specification

## Overview
Parallaxr is a command-line tool for running systematic experiments across multiple LLM providers and models, enabling prompt engineering and model comparison through parameterized testing.

## Core Functionality

### Input Processing
1. **Multiple File Context**
   - Primary prompt file (required)
   - Additional context files (optional)
   - Files are concatenated with separators: `## Document: {filename}\n\n{content}\n\n---\n\n`

2. **Template Variable Replacement**
   - Variables use double-brace syntax: `{{variable_name}}`
   - Variables are replaced across ALL input files
   - Replacement happens before sending to LLM
   - Missing variables trigger warnings but don't halt execution

### Experiment Configuration
**CSV Format** with required and optional columns:

**Required columns:**
- `provider`: LLM provider (ollama, openrouter, mock)
- `model`: Model identifier

**Optional columns:**
- Any other column name becomes a template variable
- Values in each row replace corresponding `{{column_name}}` in documents

**Example experiments.csv:**
```csv
provider,model,source
openrouter,anthropic/claude-sonnet-4,Wikipedia
ollama,llama3.2,Encyclopedia
mock,mock,TestData
```

### Output Format
**CSV file** with all input columns plus:

| Column | Description | Values |
|--------|-------------|--------|
| `status` | Execution status | `ok`, `warning`, `error` |
| `input_tokens` | Estimated input size | Integer |
| `context_window` | Model's max context (if available) | Integer or empty |
| `output_tokens` | Response token count | Integer |
| `output` | Model response text | String (CSV escaped) |
| `error_message` | Error details (if status != ok) | String or empty |

**Status definitions:**
- `ok`: Successful execution, no issues
- `warning`: Successful but issues detected (e.g., context window unknown, rate limiting, missing variables, context overflow)
- `error`: Failed execution (missing/invalid model, API error, timeout, authentication failure)

**Output Behavior:**
- CSV output file is written/appended after EACH experiment completes
- Allows real-time monitoring of progress
- Enables recovery if process is interrupted
- Headers written on first write only

## Configuration

### Environment Variables (.env file)
```bash
# OpenRouter
OPENROUTER_API_KEY=sk-or-v1-...

# Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Optional: UV virtual environment path
# UV_PROJECT_ENVIRONMENT=.venv-devcon

# Optional: For Docker/devcontainer
# OLLAMA_BASE_URL=http://host.docker.internal:11434
```

### Provider Configuration
- **openrouter**: Requires `OPENROUTER_API_KEY`
- **ollama**: Requires `OLLAMA_BASE_URL` (default: http://localhost:11434)
- **mock**: No configuration required, for testing only

## Command-Line Interface

### Basic Usage
```bash
parallaxr run --prompt prompt.txt --experiments experiments.csv --output results.csv file1 file2 ...
```

### Full Options
```bash
parallaxr run \
  --prompt PROMPT_FILE \
  --experiments EXPERIMENTS_FILE \
  --output OUTPUT_FILE \
  [--context CONTEXT_FILE ...] \
  [--verbose] \
  [--timeout SECONDS]
```

### Arguments
- `--prompt`, `-p`: Primary prompt file (required)
- `--experiments`, `-e`: Experiments CSV file (required)
- `--output`, `-o`: Output CSV file (required)
- `--context`, `-c`: Additional context files (multiple allowed)
- `--verbose`, `-v`: Enable detailed logging
- `--timeout`: Request timeout in seconds (default: 300)

## Design Decisions

### 1. **Token Counting Strategy**
- **Decision**: Character count / 4 approximation
- **Rationale**: Simple, provider-agnostic, sufficient for estimation
- **Implementation**: `estimated_tokens = len(text) // 4`

### 2. **Context Window Detection**
- **Decision**: Query provider API for model context windows
- **Fallback**: If unavailable, leave `context_window` column empty and set status to `warning`
- **Warning Message**: "Context window unknown for model {model}"

### 3. **Rate Limiting & Retry Logic**
- **Decision**: No automatic retries
- **Rate Limit Handling**: Set status to `warning`, include rate limit info in `error_message`
- **Rationale**: Explicit control, predictable behavior, avoids unexpected delays/costs

### 4. **Parallel Execution**
- **Decision**: No parallel execution in v1
- **Rationale**: Simpler implementation, easier debugging, avoids rate limit complexity
- **Sequential Processing**: One experiment at a time

### 5. **Output CSV Escaping**
- **Decision**: Standard CSV escaping using Python's `csv` module
- **Implementation**: 
  - Quotes around fields containing commas, newlines, or quotes
  - Double-quotes to escape quotes within fields
  - Handles multiline LLM outputs correctly

### 6. **Progress Indication**
- **Decision**: No built-in progress indicators in v1
- **Rationale**: Real-time CSV appending allows external monitoring (`tail -f results.csv`)
- **Verbose Mode**: Logs experiment start/completion to stderr

### 7. **Mock Provider Behavior**
- **Decision**: Simple echo implementation
- **Behavior**: Returns formatted string with input metadata
- **Output Format**: 
  ```
  MOCK RESPONSE
  Input tokens: {estimated_tokens}
  Model: mock
  Variables: {variable_dict}
  --- Original Input ---
  {input_text}
  ```

### 8. **Invalid Model Handling**
- **Decision**: Skip invalid models, continue with remaining experiments
- **Behavior**: 
  - Set status to `error`
  - Log error message: "Model {model} not found or unavailable"
  - Write row to output CSV
  - Continue to next experiment

### 9. **Missing Variable Handling**
- **Decision**: Warn but continue execution
- **Behavior**:
  - Log warning: "Variable {{variable}} in template has no value in experiment row {row_num}"
  - Leave `{{variable}}` unreplaced in text
  - Set status to `warning`
  - Include warning in `error_message` column
  - Continue with experiment

### 10. **Cost Tracking**
- **Decision**: No built-in cost tracking in v1
- **Rationale**: OpenRouter provides cost tracking in their dashboard
- **Future**: Could be added as optional feature reading provider pricing APIs

### 11. **Resume Capability**
- **Decision**: No automatic resume in v1
- **Manual Workaround**: User can manually edit experiments CSV to remove completed rows
- **Rationale**: Real-time appending provides partial results; full resume adds complexity

### 12. **Incremental Output Writing**
- **Decision**: Append to output CSV after each experiment completes
- **Benefits**:
  - Real-time monitoring possible
  - No data loss if process interrupted
  - Immediate feedback on errors
- **Implementation**:
  - Open CSV in append mode
  - Write headers on first write only
  - Flush after each write

## Technical Implementation

### Technology Stack
- **Language**: Python 3.11+
- **Dependency Manager**: uv
- **Environment**: Virtual environment via uv
- **Configuration**: python-dotenv
- **Testing**: pytest with TDD approach
- **CSV Handling**: Python standard library `csv` module

### Project Structure
```
parallaxr/
├── .gitignore
├── pyproject.toml
├── uv.lock
├── README.md
├── .env.example
├── src/
│   └── parallaxr/
│       ├── __init__.py
│       ├── cli.py              # CLI interface
│       ├── runner.py           # Experiment orchestration
│       ├── template.py         # Variable replacement
│       ├── token_counter.py    # Token estimation
│       ├── csv_writer.py       # Incremental CSV output
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py         # Provider interface
│       │   ├── openrouter.py
│       │   ├── ollama.py
│       │   └── mock.py
│       ├── models.py           # Data models (Experiment, Result)
│       └── utils.py            # Utilities
└── tests/
    ├── __init__.py
    ├── test_cli.py
    ├── test_runner.py
    ├── test_template.py
    ├── test_token_counter.py
    ├── test_csv_writer.py
    ├── test_providers.py
    └── fixtures/
        ├── prompt.txt
        ├── context.txt
        ├── experiments.csv
        └── expected_output.csv
```

### Key Components

#### 1. Template Engine (`template.py`)
```python
def replace_variables(text: str, variables: dict) -> tuple[str, list[str]]:
    """
    Replace {{variable}} with values from dict.
    Returns: (replaced_text, list_of_missing_variables)
    """
```

#### 2. Token Counter (`token_counter.py`)
```python
def estimate_tokens(text: str) -> int:
    """
    Estimate tokens using character count / 4.
    """
    return len(text) // 4
```

#### 3. CSV Writer (`csv_writer.py`)
```python
class IncrementalCSVWriter:
    """
    Handles incremental writing to CSV with proper escaping.
    Writes headers on first call, appends data subsequently.
    """
    def write_result(self, result: ExperimentResult) -> None:
        """Append single result row to CSV."""
```

#### 4. Provider Base Class (`providers/base.py`)
```python
class Provider(ABC):
    @abstractmethod
    def get_completion(self, prompt: str, model: str, **kwargs) -> ProviderResponse:
        """Get completion from provider."""
        
    @abstractmethod
    def get_context_window(self, model: str) -> Optional[int]:
        """Get model's context window size, or None if unknown."""
```

#### 5. Runner (`runner.py`)
```python
class ExperimentRunner:
    """
    Orchestrates experiment execution:
    1. Load experiments CSV
    2. Load and concatenate input files
    3. For each experiment:
       a. Replace variables
       b. Call provider
       c. Write result to CSV immediately
    4. Handle errors gracefully
    """
```

### Testing Strategy

#### Test Coverage Requirements
- **Unit tests**: Each module independently (>90% coverage)
- **Integration tests**: Full experiment runs with mock provider
- **Fixture-based**: Reusable test prompts and experiments
- **Edge cases**: Missing variables, invalid models, rate limits, context overflow

#### Key Test Scenarios
1. **Template replacement**: Variables present, missing, malformed
2. **Token counting**: Various text sizes, Unicode characters
3. **CSV writing**: Proper escaping, incremental append, header handling
4. **Provider responses**: Success, rate limit, model not found, timeout
5. **End-to-end**: Complete experiment run with mock provider
6. **Error handling**: Invalid CSV, missing files, authentication errors

### .gitignore
```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/

# Virtual environments
.venv/
.venv-devcon/
venv/
env/

# Environment
.env
.env.local

# uv
.uv/

# Testing
.pytest_cache/
.coverage
htmlcov/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Output files
results*.csv
output*.csv
experiments_output*.csv

# OS
.DS_Store
Thumbs.db
```

## Success Criteria
1. ✅ Successfully runs experiments across multiple providers
2. ✅ Correctly replaces template variables with warnings for missing ones
3. ✅ Handles errors gracefully with appropriate status codes
4. ✅ Outputs valid CSV with proper escaping after each experiment
5. ✅ Works from any directory when installed
6. ✅ Comprehensive test coverage (>90%)
7. ✅ Clear error messages for misconfiguration
8. ✅ Skips invalid models and continues processing
9. ✅ Real-time output monitoring via incremental CSV writing

## Non-Goals (for v1)
- Web interface
- Real-time streaming responses
- Built-in prompt optimization
- Automatic model selection
- Results visualization
- Database storage
- Parallel execution
- Automatic retry logic
- Built-in cost tracking
- Resume/checkpoint capability
- Progress bars or indicators

## Example Usage Scenario

### Input Files

**prompt.txt:**
```
Summarize the following information from {{source}}:

The topic is {{topic}}.
```

**context.txt:**
```
Additional context: This is a test of the parallaxr system.
```

**experiments.csv:**
```csv
provider,model,source,topic
mock,mock,Wikipedia,AI
openrouter,anthropic/claude-sonnet-4,Encyclopedia,ML
ollama,llama3.2,Database,NLP
```

### Command
```bash
parallaxr run -p prompt.txt -c context.txt -e experiments.csv -o results.csv --verbose
```

### Expected Output (results.csv)
```csv
provider,model,source,topic,status,input_tokens,context_window,output_tokens,output,error_message
mock,mock,Wikipedia,AI,ok,45,,50,"MOCK RESPONSE\nInput tokens: 45\nModel: mock...",
openrouter,anthropic/claude-sonnet-4,Encyclopedia,ML,ok,45,200000,123,"Machine learning is...",
ollama,llama3.2,Database,NLP,ok,45,8192,89,"Natural language processing...",
```

### Verbose Output (stderr)
```
[INFO] Loading prompt from prompt.txt
[INFO] Loading context from context.txt
[INFO] Loading experiments from experiments.csv
[INFO] Starting experiment 1/3: mock/mock
[INFO] Completed experiment 1/3: status=ok
[INFO] Starting experiment 2/3: openrouter/anthropic/claude-sonnet-4
[INFO] Completed experiment 2/3: status=ok
[INFO] Starting experiment 3/3: ollama/llama3.2
[INFO] Completed experiment 3/3: status=ok
[INFO] All experiments completed. Results written to results.csv
```