# Parallamr Test Fixtures Specification
## HiveMind-Tester-Delta Test Data Framework

### Overview
This document specifies all test fixtures, mock data, and sample files needed for comprehensive testing of the Parallamr project. These fixtures support unit tests, integration tests, and edge case validation.

### Directory Structure

```
tests/fixtures/
├── prompts/
│   ├── basic.txt                    # Simple prompt without variables
│   ├── with_variables.txt           # Prompt with standard variables
│   ├── complex_variables.txt        # Multiple variables, edge cases
│   ├── multiline.txt               # Multi-paragraph prompt
│   ├── unicode.txt                 # Unicode and emoji content
│   ├── empty.txt                   # Empty file
│   ├── large_prompt.txt            # Large prompt (>10KB)
│   └── malformed_variables.txt     # Invalid variable syntax
├── contexts/
│   ├── simple.txt                  # Basic context file
│   ├── technical.txt               # Technical documentation style
│   ├── multifile_context_1.txt     # First of multiple context files
│   ├── multifile_context_2.txt     # Second of multiple context files
│   ├── large_context.txt           # Large context file (>50KB)
│   ├── special_chars.txt           # Special characters and formatting
│   ├── unicode_context.txt         # International characters
│   └── empty_context.txt           # Empty context file
├── experiments/
│   ├── basic_valid.csv             # Standard valid experiments
│   ├── missing_variables.csv       # Experiments with missing variables
│   ├── invalid_providers.csv       # Invalid provider names
│   ├── invalid_models.csv          # Invalid model names
│   ├── large_dataset.csv           # Large experiment set (1000+ rows)
│   ├── edge_cases.csv              # Edge case scenarios
│   ├── malformed.csv               # Malformed CSV data
│   ├── empty.csv                   # Empty CSV file
│   ├── headers_only.csv            # CSV with only headers
│   ├── unicode_data.csv            # Unicode in CSV data
│   ├── special_chars_data.csv      # Special characters in values
│   └── mixed_providers.csv         # Multiple provider types
├── expected_outputs/
│   ├── basic_run.csv               # Expected output for basic test
│   ├── missing_vars_run.csv        # Expected output with warnings
│   ├── error_cases.csv             # Expected error outputs
│   ├── unicode_run.csv             # Expected Unicode handling
│   └── performance_baseline.csv    # Performance benchmark data
└── mock_responses/
    ├── openrouter_success.json      # Mock successful OpenRouter response
    ├── openrouter_rate_limit.json   # Mock rate limit response
    ├── openrouter_error.json        # Mock error response
    ├── ollama_success.json          # Mock successful Ollama response
    ├── ollama_not_found.json        # Mock model not found
    └── context_windows.json         # Mock context window data
```

### Prompt Fixtures

#### basic.txt
```
Please provide a comprehensive analysis of the given topic.
Focus on key concepts and practical applications.
```

#### with_variables.txt
```
Analyze the topic of {{topic}} using information from {{source}}.

Please provide:
1. A brief overview of {{topic}}
2. Key insights from {{source}}
3. Practical applications
4. Future implications

Style: {{style}}
Length: {{length}}
```

#### complex_variables.txt
```
Research Report: {{title}}
Author: {{author}}
Date: {{date}}

Executive Summary:
Investigate {{primary_topic}} in the context of {{context_area}}.
Focus on {{research_focus}} with emphasis on {{methodology}}.

Data Sources:
- Primary: {{primary_source}}
- Secondary: {{secondary_source}}
- Additional: {{additional_sources}}

Research Questions:
1. {{question_1}}
2. {{question_2}}
3. {{question_3}}

Expected Deliverables:
- {{deliverable_1}}
- {{deliverable_2}}
- {{deliverable_3}}

Timeline: {{timeline}}
Budget: {{budget}}
```

#### multiline.txt
```
# Research Analysis Framework

## Background
This analysis focuses on understanding complex systems and their interactions.
We need to examine multiple perspectives and synthesize findings.

## Methodology
Our approach will involve:
- Systematic literature review
- Data collection and analysis
- Expert interviews
- Case study development

## Expected Outcomes
The final report should provide:
1. Clear problem definition
2. Evidence-based recommendations
3. Implementation roadmap
4. Risk assessment

## Variables for Customization
Topic: {{topic}}
Industry: {{industry}}
Timeframe: {{timeframe}}
Scope: {{scope}}
```

#### unicode.txt
```
🔬 Research Topic: {{topic}}

Analyze the following aspects:
• Key concepts in {{field}}
• Impact on {{industry}}
• Future trends and implications

Sources to consider:
→ {{primary_source}}
→ {{secondary_source}}

Please ensure the analysis covers:
✓ Technical accuracy
✓ Practical relevance
✓ Cultural considerations

Languages: English, Français, Español, 中文, 日本語
Symbols: ∞ ∑ ∆ ∇ ∂ ∫ ∏ √ ∝ ≠ ≤ ≥

Émojis: 🚀 ✨ 🎯 💡 🔍 📊 📈 🎭
```

#### large_prompt.txt
```
[Large prompt content - approximately 10KB of text with repeated sections and variables]

COMPREHENSIVE RESEARCH AND ANALYSIS FRAMEWORK
==============================================

This document outlines a detailed methodology for conducting thorough research
and analysis in the field of {{research_domain}}...

[Content continues with detailed sections, multiple variables, and extensive text
to test performance with large inputs. The file should be approximately 10KB
when complete.]
```

#### malformed_variables.txt
```
This prompt has various malformed variable syntax:
- Single braces: {variable}
- Triple braces: {{{variable}}}
- Incomplete: {{incomplete
- Missing closing: variable}}
- Nested: {{outer{{inner}}}}
- Empty: {{}}
- Whitespace: {{ variable }}
- Numbers: {{123variable}}
- Special chars: {{var-with-dashes}}

The only valid variable should be: {{valid_variable}}
```

### Context Fixtures

#### simple.txt
```
This is a simple context file for testing basic functionality.
It contains straightforward text without special formatting.
```

#### technical.txt
```
Technical Documentation Context

This context provides technical background information for analysis.

Key Technologies:
- Python 3.11+
- RESTful APIs
- CSV processing
- Token estimation algorithms

Performance Metrics:
- Response time: <300ms
- Throughput: 100 requests/minute
- Memory usage: <500MB
- Token accuracy: ±5%

Error Handling:
- Graceful degradation
- Comprehensive logging
- User-friendly messages
- Automatic retry mechanisms
```

#### multifile_context_1.txt
```
## Document: Context File 1

This is the first context file in a multi-file scenario.
It contains foundational information that will be combined
with other context files to provide comprehensive background.

Key Points:
- Establishes basic framework
- Defines core concepts
- Sets up terminology
```

#### multifile_context_2.txt
```
## Document: Context File 2

This is the second context file that builds upon the first.
It provides additional details and expands on the foundation
established in the previous document.

Additional Information:
- Advanced concepts
- Implementation details
- Best practices
- Common pitfalls
```

### Experiment Fixtures

#### basic_valid.csv
```csv
provider,model,topic,source,style,length
mock,mock,Artificial Intelligence,Wikipedia,academic,detailed
mock,mock,Machine Learning,Encyclopedia,conversational,brief
mock,mock,Data Science,Research Papers,technical,comprehensive
```

#### missing_variables.csv
```csv
provider,model,topic
mock,mock,AI
mock,mock,ML
mock,mock,Data Science
```

#### invalid_providers.csv
```csv
provider,model,topic,source
invalid_provider,some_model,AI,Wikipedia
nonexistent,fake_model,ML,Encyclopedia
mock,mock,Data Science,Research Papers
```

#### invalid_models.csv
```csv
provider,model,topic,source
openrouter,nonexistent/model,AI,Wikipedia
ollama,fake_model_name,ML,Encyclopedia
mock,mock,Data Science,Research Papers
```

#### large_dataset.csv
```csv
provider,model,topic,source,iteration
mock,mock,AI,Wikipedia,1
mock,mock,ML,Encyclopedia,2
[... continues for 1000+ rows with varying data ...]
```

#### edge_cases.csv
```csv
provider,model,topic,source,special
mock,mock,"Topic with, commas","Source with ""quotes""",normal
mock,mock,"Multi
line
topic",Source with newlines,multiline
mock,mock,🚀 Unicode Topic,Unicode Source ✨,unicode
mock,mock,,Empty topic,empty_topic
mock,mock,Normal Topic,,empty_source
```

#### malformed.csv
```csv
provider,model,topic
mock,mock,"Unclosed quote
mock,mock,Normal Topic
missing_provider,,Some Topic
,missing_model,Another Topic
```

### Expected Output Fixtures

#### basic_run.csv
```csv
provider,model,topic,source,style,length,status,input_tokens,context_window,output_tokens,output,error_message
mock,mock,Artificial Intelligence,Wikipedia,academic,detailed,ok,45,,52,"MOCK RESPONSE
Input tokens: 45
Model: mock
Variables: {'topic': 'Artificial Intelligence', 'source': 'Wikipedia', 'style': 'academic', 'length': 'detailed'}
--- Original Input ---
## Document: basic.txt

Please provide a comprehensive analysis of the given topic.
Focus on key concepts and practical applications.

---

## Document: with_variables.txt

Analyze the topic of Artificial Intelligence using information from Wikipedia.

Please provide:
1. A brief overview of Artificial Intelligence
2. Key insights from Wikipedia
3. Practical applications
4. Future implications

Style: academic
Length: detailed

---
",
```

#### missing_vars_run.csv
```csv
provider,model,topic,status,input_tokens,context_window,output_tokens,output,error_message
mock,mock,AI,warning,42,,48,"MOCK RESPONSE
Input tokens: 42
Model: mock
Variables: {'topic': 'AI'}
--- Original Input ---
Analyze the topic of AI using information from {{source}}.

Style: {{style}}
Length: {{length}}
","Missing variables: source, style, length"
```

### Mock Response Fixtures

#### openrouter_success.json
```json
{
  "id": "gen-1234567890",
  "model": "anthropic/claude-sonnet-4",
  "object": "chat.completion",
  "created": 1700000000,
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "This is a simulated response from OpenRouter Claude Sonnet 4 model. The response demonstrates proper JSON structure and realistic content length for testing purposes."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 123,
    "total_tokens": 168
  }
}
```

#### openrouter_rate_limit.json
```json
{
  "error": {
    "type": "rate_limit_exceeded",
    "message": "Rate limit exceeded. Please try again in 60 seconds.",
    "code": 429
  }
}
```

#### openrouter_error.json
```json
{
  "error": {
    "type": "model_not_found",
    "message": "The requested model 'nonexistent/model' is not available.",
    "code": 404
  }
}
```

#### ollama_success.json
```json
{
  "model": "llama3.2",
  "created_at": "2024-01-01T12:00:00Z",
  "response": "This is a simulated response from Ollama's Llama 3.2 model. The response includes realistic content and demonstrates the expected JSON structure for Ollama API responses.",
  "done": true,
  "context": [1, 2, 3, 4, 5],
  "total_duration": 5000000000,
  "load_duration": 1000000000,
  "prompt_eval_count": 45,
  "prompt_eval_duration": 2000000000,
  "eval_count": 89,
  "eval_duration": 2000000000
}
```

#### context_windows.json
```json
{
  "models": {
    "anthropic/claude-sonnet-4": {
      "context_window": 200000,
      "max_output": 4096
    },
    "anthropic/claude-haiku": {
      "context_window": 200000,
      "max_output": 4096
    },
    "meta-llama/llama-3.2-8b": {
      "context_window": 8192,
      "max_output": 2048
    },
    "llama3.2": {
      "context_window": 8192,
      "max_output": 2048
    },
    "mock": {
      "context_window": null,
      "max_output": null
    }
  }
}
```

### Fixture Usage Patterns

#### Unit Test Fixture Loading
```python
import pytest
from pathlib import Path

@pytest.fixture
def fixture_dir():
    return Path(__file__).parent / "fixtures"

@pytest.fixture
def basic_prompt(fixture_dir):
    return (fixture_dir / "prompts" / "basic.txt").read_text()

@pytest.fixture
def valid_experiments(fixture_dir):
    import pandas as pd
    return pd.read_csv(fixture_dir / "experiments" / "basic_valid.csv")
```

#### Integration Test Fixture Setup
```python
@pytest.fixture
def complete_test_setup(fixture_dir):
    return {
        'prompt': fixture_dir / "prompts" / "with_variables.txt",
        'context': [fixture_dir / "contexts" / "simple.txt"],
        'experiments': fixture_dir / "experiments" / "basic_valid.csv",
        'expected': fixture_dir / "expected_outputs" / "basic_run.csv"
    }
```

### Fixture Validation

#### Content Validation Rules
1. **Text Encoding**: All text files must be UTF-8 encoded
2. **CSV Format**: All CSV files must have valid headers and proper escaping
3. **Size Limits**: Large files should be marked and have size validation
4. **Variable Syntax**: Template variables must follow {{variable_name}} format
5. **JSON Structure**: Mock responses must have valid JSON structure

#### Automated Fixture Testing
```python
def test_fixture_integrity():
    """Validate all test fixtures are properly formatted."""
    fixture_dir = Path(__file__).parent / "fixtures"

    # Test all CSV files are valid
    for csv_file in fixture_dir.glob("**/*.csv"):
        try:
            pd.read_csv(csv_file)
        except Exception as e:
            pytest.fail(f"Invalid CSV file {csv_file}: {e}")

    # Test all JSON files are valid
    for json_file in fixture_dir.glob("**/*.json"):
        try:
            json.loads(json_file.read_text())
        except Exception as e:
            pytest.fail(f"Invalid JSON file {json_file}: {e}")

    # Test all text files are UTF-8 readable
    for txt_file in fixture_dir.glob("**/*.txt"):
        try:
            txt_file.read_text(encoding='utf-8')
        except Exception as e:
            pytest.fail(f"Invalid text file {txt_file}: {e}")
```

### Fixture Maintenance

#### Version Control
- All fixtures are version controlled with the code
- Changes to fixtures require test updates
- Deprecated fixtures are clearly marked

#### Documentation
- Each fixture includes purpose and usage comments
- Complex fixtures have detailed explanations
- Edge cases are documented inline

#### Size Management
- Large fixtures (>1MB) are compressed or generated
- Performance test data is created programmatically
- Binary test data is minimized

This comprehensive fixture specification ensures thorough testing coverage for all aspects of the Parallamr project, from basic functionality to complex edge cases and error conditions.