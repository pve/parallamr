# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2025-10-07

### Added
- **OpenAI Provider Support** (#20) - Full native OpenAI API integration
  - Support for 18+ GPT model families (GPT-4o, GPT-4 Turbo, GPT-4, GPT-3.5)
  - Context windows: 128K (GPT-4o, GPT-4 Turbo), 32K (GPT-4-32k), 16K (GPT-3.5), 8K (GPT-4)
  - Bearer token authentication with optional organization ID support
  - OpenAI-compatible provider support (Azure OpenAI, LocalAI, Together AI, Groq)
  - Custom base URL override for compatible endpoints
  - Session injection for parallel processing with connection pooling
  - Comprehensive error handling (401, 403, 404, 413, 429, 500, 503)
  - Request/response transformation with graceful degradation
  - Parameter validation (temperature, max_tokens, top_p, penalties)

- **Comprehensive Test Suite for OpenAI Provider**
  - 65+ unit tests covering all functionality
  - 100% error path coverage
  - Mock API responses for all scenarios
  - Session injection validation tests
  - Azure OpenAI compatibility tests
  - LocalAI compatibility tests
  - Together AI compatibility tests

- **OpenAI Provider Documentation**
  - Complete setup guide in README.md
  - Azure OpenAI configuration examples
  - LocalAI / Together AI configuration examples
  - Supported models list with context windows
  - Troubleshooting guide (9 common scenarios)
  - Usage examples with CSV format
  - .env.example configuration updates

### Changed
- Updated provider list in README to include OpenAI
- Enhanced `providers/__init__.py` to export OpenAIProvider
- Enhanced `runner.py` to register OpenAI in default provider factory
- Updated version from 0.6.0 to 0.7.0

### Technical Details
- Implementation: `/src/parallamr/providers/openai.py` (562 lines)
- Tests: `/tests/test_openai_provider.py` (1,227 lines)
- Fixtures: `/tests/fixtures/openai_responses.py`
- Architecture follows existing provider patterns (OpenRouter, Ollama)
- Hive Mind Collective Intelligence System coordination

## [0.2.0] - 2025-10-04

### Added
- Unix-style stdin/stdout support (#6)
  - `-o` flag is now optional; when omitted, CSV output goes to stdout
  - `-p -` reads prompt from stdin
  - `-e -` reads experiments from stdin
  - Error handling when both stdin flags are used simultaneously
- Comprehensive test coverage for stdin/stdout functionality
- Project context documentation in `.claude-flow/`

### Fixed
- Async CLI command warning for `models` command (#1)
- Ollama model tag preservation (e.g., `llama3.1:latest` now preserved correctly)
- Ollama context window retrieval
- Context window warning messages now include provider/model details
- `parallamr providers` command no longer crashes without OpenRouter API key (#2)
- Fixed async test issue in `test_full_run_integration`

### Changed
- `IncrementalCSVWriter` now supports stdout output via `None` path
- `ExperimentRunner` can read from stdin when paths are `None`
- `load_experiments_from_csv()` enhanced to accept string content parameter

## [0.1.0] - 2025-09-29

### Added
- Initial release
- Support for multiple LLM providers (OpenRouter, Ollama, Mock)
- CSV-based experiment management
- Template variable substitution
- Incremental CSV result writing
- Token counting and context window validation
- CLI with commands: `run`, `init`, `providers`, `models`
- Comprehensive test suite
- TDD development methodology

[0.2.0]: https://github.com/pve/parallamr/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/pve/parallamr/releases/tag/v0.1.0
