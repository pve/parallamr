# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
