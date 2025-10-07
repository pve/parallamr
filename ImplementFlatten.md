Implementation Brief: Issue #5 - JSON Flattening Feature

  Objective

  Add --flatten flag to extract and flatten JSON from LLM outputs into separate CSV columns.

  Key Requirements (from @pve's comment)

  1. Flag: --flatten (not --decode-json)
  2. Extract JSON: Ignore markdown code fences, process JSON inside
  3. Flatten to columns: Use JSON field names, avoid conflicts with experiment CSV columns
  4. No errors for missing JSON: It's optional, not a warning
  5. Clean output: Replace \n and \r with spaces in output column (Excel compatibility)

  Technical Design

  New Module: src/parallamr/json_extractor.py

  class JSONExtractor:
      def extract_json(output: str) -> Optional[dict]
          # Try: markdown block → direct parse → inline JSON

      def flatten_json(data: dict, max_depth=5, max_array=10) -> Dict[str, Any]
          # Nested: {"a": {"b": "c"}} → {"a_b": "c"}
          # Arrays: {"x": [1, 2]} → {"x_0": 1, "x_1": 2}

      def resolve_column_names(flat_data, reserved, experiment_vars) -> Dict[str, Any]
          # Add "json_" prefix if field conflicts

  Integration Points

  1. ExperimentResult (models.py)
  - Add field: json_fields: Optional[Dict[str, Any]]
  - Update to_csv_row(): merge json_fields, clean output (linefeeds→spaces)

  2. ExperimentRunner (runner.py)
  - Add param: flatten_json: bool = False
  - After provider response: extract → flatten → resolve names → attach to result

  3. CLI (cli.py)
  - Add flag: --flatten
  - Pass to factory: create_experiment_runner(..., flatten_json=flatten)

  4. CSV Writer (csv_writer.py)
  - Handle dynamic columns from JSON fields

  Test Coverage (TDD)

  tests/test_json_extractor.py (15-20 tests)
  - Markdown extraction, direct JSON, inline JSON, no JSON
  - Nested object flattening, array flattening (with limits)
  - Column name conflicts resolution
  - Output cleaning

  tests/test_flatten_integration.py (8-10 tests)
  - End-to-end with --flatten
  - Missing JSON (no extra columns, no errors)
  - Consistent vs inconsistent JSON schemas

  Key Edge Cases

  - No JSON: Return None, no extra columns, no warnings
  - Large arrays: Limit to 10 elements, log if truncated
  - Deep nesting: Max 5 levels
  - Key conflicts: Prefix with json_ if conflicts with reserved/experiment columns
  - Special chars in keys: Sanitize (- → _)

  Implementation Order (TDD)

  1. Write JSONExtractor tests → implement class
  2. Write ExperimentResult tests → update to_csv_row()
  3. Write integration tests → update runner + CLI
  4. Update CSV writer for dynamic fields
  5. Update README with examples
  6. Verify >90% coverage

  Deliverables

  - src/parallamr/json_extractor.py (new)
  - tests/test_json_extractor.py (new)
  - tests/test_flatten_integration.py (new)
  - Updated: models.py, runner.py, cli.py, csv_writer.py
  - Updated: README.md with JSON flattening section
  - Close issue #5 with implementation summary
  - Version bump: v0.5.0 → v0.6.0

  Effort: ~8-10 hours

  Acceptance: All tests pass (>90% coverage), backward compatible

  Ready for Hive Mind parallel implementation! 🧠
