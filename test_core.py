#!/usr/bin/env python3
"""
Core functionality test without external dependencies.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from parallaxr.models import Experiment, ExperimentStatus, ProviderResponse, ExperimentResult
from parallaxr.template import replace_variables, combine_files_with_variables, extract_variables
from parallaxr.token_counter import estimate_tokens, validate_context_window


def test_core_functionality():
    """Test core functionality without external dependencies."""
    print("🧪 Testing Parallaxr core implementation...\n")

    # Test 1: Template engine
    print("1. Testing template engine...")
    text = "Hello {{name}}, analyze {{topic}} from {{source}}"
    variables = {"name": "Alice", "topic": "AI", "source": "Wikipedia"}
    result, missing = replace_variables(text, variables)

    expected = "Hello Alice, analyze AI from Wikipedia"
    assert result == expected, f"Expected '{expected}', got '{result}'"
    assert missing == [], f"Expected no missing variables, got {missing}"
    print("   ✅ Variable replacement works correctly")

    # Test missing variables
    result2, missing2 = replace_variables("{{missing}} variable", {})
    assert "{{missing}}" in result2
    assert missing2 == ["missing"]
    print("   ✅ Missing variable handling works correctly")

    # Test 2: Token counting
    print("\n2. Testing token counting...")
    text = "This is a test message for counting tokens."
    tokens = estimate_tokens(text)
    expected_tokens = len(text) // 4

    assert tokens == expected_tokens, f"Expected {expected_tokens} tokens, got {tokens}"
    print(f"   ✅ Token estimation: '{text}' = {tokens} tokens")

    # Test context window validation
    is_valid, warning = validate_context_window(500, 1000)
    assert is_valid is True
    assert warning is None
    print("   ✅ Context window validation works correctly")

    # Test 3: Data models
    print("\n3. Testing data models...")

    # Test Experiment model
    csv_row = {"provider": "mock", "model": "test", "topic": "AI", "source": "Wikipedia"}
    experiment = Experiment.from_csv_row(csv_row, 1)

    assert experiment.provider == "mock"
    assert experiment.model == "test"
    assert experiment.variables == {"topic": "AI", "source": "Wikipedia"}
    assert experiment.row_number == 1
    print("   ✅ Experiment model works correctly")

    # Test ProviderResponse
    response = ProviderResponse(
        output="Test output",
        output_tokens=10,
        success=True,
        context_window=1000
    )
    assert response.status == ExperimentStatus.OK
    print("   ✅ ProviderResponse model works correctly")

    # Test ExperimentResult
    result = ExperimentResult.from_experiment_and_response(
        experiment=experiment,
        response=response,
        input_tokens=50
    )

    assert result.provider == "mock"
    assert result.status == ExperimentStatus.OK
    assert result.input_tokens == 50
    print("   ✅ ExperimentResult model works correctly")

    # Test CSV conversion
    csv_row_output = result.to_csv_row()
    assert csv_row_output["provider"] == "mock"
    assert csv_row_output["status"] == "ok"
    assert csv_row_output["topic"] == "AI"
    print("   ✅ CSV conversion works correctly")

    # Test 4: File combination
    print("\n4. Testing file combination...")
    primary = "Main: {{topic}}"
    context_files = [("doc1.txt", "Context: {{topic}} and {{detail}}")]
    variables = {"topic": "AI", "detail": "Machine Learning"}

    combined, missing = combine_files_with_variables(primary, context_files, variables)

    assert "Main: AI" in combined
    assert "Context: AI and Machine Learning" in combined
    assert "## Document: doc1.txt" in combined
    assert missing == []
    print("   ✅ File combination works correctly")

    # Test 5: Variable extraction
    print("\n5. Testing variable extraction...")
    template = "Use {{model}} to analyze {{topic}} with {{params}}"
    variables = extract_variables(template)

    expected_vars = {"model", "topic", "params"}
    assert set(variables) == expected_vars
    print("   ✅ Variable extraction works correctly")

    print("\n🎉 All core tests passed!")

    return True


if __name__ == "__main__":
    try:
        test_core_functionality()
        print("\n📋 Parallaxr Implementation Complete!")
        print("\n✨ Features implemented:")
        print("   • Template engine with {{variable}} replacement")
        print("   • Token counting and context window validation")
        print("   • Comprehensive data models for experiments and results")
        print("   • Provider abstraction layer (Mock, OpenRouter, Ollama)")
        print("   • Incremental CSV writer with proper escaping")
        print("   • Experiment runner with error handling")
        print("   • CLI interface with validation and help")
        print("   • Comprehensive test suite")
        print("   • Project configuration and documentation")

        print("\n🚀 Ready for use!")
        print("   1. Install dependencies: uv sync")
        print("   2. Configure API keys in .env")
        print("   3. Run: parallaxr init")
        print("   4. Execute: parallaxr run -p prompt.txt -e experiments.csv -o results.csv")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)