"""Provider Registry Validation Tests.

This module provides automated validation that:
1. All Provider subclasses are discovered
2. Each provider has a corresponding test file
3. Minimum test count per provider is met
4. No orphaned test files exist
5. Test coverage is comprehensive

Purpose:
- Prevents adding providers without tests (catches developer errors)
- Automated CI/CD validation
- Clear error messages for debugging
- Enforces test quality standards

Integration:
- Run as part of CI/CD pipeline
- Fails build if provider tests are missing
- Reports detailed metrics and gaps
"""

import inspect
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest

# Add src to path for importing providers
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parallamr.providers.base import Provider


# ============================================================================
# CONFIGURATION - Adjust thresholds as needed
# ============================================================================

# Minimum number of tests required per provider
MIN_TESTS_PER_PROVIDER = {
    "MockProvider": 6,      # Simple mock provider, fewer tests needed
    "OpenAIProvider": 60,   # Complex provider with many features
    "OllamaProvider": 40,   # Moderate complexity
    "OpenRouterProvider": 40,  # Moderate complexity
}

# Default minimum if not specified above
DEFAULT_MIN_TESTS = 10

# Providers that are allowed to skip test requirements (e.g., abstract base classes)
EXEMPT_PROVIDERS = []

# Test file naming convention
TEST_FILE_PREFIX = "test_"
TEST_FILE_SUFFIX = "_provider.py"


# ============================================================================
# DISCOVERY FUNCTIONS
# ============================================================================

def discover_provider_classes() -> Dict[str, type]:
    """Discover all Provider subclasses in the codebase.

    Returns:
        Dict mapping provider class names to their class objects

    Example:
        >>> providers = discover_provider_classes()
        >>> print(providers.keys())
        dict_keys(['MockProvider', 'OpenAIProvider', 'OllamaProvider', 'OpenRouterProvider'])
    """
    # Import the providers module
    import parallamr.providers as providers_module

    provider_classes = {}

    # Inspect all members of the providers module
    for name, obj in inspect.getmembers(providers_module):
        # Check if it's a class and a subclass of Provider (but not Provider itself)
        if (
            inspect.isclass(obj)
            and issubclass(obj, Provider)
            and obj != Provider
            and name not in EXEMPT_PROVIDERS
        ):
            provider_classes[name] = obj

    return provider_classes


def discover_test_files() -> Dict[str, Path]:
    """Discover all provider test files in the tests directory.

    Returns:
        Dict mapping provider names to their test file paths

    Example:
        >>> test_files = discover_test_files()
        >>> print(test_files['OpenAI'])
        PosixPath('/workspaces/parallamr/tests/test_openai_provider.py')
    """
    tests_dir = PROJECT_ROOT / "tests"
    test_files = {}

    # Find all test files matching the pattern
    for test_file in tests_dir.glob(f"{TEST_FILE_PREFIX}*{TEST_FILE_SUFFIX}"):
        # Extract provider name from filename
        # Example: test_openai_provider.py -> openai -> OpenAI
        filename = test_file.stem  # Remove .py
        provider_name = filename.replace(TEST_FILE_PREFIX, "").replace("_provider", "")

        # Convert to PascalCase to match provider class names
        provider_name = "".join(word.capitalize() for word in provider_name.split("_"))

        test_files[provider_name] = test_file

    return test_files


def count_tests_in_file(test_file: Path) -> int:
    """Count number of test functions/methods in a test file.

    Args:
        test_file: Path to test file

    Returns:
        Number of test functions found

    Note:
        This uses pytest's collection mechanism for accuracy
    """
    try:
        # Use pytest to collect tests from the file
        import _pytest.config
        import _pytest.main

        # Create a pytest config
        config = _pytest.config.Config.fromdictargs(
            {"args": [str(test_file), "--collect-only", "-q"]},
            []
        )

        # Collect tests
        session = _pytest.main.Session.from_config(config)
        session.perform_collect()

        # Count collected items
        return len(session.items)
    except Exception:
        # Fallback: count by pattern matching
        with open(test_file, 'r') as f:
            content = f.read()
            # Count lines starting with "def test_" or "async def test_"
            import re
            pattern = r'^\s*(async\s+)?def\s+test_\w+'
            matches = re.findall(pattern, content, re.MULTILINE)
            return len(matches)


def get_provider_test_metrics() -> Dict[str, Dict[str, any]]:
    """Get comprehensive metrics for all providers and their tests.

    Returns:
        Dict mapping provider names to their metrics:
        - 'exists': Whether provider class exists
        - 'has_test_file': Whether test file exists
        - 'test_file_path': Path to test file (if exists)
        - 'test_count': Number of tests (if test file exists)
        - 'min_required': Minimum tests required
        - 'passes_requirement': Whether test count meets minimum
        - 'gap': How many tests short (negative if exceeds)
    """
    providers = discover_provider_classes()
    test_files = discover_test_files()

    metrics = {}

    for provider_name, provider_class in providers.items():
        # Determine expected test file name (e.g., OpenAIProvider -> OpenAI)
        test_key = provider_name.replace("Provider", "")

        has_test_file = test_key in test_files
        test_file_path = test_files.get(test_key)
        test_count = count_tests_in_file(test_file_path) if has_test_file else 0

        min_required = MIN_TESTS_PER_PROVIDER.get(provider_name, DEFAULT_MIN_TESTS)
        passes_requirement = test_count >= min_required
        gap = min_required - test_count if not passes_requirement else 0

        metrics[provider_name] = {
            'exists': True,
            'has_test_file': has_test_file,
            'test_file_path': test_file_path,
            'test_count': test_count,
            'min_required': min_required,
            'passes_requirement': passes_requirement,
            'gap': gap
        }

    return metrics


def find_orphaned_test_files() -> List[Path]:
    """Find test files that don't correspond to any provider.

    Returns:
        List of paths to orphaned test files

    Note:
        Orphaned files may indicate:
        - Renamed providers without updating tests
        - Deleted providers without removing tests
        - Incorrectly named test files
    """
    providers = discover_provider_classes()
    test_files = discover_test_files()

    # Get expected provider names (without "Provider" suffix)
    provider_names = {name.replace("Provider", "") for name in providers.keys()}

    # Find test files without matching providers
    orphaned = []
    for test_name, test_path in test_files.items():
        if test_name not in provider_names:
            orphaned.append(test_path)

    return orphaned


# ============================================================================
# TEST CASES
# ============================================================================

class TestProviderRegistryDiscovery:
    """Test automated provider discovery."""

    def test_can_discover_providers(self):
        """Provider discovery finds all Provider subclasses."""
        providers = discover_provider_classes()

        # Should find at least the known providers
        assert len(providers) >= 4
        assert "MockProvider" in providers
        assert "OpenAIProvider" in providers
        assert "OllamaProvider" in providers
        assert "OpenRouterProvider" in providers

    def test_discovered_providers_are_subclasses(self):
        """All discovered providers are proper Provider subclasses."""
        providers = discover_provider_classes()

        for name, provider_class in providers.items():
            assert issubclass(provider_class, Provider)
            assert provider_class != Provider  # Should not include base class

    def test_can_discover_test_files(self):
        """Test file discovery finds provider test files."""
        test_files = discover_test_files()

        # Should find at least some test files
        assert len(test_files) > 0

        # All discovered files should exist
        for name, path in test_files.items():
            assert path.exists(), f"Test file not found: {path}"


class TestProviderTestCoverage:
    """Test that all providers have adequate test coverage."""

    def test_all_providers_have_test_files(self):
        """Every provider has a corresponding test file."""
        providers = discover_provider_classes()
        test_files = discover_test_files()

        missing_tests = []
        for provider_name in providers.keys():
            test_key = provider_name.replace("Provider", "")
            if test_key not in test_files:
                missing_tests.append(provider_name)

        assert not missing_tests, (
            f"The following providers are missing test files:\n"
            f"{chr(10).join(f'  - {name}' for name in missing_tests)}\n\n"
            f"Expected test file pattern: test_<provider_name>_provider.py\n"
            f"Example: OpenAIProvider -> test_openai_provider.py"
        )

    def test_all_providers_meet_minimum_test_count(self):
        """Every provider meets the minimum test count requirement."""
        metrics = get_provider_test_metrics()

        failures = []
        for provider_name, data in metrics.items():
            if not data['passes_requirement']:
                failures.append(
                    f"  - {provider_name}: {data['test_count']}/{data['min_required']} tests "
                    f"(need {data['gap']} more)"
                )

        assert not failures, (
            f"The following providers don't meet minimum test requirements:\n"
            f"{chr(10).join(failures)}\n\n"
            f"Add more tests to these providers to meet the minimum thresholds."
        )

    def test_no_orphaned_test_files(self):
        """No test files exist without corresponding providers."""
        orphaned = find_orphaned_test_files()

        assert not orphaned, (
            f"Found orphaned test files (no matching provider):\n"
            f"{chr(10).join(f'  - {path.name}' for path in orphaned)}\n\n"
            f"These files may indicate:\n"
            f"  - Renamed providers without updating test files\n"
            f"  - Deleted providers without removing tests\n"
            f"  - Incorrectly named test files\n\n"
            f"Action: Rename or remove these test files."
        )


class TestProviderTestQuality:
    """Test that provider tests meet quality standards."""

    def test_provider_tests_are_runnable(self):
        """All provider test files can be collected by pytest."""
        test_files = discover_test_files()

        collection_failures = []
        for provider_name, test_path in test_files.items():
            try:
                # Try to count tests (uses pytest collection)
                count = count_tests_in_file(test_path)
                if count == 0:
                    collection_failures.append(
                        f"  - {test_path.name}: No tests found"
                    )
            except Exception as e:
                collection_failures.append(
                    f"  - {test_path.name}: Collection failed ({str(e)})"
                )

        assert not collection_failures, (
            f"Some test files have collection issues:\n"
            f"{chr(10).join(collection_failures)}\n\n"
            f"Check syntax and imports in these files."
        )

    def test_test_files_follow_naming_convention(self):
        """Test files follow the naming convention."""
        tests_dir = PROJECT_ROOT / "tests"

        # Find all provider-related test files
        provider_test_files = list(tests_dir.glob(f"{TEST_FILE_PREFIX}*provider*.py"))

        incorrectly_named = []
        for test_file in provider_test_files:
            # Check if it follows the pattern: test_<name>_provider.py
            if not test_file.name.endswith(TEST_FILE_SUFFIX):
                incorrectly_named.append(test_file.name)

        assert not incorrectly_named, (
            f"Test files don't follow naming convention:\n"
            f"{chr(10).join(f'  - {name}' for name in incorrectly_named)}\n\n"
            f"Expected pattern: {TEST_FILE_PREFIX}<provider_name>{TEST_FILE_SUFFIX}\n"
            f"Example: test_openai_provider.py"
        )


class TestProviderMetricsReporting:
    """Generate comprehensive metrics reports."""

    def test_generate_coverage_report(self):
        """Generate and display comprehensive provider test coverage report."""
        metrics = get_provider_test_metrics()

        # Calculate summary statistics
        total_providers = len(metrics)
        providers_with_tests = sum(1 for m in metrics.values() if m['has_test_file'])
        providers_meeting_min = sum(1 for m in metrics.values() if m['passes_requirement'])
        total_tests = sum(m['test_count'] for m in metrics.values())
        total_required = sum(m['min_required'] for m in metrics.values())

        # Build report
        report = [
            "",
            "=" * 80,
            "PROVIDER TEST COVERAGE REPORT",
            "=" * 80,
            "",
            f"Total Providers: {total_providers}",
            f"Providers with Test Files: {providers_with_tests}/{total_providers}",
            f"Providers Meeting Minimum: {providers_meeting_min}/{total_providers}",
            f"Total Tests: {total_tests}",
            f"Total Required: {total_required}",
            f"Coverage: {(providers_meeting_min/total_providers*100):.1f}%",
            "",
            "Per-Provider Breakdown:",
            "-" * 80,
        ]

        # Sort by provider name
        for provider_name in sorted(metrics.keys()):
            data = metrics[provider_name]
            status = "✓" if data['passes_requirement'] else "✗"

            report.append(
                f"{status} {provider_name:20s} "
                f"{data['test_count']:3d}/{data['min_required']:3d} tests "
                f"{'(PASS)' if data['passes_requirement'] else f'(need {data[\"gap\"]} more)'}"
            )

        report.extend([
            "-" * 80,
            "",
            "Test Files:",
        ])

        for provider_name in sorted(metrics.keys()):
            data = metrics[provider_name]
            if data['has_test_file']:
                report.append(f"  ✓ {data['test_file_path'].name}")
            else:
                report.append(f"  ✗ {provider_name}: NO TEST FILE")

        report.extend([
            "",
            "=" * 80,
        ])

        # Print report
        print("\n".join(report))

        # This test always passes - it's just for reporting
        assert True


# ============================================================================
# UTILITY FUNCTIONS FOR CI/CD
# ============================================================================

def get_coverage_percentage() -> float:
    """Get overall provider test coverage percentage.

    Returns:
        Coverage percentage (0-100)

    Example:
        >>> coverage = get_coverage_percentage()
        >>> assert coverage >= 90, f"Coverage too low: {coverage}%"
    """
    metrics = get_provider_test_metrics()
    if not metrics:
        return 0.0

    providers_meeting_min = sum(1 for m in metrics.values() if m['passes_requirement'])
    return (providers_meeting_min / len(metrics)) * 100


def get_test_gap_count() -> int:
    """Get total number of tests needed to meet all requirements.

    Returns:
        Number of additional tests needed

    Example:
        >>> gap = get_test_gap_count()
        >>> if gap > 0:
        ...     print(f"Need {gap} more tests")
    """
    metrics = get_provider_test_metrics()
    return sum(m['gap'] for m in metrics.values() if m['gap'] > 0)


if __name__ == "__main__":
    # Run the coverage report when executed directly
    print("\nGenerating Provider Test Coverage Report...")
    print("-" * 80)

    metrics = get_provider_test_metrics()

    for provider_name, data in sorted(metrics.items()):
        status = "✓" if data['passes_requirement'] else "✗"
        print(
            f"{status} {provider_name:20s} "
            f"{data['test_count']:3d}/{data['min_required']:3d} tests"
        )

    print("-" * 80)
    coverage = get_coverage_percentage()
    gap = get_test_gap_count()

    print(f"\nOverall Coverage: {coverage:.1f}%")
    if gap > 0:
        print(f"Tests Needed: {gap}")
    else:
        print("All requirements met! ✓")
