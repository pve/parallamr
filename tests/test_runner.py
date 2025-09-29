"""Tests for experiment runner."""

import pytest

from parallaxr.models import ExperimentStatus
from parallaxr.runner import ExperimentRunner


class TestExperimentRunner:
    """Test experiment runner functionality."""

    def test_initialization(self):
        """Test runner initialization."""
        runner = ExperimentRunner(timeout=60, verbose=True)

        assert runner.timeout == 60
        assert runner.verbose is True
        assert "mock" in runner.providers
        assert "openrouter" in runner.providers
        assert "ollama" in runner.providers

    def test_list_providers(self):
        """Test listing available providers."""
        runner = ExperimentRunner()

        providers = runner.list_providers()

        assert isinstance(providers, list)
        assert "mock" in providers
        assert "openrouter" in providers
        assert "ollama" in providers

    def test_add_custom_provider(self):
        """Test adding custom provider."""
        from parallaxr.providers import MockProvider

        runner = ExperimentRunner()
        custom_provider = MockProvider()

        runner.add_provider("custom", custom_provider)

        assert "custom" in runner.providers
        assert runner.providers["custom"] == custom_provider

    @pytest.mark.asyncio
    async def test_validate_experiments_valid(self, tmp_path):
        """Test validating valid experiments."""
        # Create test CSV
        csv_file = tmp_path / "experiments.csv"
        csv_content = """provider,model,topic
mock,mock,AI"""
        csv_file.write_text(csv_content, encoding='utf-8')

        runner = ExperimentRunner()
        result = await runner.validate_experiments(csv_file)

        assert result["valid"] is True
        assert result["experiments"] == 1
        assert "mock" in result["providers"]
        assert result["providers"]["mock"]["available"] is True

    @pytest.mark.asyncio
    async def test_validate_experiments_invalid_csv(self, tmp_path):
        """Test validating invalid CSV."""
        # Create invalid CSV (missing required columns)
        csv_file = tmp_path / "experiments.csv"
        csv_content = """topic
AI"""
        csv_file.write_text(csv_content, encoding='utf-8')

        runner = ExperimentRunner()
        result = await runner.validate_experiments(csv_file)

        assert result["valid"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_validate_experiments_unknown_provider(self, tmp_path):
        """Test validating experiments with unknown provider."""
        # Create CSV with unknown provider
        csv_file = tmp_path / "experiments.csv"
        csv_content = """provider,model,topic
unknown,model,AI"""
        csv_file.write_text(csv_content, encoding='utf-8')

        runner = ExperimentRunner()
        result = await runner.validate_experiments(csv_file)

        assert result["valid"] is True  # Still valid, just warnings
        assert len(result["warnings"]) > 0
        assert any("Unknown provider" in warning for warning in result["warnings"])

    @pytest.mark.asyncio
    async def test_run_single_experiment_success(self, tmp_path):
        """Test running a single experiment successfully."""
        from parallaxr.models import Experiment

        runner = ExperimentRunner()

        experiment = Experiment(
            provider="mock",
            model="mock",
            variables={"topic": "AI", "source": "Wikipedia"},
            row_number=1
        )

        primary_content = "Test prompt about {{topic}} from {{source}}"
        context_files = []

        result = await runner._run_single_experiment(
            experiment, primary_content, context_files
        )

        assert result.status == ExperimentStatus.OK
        assert result.provider == "mock"
        assert result.model == "mock"
        assert result.variables == {"topic": "AI", "source": "Wikipedia"}
        assert result.input_tokens > 0
        assert result.output_tokens > 0
        assert "MOCK RESPONSE" in result.output
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_run_single_experiment_with_missing_variables(self, tmp_path):
        """Test running experiment with missing template variables."""
        from parallaxr.models import Experiment

        runner = ExperimentRunner()

        experiment = Experiment(
            provider="mock",
            model="mock",
            variables={"topic": "AI"},  # Missing 'source' variable
            row_number=1
        )

        primary_content = "Test prompt about {{topic}} from {{source}}"
        context_files = []

        result = await runner._run_single_experiment(
            experiment, primary_content, context_files
        )

        assert result.status == ExperimentStatus.WARNING
        assert "Variable 'source' not found" in result.error_message

    @pytest.mark.asyncio
    async def test_run_single_experiment_unknown_provider(self, tmp_path):
        """Test running experiment with unknown provider."""
        from parallaxr.models import Experiment

        runner = ExperimentRunner()

        experiment = Experiment(
            provider="unknown",
            model="unknown-model",
            variables={"topic": "AI"},
            row_number=1
        )

        primary_content = "Test prompt about {{topic}}"
        context_files = []

        result = await runner._run_single_experiment(
            experiment, primary_content, context_files
        )

        assert result.status == ExperimentStatus.ERROR
        assert "Unknown provider" in result.error_message

    @pytest.mark.asyncio
    async def test_run_experiments_integration(self, tmp_path):
        """Test full experiment run integration."""
        # Create test files
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Test prompt about {{topic}}", encoding='utf-8')

        context_file = tmp_path / "context.txt"
        context_file.write_text("Additional context", encoding='utf-8')

        experiments_file = tmp_path / "experiments.csv"
        experiments_content = """provider,model,topic
mock,mock,AI
mock,mock,ML"""
        experiments_file.write_text(experiments_content, encoding='utf-8')

        output_file = tmp_path / "results.csv"

        runner = ExperimentRunner()

        # Run experiments
        await runner.run_experiments(
            prompt_file=prompt_file,
            experiments_file=experiments_file,
            output_file=output_file,
            context_files=[context_file]
        )

        # Verify output file was created
        assert output_file.exists()

        # Verify content
        import csv
        with open(output_file, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        assert len(rows) == 2
        assert all(row["status"] == "ok" for row in rows)
        assert all("MOCK RESPONSE" in row["output"] for row in rows)