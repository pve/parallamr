"""Tests for data models."""

import pytest

from parallamr.models import Experiment, ExperimentResult, ExperimentStatus, ProviderResponse


class TestExperiment:
    """Test Experiment model."""

    def test_from_csv_row(self):
        """Test creating experiment from CSV row."""
        row = {
            "provider": "mock",
            "model": "test-model",
            "source": "Wikipedia",
            "topic": "AI"
        }

        experiment = Experiment.from_csv_row(row, 1)

        assert experiment.provider == "mock"
        assert experiment.model == "test-model"
        assert experiment.variables == {"source": "Wikipedia", "topic": "AI"}
        assert experiment.row_number == 1

    def test_from_csv_row_missing_provider(self):
        """Test creating experiment with missing provider."""
        row = {
            "model": "test-model",
            "source": "Wikipedia"
        }

        with pytest.raises(KeyError):
            Experiment.from_csv_row(row, 1)

    def test_from_csv_row_missing_model(self):
        """Test creating experiment with missing model."""
        row = {
            "provider": "mock",
            "source": "Wikipedia"
        }

        with pytest.raises(KeyError):
            Experiment.from_csv_row(row, 1)


class TestProviderResponse:
    """Test ProviderResponse model."""

    def test_status_ok(self):
        """Test OK status determination."""
        response = ProviderResponse(
            output="Test output",
            output_tokens=10,
            success=True
        )

        assert response.status == ExperimentStatus.OK

    def test_status_warning(self):
        """Test WARNING status determination."""
        response = ProviderResponse(
            output="Test output",
            output_tokens=10,
            success=True,
            error_message="Context window unknown"
        )

        assert response.status == ExperimentStatus.WARNING

    def test_status_error(self):
        """Test ERROR status determination."""
        response = ProviderResponse(
            output="",
            output_tokens=0,
            success=False,
            error_message="Model not found"
        )

        assert response.status == ExperimentStatus.ERROR


class TestExperimentResult:
    """Test ExperimentResult model."""

    def test_from_experiment_and_response(self):
        """Test creating result from experiment and response."""
        experiment = Experiment(
            provider="mock",
            model="test-model",
            variables={"source": "Wikipedia", "topic": "AI"},
            row_number=1
        )

        response = ProviderResponse(
            output="Test output",
            output_tokens=10,
            success=True,
            context_window=8192
        )

        result = ExperimentResult.from_experiment_and_response(
            experiment=experiment,
            response=response,
            input_tokens=50
        )

        assert result.provider == "mock"
        assert result.model == "test-model"
        assert result.variables == {"source": "Wikipedia", "topic": "AI"}
        assert result.row_number == 1
        assert result.status == ExperimentStatus.OK
        assert result.input_tokens == 50
        assert result.context_window == 8192
        assert result.output_tokens == 10
        assert result.output == "Test output"
        assert result.error_message is None

    def test_from_experiment_and_response_with_warnings(self):
        """Test creating result with template warnings."""
        experiment = Experiment(
            provider="mock",
            model="test-model",
            variables={"source": "Wikipedia"},
            row_number=1
        )

        response = ProviderResponse(
            output="Test output",
            output_tokens=10,
            success=True
        )

        template_warnings = ["Variable 'topic' not found"]

        result = ExperimentResult.from_experiment_and_response(
            experiment=experiment,
            response=response,
            input_tokens=50,
            template_warnings=template_warnings
        )

        assert result.status == ExperimentStatus.WARNING
        assert result.error_message == "Variable 'topic' not found"

    def test_to_csv_row(self):
        """Test converting result to CSV row."""
        result = ExperimentResult(
            provider="mock",
            model="test-model",
            variables={"source": "Wikipedia", "topic": "AI"},
            row_number=1,
            status=ExperimentStatus.OK,
            input_tokens=50,
            context_window=8192,
            output_tokens=10,
            output="Test output",
            error_message=None
        )

        row = result.to_csv_row()

        expected = {
            "provider": "mock",
            "model": "test-model",
            "source": "Wikipedia",
            "topic": "AI",
            "status": "ok",
            "input_tokens": 50,
            "context_window": 8192,
            "output_tokens": 10,
            "output": "Test output",
            "error_message": ""
        }

        assert row == expected

    def test_to_csv_row_with_none_values(self):
        """Test converting result with None values to CSV row."""
        result = ExperimentResult(
            provider="mock",
            model="test-model",
            variables={},
            row_number=1,
            status=ExperimentStatus.ERROR,
            input_tokens=0,
            context_window=None,
            output_tokens=0,
            output="",
            error_message="Test error"
        )

        row = result.to_csv_row()

        assert row["context_window"] == ""
        assert row["error_message"] == "Test error"