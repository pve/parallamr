"""Data models for parallamr experiments and results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class ExperimentStatus(str, Enum):
    """Status of an experiment execution."""

    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Experiment:
    """Represents a single experiment configuration."""

    provider: str
    model: str
    variables: Dict[str, Any]
    row_number: int

    @classmethod
    def from_csv_row(cls, row: Dict[str, Any], row_number: int) -> Experiment:
        """Create an Experiment from a CSV row."""
        provider = row.pop("provider")
        model = row.pop("model")
        variables = row  # Remaining columns become variables

        return cls(
            provider=provider,
            model=model,
            variables=variables,
            row_number=row_number
        )


@dataclass
class ProviderResponse:
    """Response from an LLM provider."""

    output: str
    output_tokens: int
    success: bool
    error_message: Optional[str] = None
    context_window: Optional[int] = None

    @property
    def status(self) -> ExperimentStatus:
        """Determine status based on response state."""
        if not self.success:
            return ExperimentStatus.ERROR
        elif self.error_message:
            return ExperimentStatus.WARNING
        else:
            return ExperimentStatus.OK


@dataclass
class ExperimentResult:
    """Complete result of an experiment execution."""

    # Original experiment data
    provider: str
    model: str
    variables: Dict[str, Any]
    row_number: int

    # Execution results
    status: ExperimentStatus
    input_tokens: int
    context_window: Optional[int]
    output_tokens: int
    output: str
    error_message: Optional[str] = None
    json_fields: Optional[Dict[str, Any]] = None

    @classmethod
    def from_experiment_and_response(
        cls,
        experiment: Experiment,
        response: ProviderResponse,
        input_tokens: int,
        template_warnings: Optional[list[str]] = None,
        json_fields: Optional[Dict[str, Any]] = None,
    ) -> ExperimentResult:
        """Create a result from an experiment and provider response."""
        # Combine any template warnings with provider error message
        error_messages = []
        if template_warnings:
            error_messages.extend(template_warnings)
        if response.error_message:
            error_messages.append(response.error_message)

        combined_error = "; ".join(error_messages) if error_messages else None

        # Determine final status
        final_status = response.status
        if template_warnings and final_status == ExperimentStatus.OK:
            final_status = ExperimentStatus.WARNING

        return cls(
            provider=experiment.provider,
            model=experiment.model,
            variables=experiment.variables,
            row_number=experiment.row_number,
            status=final_status,
            input_tokens=input_tokens,
            context_window=response.context_window,
            output_tokens=response.output_tokens,
            output=response.output,
            error_message=combined_error,
            json_fields=json_fields,
        )

    def to_csv_row(self) -> Dict[str, Any]:
        """Convert result to a CSV row dictionary."""
        row = self.variables.copy()

        # Add JSON fields if present
        if self.json_fields:
            row.update(self.json_fields)

        # Clean output: replace newlines and carriage returns with spaces for Excel compatibility
        cleaned_output = self.output.replace('\n', ' ').replace('\r', ' ') if self.output else ""

        row.update({
            "provider": self.provider,
            "model": self.model,
            "status": self.status.value,
            "input_tokens": self.input_tokens,
            "context_window": self.context_window or "",
            "output_tokens": self.output_tokens,
            "output": cleaned_output,
            "error_message": self.error_message or "",
        })
        return row