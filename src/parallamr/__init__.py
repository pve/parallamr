"""Parallamr: A command-line tool for running systematic experiments across multiple LLM providers."""

__version__ = "0.4.0"
__author__ = "Peter's HiveMind Collective"
__email__ = "contact@parallamr.dev"

from .models import Experiment, ExperimentResult, ProviderResponse

__all__ = ["Experiment", "ExperimentResult", "ProviderResponse"]