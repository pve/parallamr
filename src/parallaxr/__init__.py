"""Parallaxr: A command-line tool for running systematic experiments across multiple LLM providers."""

__version__ = "0.1.0"
__author__ = "HiveMind Collective"
__email__ = "contact@parallaxr.dev"

from .models import Experiment, ExperimentResult, ProviderResponse

__all__ = ["Experiment", "ExperimentResult", "ProviderResponse"]