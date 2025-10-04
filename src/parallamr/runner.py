"""Experiment runner for orchestrating LLM experiments."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .csv_writer import IncrementalCSVWriter
from .models import Experiment, ExperimentResult, ExperimentStatus
from .providers import MockProvider, OllamaProvider, OpenRouterProvider, Provider
from .template import combine_files_with_variables
from .token_counter import estimate_tokens, validate_context_window
from .utils import format_experiment_summary, load_context_files, load_experiments_from_csv, load_file_content, validate_output_path


class ExperimentRunner:
    """
    Orchestrates experiment execution across multiple providers and models.
    """

    def __init__(
        self,
        timeout: int = 300,
        verbose: bool = False,
        providers: Optional[Dict[str, Provider]] = None
    ):
        """
        Initialize the experiment runner.

        Args:
            timeout: Request timeout in seconds
            verbose: Enable verbose logging
            providers: Optional provider dictionary (defaults to standard providers)
        """
        self.timeout = timeout
        self.verbose = verbose

        # Use injected providers or create defaults
        if providers is not None:
            self.providers = providers
        else:
            self.providers = self._create_default_providers(timeout)

        self._setup_logging()

    def _create_default_providers(self, timeout: int) -> Dict[str, Provider]:
        """
        Create default provider instances.

        Separated for easier testing and configuration.

        Args:
            timeout: Request timeout in seconds

        Returns:
            Dictionary of provider name to provider instance
        """
        return {
            "mock": MockProvider(timeout=timeout),
            "openrouter": OpenRouterProvider(timeout=timeout),
            "ollama": OllamaProvider(timeout=timeout),
        }

    def _setup_logging(self) -> None:
        """Setup logging configuration for this runner instance."""
        level = logging.INFO if self.verbose else logging.WARNING

        # Use instance-specific logger instead of root logger
        self.logger = logging.getLogger(f"{__name__}.{id(self)}")
        self.logger.setLevel(level)

        # Only add handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter('[%(levelname)s] %(message)s')
            )
            self.logger.addHandler(handler)

        # Prevent propagation to root logger
        self.logger.propagate = False

    async def run_experiments(
        self,
        prompt_file: Optional[str | Path],
        experiments_file: Optional[str | Path],
        output_file: Optional[str | Path],
        context_files: Optional[List[str | Path]] = None,
        read_prompt_stdin: bool = False,
        read_experiments_stdin: bool = False,
    ) -> None:
        """
        Run all experiments and write results to CSV.

        Args:
            prompt_file: Path to the primary prompt file, or None if reading from stdin
            experiments_file: Path to the experiments CSV file, or None if reading from stdin
            output_file: Path to the output CSV file, or None for stdout
            context_files: Optional list of context files to include
            read_prompt_stdin: Whether to read prompt from stdin
            read_experiments_stdin: Whether to read experiments from stdin

        Raises:
            FileNotFoundError: If input files don't exist
            ValueError: If configuration is invalid
        """
        # Load and validate inputs
        if read_prompt_stdin:
            self.logger.info("Reading prompt from stdin")
            primary_content = sys.stdin.read()
        else:
            self.logger.info(f"Loading prompt from {prompt_file}")
            primary_content = load_file_content(prompt_file)

        if read_experiments_stdin:
            self.logger.info("Reading experiments from stdin")
            experiments_content = sys.stdin.read()
            experiments = load_experiments_from_csv(csv_content=experiments_content)
        else:
            self.logger.info(f"Loading experiments from {experiments_file}")
            experiments = load_experiments_from_csv(csv_path=experiments_file)

        context_file_contents = []
        if context_files:
            self.logger.info(f"Loading {len(context_files)} context file(s)")
            context_file_contents = load_context_files(context_files)

        # Validate output path
        output_path = validate_output_path(output_file)
        csv_writer = IncrementalCSVWriter(output_path)

        # Log experiment summary
        self.logger.info(format_experiment_summary(experiments))
        if output_path:
            self.logger.info(f"Results will be written to {output_path}")
        else:
            self.logger.info("Results will be written to stdout")

        # Run experiments sequentially
        total_experiments = len(experiments)
        for i, experiment in enumerate(experiments, 1):
            self.logger.info(f"Starting experiment {i}/{total_experiments}: {experiment.provider}/{experiment.model}")

            result = await self._run_single_experiment(
                experiment=experiment,
                primary_content=primary_content,
                context_files=context_file_contents,
            )

            # Write result immediately
            csv_writer.write_result(result)

            self.logger.info(f"Completed experiment {i}/{total_experiments}: status={result.status.value}")

            # Log any warnings or errors
            if result.error_message:
                if result.status == ExperimentStatus.WARNING:
                    self.logger.warning(f"Experiment {i} warning: {result.error_message}")
                else:
                    self.logger.error(f"Experiment {i} error: {result.error_message}")

        if output_path:
            self.logger.info(f"All experiments completed. Results written to {output_path}")
        else:
            self.logger.info("All experiments completed")

    async def _run_single_experiment(
        self,
        experiment: Experiment,
        primary_content: str,
        context_files: List[Tuple[str, str]],
    ) -> ExperimentResult:
        """
        Run a single experiment.

        Args:
            experiment: Experiment configuration
            primary_content: Primary prompt file content
            context_files: List of (filename, content) tuples for context files

        Returns:
            ExperimentResult containing the execution result
        """
        try:
            # Get the appropriate provider
            provider = self.providers.get(experiment.provider)
            if not provider:
                return self._create_error_result(
                    experiment,
                    f"Unknown provider: {experiment.provider}"
                )

            # Combine files and replace variables
            combined_content, missing_variables = combine_files_with_variables(
                primary_content,
                context_files,
                experiment.variables
            )

            # Estimate input tokens
            input_tokens = estimate_tokens(combined_content)

            # Get model context window
            context_window = await provider.get_context_window(experiment.model)

            # Validate context window
            context_valid, context_warning = validate_context_window(
                input_tokens,
                context_window,
                model=experiment.model,
                provider=experiment.provider
            )

            # Prepare warnings list
            warnings = []
            if missing_variables:
                for var in missing_variables:
                    warnings.append(f"Variable '{{{{{var}}}}}' in template has no value in experiment row {experiment.row_number}")

            if context_warning:
                warnings.append(context_warning)

            if not context_valid:
                return self._create_error_result(
                    experiment,
                    f"Input tokens ({input_tokens}) exceed model context window ({context_window})",
                    input_tokens
                )

            # Get completion from provider
            provider_response = await provider.get_completion(
                prompt=combined_content,
                model=experiment.model,
                variables=experiment.variables  # Pass variables for providers that might use them
            )

            # Create result
            result = ExperimentResult.from_experiment_and_response(
                experiment=experiment,
                response=provider_response,
                input_tokens=input_tokens,
                template_warnings=warnings if warnings else None
            )

            return result

        except Exception as e:
            self.logger.exception(f"Unexpected error in experiment {experiment.row_number}")
            return self._create_error_result(
                experiment,
                f"Unexpected error: {str(e)}"
            )

    def _create_error_result(
        self,
        experiment: Experiment,
        error_message: str,
        input_tokens: int = 0
    ) -> ExperimentResult:
        """
        Create an error result for a failed experiment.

        Args:
            experiment: The experiment that failed
            error_message: Error description
            input_tokens: Estimated input tokens (if available)

        Returns:
            ExperimentResult with error status
        """
        return ExperimentResult(
            provider=experiment.provider,
            model=experiment.model,
            variables=experiment.variables,
            row_number=experiment.row_number,
            status=ExperimentStatus.ERROR,
            input_tokens=input_tokens,
            context_window=None,
            output_tokens=0,
            output="",
            error_message=error_message
        )

    def add_provider(self, name: str, provider: Provider) -> None:
        """
        Add a custom provider.

        Args:
            name: Provider name
            provider: Provider instance
        """
        self.providers[name] = provider

    def list_providers(self) -> List[str]:
        """
        List available provider names.

        Returns:
            List of provider names
        """
        return list(self.providers.keys())

    async def validate_experiments(
        self,
        experiments_file: Optional[str | Path],
        read_stdin: bool = False
    ) -> Dict[str, Any]:
        """
        Validate experiments without running them.

        Args:
            experiments_file: Path to experiments CSV file, or None if reading from stdin
            read_stdin: Whether to read from stdin

        Returns:
            Dictionary with validation results
        """
        try:
            if read_stdin:
                experiments_content = sys.stdin.read()
                experiments = load_experiments_from_csv(csv_content=experiments_content)
            else:
                experiments = load_experiments_from_csv(csv_path=experiments_file)
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "experiments": 0
            }

        validation_results = {
            "valid": True,
            "experiments": len(experiments),
            "providers": {},
            "warnings": []
        }

        # Check each experiment
        for experiment in experiments:
            provider_name = experiment.provider

            if provider_name not in validation_results["providers"]:
                validation_results["providers"][provider_name] = {
                    "available": provider_name in self.providers,
                    "models": []
                }

            if provider_name in self.providers:
                provider = self.providers[provider_name]
                model_available = provider.is_model_available(experiment.model)
                validation_results["providers"][provider_name]["models"].append({
                    "model": experiment.model,
                    "available": model_available
                })

                if not model_available:
                    validation_results["warnings"].append(
                        f"Model {experiment.model} may not be available for provider {provider_name}"
                    )
            else:
                validation_results["warnings"].append(
                    f"Unknown provider: {provider_name}"
                )

        return validation_results