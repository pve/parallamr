"""Experiment runner for orchestrating LLM experiments."""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .csv_writer import IncrementalCSVWriter
from .file_loader import FileLoader
from .models import Experiment, ExperimentResult, ExperimentStatus
from .path_template import PathSubstitutionError, substitute_path_template
from .providers import MockProvider, OllamaProvider, OpenRouterProvider, Provider
from .template import combine_files_with_variables
from .token_counter import estimate_tokens, validate_context_window
from .utils import format_experiment_summary, validate_output_path


@dataclass
class TokenValidation:
    """Result of token limit validation."""
    valid: bool
    input_tokens: int
    warnings: List[str]
    error_message: Optional[str] = None


class ExperimentRunner:
    """
    Orchestrates experiment execution across multiple providers and models.
    """

    def __init__(
        self,
        timeout: int = 300,
        verbose: bool = False,
        providers: Optional[Dict[str, Provider]] = None,
        file_loader: Optional[FileLoader] = None
    ):
        """
        Initialize the experiment runner.

        Args:
            timeout: Request timeout in seconds
            verbose: Enable verbose logging
            providers: Optional provider dictionary (defaults to standard providers)
            file_loader: Optional file loader (defaults to FileLoader instance)
        """
        self.timeout = timeout
        self.verbose = verbose

        # Use injected providers or create defaults
        if providers is not None:
            self.providers = providers
        else:
            self.providers = self._create_default_providers(timeout)

        # Use injected file loader or create default
        self.file_loader = file_loader or FileLoader()

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

    def _has_template_variables(self, path: Optional[str | Path]) -> bool:
        """Check if a path contains template variables {{}}."""
        if path is None:
            return False
        path_str = str(path)
        return "{{" in path_str and "}}" in path_str

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

        Supports template variable substitution in output_file path.
        If output_file contains {{variable}}, experiments are grouped by
        their resolved output path and written to separate files.

        Args:
            prompt_file: Path to the primary prompt file, or None if reading from stdin
            experiments_file: Path to the experiments CSV file, or None if reading from stdin
            output_file: Path to the output CSV file (supports {{variable}} templates), or None for stdout
            context_files: Optional list of context files to include
            read_prompt_stdin: Whether to read prompt from stdin
            read_experiments_stdin: Whether to read experiments from stdin

        Raises:
            FileNotFoundError: If input files don't exist
            ValueError: If configuration is invalid
            PathSubstitutionError: If template variables are missing or path is unsafe
        """
        # Load and validate inputs using FileLoader
        if read_prompt_stdin:
            self.logger.info("Reading prompt from stdin")
        else:
            self.logger.info(f"Loading prompt from {prompt_file}")
        primary_content = self.file_loader.load_prompt(
            Path(prompt_file) if prompt_file else None,
            read_prompt_stdin
        )

        if read_experiments_stdin:
            self.logger.info("Reading experiments from stdin")
        else:
            self.logger.info(f"Loading experiments from {experiments_file}")
        experiments = self.file_loader.load_experiments(
            Path(experiments_file) if experiments_file else None,
            read_experiments_stdin
        )

        context_file_contents = []
        if context_files:
            self.logger.info(f"Loading {len(context_files)} context file(s)")
            context_file_contents = self.file_loader.load_context(
                [Path(f) for f in context_files]
            )

        # Check if output path contains template variables
        if self._has_template_variables(output_file):
            # Group experiments by resolved output path
            await self._run_experiments_with_templated_output(
                experiments=experiments,
                primary_content=primary_content,
                context_file_contents=context_file_contents,
                output_template=str(output_file)
            )
        else:
            # Original behavior: single output file
            await self._run_experiments_to_single_output(
                experiments=experiments,
                primary_content=primary_content,
                context_file_contents=context_file_contents,
                output_file=output_file
            )

    async def _run_experiments_to_single_output(
        self,
        experiments: List[Experiment],
        primary_content: str,
        context_file_contents: List[Tuple[str, str]],
        output_file: Optional[str | Path]
    ) -> None:
        """Run experiments and write to a single output file."""
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

        # Close writer
        csv_writer.close()

        if output_path:
            self.logger.info(f"All experiments completed. Results written to {output_path}")
        else:
            self.logger.info("All experiments completed")

    async def _run_experiments_with_templated_output(
        self,
        experiments: List[Experiment],
        primary_content: str,
        context_file_contents: List[Tuple[str, str]],
        output_template: str
    ) -> None:
        """Run experiments with template variable substitution in output paths."""
        # Group experiments by their resolved output path
        output_groups: Dict[Path, List[Experiment]] = defaultdict(list)

        for experiment in experiments:
            try:
                resolved_path = substitute_path_template(
                    output_template,
                    experiment.variables
                )
                output_groups[resolved_path].append(experiment)
            except PathSubstitutionError as e:
                self.logger.error(f"Failed to resolve output path for experiment {experiment.row_number}: {e}")
                # Create error result for this experiment
                error_result = self._create_error_result(experiment, str(e))
                # Since we can't determine output file, log the error but continue
                continue

        # Log summary
        total_experiments = len(experiments)
        num_output_files = len(output_groups)
        self.logger.info(f"Running {total_experiments} experiments across {num_output_files} output file(s)")

        # Create directories for all output paths
        for output_path in output_groups.keys():
            if output_path.parent != Path('.'):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created directory: {output_path.parent}")

        # Run experiments grouped by output file
        experiment_counter = 0
        for output_path, exp_group in output_groups.items():
            self.logger.info(f"Writing {len(exp_group)} experiment(s) to {output_path}")
            csv_writer = IncrementalCSVWriter(output_path)

            for experiment in exp_group:
                experiment_counter += 1
                self.logger.info(f"Starting experiment {experiment_counter}/{total_experiments}: {experiment.provider}/{experiment.model}")

                result = await self._run_single_experiment(
                    experiment=experiment,
                    primary_content=primary_content,
                    context_files=context_file_contents,
                )

                # Write result immediately
                csv_writer.write_result(result)

                self.logger.info(f"Completed experiment {experiment_counter}/{total_experiments}: status={result.status.value}")

                # Log any warnings or errors
                if result.error_message:
                    if result.status == ExperimentStatus.WARNING:
                        self.logger.warning(f"Experiment {experiment_counter} warning: {result.error_message}")
                    else:
                        self.logger.error(f"Experiment {experiment_counter} error: {result.error_message}")

            # Close writer for this output file
            csv_writer.close()

        self.logger.info(f"All experiments completed. Results written to {num_output_files} file(s)")

    def _get_provider(self, experiment: Experiment) -> Optional[Provider]:
        """
        Get provider for experiment.

        Args:
            experiment: Experiment configuration

        Returns:
            Provider instance or None if not found
        """
        return self.providers.get(experiment.provider)

    def _prepare_prompt(
        self,
        experiment: Experiment,
        primary_content: str,
        context_files: List[Tuple[str, str]]
    ) -> Tuple[str, List[str]]:
        """
        Prepare prompt with variable substitution.

        Args:
            experiment: Experiment configuration
            primary_content: Primary prompt file content
            context_files: List of (filename, content) tuples

        Returns:
            Tuple of (combined_content, warnings)
        """
        combined_content, missing_variables = combine_files_with_variables(
            primary_content,
            context_files,
            experiment.variables
        )

        warnings = []
        if missing_variables:
            for var in missing_variables:
                warnings.append(
                    f"Variable '{{{{{var}}}}}' in template has no value in experiment row {experiment.row_number}"
                )

        return combined_content, warnings

    async def _validate_token_limits(
        self,
        experiment: Experiment,
        provider: Provider,
        combined_content: str
    ) -> TokenValidation:
        """
        Validate token limits for experiment.

        Args:
            experiment: Experiment configuration
            provider: Provider instance
            combined_content: Prepared prompt content

        Returns:
            TokenValidation result
        """
        input_tokens = estimate_tokens(combined_content)
        context_window = await provider.get_context_window(experiment.model)

        context_valid, context_warning = validate_context_window(
            input_tokens,
            context_window,
            model=experiment.model,
            provider=experiment.provider
        )

        warnings = []
        if context_warning:
            warnings.append(context_warning)

        if not context_valid:
            return TokenValidation(
                valid=False,
                input_tokens=input_tokens,
                warnings=warnings,
                error_message=f"Input tokens ({input_tokens}) exceed model context window ({context_window})"
            )

        return TokenValidation(
            valid=True,
            input_tokens=input_tokens,
            warnings=warnings
        )

    async def _run_single_experiment(
        self,
        experiment: Experiment,
        primary_content: str,
        context_files: List[Tuple[str, str]],
    ) -> ExperimentResult:
        """
        Run a single experiment (orchestrator method).

        This method coordinates the experiment execution by:
        1. Getting the provider
        2. Preparing the prompt with variable substitution
        3. Validating token limits
        4. Calling the provider
        5. Creating the result

        Args:
            experiment: Experiment configuration
            primary_content: Primary prompt file content
            context_files: List of (filename, content) tuples for context files

        Returns:
            ExperimentResult containing the execution result
        """
        try:
            # Step 1: Get provider
            provider = self._get_provider(experiment)
            if not provider:
                return self._create_error_result(
                    experiment,
                    f"Unknown provider: {experiment.provider}"
                )

            # Step 2: Prepare prompt
            combined_content, template_warnings = self._prepare_prompt(
                experiment,
                primary_content,
                context_files
            )

            # Step 3: Validate token limits
            validation = await self._validate_token_limits(
                experiment,
                provider,
                combined_content
            )

            if not validation.valid:
                return self._create_error_result(
                    experiment,
                    validation.error_message,
                    validation.input_tokens
                )

            # Step 4: Get completion from provider
            provider_response = await provider.get_completion(
                prompt=combined_content,
                model=experiment.model,
                variables=experiment.variables
            )

            # Step 5: Create result
            all_warnings = template_warnings + validation.warnings
            result = ExperimentResult.from_experiment_and_response(
                experiment=experiment,
                response=provider_response,
                input_tokens=validation.input_tokens,
                template_warnings=all_warnings if all_warnings else None
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
            experiments = self.file_loader.load_experiments(
                Path(experiments_file) if experiments_file else None,
                read_stdin
            )
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