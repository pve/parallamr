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
from .providers import MockProvider, OllamaProvider, OpenAIProvider, OpenRouterProvider, Provider
from .template import combine_files_with_variables
from .token_counter import estimate_tokens, validate_context_window
from .utils import format_experiment_summary, load_file_content, validate_output_path


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
        file_loader: Optional[FileLoader] = None,
        flatten_json: bool = False,
        max_concurrent: Optional[int] = None,
        sequential: bool = False,
        provider_concurrency: Optional[Dict[str, int]] = None
    ):
        """
        Initialize the experiment runner.

        Args:
            timeout: Request timeout in seconds
            verbose: Enable verbose logging
            providers: Optional provider dictionary (defaults to standard providers)
            file_loader: Optional file loader (defaults to FileLoader instance)
            flatten_json: Enable JSON extraction and flattening from LLM outputs
            max_concurrent: Global maximum concurrent experiments (overrides provider limits)
            sequential: Force sequential execution (equivalent to max_concurrent=1)
            provider_concurrency: Per-provider concurrency limits (e.g., {"openrouter": 5})
        """
        self.timeout = timeout
        self.verbose = verbose
        self.flatten_json = flatten_json

        # Use injected providers or create defaults
        if providers is not None:
            self.providers = providers
        else:
            self.providers = self._create_default_providers(timeout)

        # Use injected file loader or create default
        self.file_loader = file_loader or FileLoader()

        # Setup logging first (needed for concurrency setup)
        self._setup_logging()

        # Setup concurrency control
        self._setup_concurrency(max_concurrent, sequential, provider_concurrency)

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
            "openai": OpenAIProvider(timeout=timeout),
        }

    def _setup_concurrency(
        self,
        max_concurrent: Optional[int],
        sequential: bool,
        provider_concurrency: Optional[Dict[str, int]]
    ) -> None:
        """
        Setup concurrency control with provider-specific semaphores.

        Args:
            max_concurrent: Global maximum concurrent experiments
            sequential: Force sequential execution
            provider_concurrency: Per-provider concurrency limits
        """
        # Default concurrency limits per provider
        default_limits = {
            "openrouter": 10,
            "ollama": 1,
            "openai": 10,
            "mock": 50
        }

        # Override with user-provided limits
        if provider_concurrency:
            default_limits.update(provider_concurrency)

        # Handle sequential flag
        if sequential:
            max_concurrent = 1

        # Create provider-specific semaphores
        self._provider_semaphores: Dict[str, asyncio.Semaphore] = {}
        for provider_name in self.providers.keys():
            limit = default_limits.get(provider_name, 10)  # Default to 10 if not specified
            if max_concurrent is not None:
                limit = min(limit, max_concurrent)
            self._provider_semaphores[provider_name] = asyncio.Semaphore(limit)

        # Global semaphore if max_concurrent is set
        self._global_semaphore: Optional[asyncio.Semaphore] = None
        if max_concurrent is not None:
            self._global_semaphore = asyncio.Semaphore(max_concurrent)

        self.logger.debug(f"Concurrency limits: {[(k, v._value) for k, v in self._provider_semaphores.items()]}")

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

    def _resolve_and_load_input_files(
        self,
        experiment: Experiment,
        prompt_template: Optional[str | Path],
        context_templates: Optional[List[str | Path]],
        read_prompt_stdin: bool
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Resolve template variables in input file paths and load the files.

        Args:
            experiment: Experiment with variables for substitution
            prompt_template: Prompt file path (may contain {{variables}})
            context_templates: Context file paths (may contain {{variables}})
            read_prompt_stdin: Whether prompt should be read from stdin

        Returns:
            Tuple of (prompt_content, context_file_contents)

        Raises:
            PathSubstitutionError: If template variables are missing or path is unsafe
            FileNotFoundError: If resolved file doesn't exist
        """
        # Load prompt
        if read_prompt_stdin:
            prompt_content = self.file_loader.load_prompt(None, use_stdin=True)
        elif self._has_template_variables(prompt_template):
            # Resolve template in prompt path
            resolved_prompt_path = substitute_path_template(
                str(prompt_template),
                experiment.variables
            )
            self.logger.debug(f"Resolved prompt path: {resolved_prompt_path}")
            prompt_content = self.file_loader.load_prompt(resolved_prompt_path, use_stdin=False)
        else:
            # No template, load normally
            prompt_content = self.file_loader.load_prompt(
                Path(prompt_template) if prompt_template else None,
                use_stdin=False
            )

        # Load context files
        context_file_contents = []
        if context_templates:
            for context_template in context_templates:
                if self._has_template_variables(context_template):
                    # Resolve template in context path
                    resolved_context_path = substitute_path_template(
                        str(context_template),
                        experiment.variables
                    )
                    self.logger.debug(f"Resolved context path: {resolved_context_path}")
                    content = load_file_content(resolved_context_path)
                    context_file_contents.append((str(resolved_context_path.name), content))
                else:
                    # No template, load normally
                    content = load_file_content(Path(context_template))
                    context_file_contents.append((Path(context_template).name, content))

        return prompt_content, context_file_contents

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

        Supports template variable substitution in all file paths (prompt, context, output).
        If any path contains {{variable}}, it will be resolved per-experiment using
        that experiment's variables.

        Args:
            prompt_file: Path to prompt file (supports {{variable}} templates), or None for stdin
            experiments_file: Path to experiments CSV file (NO template support)
            output_file: Path to output CSV file (supports {{variable}} templates), or None for stdout
            context_files: Context file paths (each supports {{variable}} templates)
            read_prompt_stdin: Whether to read prompt from stdin
            read_experiments_stdin: Whether to read experiments from stdin

        Raises:
            FileNotFoundError: If input files don't exist
            ValueError: If configuration is invalid
            PathSubstitutionError: If template variables are missing or path is unsafe
        """
        # Load experiments first (always needed, no templating)
        if read_experiments_stdin:
            self.logger.info("Reading experiments from stdin")
        else:
            self.logger.info(f"Loading experiments from {experiments_file}")
        experiments = self.file_loader.load_experiments(
            Path(experiments_file) if experiments_file else None,
            read_experiments_stdin
        )

        # Check if ANY input path has templates
        has_templated_prompt = self._has_template_variables(prompt_file) and not read_prompt_stdin
        has_templated_context = any(
            self._has_template_variables(ctx) for ctx in (context_files or [])
        )
        has_templated_output = self._has_template_variables(output_file)
        has_any_templates = has_templated_prompt or has_templated_context or has_templated_output

        if has_any_templates:
            # Per-experiment resolution (load files individually for each experiment)
            self.logger.info("Detected templated paths - using per-experiment file resolution")
            await self._run_experiments_with_templated_paths(
                experiments=experiments,
                prompt_template=prompt_file,
                context_templates=list(context_files) if context_files else None,
                output_template=str(output_file) if output_file else None,
                read_prompt_stdin=read_prompt_stdin
            )
        else:
            # Original behavior: load once, use for all experiments
            if read_prompt_stdin:
                self.logger.info("Reading prompt from stdin")
            else:
                self.logger.info(f"Loading prompt from {prompt_file}")
            primary_content = self.file_loader.load_prompt(
                Path(prompt_file) if prompt_file else None,
                read_prompt_stdin
            )

            context_file_contents = []
            if context_files:
                self.logger.info(f"Loading {len(context_files)} context file(s)")
                context_file_contents = self.file_loader.load_context(
                    [Path(f) for f in context_files]
                )

            await self._run_experiments_to_single_output(
                experiments=experiments,
                primary_content=primary_content,
                context_file_contents=context_file_contents,
                output_file=output_file
            )

    async def _run_experiments_with_templated_paths(
        self,
        experiments: List[Experiment],
        prompt_template: Optional[str | Path],
        context_templates: Optional[List[str | Path]],
        output_template: Optional[str],
        read_prompt_stdin: bool
    ) -> None:
        """
        Run experiments with template variable substitution in input/output paths.

        For each experiment, resolves templates, loads files, and runs the experiment.
        Handles missing files gracefully by creating error results.
        """
        # Group by output path if output is templated
        if output_template and self._has_template_variables(output_template):
            # Group experiments by resolved output path
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
                    continue

            # Create directories for output paths
            for output_path in output_groups.keys():
                if output_path.parent != Path('.'):
                    output_path.parent.mkdir(parents=True, exist_ok=True)

            # Run experiments grouped by output file
            total_experiments = len(experiments)
            experiment_counter = 0

            for output_path, exp_group in output_groups.items():
                self.logger.info(f"Writing {len(exp_group)} experiment(s) to {output_path}")
                csv_writer = IncrementalCSVWriter(output_path)

                for experiment in exp_group:
                    experiment_counter += 1
                    result = await self._run_single_experiment_with_templated_inputs(
                        experiment=experiment,
                        prompt_template=prompt_template,
                        context_templates=context_templates,
                        read_prompt_stdin=read_prompt_stdin,
                        experiment_num=experiment_counter,
                        total=total_experiments
                    )
                    await csv_writer.write_result(result)

                await csv_writer.close()

            self.logger.info(f"All experiments completed. Results written to {len(output_groups)} file(s)")

        else:
            # Single output file, but inputs may be templated
            output_path = validate_output_path(output_template)
            csv_writer = IncrementalCSVWriter(output_path)

            total_experiments = len(experiments)
            for i, experiment in enumerate(experiments, 1):
                result = await self._run_single_experiment_with_templated_inputs(
                    experiment=experiment,
                    prompt_template=prompt_template,
                    context_templates=context_templates,
                    read_prompt_stdin=read_prompt_stdin,
                    experiment_num=i,
                    total=total_experiments
                )
                await csv_writer.write_result(result)

            await csv_writer.close()

            if output_path:
                self.logger.info(f"All experiments completed. Results written to {output_path}")
            else:
                self.logger.info("All experiments completed")

    async def _run_single_experiment_with_templated_inputs(
        self,
        experiment: Experiment,
        prompt_template: Optional[str | Path],
        context_templates: Optional[List[str | Path]],
        read_prompt_stdin: bool,
        experiment_num: int,
        total: int
    ) -> ExperimentResult:
        """
        Run a single experiment with template resolution for input files.

        Handles file loading errors gracefully by creating error results.
        """
        self.logger.info(f"Starting experiment {experiment_num}/{total}: {experiment.provider}/{experiment.model}")

        try:
            # Resolve and load input files for this experiment
            primary_content, context_file_contents = self._resolve_and_load_input_files(
                experiment=experiment,
                prompt_template=prompt_template,
                context_templates=context_templates,
                read_prompt_stdin=read_prompt_stdin
            )

            # Run the experiment
            result = await self._run_single_experiment(
                experiment=experiment,
                primary_content=primary_content,
                context_files=context_file_contents,
            )

        except (FileNotFoundError, PathSubstitutionError) as e:
            # File loading error - create error result
            self.logger.error(f"File loading error for experiment {experiment_num}: {e}")
            result = self._create_error_result(
                experiment,
                f"File loading error: {str(e)}"
            )
        except Exception as e:
            # Unexpected error
            self.logger.exception(f"Unexpected error in experiment {experiment_num}")
            result = self._create_error_result(
                experiment,
                f"Unexpected error: {str(e)}"
            )

        self.logger.info(f"Completed experiment {experiment_num}/{total}: status={result.status.value}")

        if result.error_message:
            if result.status == ExperimentStatus.WARNING:
                self.logger.warning(f"Experiment {experiment_num} warning: {result.error_message}")
            else:
                self.logger.error(f"Experiment {experiment_num} error: {result.error_message}")

        return result

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

        # Run experiments concurrently
        total_experiments = len(experiments)

        # Create wrapped experiment tasks
        tasks = []
        for i, experiment in enumerate(experiments, 1):
            task = self._run_experiment_with_semaphore(
                experiment=experiment,
                primary_content=primary_content,
                context_files=context_file_contents,
                experiment_num=i,
                total=total_experiments
            )
            tasks.append(task)

        # Execute all experiments concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Write results as they complete and handle any exceptions
        for i, result in enumerate(results, 1):
            experiment = experiments[i - 1]

            # Handle exceptions from gather
            if isinstance(result, Exception):
                self.logger.exception(f"Unexpected error in experiment {i}")
                result = self._create_error_result(
                    experiment,
                    f"Unexpected error: {str(result)}"
                )

            # Write result immediately
            await csv_writer.write_result(result)

            # Log completion
            self.logger.info(f"Completed experiment {i}/{total_experiments}: status={result.status.value}")

            # Log any warnings or errors
            if result.error_message:
                if result.status == ExperimentStatus.WARNING:
                    self.logger.warning(f"Experiment {i} warning: {result.error_message}")
                else:
                    self.logger.error(f"Experiment {i} error: {result.error_message}")

        # Close writer
        await csv_writer.close()

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
                await csv_writer.write_result(result)

                self.logger.info(f"Completed experiment {experiment_counter}/{total_experiments}: status={result.status.value}")

                # Log any warnings or errors
                if result.error_message:
                    if result.status == ExperimentStatus.WARNING:
                        self.logger.warning(f"Experiment {experiment_counter} warning: {result.error_message}")
                    else:
                        self.logger.error(f"Experiment {experiment_counter} error: {result.error_message}")

            # Close writer for this output file
            await csv_writer.close()

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

    async def _run_experiment_with_semaphore(
        self,
        experiment: Experiment,
        primary_content: str,
        context_files: List[Tuple[str, str]],
        experiment_num: int,
        total: int
    ) -> ExperimentResult:
        """
        Run a single experiment with semaphore-based rate limiting.

        This wrapper applies both provider-specific and global semaphores
        to control concurrency and prevent rate limiting.

        Args:
            experiment: Experiment configuration
            primary_content: Primary prompt file content
            context_files: List of (filename, content) tuples
            experiment_num: Experiment number for logging
            total: Total number of experiments for logging

        Returns:
            ExperimentResult containing the execution result
        """
        # Get the appropriate semaphore for this provider
        provider_semaphore = self._provider_semaphores.get(experiment.provider)

        self.logger.info(f"Starting experiment {experiment_num}/{total}: {experiment.provider}/{experiment.model}")

        try:
            # Acquire both global and provider-specific semaphores
            async with provider_semaphore:
                if self._global_semaphore:
                    async with self._global_semaphore:
                        result = await self._run_single_experiment(
                            experiment=experiment,
                            primary_content=primary_content,
                            context_files=context_files,
                        )
                else:
                    result = await self._run_single_experiment(
                        experiment=experiment,
                        primary_content=primary_content,
                        context_files=context_files,
                    )

            return result

        except Exception as e:
            self.logger.exception(f"Unexpected error in experiment {experiment_num}")
            return self._create_error_result(
                experiment,
                f"Unexpected error: {str(e)}"
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

            # Step 4b: Extract and flatten JSON if enabled
            json_fields = None
            if self.flatten_json and provider_response.success:
                json_fields = self._extract_and_flatten_json(
                    output=provider_response.output,
                    experiment=experiment
                )

            # Step 5: Create result
            all_warnings = template_warnings + validation.warnings
            result = ExperimentResult.from_experiment_and_response(
                experiment=experiment,
                response=provider_response,
                input_tokens=validation.input_tokens,
                template_warnings=all_warnings if all_warnings else None,
                json_fields=json_fields
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

    def _extract_and_flatten_json(
        self,
        output: str,
        experiment: Experiment
    ) -> Optional[Dict[str, Any]]:
        """
        Extract and flatten JSON from provider output.

        Args:
            output: Raw provider output string
            experiment: Experiment configuration for column name resolution

        Returns:
            Dictionary of flattened JSON fields, or None if no valid JSON found
        """
        from . import json_extractor

        try:
            # Extract JSON from output (handles markdown code fences)
            json_data = json_extractor.extract_json(output)
            if json_data is None:
                # No JSON found - this is OK, not an error
                return None

            # Flatten the JSON structure
            flat_data = json_extractor.flatten_json(json_data)
            if not flat_data:
                return None

            # Resolve column name conflicts
            reserved_columns = {
                "provider", "model", "status", "input_tokens",
                "context_window", "output_tokens", "output", "error_message"
            }
            resolved_data = json_extractor.resolve_column_names(
                flat_data=flat_data,
                reserved_columns=reserved_columns,
                experiment_vars=set(experiment.variables.keys())
            )

            return resolved_data

        except Exception as e:
            # Log but don't fail the experiment - JSON extraction is optional
            self.logger.debug(f"JSON extraction failed for experiment {experiment.row_number}: {e}")
            return None

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