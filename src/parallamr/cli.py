"""Command-line interface for parallamr."""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Optional

import click
from dotenv import load_dotenv, find_dotenv

from . import __version__
from .runner import ExperimentRunner


def load_environment() -> None:
    """Load environment variables from .env file crawling up to the top."""
    load_dotenv(find_dotenv())


@click.group()
@click.version_option(version=__version__, prog_name="parallamr")
def cli() -> None:
    """
    Parallamr: A command-line tool for running systematic experiments
    across multiple LLM providers and models.
    """
    load_environment()


@cli.command()
@click.option(
    "--prompt", "-p",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Primary prompt file (required)"
)
@click.option(
    "--experiments", "-e",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Experiments CSV file (required)"
)
@click.option(
    "--output", "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output CSV file (required)"
)
@click.option(
    "--context", "-c",
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Additional context files (multiple allowed)"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable detailed logging"
)
@click.option(
    "--timeout",
    type=int,
    default=300,
    help="Request timeout in seconds (default: 300)"
)
@click.option(
    "--validate-only",
    is_flag=True,
    help="Validate experiments without running them"
)
def run(
    prompt: Path,
    experiments: Path,
    output: Path,
    context: tuple[Path, ...],
    verbose: bool,
    timeout: int,
    validate_only: bool
) -> None:
    """
    Run experiments across multiple LLM providers and models.

    This command loads experiments from a CSV file and executes them
    sequentially across the specified providers and models, writing
    results to an output CSV file after each experiment completes.

    Examples:

        # Basic usage
        parallamr run -p prompt.txt -e experiments.csv -o results.csv

        # With context files
        parallamr run -p prompt.txt -c context1.txt -c context2.txt -e experiments.csv -o results.csv

        # With verbose logging
        parallamr run -p prompt.txt -e experiments.csv -o results.csv --verbose

        # Validate experiments only
        parallamr run -p prompt.txt -e experiments.csv -o results.csv --validate-only
    """
    if timeout <= 0:
        click.echo("Error: Timeout must be positive", err=True)
        sys.exit(1)

    # Create runner
    runner = ExperimentRunner(timeout=timeout, verbose=verbose)

    if validate_only:
        # Validate experiments without running them
        asyncio.run(_validate_experiments(runner, experiments))
    else:
        # Run experiments
        asyncio.run(_run_experiments(
            runner=runner,
            prompt_file=prompt,
            experiments_file=experiments,
            output_file=output,
            context_files=list(context) if context else None
        ))


async def _run_experiments(
    runner: ExperimentRunner,
    prompt_file: Path,
    experiments_file: Path,
    output_file: Path,
    context_files: Optional[List[Path]]
) -> None:
    """Run the experiments asynchronously."""
    try:
        await runner.run_experiments(
            prompt_file=prompt_file,
            experiments_file=experiments_file,
            output_file=output_file,
            context_files=context_files
        )
    except KeyboardInterrupt:
        click.echo("\nExperiment run interrupted by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error running experiments: {e}", err=True)
        sys.exit(1)


async def _validate_experiments(
    runner: ExperimentRunner,
    experiments_file: Path
) -> None:
    """Validate experiments asynchronously."""
    try:
        click.echo("Validating experiments...")
        results = await runner.validate_experiments(experiments_file)

        if not results["valid"]:
            click.echo(f"❌ Validation failed: {results['error']}", err=True)
            sys.exit(1)

        # Display validation results
        click.echo(f"✅ Validation successful: {results['experiments']} experiments found")

        for provider_name, provider_info in results["providers"].items():
            status = "✅" if provider_info["available"] else "❌"
            click.echo(f"{status} Provider '{provider_name}': {'available' if provider_info['available'] else 'not available'}")

            for model_info in provider_info["models"]:
                model_status = "✅" if model_info["available"] else "⚠️"
                click.echo(f"  {model_status} Model '{model_info['model']}': {'available' if model_info['available'] else 'may not be available'}")

        if results["warnings"]:
            click.echo("\n⚠️  Warnings:")
            for warning in results["warnings"]:
                click.echo(f"  - {warning}")

    except Exception as e:
        click.echo(f"Error validating experiments: {e}", err=True)
        sys.exit(1)


@cli.command()
def providers() -> None:
    """List available LLM providers and their configuration."""
    runner = ExperimentRunner()

    click.echo("Available providers:")
    for provider_name in runner.list_providers():
        click.echo(f"  - {provider_name}")

    click.echo("\nConfiguration:")

    # Check OpenRouter
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        click.echo("  ✅ OpenRouter: API key configured")
    else:
        click.echo("  ❌ OpenRouter: API key not found (set OPENROUTER_API_KEY)")

    # Check Ollama
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    click.echo(f"  📍 Ollama: {ollama_url}")

    # Mock provider is always available
    click.echo("  ✅ Mock: Always available (for testing)")


@cli.command()
@click.argument("provider", type=click.Choice(["openrouter", "ollama"]))
async def models(provider: str) -> None:
    """List available models for a specific provider."""
    runner = ExperimentRunner()

    if provider not in runner.providers:
        click.echo(f"Provider '{provider}' not available", err=True)
        sys.exit(1)

    try:
        click.echo(f"Fetching models for {provider}...")
        provider_instance = runner.providers[provider]
        model_list = await provider_instance.list_models()

        if model_list:
            click.echo(f"\nAvailable models for {provider}:")
            for model in sorted(model_list):
                click.echo(f"  - {model}")
        else:
            click.echo(f"No models found for {provider}")

    except Exception as e:
        click.echo(f"Error fetching models: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default="experiments.csv",
    help="Output file for example CSV (default: experiments.csv)"
)
def init(output: Path) -> None:
    """Create example experiment files to get started."""
    # Create example experiments.csv
    experiments_content = """provider,model,source,topic
mock,mock,Wikipedia,AI
openrouter,anthropic/claude-sonnet-4,Encyclopedia,Machine Learning
ollama,llama3.2,Database,Natural Language Processing"""

    output.write_text(experiments_content, encoding='utf-8')
    click.echo(f"✅ Created example experiments file: {output}")

    # Create example prompt.txt
    prompt_file = output.parent / "prompt.txt"
    prompt_content = """Summarize the following information from {{source}}:

The topic is {{topic}}.

Please provide a concise summary that covers the key concepts and applications."""

    prompt_file.write_text(prompt_content, encoding='utf-8')
    click.echo(f"✅ Created example prompt file: {prompt_file}")

    # Create example context file
    context_file = output.parent / "context.txt"
    context_content = """Additional context: This is a test of the parallamr system.
The system allows for systematic comparison of LLM responses across different providers and models."""

    context_file.write_text(context_content, encoding='utf-8')
    click.echo(f"✅ Created example context file: {context_file}")

    # Create .env.example if it doesn't exist
    env_example = output.parent / ".env.example"
    if not env_example.exists():
        env_content = """# OpenRouter API Configuration
OPENROUTER_API_KEY=sk-or-v1-your-api-key-here

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# Optional: Request timeout in seconds
# DEFAULT_TIMEOUT=300

# Optional: Debug mode
# DEBUG=false"""

        env_example.write_text(env_content, encoding='utf-8')
        click.echo(f"✅ Created environment example: {env_example}")

    click.echo("\n🚀 Getting started:")
    click.echo("1. Copy .env.example to .env and configure your API keys")
    click.echo("2. Edit prompt.txt and experiments.csv for your use case")
    click.echo("3. Run: parallamr run -p prompt.txt -e experiments.csv -o results.csv")


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()